"""Seven-stage proposal drafting pipeline.

Each stage is recorded as a ``ToolCall`` so the whole run stays replayable.
Model-backed stages degrade to deterministic fallbacks instead of inventing
content: gaps without evidence become placeholders, sections without sources
become material compilations, and every degradation is kept in warnings.
"""

import json
import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from grantscout.agent.loop import GrantScoutAgent
from grantscout.config import Settings
from grantscout.models.llm import ModelClientError, OpenAICompatibleClient, strip_thinking
from grantscout.models.schemas import EvidenceItem, ToolCall
from grantscout.proposal.renderer import proposal_word_count, render_proposal_markdown
from grantscout.proposal.schemas import (
    DraftCitation,
    GapStatement,
    IdeaBrief,
    IndicatorItem,
    LiteratureDossier,
    Outline,
    OutlineSection,
    ProposalDocument,
    ProposalRequest,
    ProposalReview,
    ScienceQuestion,
    SectionDraft,
)
from grantscout.proposal.templates import get_template
from grantscout.retrieval.store import CorpusStore

logger = logging.getLogger(__name__)

_MARKER_PATTERN = re.compile(r"【文献(\d{1,3})】")
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?。！？])\s+")


class ProposalPipelineError(RuntimeError):
    """Raised when the proposal cannot be produced from the given corpus."""


def _build_model_client(settings: Settings) -> OpenAICompatibleClient | None:
    """Mirror the agent's model routing: research endpoint when configured."""
    if not settings.use_model_reasoning:
        return None
    base_url = settings.research_model_base_url or settings.model_base_url
    return OpenAICompatibleClient(
        base_url=base_url,
        api_key=(
            settings.research_model_api_key if settings.research_model_base_url else settings.model_api_key
        ),
        model=settings.research_model_name or settings.model_name,
        timeout_seconds=settings.model_timeout_seconds,
        allow_research_endpoint=bool(settings.research_model_base_url),
    )


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in _SENTENCE_PATTERN.split(text) if sentence.strip()]


class ProposalPipeline:
    def __init__(
        self,
        store: CorpusStore,
        settings: Settings,
        template_id: str = "project-plan-v1",
        output_language: str = "zh",
        on_tool_call=None,
    ) -> None:
        self.store = store
        self.settings = settings
        self.template = get_template(template_id)
        self.output_language = output_language
        self.on_tool_call = on_tool_call
        self.model_client = _build_model_client(settings)

    def run(self, request: ProposalRequest) -> ProposalDocument:
        if request.template_id != self.template.id:
            self.template = get_template(request.template_id)
        document = ProposalDocument(
            id=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8],
            title=request.topic.strip()[:120],
            template_id=self.template.id,
            locale=request.locale,
            corpus_path=str(self.store.path),
            idea_brief=IdeaBrief(
                topic=request.topic.strip(),
                goals=list(request.goals),
                duration_months=request.duration_months,
                constraints=request.constraints,
            ),
        )
        try:
            self._clarify(document)
            self._build_dossier(document)
            self._build_outline(document)
            for spec in self.template.sections:
                document.add_draft(self._draft_section(document, spec.key))
            document.review = self._self_check(document)
            document.status = "completed"
        except Exception as error:
            document.status = "failed"
            document.warnings.append(f"Pipeline failed: {error}")
            logger.exception("Proposal pipeline failed")
            raise ProposalPipelineError(str(error)) from error
        finally:
            self._persist(document)
        return document

    def redraft_section(self, document: ProposalDocument, section_key: str) -> ProposalDocument:
        """Regenerate one section as a new version (human-gated workflow)."""
        if self.template.id != document.template_id:
            self.template = get_template(document.template_id)
        spec = next((item for item in self.template.sections if item.key == section_key), None)
        if spec is None or not spec.generatable:
            raise ValueError(f"Section is not generatable: {section_key}")
        if document.outline is None:
            self._build_outline(document)
        document.add_draft(self._draft_section(document, section_key))
        document.review = self._self_check(document)
        self._persist(document)
        return document

    def _record(self, document: ProposalDocument, tool: str, input_data: dict, operation):
        started = time.perf_counter()
        try:
            output = operation()
            status = "ok"
        except Exception as error:
            output = {"error": str(error)}
            status = "error"
            document.warnings.append(f"Stage {tool} failed: {error}")
            logger.warning("Proposal stage %s failed: %s", tool, error)
        duration_ms = int((time.perf_counter() - started) * 1000)
        tool_call = ToolCall(
            tool=tool,
            input=input_data,
            output=self._summarize(output) if status == "ok" else {"error": str(output["error"])},
            status=status,
            duration_ms=duration_ms,
        )
        document.tool_history.append(tool_call)
        self._notify(tool_call)
        return output if status == "ok" else None

    @staticmethod
    def _summarize(value):
        if isinstance(value, list):
            return {"count": len(value)}
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return value
        return {"value": str(value)}

    def _notify(self, tool_call: ToolCall) -> None:
        if self.on_tool_call is None:
            return
        try:
            self.on_tool_call(tool_call)
        except Exception as error:
            logger.warning("Proposal tool-call observer failed: %s", error)

    # Stage 1 ---------------------------------------------------------------
    def _clarify(self, document: ProposalDocument) -> None:
        def operation() -> dict:
            brief = document.idea_brief
            questions: list[str] = []
            if not brief.goals:
                questions.append("本项目的可量化研究目标是什么?")
            if brief.duration_months is None:
                questions.append("研究周期是多少个月?")
            if len(brief.topic) < 12:
                questions.append("选题的研究对象与边界是什么?")
            brief.open_questions = questions
            return {"open_questions": questions}

        self._record(document, "clarify_topic", {"topic": document.idea_brief.topic}, operation)

    # Stage 2 ---------------------------------------------------------------
    def _build_dossier(self, document: ProposalDocument) -> None:
        def operation() -> dict:
            dossier_settings = self.settings.model_copy(update={"max_papers": min(self.settings.max_papers, 10)})
            agent = GrantScoutAgent(
                self.store,
                dossier_settings,
                decompose=False,
                output_language=self.output_language,
            )
            state = agent.run(document.idea_brief.topic)
            if state.status != "completed":
                raise RuntimeError(f" Literature research failed: {state.warnings}")
            document.warnings.extend(state.warnings)
            dossier = LiteratureDossier(topic=document.idea_brief.topic, state=state)
            dossier.gaps = self._extract_gaps(document, state)
            dossier.science_questions = self._derive_questions(dossier.gaps)
            document.dossier = dossier
            return {
                "papers": len(state.selected_papers),
                "evidence": len(state.evidence_items),
                "gaps": len(dossier.gaps),
            }

        self._record(document, "literature_dossier", {"topic": document.idea_brief.topic}, operation)
        if document.dossier is None:
            raise ProposalPipelineError("Literature dossier stage produced no dossier")

    def _extract_gaps(self, document: ProposalDocument, state) -> list[GapStatement]:
        gaps: list[GapStatement] = []
        if self.model_client is not None:
            try:
                gaps = self._gaps_with_model(state)
            except ModelClientError as error:
                document.warnings.append(f"模型研究空白提炼失败,已降级为事实改写:{error}")
        if not gaps:
            gaps = self._gaps_from_limitations(state)
        if not gaps:
            gaps = [
                GapStatement(
                    id="gap:0001",
                    statement="证据库中未检索到足够的局限性证据,研究空白需人工补充。",
                    evidence_ids=[],
                    confidence=0.0,
                )
            ]
        return gaps[:4]

    def _gaps_with_model(self, state) -> list[GapStatement]:
        evidence_pool = state.evidence_items[:40]
        payload = self.model_client.chat_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research-gap analyst for project proposals. From the supplied "
                        "evidence only, state 2-4 research gaps that existing work has not solved. "
                        "Each gap must cite evidence_ids copied exactly from the input; never invent "
                        "evidence or facts. Return a JSON object {\"gaps\": [{\"statement\": str, "
                        "\"evidence_ids\": [str]}]}."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "topic": state.question,
                            "evidence": [
                                {"evidence_id": item.id, "text": item.text} for item in evidence_pool
                            ],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            max_tokens=1024,
            temperature=0.0,
        )
        known = {item.id for item in evidence_pool}
        gaps: list[GapStatement] = []
        for value in payload.get("gaps", []) if isinstance(payload, dict) else []:
            if not isinstance(value, dict):
                continue
            statement = value.get("statement")
            evidence_ids = [item for item in value.get("evidence_ids", []) if item in known]
            if not isinstance(statement, str) or not statement.strip() or not evidence_ids:
                continue
            gaps.append(
                GapStatement(
                    id=f"gap:{len(gaps):04d}",
                    statement=statement.strip(),
                    evidence_ids=evidence_ids,
                    confidence=0.8,
                )
            )
        if not gaps:
            raise ModelClientError("Gap extraction returned no evidence-backed gaps")
        return gaps

    def _gaps_from_limitations(self, state) -> list[GapStatement]:
        gaps: list[GapStatement] = []
        for facts in state.facts:
            for fact in facts.limitations:
                prefix = "现有工作的局限:" if self.output_language == "zh" else "Reported limitation: "
                gaps.append(
                    GapStatement(
                        id=f"gap:{len(gaps):04d}",
                        statement=f"{prefix}{fact.text}",
                        evidence_ids=[fact.evidence_id],
                        confidence=0.3,
                    )
                )
                if len(gaps) >= 4:
                    return gaps
        return gaps

    def _derive_questions(self, gaps: list[GapStatement]) -> list[ScienceQuestion]:
        questions: list[ScienceQuestion] = []
        for gap in gaps:
            if not gap.evidence_ids:
                continue
            template = "如何克服以下研究空白:{}" if self.output_language == "zh" else "How can we address: {}"
            questions.append(
                ScienceQuestion(
                    id=f"sq:{len(questions):04d}",
                    question=template.format(gap.statement),
                    gap_ids=[gap.id],
                    evidence_ids=list(gap.evidence_ids),
                )
            )
        return questions

    # Stage 3 ---------------------------------------------------------------
    def _build_outline(self, document: ProposalDocument) -> None:
        def operation() -> dict:
            outline = Outline(template_id=self.template.id)
            for spec in self.template.sections:
                outline.sections.append(
                    OutlineSection(
                        key=spec.key,
                        title=spec.title,
                        word_budget=spec.max_words or spec.min_words,
                    )
                )
            if self.model_client is not None and document.dossier is not None:
                try:
                    self._outline_bullets_with_model(document, outline)
                except ModelClientError as error:
                    document.warnings.append(f"模型大纲要点生成失败,已用证据回填:{error}")
            for section in outline.sections:
                if not section.bullet_points:
                    self._outline_bullets_from_evidence(document, section)
            document.outline = outline
            return {"sections": len(outline.sections)}

        self._record(document, "build_outline", {"template": self.template.id}, operation)

    def _outline_bullets_with_model(self, document: ProposalDocument, outline: Outline) -> None:
        dossier = document.dossier
        assert dossier is not None
        evidence_pool = dossier.state.evidence_items[:30]
        payload = self.model_client.chat_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are drafting the outline of a project proposal. For each requested "
                        "section produce 3-5 bullet points based only on the supplied evidence and "
                        "topic. Return JSON {\"sections\": [{\"key\": str, \"bullet_points\": [str], "
                        "\"evidence_ids\": [str]}]} with keys copied exactly from the input."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "topic": document.idea_brief.topic,
                            "sections": [
                                {"key": spec.key, "title": spec.title, "guidance": spec.guidance}
                                for spec in self.template.sections
                            ],
                            "gaps": [gap.statement for gap in dossier.gaps],
                            "evidence": [
                                {"evidence_id": item.id, "text": item.text} for item in evidence_pool
                            ],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            max_tokens=1200,
            temperature=0.1,
        )
        known = {item.id for item in evidence_pool}
        by_key = {value.get("key"): value for value in payload.get("sections", []) if isinstance(value, dict)}
        for section in outline.sections:
            value = by_key.get(section.key, {})
            bullets = value.get("bullet_points")
            if isinstance(bullets, list):
                section.bullet_points = [item for item in bullets if isinstance(item, str) and item.strip()][:5]
            evidence_ids = value.get("evidence_ids")
            if isinstance(evidence_ids, list):
                section.evidence_ids = [item for item in evidence_ids if item in known]

    def _outline_bullets_from_evidence(self, document: ProposalDocument, section: OutlineSection) -> None:
        query = f"{document.idea_brief.topic} {section.title}"
        bullets: list[str] = []
        for result in self.store.search(query, top_k=5):
            sentence = next(iter(_sentences(result.evidence.text)), result.evidence.text[:120])
            if sentence not in bullets:
                bullets.append(sentence)
                section.evidence_ids.append(result.evidence.id)
            if len(bullets) >= 3:
                break
        section.bullet_points = bullets

    # Stage 4 ---------------------------------------------------------------
    def _draft_section(self, document: ProposalDocument, section_key: str) -> SectionDraft:
        spec = next((item for item in self.template.sections if item.key == section_key), None)
        if spec is None:
            raise ProposalPipelineError(f"Template section missing: {section_key}")
        if not spec.generatable:
            draft = SectionDraft(
                section_key=section_key,
                content_md="【本节为表格/签章/附件项,由人工填写,系统不自动生成。】",
            )
            draft.warnings.append("manual-only section; skipped by design")
            return draft
        outline_bullets: list[str] = []
        if document.outline is not None:
            outline_section = next(
                (item for item in document.outline.sections if item.key == spec.key), None
            )
            if outline_section is not None:
                outline_bullets = outline_section.bullet_points
        sources = self._section_sources(document, section_key)
        draft = SectionDraft(section_key=section_key, content_md="")
        if not sources:
            draft.content_md = "【待补充】未检索到可用资料,本节需要人工撰写或扩充文献库。"
            draft.warnings.append("no evidence retrieved for this section")
            draft.word_count = proposal_word_count(draft.content_md)
            return draft
        if self.model_client is not None:
            try:
                self._draft_with_model(document, spec, outline_bullets, sources, draft)
            except ModelClientError as error:
                draft.warnings.append(f"模型撰写失败,已降级为素材整理稿:{error}")
        if not draft.content_md:
            self._draft_material(document, spec, sources, draft)
        if self.model_client is not None and draft.source == "agent":
            draft = self._expand_to_budget(document, spec, outline_bullets, sources, draft)
        draft.word_count = proposal_word_count(draft.content_md)
        if spec.max_words and draft.word_count > spec.max_words:
            draft.warnings.append(
                f"字数 {draft.word_count} 超出预算 {spec.max_words},需要删减。"
            )
        return draft

    def _expand_to_budget(
        self,
        document: ProposalDocument,
        spec,
        outline_bullets: list[str],
        sources: list[EvidenceItem],
        draft: SectionDraft,
    ) -> SectionDraft:
        """Run bounded expansion passes until the section approaches its budget."""
        target = spec.min_words or int((spec.max_words or 0) * 0.6)
        if not target or draft.word_count >= target:
            return draft
        passes = self.settings.proposal_expansion_passes
        while passes > 0 and draft.word_count < target:
            passes -= 1
            previous = draft.word_count
            try:
                expanded = self._expand_with_model(document, spec, outline_bullets, sources, draft, target)
            except ModelClientError as error:
                draft.warnings.append(f"扩写失败,保留当前稿:{error}")
                break
            if proposal_word_count(expanded) <= int(previous * 1.05):
                break
            draft.content_md = self._sanitize(
                strip_thinking(expanded),
                sources,
                document,
                draft,
                structured=spec.structured_indicators,
            )
            if spec.structured_indicators:
                self._register_indicator_citations(document, sources, draft)
            draft.word_count = proposal_word_count(draft.content_md)
        return draft

    def _expand_with_model(
        self,
        document: ProposalDocument,
        spec,
        outline_bullets: list[str],
        sources: list[EvidenceItem],
        draft: SectionDraft,
        target: int,
    ) -> str:
        locale_note = "用中文撰写" if self.output_language == "zh" else "Write in English"
        return self.model_client.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are expanding one section ({spec.title}) of a project proposal. "
                        f"{locale_note}。在现有正文基础上扩写到约 {target} 字:展开论证、补充细节与"
                        "过渡,不得引入资料之外的新事实,不得编造引用;保留并复用现有【文献n】标记,"
                        "新引用同样只能使用资料编号。直接输出完整的 markdown 正文(整体替换原稿)。"
                        f"写作要点:{spec.guidance}"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            **self._section_user_payload(document, spec, outline_bullets, sources),
                            "current_draft": draft.content_md,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            max_tokens=max(self.settings.model_max_tokens, 2048),
            temperature=0.4,
        ).content

    def _section_sources(self, document: ProposalDocument, section_key: str) -> list[EvidenceItem]:
        spec = next(item for item in self.template.sections if item.key == section_key)
        query = f"{document.idea_brief.topic} {spec.title}"
        seen: set[str] = set()
        sources: list[EvidenceItem] = []
        for result in self.store.search(query, top_k=12):
            if result.evidence.id in seen:
                continue
            seen.add(result.evidence.id)
            sources.append(result.evidence)
            if len(sources) >= 8:
                break
        if not sources:
            # CJK 词法检索在 unicode61 分词下偏弱;trigram 句子索引按子串兜底。
            from grantscout.retrieval.sentences import SentenceIndex

            for hit in SentenceIndex(self.store).continue_fragment(query, top_k=8):
                sources.append(hit.evidence)
        return sources

    def _draft_with_model(
        self,
        document: ProposalDocument,
        spec,
        outline_bullets: list[str],
        sources: list[EvidenceItem],
        draft: SectionDraft,
    ) -> None:
        if spec.structured_indicators:
            self._draft_indicators_with_model(document, spec, outline_bullets, sources, draft)
        else:
            content = self._section_chat(document, spec, outline_bullets, sources, guidance_note=(
                "直接输出 markdown 正文,不要输出节标题。"
            ))
            draft.content_md = self._sanitize(strip_thinking(content), sources, document, draft)
        draft.source = "agent"

    def _section_user_payload(
        self,
        document: ProposalDocument,
        spec,
        outline_bullets: list[str],
        sources: list[EvidenceItem],
    ) -> dict:
        return {
            "topic": document.idea_brief.topic,
            "outline": outline_bullets,
            "sources": [
                {
                    "marker": f"【文献{index}】",
                    "paper": self._paper_title(document, item.paper_id),
                    "text": item.text,
                }
                for index, item in enumerate(sources, start=1)
            ],
        }

    def _section_chat(
        self,
        document: ProposalDocument,
        spec,
        outline_bullets: list[str],
        sources: list[EvidenceItem],
        guidance_note: str,
        temperature: float = 0.3,
        json_mode: bool = False,
    ):
        locale_note = "用中文撰写" if self.output_language == "zh" else "Write in English"
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are drafting one section ({spec.title}) of a project proposal. "
                    f"{locale_note}。只能依据提供的资料撰写;每个使用资料的句子末尾标注"
                    "【文献n】(n 为资料编号);资料未覆盖的要点写【待补充】;不要编造引用、"
                    f"数据或文献。写作要点:{spec.guidance}{guidance_note}"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    self._section_user_payload(document, spec, outline_bullets, sources),
                    ensure_ascii=False,
                ),
            },
        ]
        if json_mode:
            return self.model_client.chat_json(messages=messages, max_tokens=max(self.settings.model_max_tokens, 2048), temperature=temperature)
        return self.model_client.chat(messages=messages, max_tokens=max(self.settings.model_max_tokens, 2048), temperature=temperature).content

    def _draft_indicators_with_model(
        self,
        document: ProposalDocument,
        spec,
        outline_bullets: list[str],
        sources: list[EvidenceItem],
        draft: SectionDraft,
    ) -> None:
        payload = self._section_chat(
            document,
            spec,
            outline_bullets,
            sources,
            guidance_note=(
                '返回 JSON:{"content_md": str, "indicators": [{"name": str, "target_value": str, '
                '"test_conditions": str, "test_method": str, "acceptance_materials": str, '
                '"source_basis": str}]}。indicators 须覆盖全部考核指标,六个字段分别是指标名称、'
                '目标值、测试条件、测试方法、验收材料、来源依据;资料无法支撑的字段填"待补充"。'
            ),
            temperature=0.1,
            json_mode=True,
        )
        content = payload.get("content_md") if isinstance(payload, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ModelClientError("Indicator drafting returned no content_md")
        draft.content_md = self._sanitize(strip_thinking(content), sources, document, draft, structured=True)
        raw_indicators = payload.get("indicators", []) if isinstance(payload, dict) else []
        for value in raw_indicators:
            if not isinstance(value, dict):
                continue
            name = value.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            fields = {
                field: str(value.get(field) or "").strip()[:500]
                for field in ("target_value", "test_conditions", "test_method", "acceptance_materials", "source_basis")
            }
            draft.indicators.append(IndicatorItem(name=name.strip()[:200], **fields))
            if len(draft.indicators) >= 15:
                break
        self._register_indicator_citations(document, sources, draft)

    def _register_indicator_citations(
        self, document: ProposalDocument, sources: list[EvidenceItem], draft: SectionDraft
    ) -> None:
        """Indicator tables carry citations in 来源依据; register them too."""
        known = {citation.marker for citation in draft.citations}
        for indicator in draft.indicators:
            for match in _MARKER_PATTERN.finditer(indicator.source_basis):
                marker = match.group(0)
                index = int(match.group(1))
                if marker in known or not 1 <= index <= len(sources):
                    continue
                evidence = sources[index - 1]
                draft.citations.append(
                    DraftCitation(
                        marker=marker,
                        evidence_id=evidence.id,
                        paper_title=self._paper_title(document, evidence.paper_id),
                        page=evidence.page,
                    )
                )
                known.add(marker)

    def _paper_title(self, document: ProposalDocument, paper_id: str) -> str:
        dossier = document.dossier
        if dossier is not None:
            for paper in dossier.state.selected_papers:
                if paper.id == paper_id:
                    return paper.title
        paper = self.store.get_paper(paper_id)
        return paper.title if paper else paper_id

    def _sanitize(
        self,
        content: str,
        sources: list[EvidenceItem],
        document: ProposalDocument,
        draft: SectionDraft,
        structured: bool = False,
    ) -> str:
        """Validate citation markers; drop invalid ones and record valid citations."""

        def replace(match: re.Match[str]) -> str:
            index = int(match.group(1))
            if 1 <= index <= len(sources):
                return match.group(0)
            draft.warnings.append(f"剔除非法引用标记 {match.group(0)}(超出资料范围)。")
            return ""

        cleaned = _MARKER_PATTERN.sub(replace, content).strip()
        draft.citations = []
        used = {match.group(0) for match in _MARKER_PATTERN.finditer(cleaned)}
        known_titles = {
            item.id: self._paper_title(document, item.paper_id) for item in sources
        }
        for marker in sorted(used):
            index = int(marker[3:-1])
            evidence = sources[index - 1]
            draft.citations.append(
                DraftCitation(
                    marker=marker,
                    evidence_id=evidence.id,
                    paper_title=known_titles[evidence.id],
                    page=evidence.page,
                )
            )
        if not draft.citations and sources and not structured:
            draft.warnings.append("模型正文未包含任何引用标记,请人工核对。")
        return cleaned

    def _draft_material(
        self, document: ProposalDocument, spec, sources: list[EvidenceItem], draft: SectionDraft
    ) -> None:
        """Deterministic fallback: a clearly-labelled evidence compilation."""
        lines = ["> 素材整理稿(模型未启用或调用失败),请基于以下资料人工改写:", ""]
        for index, item in enumerate(sources, start=1):
            sentence = next(iter(_sentences(item.text)), item.text[:160])
            lines.append(f"- 【文献{index}】{sentence}")
            draft.citations.append(
                DraftCitation(
                    marker=f"【文献{index}】",
                    evidence_id=item.id,
                    paper_title=self._paper_title(document, item.paper_id),
                    page=item.page,
                )
            )
        draft.content_md = "\n".join(lines)
        draft.source = "material"
        draft.warnings.append("本节为素材整理稿,非最终行文。")

    # Stage 5 ---------------------------------------------------------------
    def _self_check(self, document: ProposalDocument) -> ProposalReview:
        def operation() -> dict:
            review = ProposalReview()
            evidence_ids: set[str] = set()
            if document.dossier is not None:
                evidence_ids = {item.id for item in document.dossier.state.evidence_items}
            for spec in self.template.sections:
                draft = document.latest_draft(spec.key)
                if draft is None:
                    if spec.generatable:
                        review.sections_missing_evidence.append(spec.key)
                    continue
                review.word_counts[spec.key] = draft.word_count
                review.total_citations += len(draft.citations)
                review.unverified_citations += sum(
                    1 for citation in draft.citations if evidence_ids and citation.evidence_id not in evidence_ids
                )
                if spec.generatable and spec.requires_citations and not draft.citations:
                    review.sections_missing_evidence.append(spec.key)
                if spec.generatable and spec.min_words and draft.word_count < spec.min_words:
                    review.budget_warnings.append(
                        f"{spec.title}:字数 {draft.word_count} 低于预算下限 {spec.min_words}。"
                    )
            if review.unverified_citations:
                review.status = "failed"
            elif review.sections_missing_evidence or review.budget_warnings:
                review.status = "warning"
            else:
                review.status = "passed"
            document.warnings.extend(review.budget_warnings)
            return review

        result = self._record(document, "self_check", {"sections": len(self.template.sections)}, operation)
        return result if isinstance(result, ProposalReview) else ProposalReview(status="failed")

    # Persistence -----------------------------------------------------------
    def _persist(self, document: ProposalDocument) -> None:
        runs_dir = Path(self.settings.runs_dir)
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / f"{document.id}.proposal.json").write_text(
            json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (runs_dir / f"{document.id}.proposal.md").write_text(
            render_proposal_markdown(document), encoding="utf-8"
        )
        try:
            from grantscout.proposal.exporter import export_proposal_docx

            export_proposal_docx(document, runs_dir / f"{document.id}.proposal.docx")
        except ImportError:
            logger.warning("python-docx unavailable; docx export skipped for %s", document.id)
