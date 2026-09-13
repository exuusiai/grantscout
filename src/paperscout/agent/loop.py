import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from paperscout.config import Settings
from paperscout.models.llm import OpenAICompatibleClient
from paperscout.models.schemas import Claim, ResearchState, StructuredFacts, ToolCall
from paperscout.reports.renderer import render_markdown
from paperscout.retrieval.store import CorpusStore
from paperscout.retrieval.semantic import SemanticIndex, SemanticIndexError
from paperscout.retrieval.reranker import CrossEncoderReranker, RerankerError
from paperscout.tools.audit import audit_citations
from paperscout.tools.comparison import compare_papers, find_contradictions
from paperscout.tools.evidence import extract_structured_facts, synthesize_claims
from paperscout.tools.search import retrieve_evidence, search_papers

logger = logging.getLogger(__name__)


class PaperScoutAgent:
    def __init__(
        self,
        store: CorpusStore,
        settings: Settings,
        decompose: bool = True,
        audit: bool = True,
        rerank: bool | None = None,
        detect_conflicts: bool = True,
        on_tool_call: Callable[[ToolCall], None] | None = None,
        semantic_index: SemanticIndex | None = None,
        reranker_instance: CrossEncoderReranker | None = None,
        output_language: str = "en",
    ) -> None:
        self.store = store
        self.settings = settings
        self.decompose_enabled = decompose
        self.audit_enabled = audit
        self.rerank_enabled = settings.use_reranker if rerank is None else rerank
        self.conflict_detection_enabled = detect_conflicts
        self.on_tool_call = on_tool_call
        self.model_client = (
            OpenAICompatibleClient(
                base_url=settings.model_base_url,
                api_key=settings.model_api_key,
                model=settings.model_name,
                timeout_seconds=settings.model_timeout_seconds,
            )
            if settings.use_model_reasoning
            else None
        )
        self.semantic_index = semantic_index
        self.reranker = reranker_instance
        self.output_language = output_language
        if settings.retrieval_mode.lower() == "semantic" and self.semantic_index is None:
            self.semantic_index = SemanticIndex(settings.vector_index_path, settings.embedding_model)
            try:
                self.semantic_index.load()
            except SemanticIndexError as error:
                logger.warning("Semantic index unavailable; falling back to lexical retrieval: %s", error)
                self.semantic_index = None
        if self.rerank_enabled and self.reranker is None:
            self.reranker = CrossEncoderReranker(settings.reranker_model)
            try:
                self.reranker._load_model()
            except RerankerError as error:
                logger.warning("Reranker unavailable; continuing without reranking: %s", error)
                self.reranker = None

    def run(self, question: str, search_query: str | None = None) -> ResearchState:
        state = ResearchState(run_id=self._run_id(), question=question)
        self._record(state, "plan_question", {"question": question}, lambda: {"status": "deterministic"})
        state.sub_questions = self._decompose(state, question)
        retrieval_questions = [search_query] if search_query else state.sub_questions

        candidate_by_id = {}
        for sub_question in retrieval_questions:
            if self._budget_exhausted(state):
                state.warnings.append("Agent budget reached before all sub-questions were searched.")
                break
            candidates = self._search(state, sub_question)
            for candidate in candidates:
                existing = candidate_by_id.get(candidate.paper.id)
                if existing is None or candidate.score > existing.score:
                    candidate_by_id[candidate.paper.id] = candidate
        state.candidate_papers = sorted(
            candidate_by_id.values(), key=lambda candidate: candidate.score, reverse=True
        )[: self.settings.max_papers]
        state.selected_papers = [candidate.paper for candidate in state.candidate_papers]
        state.budget.papers = len(state.selected_papers)
        if not state.selected_papers:
            state.warnings.append("No local papers matched the research question.")

        for paper in state.selected_papers:
            if self._budget_exhausted(state):
                state.warnings.append("Agent budget reached before all selected papers were processed.")
                break
            evidence_results = self._retrieve(state, paper.id, search_query or question)
            paper_evidence = [result.evidence for result in evidence_results]
            state.evidence_items.extend(paper_evidence)
            facts = self._extract_facts(state, paper.id, paper_evidence)
            state.facts.append(facts)
            self._append_claims(state, facts)

        if self.model_client is not None and state.evidence_items:
            synthesized = self._record(
                state,
                "model_synthesize_findings",
                {"question": question, "evidence_count": len(state.evidence_items)},
                lambda: synthesize_claims(
                    question,
                    state.evidence_items,
                    self.model_client,
                    "Chinese" if self.output_language == "zh" else "English",
                ),
            )
            if isinstance(synthesized, list) and synthesized:
                state.claims = synthesized

        state.comparison = compare_papers(
            state.facts, prefer_localized=self.output_language == "zh"
        )
        if self.conflict_detection_enabled:
            state.conflicts = find_contradictions(state.facts)
            state.warnings.extend(conflict.message for conflict in state.conflicts)
        if self.audit_enabled:
            state.citation_audit = audit_citations(state.claims, state.evidence_items)
        if state.citation_audit and state.citation_audit.unsupported_claims:
            state.warnings.append("One or more claims did not pass the evidence overlap check.")
        state.status = "completed"
        state.finished_at = datetime.now(UTC)
        self._persist(state)
        return state

    @staticmethod
    def _run_id() -> str:
        return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]

    def _decompose(self, state: ResearchState, question: str) -> list[str]:
        from paperscout.agent.planner import decompose_question

        if not self.decompose_enabled:
            return [question]
        if self.model_client is not None:
            modeled = self._record(
                state,
                "model_decompose_question",
                {"question": question},
                lambda: decompose_question(question, model_client=self.model_client),
            )
            if isinstance(modeled, list) and modeled:
                return modeled
        return decompose_question(question)

    def _extract_facts(
        self, state: ResearchState, paper_id: str, evidence_items
    ) -> StructuredFacts:
        if self.model_client is not None:
            modeled = self._record(
                state,
                "model_extract_structured_facts",
                {"paper_id": paper_id, "evidence_count": len(evidence_items)},
                lambda: extract_structured_facts(
                    self.store,
                    paper_id,
                    evidence_items,
                    model_client=self.model_client,
                    output_language="Chinese" if self.output_language == "zh" else "English",
                ),
            )
            if isinstance(modeled, StructuredFacts):
                return modeled
        extracted = self._record(
            state,
            "extract_structured_facts",
            {"paper_id": paper_id, "evidence_count": len(evidence_items)},
            lambda: extract_structured_facts(self.store, paper_id, evidence_items),
        )
        return extracted if isinstance(extracted, StructuredFacts) else StructuredFacts(paper_id=paper_id)

    def _search(self, state: ResearchState, query: str):
        return self._record(
            state,
            "search_papers",
            {"query": query, "top_k": self.settings.max_papers},
            lambda: search_papers(
                self.store, query, top_k=self.settings.max_papers, semantic_index=self.semantic_index
                , reranker=self.reranker
            ),
        )

    def _retrieve(self, state: ResearchState, paper_id: str, query: str):
        return self._record(
            state,
            "retrieve_evidence",
            {"paper_id": paper_id, "query": query, "top_k": 5},
            lambda: retrieve_evidence(
                self.store, paper_id, query, top_k=5, semantic_index=self.semantic_index
                , reranker=self.reranker
            ),
        )

    def _record(self, state: ResearchState, tool: str, input_data: dict, operation):
        started = time.perf_counter()
        try:
            output = operation()
            duration_ms = int((time.perf_counter() - started) * 1000)
            state.budget.steps += 1
            state.budget.tool_calls += 1
            tool_call = ToolCall(
                tool=tool,
                input=input_data,
                output=self._summarize(output),
                duration_ms=duration_ms,
            )
            state.tool_history.append(tool_call)
            self._notify_tool_call(tool_call)
            return output
        except Exception as error:
            duration_ms = int((time.perf_counter() - started) * 1000)
            state.budget.steps += 1
            state.budget.tool_calls += 1
            tool_call = ToolCall(
                tool=tool,
                input=input_data,
                output={"error": str(error)},
                status="error",
                duration_ms=duration_ms,
                retryable=False,
            )
            state.tool_history.append(tool_call)
            self._notify_tool_call(tool_call)
            state.warnings.append(f"Tool {tool} failed: {error}")
            return []

    def _append_claims(self, state: ResearchState, facts) -> None:
        for fact in facts.conclusions[:5]:
            claim_id = f"claim:{len(state.claims):04d}"
            state.claims.append(
                Claim(
                    id=claim_id,
                    text=fact.text,
                    localized_text=fact.localized_text,
                    evidence_ids=[fact.evidence_id],
                    confidence=0.0,
                )
            )
        if not facts.conclusions and facts.methods:
            fact = facts.methods[0]
            state.claims.append(
                Claim(
                    id=f"claim:{len(state.claims):04d}",
                    text=fact.text,
                    localized_text=fact.localized_text,
                    evidence_ids=[fact.evidence_id],
                    confidence=0.0,
                )
            )

    def _budget_exhausted(self, state: ResearchState) -> bool:
        return state.budget.steps >= self.settings.max_steps or state.budget.tool_calls >= self.settings.max_tool_calls

    @staticmethod
    def _summarize(value) -> dict:
        if isinstance(value, list):
            return {
                "count": len(value),
                "items": [
                    item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                    for item in value[:10]
                ],
            }
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return value
        return {"value": str(value)}

    def _notify_tool_call(self, tool_call: ToolCall) -> None:
        if self.on_tool_call is None:
            return
        try:
            self.on_tool_call(tool_call)
        except Exception as error:
            logger.warning("Tool-call observer failed: %s", error)

    def _persist(self, state: ResearchState) -> None:
        run_dir = Path(self.settings.runs_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        state_path = run_dir / f"{state.run_id}.json"
        trajectory_path = run_dir / f"{state.run_id}.jsonl"
        report_path = run_dir / f"{state.run_id}.md"
        html_report_path = run_dir / f"{state.run_id}.html"
        state_path.write_text(json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
        trajectory_path.write_text(
            "".join(json.dumps(item.model_dump(mode="json"), ensure_ascii=False) + "\n" for item in state.tool_history),
            encoding="utf-8",
        )
        report_path.write_text(render_markdown(state), encoding="utf-8")
        from paperscout.reports.renderer import render_html

        html_report_path.write_text(render_html(state), encoding="utf-8")
