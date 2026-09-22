"""Markdown rendering for proposal documents."""

import re

from grantscout.proposal.schemas import ProposalDocument, SectionDraft

_MARKER_PATTERN = re.compile(r"【文献(\d{1,3})】")
_CJK_PATTERN = re.compile(r"[\u3400-\u9fff]")
_LATIN_WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9+._-]*")


def proposal_word_count(text: str) -> int:
    """Count CJK characters plus Latin technical terms as one word each."""
    return len(_CJK_PATTERN.findall(text)) + len(_LATIN_WORD_PATTERN.findall(text))


def _references(document: ProposalDocument) -> tuple[list[tuple[str, str]], dict[str, dict[str, int]]]:
    """Collect global references and per-section marker-to-number mappings.

    Section drafts number their sources locally (【文献1】 per section); the
    rendered document renumbers them globally in order of first appearance.
    """
    references: list[tuple[str, str]] = []
    index_by_evidence: dict[str, int] = {}
    mappings: dict[str, dict[str, int]] = {}
    for section_key in document.sections:
        mapping: dict[str, int] = {}
        draft = document.latest_draft(section_key)
        if draft is None:
            mappings[section_key] = mapping
            continue
        for citation in draft.citations:
            if citation.evidence_id not in index_by_evidence:
                index_by_evidence[citation.evidence_id] = len(references)
                references.append((citation.evidence_id, citation.paper_title))
            mapping[citation.marker] = index_by_evidence[citation.evidence_id] + 1
        mappings[section_key] = mapping
    return references, mappings


def _renumber(content_md: str, mapping: dict[str, int]) -> str:
    """Rewrite local 【文献n】 markers using the global numbering."""

    def replace(match: re.Match[str]) -> str:
        number = mapping.get(match.group(0))
        return f"【文献{number}】" if number else match.group(0)

    return _MARKER_PATTERN.sub(replace, content_md)


def render_proposal_markdown(document: ProposalDocument) -> str:
    lines: list[str] = [f"# {document.title}", ""]
    brief = document.idea_brief
    lines.append("## 选题简报")
    lines.append(f"- 选题:{brief.topic}")
    if brief.goals:
        lines.append("- 研究目标:" + ";".join(brief.goals))
    if brief.innovation_points:
        lines.append("- 创新点:" + ";".join(brief.innovation_points))
    if brief.duration_months:
        lines.append(f"- 研究周期:{brief.duration_months} 个月")
    if brief.open_questions:
        for question in brief.open_questions:
            lines.append(f"- 待澄清:{question}")
    lines.append("")

    dossier = document.dossier
    if dossier is not None:
        lines.append("## 研究空白与科学问题")
        for gap in dossier.gaps:
            marker = "" if gap.evidence_ids else "(无证据,待人工补充)"
            lines.append(f"- 空白:{gap.statement}{marker}")
        for science_question in dossier.science_questions:
            lines.append(f"- 科学问题:{science_question.question}")
        lines.append("")

    if document.outline is not None:
        lines.append("## 写作大纲(未经人工确认)")
        for section in document.outline.sections:
            bullets = ";".join(section.bullet_points) if section.bullet_points else "(要点待补)"
            lines.append(f"- {section.title}(预算 {section.word_budget} 字):{bullets}")
        lines.append("")

    lines.append("## 正文草稿")
    references, mappings = _references(document)
    for section_key, versions in document.sections.items():
        draft: SectionDraft | None = versions[-1] if versions else None
        if draft is None:
            continue
        lines.append("")
        lines.append(f"### {draft.section_key}")
        lines.append(_renumber(draft.content_md, mappings.get(section_key, {})))
        if draft.indicators:
            lines.append("")
            lines.append("| 指标名称 | 目标值 | 测试条件 | 测试方法 | 验收材料 | 来源依据 |")
            lines.append("|---|---|---|---|---|---|")
            for indicator in draft.indicators:
                cells = [
                    indicator.name,
                    indicator.target_value,
                    indicator.test_conditions,
                    indicator.test_method,
                    indicator.acceptance_materials,
                    indicator.source_basis,
                ]
                lines.append("| " + " | ".join(cell.replace("|", "\\|") or "-" for cell in cells) + " |")
        for warning in draft.warnings:
            lines.append(f"> 警告:{warning}")

    if references:
        lines.append("")
        lines.append("## 参考文献")
        for number, (evidence_id, paper_title) in enumerate(references, start=1):
            lines.append(f"{number}. {paper_title}(证据 {evidence_id})")

    if document.review is not None:
        lines.append("")
        lines.append("## 自检结果")
        lines.append(f"- 状态:{document.review.status}")
        lines.append(f"- 引用总数:{document.review.total_citations}")
        if document.review.sections_missing_evidence:
            lines.append(
                "- 缺少证据支撑的章节:" + "、".join(document.review.sections_missing_evidence)
            )
        for warning in document.review.budget_warnings:
            lines.append(f"- {warning}")

    if document.warnings:
        lines.append("")
        lines.append("## 全局提示")
        for warning in document.warnings:
            lines.append(f"- {warning}")
    lines.append("")
    return "\n".join(lines)
