from __future__ import annotations

from datetime import UTC
from html import escape
from typing import Iterable

from paperscout.models.schemas import EvidenceItem, ResearchState


def _duration_seconds(state: ResearchState) -> float | None:
    if state.finished_at is None:
        return None
    return round((state.finished_at - state.started_at).total_seconds(), 3)


def _citations_markdown(evidence_ids: Iterable[str]) -> str:
    return ", ".join(f"`{evidence_id}`" for evidence_id in evidence_ids) or "no evidence"


def _citations_html(evidence_ids: Iterable[str]) -> str:
    links = []
    for evidence_id in evidence_ids:
        safe_id = escape(evidence_id, quote=True)
        links.append(f'<a href="#evidence-{safe_id}"><code>{safe_id}</code></a>')
    return ", ".join(links) or "no evidence"


def _fact_lines(state: ResearchState, field: str) -> list[str]:
    lines = []
    for facts in state.facts:
        for fact in getattr(facts, field):
            citation = _citations_markdown([fact.evidence_id])
            lines.append(f"- `{facts.paper_id}`: {fact.text} [{citation}]")
    return lines


def _comparison_markdown(state: ResearchState, fields: tuple[str, ...]) -> list[str]:
    lines: list[str] = []
    for row in state.comparison:
        lines.append(f"### {row['paper_id']}")
        for field in fields:
            values = row.get(field) or ["Not extracted"]
            label = field.replace("_", " ").title()
            lines.append(f"- {label}: {' | '.join(str(value) for value in values)}")
    return lines or ["- No structured facts were extracted."]


def render_markdown(state: ResearchState) -> str:
    """Render a traceable, deterministic research report from one agent state."""
    duration = _duration_seconds(state)
    lines = [f"# PaperScout Report: {state.question}", "", "## Executive Summary", ""]
    model_calls = sum(call.tool.startswith("model_") for call in state.tool_history)
    lines.append(
        "- System synthesis: "
        f"selected {len(state.selected_papers)} paper(s), retrieved {len(state.evidence_items)} "
        f"evidence item(s), and produced {len(state.claims)} traceable claim(s)."
    )
    lines.append(
        "- Model-assisted calls: "
        f"{model_calls}; findings remain bounded by retrieved evidence IDs and citation audit."
    )
    lines.extend(["", "## Research Question Decomposition", ""])
    lines.extend(
        f"- {question}" for question in state.sub_questions or ["No sub-question generated."]
    )
    lines.extend(["", "## Selected Papers", ""])
    if state.selected_papers:
        candidate_scores = {candidate.paper.id: candidate for candidate in state.candidate_papers}
        for paper in state.selected_papers:
            candidate = candidate_scores.get(paper.id)
            score_text = (
                f" - score {candidate.score:.3f}; {candidate.relevance_reason}"
                if candidate
                else ""
            )
            year_text = f" ({paper.year})" if paper.year else ""
            lines.append(f"- `{paper.id}`: {paper.title}{year_text}{score_text}")
    else:
        lines.append("- No local paper matched.")
    lines.extend(["", "## Main Findings", ""])
    if state.claims:
        for claim in state.claims:
            citations = _citations_markdown(claim.evidence_ids)
            lines.append(f"- {claim.text} [{claim.support_status}; {citations}]")
    else:
        lines.append("- No traceable claims were generated.")
    lines.extend(["", "## Method Comparison", ""])
    lines.extend(_comparison_markdown(state, ("methods", "conclusions", "limitations")))
    lines.extend(["", "## Dataset and Experimental Setup Comparison", ""])
    lines.extend(_comparison_markdown(state, ("datasets", "experimental_settings", "metrics")))
    lines.extend(["", "## Limitations", ""])
    lines.extend(_fact_lines(state, "limitations") or ["- No limitations were extracted from retrieved evidence."])
    lines.extend(["", "## Conflicting Evidence", ""])
    if state.conflicts:
        for conflict in state.conflicts:
            citations = _citations_markdown(
                [*conflict.positive_evidence_ids, *conflict.negative_evidence_ids]
            )
            lines.append(f"- {conflict.message} [{citations}]")
    else:
        lines.append("- No cross-paper conflict was detected in the retrieved evidence.")
    lines.extend(["", "## Open Questions", ""])
    if state.warnings:
        lines.extend(f"- {warning}" for warning in state.warnings)
    else:
        lines.append("- Which experimental conditions would change these evidence-backed findings?")
    lines.extend(["", "## Evidence Table", ""])
    if state.evidence_items:
        lines.append("| Evidence ID | Paper | Page | Text |")
        lines.append("| --- | --- | --- | --- |")
        papers = {paper.id: paper.title for paper in state.selected_papers}
        for item in state.evidence_items:
            text = item.text.replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| `{item.id}` | {papers.get(item.paper_id, item.paper_id)} | "
                f"{item.page or '-'} | {text} |"
            )
    else:
        lines.append("- No evidence was retrieved.")
    lines.extend(["", "## Citation Audit", ""])
    if state.citation_audit:
        audit = state.citation_audit
        lines.append(
            f"Status: **{audit.status}**; supported claims: {audit.supported_claims}; "
            f"unsupported claims: {audit.unsupported_claims}."
        )
        for item in audit.items:
            status = "supported" if item.supported else "needs review"
            lines.append(f"- `{item.claim_id}`: {status} - {item.reason}")
    else:
        lines.append("Citation audit was not run.")
    lines.extend(["", "## Run Metadata", ""])
    lines.append(f"- Run ID: `{state.run_id}`")
    lines.append(f"- Status: `{state.status}`")
    lines.append(f"- Started at: {state.started_at.astimezone(UTC).isoformat()}")
    if state.finished_at:
        lines.append(f"- Finished at: {state.finished_at.astimezone(UTC).isoformat()}")
    lines.append(f"- Duration seconds: {duration if duration is not None else 'in progress'}")
    lines.append(f"- Tool calls: {len(state.tool_history)}")
    lines.append(f"- Failed tool calls: {sum(call.status == 'error' for call in state.tool_history)}")
    lines.append(f"- Model calls: {model_calls}")
    return "\n".join(lines) + "\n"


def _render_html_english(state: ResearchState) -> str:
    """Render a self-contained HTML report with evidence anchors and citations."""
    duration = _duration_seconds(state)
    model_calls = sum(call.tool.startswith("model_") for call in state.tool_history)
    papers = {paper.id: paper.title for paper in state.selected_papers}
    selected_papers = "".join(
        f"<li><code>{escape(paper.id)}</code>: {escape(paper.title)}"
        f"{f' ({paper.year})' if paper.year else ''}</li>"
        for paper in state.selected_papers
    ) or "<li>No local paper matched.</li>"
    findings = "".join(
        "<li>"
        f"{escape(claim.text)} [{escape(claim.support_status)}; "
        f"{_citations_html(claim.evidence_ids)}]"
        "</li>"
        for claim in state.claims
    ) or "<li>No traceable claims were generated.</li>"
    evidence_rows = "".join(_evidence_row(item, papers) for item in state.evidence_items)
    conflicts = "".join(
        f"<li>{escape(conflict.message)} ["
        f"{_citations_html([*conflict.positive_evidence_ids, *conflict.negative_evidence_ids])}]</li>"
        for conflict in state.conflicts
    ) or "<li>No cross-paper conflict was detected in the retrieved evidence.</li>"
    limitations = _facts_html(state, "limitations") or (
        "<li>No limitations were extracted from retrieved evidence.</li>"
    )
    audit = state.citation_audit
    audit_items = "".join(
        f"<li><code>{escape(item.claim_id)}</code>: "
        f"{'supported' if item.supported else 'needs review'} - {escape(item.reason)}</li>"
        for item in (audit.items if audit else [])
    ) or "<li>Citation audit was not run.</li>"
    open_questions = "".join(f"<li>{escape(warning)}</li>" for warning in state.warnings) or (
        "<li>Which experimental conditions would change these evidence-backed findings?</li>"
    )
    audit_status = escape(audit.status) if audit else "disabled"
    decomposed_questions = "".join(
        f"<li>{escape(question)}</li>" for question in state.sub_questions
    ) or "<li>No sub-question generated.</li>"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>PaperScout Report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#172033}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #cbd5e1;padding:8px;text-align:left;vertical-align:top}}th{{background:#e2e8f0}}code{{overflow-wrap:anywhere}}a{{color:#075985}}</style>
</head><body>
<h1>PaperScout Report: {escape(state.question)}</h1>
<h2>Executive Summary</h2><ul><li>System synthesis: selected {len(state.selected_papers)} paper(s), retrieved {len(state.evidence_items)} evidence item(s), and produced {len(state.claims)} traceable claim(s).</li><li>Model-assisted calls: {model_calls}; findings remain bounded by retrieved evidence IDs and citation audit.</li></ul>
<h2>Research Question Decomposition</h2><ul>{decomposed_questions}</ul>
<h2>Selected Papers</h2><ul>{selected_papers}</ul>
<h2>Main Findings</h2><ul>{findings}</ul>
<h2>Method Comparison</h2>{_comparison_html(state, ('methods', 'conclusions', 'limitations'))}
<h2>Dataset and Experimental Setup Comparison</h2>{_comparison_html(state, ('datasets', 'experimental_settings', 'metrics'))}
<h2>Limitations</h2><ul>{limitations}</ul>
<h2>Conflicting Evidence</h2><ul>{conflicts}</ul>
<h2>Open Questions</h2><ul>{open_questions}</ul>
<h2>Evidence Table</h2><table><thead><tr><th>Evidence ID</th><th>Paper</th><th>Page</th><th>Text</th></tr></thead><tbody>{evidence_rows or '<tr><td colspan="4">No evidence was retrieved.</td></tr>'}</tbody></table>
<h2>Citation Audit</h2><p>Status: <strong>{audit_status}</strong></p><ul>{audit_items}</ul>
<h2>Run Metadata</h2><ul><li>Run ID: <code>{escape(state.run_id)}</code></li><li>Status: {escape(state.status)}</li><li>Duration seconds: {duration if duration is not None else 'in progress'}</li><li>Tool calls: {len(state.tool_history)}</li><li>Failed tool calls: {sum(call.status == 'error' for call in state.tool_history)}</li><li>Model calls: {model_calls}</li></ul>
</body></html>"""


_ZH_REPORT_COPY = {
    '<html lang="en">': '<html lang="zh-CN">',
    "PaperScout Report": "PaperScout 研究笔记",
    "Executive Summary": "摘要",
    "System synthesis:": "系统汇总：",
    "paper(s), retrieved": "篇论文，检索到",
    "evidence item(s), and produced": "条证据，并生成",
    "traceable claim(s).": "条可追溯结论。",
    "Model-assisted calls:": "模型辅助调用：",
    "findings remain bounded by retrieved evidence IDs and citation audit.": "所有发现均受检索证据编号与引用审计约束。",
    "Research Question Decomposition": "研究问题拆解",
    "Selected Papers": "入选论文",
    "Main Findings": "主要发现",
    "Method Comparison": "方法对比",
    "Dataset and Experimental Setup Comparison": "数据集与实验设置对比",
    "Limitations": "局限性",
    "Conflicting Evidence": "冲突证据",
    "Open Questions": "待解决问题",
    "Evidence Table": "证据表",
    "Evidence ID": "证据编号",
    "<th>Paper</th>": "<th>论文</th>",
    "Page": "页码",
    "Text": "原文",
    "Citation Audit": "引用审计",
    "Run Metadata": "运行信息",
    "Run ID:": "运行编号：",
    "Status:": "状态：",
    "Duration seconds:": "耗时（秒）：",
    "Tool calls:": "工具调用：",
    "Failed tool calls:": "失败工具调用：",
    "Model calls:": "模型调用：",
    "No local paper matched.": "未找到匹配论文。",
    "No traceable claims were generated.": "未生成可追溯结论。",
    "No cross-paper conflict was detected in the retrieved evidence.": "检索证据中未发现跨论文冲突。",
    "No limitations were extracted from retrieved evidence.": "检索证据中未提取到局限性。",
    "Citation audit was not run.": "未运行引用审计。",
    "No sub-question generated.": "未生成子问题。",
    "No structured facts were extracted.": "未提取到结构化事实。",
    "No evidence was retrieved.": "未检索到证据。",
    "Which experimental conditions would change these evidence-backed findings?": "哪些实验条件可能改变这些有证据支持的发现？",
    "Not Extracted": "未提取",
    "supported": "有支持",
    "needs review": "需复核",
    "disabled": "未启用",
    "in progress": "进行中",
}


def render_html(state: ResearchState, locale: str = "en") -> str:
    """Render a self-contained, localized HTML research note."""
    document = _render_html_english(state)
    if locale == "zh":
        for source, target in _ZH_REPORT_COPY.items():
            document = document.replace(source, target)
    return document


def render_html_fragment(state: ResearchState, locale: str = "en") -> str:
    """Return only the trusted report body for embedding in the web interface."""
    document = render_html(state, locale)
    return document.split("<body>", 1)[1].rsplit("</body>", 1)[0]


def _evidence_row(item: EvidenceItem, papers: dict[str, str]) -> str:
    evidence_id = escape(item.id, quote=True)
    paper_title = escape(papers.get(item.paper_id, item.paper_id))
    return (
        f'<tr id="evidence-{evidence_id}"><td><code>{evidence_id}</code></td>'
        f"<td>{paper_title}</td><td>{item.page or '-'}</td><td>{escape(item.text)}</td></tr>"
    )


def _comparison_html(state: ResearchState, fields: tuple[str, ...]) -> str:
    rows = ""
    for row in state.comparison:
        rows += f"<h3>{escape(str(row['paper_id']))}</h3><ul>"
        for field in fields:
            values = row.get(field) or ["Not extracted"]
            label = escape(field.replace("_", " ").title())
            contents = escape(" | ".join(str(value) for value in values))
            rows += f"<li><strong>{label}:</strong> {contents}</li>"
        rows += "</ul>"
    return rows or "<p>No structured facts were extracted.</p>"


def _facts_html(state: ResearchState, field: str) -> str:
    items = []
    for facts in state.facts:
        for fact in getattr(facts, field):
            citation = _citations_html([fact.evidence_id])
            items.append(f"<li><code>{escape(facts.paper_id)}</code>: {escape(fact.text)} [{citation}]</li>")
    return "".join(items)
