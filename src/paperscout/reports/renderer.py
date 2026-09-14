from __future__ import annotations

from html import escape
from typing import Iterable

from paperscout.models.schemas import ResearchState


def _citations_markdown(evidence_ids: Iterable[str]) -> str:
    return ", ".join(f"`{evidence_id}`" for evidence_id in evidence_ids) or "no evidence"


def _citations_html(evidence_ids: Iterable[str]) -> str:
    return ", ".join(f"<code>{escape(value)}</code>" for value in evidence_ids) or "no evidence"


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
    lines = [f"# PaperScout Report: {state.question}", "", "## Executive Summary", ""]
    lines.append(
        "- System synthesis: "
        f"selected {len(state.selected_papers)} paper(s), retrieved {len(state.evidence_items)} "
        f"evidence item(s), and produced {len(state.claims)} traceable claim(s)."
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
    if state.comparability:
        lines.extend(["", "## Comparability Gate", ""])
        for item in state.comparability:
            verdict = "Comparable" if item.comparable else "Not directly comparable"
            lines.append(f"- `{' / '.join(item.paper_ids)}`: {verdict}. {item.reason}")
    lines.extend(["", "## Conflicting Evidence", ""])
    if state.conflicts:
        for conflict in state.conflicts:
            citations = _citations_markdown(
                [*conflict.positive_evidence_ids, *conflict.negative_evidence_ids]
            )
            lines.append(f"- {conflict.message} [{citations}]")
    else:
        lines.append("- No cross-paper conflict was detected in the retrieved evidence.")
    lines.extend(["", "## Research Decisions", ""])
    if state.decisions:
        for decision in state.decisions:
            missing = "; ".join(decision.missing_information) or "none"
            lines.append(
                f"- `{decision.paper_id}`: {decision.recommendation}; readiness "
                f"{decision.readiness_score}/100. {decision.reason} Missing: {missing}."
            )
    else:
        lines.append("- No candidate was available for a research decision.")
    lines.extend(["", "## Open Questions", ""])
    if state.warnings:
        lines.extend(f"- {warning}" for warning in state.warnings)
    else:
        lines.append("- Which experimental conditions would change these evidence-backed findings?")
    return "\n".join(lines) + "\n"


def _render_html_english(state: ResearchState, locale: str = "en") -> str:
    """Render a self-contained HTML report with evidence anchors and citations."""
    selected_papers = "".join(
        f"<li><code>{escape(paper.id)}</code>: {escape(paper.title)}"
        f"{f' ({paper.year})' if paper.year else ''}</li>"
        for paper in state.selected_papers
    ) or "<li>No local paper matched.</li>"
    findings = "".join(
        "<li>"
        f"{escape(claim.localized_text or claim.text) if locale == 'zh' else escape(claim.text)} "
        f"[{escape(claim.support_status)}; "
        f"{_citations_html(claim.evidence_ids)}]"
        "</li>"
        for claim in state.claims
    ) or "<li>No traceable claims were generated.</li>"
    conflicts = "".join(
        f"<li>{escape(conflict.message)} ["
        f"{_citations_html([*conflict.positive_evidence_ids, *conflict.negative_evidence_ids])}]</li>"
        for conflict in state.conflicts
    ) or "<li>No cross-paper conflict was detected in the retrieved evidence.</li>"
    comparability = "".join(
        f"<li><code>{escape(' / '.join(item.paper_ids))}</code>: "
        f"{'Comparable' if item.comparable else 'Not directly comparable'}. {escape(item.reason)}</li>"
        for item in state.comparability
    ) or "<li>Not enough paper pairs were available for comparison.</li>"
    decisions = "".join(
        f"<li><code>{escape(item.paper_id)}</code>: {escape(item.recommendation)}; "
        f"readiness {item.readiness_score}/100. {escape(item.reason)} "
        f"Missing: {escape('; '.join(item.missing_information) or 'none')}.</li>"
        for item in state.decisions
    ) or "<li>No candidate was available for a research decision.</li>"
    limitations = _facts_html(state, "limitations", locale) or (
        "<li>No limitations were extracted from retrieved evidence.</li>"
    )
    open_questions = "".join(f"<li>{escape(warning)}</li>" for warning in state.warnings) or (
        "<li>Which experimental conditions would change these evidence-backed findings?</li>"
    )
    decomposed_questions = "".join(
        f"<li>{escape(question)}</li>" for question in state.sub_questions
    ) or "<li>No sub-question generated.</li>"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>PaperScout Report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#172033}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #cbd5e1;padding:8px;text-align:left;vertical-align:top}}th{{background:#e2e8f0}}code{{overflow-wrap:anywhere}}a{{color:#075985}}</style>
</head><body>
<h1>PaperScout Report: {escape(state.question)}</h1>
<h2>Executive Summary</h2><ul><li>System synthesis: selected {len(state.selected_papers)} paper(s), retrieved {len(state.evidence_items)} evidence item(s), and produced {len(state.claims)} traceable claim(s).</li></ul>
<h2>Research Question Decomposition</h2><ul>{decomposed_questions}</ul>
<h2>Selected Papers</h2><ul>{selected_papers}</ul>
<h2>Main Findings</h2><ul>{findings}</ul>
<h2>Method Comparison</h2>{_comparison_html(state, ('methods', 'conclusions', 'limitations'), locale)}
<h2>Dataset and Experimental Setup Comparison</h2>{_comparison_html(state, ('datasets', 'experimental_settings', 'metrics'), locale)}
<h2>Limitations</h2><ul>{limitations}</ul>
<h2>Comparability Gate</h2><ul>{comparability}</ul>
<h2>Conflicting Evidence</h2><ul>{conflicts}</ul>
<h2>Research Decisions</h2><ul>{decisions}</ul>
<h2>Open Questions</h2><ul>{open_questions}</ul>
</body></html>"""


_ZH_REPORT_COPY = {
    '<html lang="en">': '<html lang="zh-CN">',
    "PaperScout Report": "PaperScout 研究笔记",
    "Executive Summary": "摘要",
    "System synthesis:": "系统汇总：",
    "selected ": "入选 ",
    "paper(s), retrieved": "篇论文，检索到",
    "evidence item(s), and produced": "条证据，并生成",
    "traceable claim(s).": "条可追溯结论。",
    "Model-assisted calls:": "模型辅助调用：",
    "findings remain bounded by retrieved evidence IDs and citation audit.": "所有发现均受检索证据编号与引用审计约束。",
    "Research Question Decomposition": "研究问题拆解",
    "Selected Papers": "入选论文",
    "Main Findings": "主要发现",
    "Method Comparison": "方法对比",
    "Methods:": "方法：",
    "Conclusions:": "结论：",
    "Dataset and Experimental Setup Comparison": "数据集与实验设置对比",
    "Datasets:": "数据集：",
    "Experimental Settings:": "实验设置：",
    "Metrics:": "指标：",
    "Not extracted": "未提取",
    "Limitations": "局限性",
    "Conflicting Evidence": "冲突证据",
    "Comparability Gate": "可比性门禁",
    "Comparable": "可直接比较",
    "Not directly comparable": "条件不同，不可直接比较",
    "Not enough paper pairs were available for comparison.": "论文对不足，无法进行可比性判断。",
    "Open Questions": "待解决问题",
    "Research Decisions": "研究决策建议",
    "readiness": "复现准备度",
    "Missing:": "尚缺：",
    "No candidate was available for a research decision.": "没有可用于研究决策的候选论文。",
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
    document = _render_html_english(state, locale)
    if locale == "zh":
        for source, target in _ZH_REPORT_COPY.items():
            document = document.replace(source, target)
    return document


def render_html_fragment(state: ResearchState, locale: str = "en") -> str:
    """Return only the trusted report body for embedding in the web interface."""
    document = render_html(state, locale)
    return document.split("<body>", 1)[1].rsplit("</body>", 1)[0]


def _comparison_html(
    state: ResearchState, fields: tuple[str, ...], locale: str = "en"
) -> str:
    rows = ""
    facts_by_id = {facts.paper_id: facts for facts in state.facts}
    for row in state.comparison:
        paper_id = str(row["paper_id"])
        facts = facts_by_id.get(paper_id)
        rows += f"<h3>{escape(paper_id)}</h3><ul>"
        for field in fields:
            values = row.get(field) or ["Not extracted"]
            if facts is not None:
                source_facts = getattr(facts, field)
                values = [
                    fact.localized_text if locale == "zh" and fact.localized_text else fact.text
                    for fact in source_facts[:5]
                ] or ["Not extracted"]
            label = escape(field.replace("_", " ").title())
            contents = escape(" | ".join(str(value) for value in values))
            rows += f"<li><strong>{label}:</strong> {contents}</li>"
        rows += "</ul>"
    return rows or "<p>No structured facts were extracted.</p>"


def _facts_html(state: ResearchState, field: str, locale: str = "en") -> str:
    items = []
    for facts in state.facts:
        for fact in getattr(facts, field):
            citation = _citations_html([fact.evidence_id])
            text = fact.localized_text if locale == "zh" and fact.localized_text else fact.text
            items.append(f"<li><code>{escape(facts.paper_id)}</code>: {escape(text)} [{citation}]</li>")
    return "".join(items)
