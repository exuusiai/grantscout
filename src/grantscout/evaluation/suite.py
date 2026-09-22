from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from grantscout import __version__
from grantscout.config import Settings
from grantscout.evaluation.ablation import run_ablation, run_comparison
from grantscout.evaluation.runner import run_evaluation
from grantscout.retrieval.semantic import SemanticIndex


def run_evaluation_suite(
    corpus: Path,
    queries: Path,
    settings: Settings,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run retrieval, baseline, and ablation evaluation in one reproducible bundle."""
    started_at = datetime.now(UTC)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ") + "-evaluation-" + uuid4().hex[:8]
    semantic_index = None
    if settings.retrieval_mode.lower() == "semantic":
        semantic_index = SemanticIndex(settings.vector_index_path, settings.embedding_model)
        semantic_index.load()
    payload: dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "paper_scout_version": __version__,
        "corpus": str(corpus),
        "queries": str(queries),
        "top_k": top_k,
        "retrieval": run_evaluation(
            corpus,
            queries,
            top_k=top_k,
            mode=settings.retrieval_mode.lower(),
            settings=settings,
            semantic_index=semantic_index,
        ),
        "benchmark": run_comparison(
            corpus, queries, settings, top_k=top_k, semantic_index=semantic_index
        ),
        "ablation": run_ablation(
            corpus, queries, settings, top_k=top_k, semantic_index=semantic_index
        ),
    }
    finished_at = datetime.now(UTC)
    payload["finished_at"] = finished_at.isoformat()
    payload["duration_seconds"] = round((finished_at - started_at).total_seconds(), 3)
    return payload


def persist_evaluation_suite(payload: dict[str, Any], output_dir: Path) -> dict[str, str]:
    """Persist JSON source data and a concise human-readable comparison report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(payload["run_id"])
    json_path = output_dir / f"{run_id}.json"
    markdown_path = output_dir / f"{run_id}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_evaluation_markdown(payload), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def render_evaluation_markdown(payload: dict[str, Any]) -> str:
    retrieval = payload["retrieval"]
    benchmark = payload["benchmark"]
    ablation = payload["ablation"]
    lines = ["# GrantScout Evaluation Suite", ""]
    lines.extend(
        [
            f"- Run ID: `{payload['run_id']}`",
            f"- GrantScout version: `{payload['paper_scout_version']}`",
            f"- Corpus: `{payload['corpus']}`",
            f"- Queries: `{payload['queries']}`",
            f"- Duration seconds: {payload['duration_seconds']}",
            "",
            "## Retrieval Metrics",
            "",
            "| Recall@K | Evidence Recall@K | MRR | Queries |",
            "| --- | --- | --- | --- |",
            (
                f"| {retrieval['recall_at_k']:.4f} | {retrieval['evidence_recall_at_k']:.4f} | "
                f"{retrieval['mrr']:.4f} | {retrieval['queries']} |"
            ),
            "",
            "## Baseline Comparison",
            "",
            "| System | Paper Recall | Evidence Recall |",
            "| --- | --- | --- |",
        ]
    )
    for name in ("single_pass_context", "fixed_rag", "paperscout_agentic_rag"):
        score = benchmark[name]
        lines.append(f"| {name} | {score['paper_recall']:.4f} | {score['evidence_recall']:.4f} |")
    lines.extend(
        [
            "",
            "## Ablations",
            "",
            "| Variant | Paper Recall | Evidence Recall | Runs |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, variant in ablation["variants"].items():
        runs = variant["runs"]
        summary = variant["summary"]
        lines.append(
            f"| {name} | {summary['paper_recall']:.4f} | "
            f"{summary['evidence_recall']:.4f} | {len(runs)} |"
        )
    lines.extend(
        [
            "",
            "## Traceability and Conflict Metrics",
            "",
            "| Variant | Citation coverage | Audit execution | Citation support precision | Unsupported claim rate | Conflict F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, variant in ablation["variants"].items():
        summary = variant["summary"]
        display = lambda value: "N/A" if value is None else f"{value:.4f}"
        lines.append(
            f"| {name} | {display(summary['citation_coverage'])} | "
            f"{display(summary['audit_execution_rate'])} | "
            f"{display(summary['citation_support_precision'])} | "
            f"{display(summary['unsupported_claim_rate'])} | "
            f"{display(summary['conflict_f1'])} |"
        )
    lines.extend(
        [
            "",
            "## Artifact Notes",
            "",
            "- The companion JSON stores per-query retrieval IDs, Agent run IDs, warnings, and ablation outputs.",
            "- Agent run artifacts store traceable evidence, claims, citation audits, Markdown reports, and HTML reports.",
        ]
    )
    return "\n".join(lines) + "\n"
