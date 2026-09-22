from __future__ import annotations

from typing import Any

from grantscout.agent.loop import GrantScoutAgent
from grantscout.config import Settings
from grantscout.evaluation.baselines import (
    retrieval_metrics_for_results,
    single_pass_context,
)
from grantscout.evaluation.datasets import evaluation_records
from grantscout.retrieval.store import CorpusStore
from grantscout.retrieval.semantic import SemanticIndex


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _metrics_summary(scores: list[dict[str, float]]) -> dict[str, float]:
    return {
        "paper_recall": _mean([item["paper_recall"] for item in scores]),
        "evidence_recall": _mean([item["evidence_recall"] for item in scores]),
    }


def _quality_metrics(state, record: dict[str, Any]) -> dict[str, float | bool | None]:
    claims = state.claims
    cited_claims = sum(bool(claim.evidence_ids) for claim in claims)
    audit = state.citation_audit
    expected_conflict = record.get("conflict_expected")
    return {
        "citation_coverage": cited_claims / max(len(claims), 1),
        "audit_execution_rate": float(audit is not None),
        "citation_support_precision": (
            audit.supported_claims / max(len(audit.items), 1) if audit is not None else None
        ),
        "unsupported_claim_rate": (
            audit.unsupported_claims / max(len(audit.items), 1) if audit is not None else None
        ),
        "conflict_expected": bool(expected_conflict) if expected_conflict is not None else None,
        "conflict_detected": bool(state.conflicts),
        "model_calls": sum(call.tool.startswith("model_") for call in state.tool_history),
        "tool_duration_ms": sum(call.duration_ms for call in state.tool_history),
    }


def _variant_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "paper_recall": _mean([float(item["paper_recall"]) for item in runs]),
        "evidence_recall": _mean([float(item["evidence_recall"]) for item in runs]),
        "citation_coverage": _mean([float(item["citation_coverage"]) for item in runs]),
        "audit_execution_rate": _mean([float(item["audit_execution_rate"]) for item in runs]),
        "mean_model_calls": _mean([float(item["model_calls"]) for item in runs]),
        "mean_tool_duration_ms": _mean([float(item["tool_duration_ms"]) for item in runs]),
    }
    audited = [item for item in runs if item["citation_support_precision"] is not None]
    summary["citation_support_precision"] = (
        _mean([float(item["citation_support_precision"]) for item in audited]) if audited else None
    )
    summary["unsupported_claim_rate"] = (
        _mean([float(item["unsupported_claim_rate"]) for item in audited]) if audited else None
    )
    conflict_labeled = [item for item in runs if item["conflict_expected"] is not None]
    if conflict_labeled:
        true_positive = sum(item["conflict_expected"] and item["conflict_detected"] for item in conflict_labeled)
        false_positive = sum(not item["conflict_expected"] and item["conflict_detected"] for item in conflict_labeled)
        false_negative = sum(item["conflict_expected"] and not item["conflict_detected"] for item in conflict_labeled)
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        summary.update(
            {
                "conflict_labeled_queries": len(conflict_labeled),
                "conflict_precision": round(precision, 4),
                "conflict_recall": round(recall, 4),
                "conflict_f1": round(2 * precision * recall / max(precision + recall, 1e-12), 4),
            }
        )
    else:
        summary.update(
            {
                "conflict_labeled_queries": 0,
                "conflict_precision": None,
                "conflict_recall": None,
                "conflict_f1": None,
            }
        )
    return summary


def _agent_metrics(state, relevant_papers: set[str], relevant_evidence: set[str]) -> dict[str, float]:
    selected_papers = {paper.id for paper in state.selected_papers}
    selected_evidence = {item.id for item in state.evidence_items}
    return {
        "paper_recall": len(selected_papers & relevant_papers) / max(len(relevant_papers), 1),
        "evidence_recall": len(selected_evidence & relevant_evidence) / max(len(relevant_evidence), 1),
    }


def run_comparison(
    corpus,
    queries,
    settings: Settings,
    top_k: int = 5,
    semantic_index: SemanticIndex | None = None,
) -> dict[str, Any]:
    """Compare deterministic single-pass, fixed RAG, and Agentic RAG retrieval."""
    records = evaluation_records(queries)
    single_pass_scores: list[dict[str, float]] = []
    baseline_scores: list[dict[str, float]] = []
    agent_scores: list[dict[str, float]] = []
    per_query: list[dict[str, Any]] = []
    agent_settings = settings.model_copy(update={"max_papers": top_k})
    with CorpusStore(corpus) as store:
        agent = GrantScoutAgent(store, agent_settings, semantic_index=semantic_index)
        for record in records:
            query = str(record["query"])
            relevant_papers = {str(value) for value in record.get("relevant_paper_ids", [])}
            relevant_evidence = {str(value) for value in record.get("relevant_evidence_ids", [])}
            single_pass_results = store.search(query, top_k=top_k)
            single_pass = single_pass_context(store, query, top_k, results=single_pass_results)
            single_pass_metric = retrieval_metrics_for_results(
                single_pass_results, relevant_papers, relevant_evidence
            )
            fixed_results = single_pass_results
            fixed_metric = retrieval_metrics_for_results(
                fixed_results, relevant_papers, relevant_evidence
            )
            state = agent.run(query)
            agent_metric = _agent_metrics(state, relevant_papers, relevant_evidence)
            single_pass_scores.append(single_pass_metric)
            baseline_scores.append(fixed_metric)
            agent_scores.append(agent_metric)
            per_query.append(
                {
                    "query": query,
                    "single_pass_context": {
                        **single_pass_metric,
                        "evidence_ids": single_pass["evidence_ids"],
                    },
                    "fixed_rag": {
                        **fixed_metric,
                        "evidence_ids": [result.evidence.id for result in fixed_results],
                    },
                    "paperscout_agentic_rag": {
                        **agent_metric,
                        "run_id": state.run_id,
                        "paper_ids": [paper.id for paper in state.selected_papers],
                        "evidence_ids": [item.id for item in state.evidence_items],
                        "tool_calls": len(state.tool_history),
                        "warnings": state.warnings,
                    },
                }
            )
    return {
        "queries": len(records),
        "top_k": top_k,
        "single_pass_context": _metrics_summary(single_pass_scores),
        "fixed_rag": _metrics_summary(baseline_scores),
        "paperscout_agentic_rag": _metrics_summary(agent_scores),
        "per_query": per_query,
    }


def run_ablation(
    corpus,
    queries,
    settings: Settings,
    top_k: int = 5,
    semantic_index: SemanticIndex | None = None,
) -> dict[str, Any]:
    """Measure the impact of question planning, reranking, audit, and conflicts."""
    records = evaluation_records(queries)
    variants = {
        "full": {"decompose": True, "audit": True, "rerank": None, "detect_conflicts": True},
        "without_question_decomposition": {
            "decompose": False,
            "audit": True,
            "rerank": None,
            "detect_conflicts": True,
        },
        "without_reranker": {
            "decompose": True,
            "audit": True,
            "rerank": False,
            "detect_conflicts": True,
        },
        "without_citation_audit": {
            "decompose": True,
            "audit": False,
            "rerank": None,
            "detect_conflicts": True,
        },
        "without_conflict_detection": {
            "decompose": True,
            "audit": True,
            "rerank": None,
            "detect_conflicts": False,
        },
    }
    output: dict[str, Any] = {"queries": len(records), "top_k": top_k, "variants": {}}
    agent_settings = settings.model_copy(update={"max_papers": top_k})
    with CorpusStore(corpus) as store:
        for name, options in variants.items():
            agent = GrantScoutAgent(
                store,
                agent_settings,
                semantic_index=semantic_index,
                **options,
            )
            runs = []
            for record in records:
                query = str(record["query"])
                relevant_papers = {str(value) for value in record.get("relevant_paper_ids", [])}
                relevant_evidence = {str(value) for value in record.get("relevant_evidence_ids", [])}
                state = agent.run(query)
                runs.append(
                    {
                        "query": query,
                        **_agent_metrics(state, relevant_papers, relevant_evidence),
                        "run_id": state.run_id,
                        "paper_ids": [paper.id for paper in state.selected_papers],
                        "evidence_ids": [item.id for item in state.evidence_items],
                        "claims": len(state.claims),
                        "citation_audit": (
                            state.citation_audit.status if state.citation_audit else "disabled"
                        ),
                        "conflicts": len(state.conflicts),
                        "tool_calls": len(state.tool_history),
                        "warnings": state.warnings,
                        **_quality_metrics(state, record),
                    }
                )
            output["variants"][name] = {
                "options": options,
                "summary": _variant_summary(runs),
                "runs": runs,
            }
    return output
