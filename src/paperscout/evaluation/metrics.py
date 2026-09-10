from typing import Any

from paperscout.retrieval.semantic import SemanticIndex
from paperscout.retrieval.store import CorpusStore


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def evaluate_queries(
    store: CorpusStore,
    records: list[dict[str, Any]],
    top_k: int = 10,
    semantic_index: SemanticIndex | None = None,
) -> dict[str, Any]:
    per_query: list[dict[str, Any]] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    evidence_recalls: list[float] = []
    for record in records:
        query = str(record["query"])
        relevant_papers = {str(value) for value in record.get("relevant_paper_ids", [])}
        relevant_evidence = {str(value) for value in record.get("relevant_evidence_ids", [])}
        results = (
            semantic_index.search(query, top_k=top_k)
            if semantic_index is not None
            else store.search(query, top_k=top_k)
        )
        retrieved_papers = [result.paper.id for result in results]
        retrieved_evidence = [result.evidence.id for result in results]
        paper_recall = len(set(retrieved_papers) & relevant_papers) / max(len(relevant_papers), 1)
        evidence_recall = (
            len(set(retrieved_evidence) & relevant_evidence) / len(relevant_evidence)
            if relevant_evidence
            else None
        )
        reciprocal_rank = 0.0
        for rank, paper_id in enumerate(retrieved_papers, start=1):
            if paper_id in relevant_papers:
                reciprocal_rank = 1.0 / rank
                break
        recalls.append(paper_recall)
        if evidence_recall is not None:
            evidence_recalls.append(evidence_recall)
        reciprocal_ranks.append(reciprocal_rank)
        per_query.append(
            {
                "query": query,
                "retrieved_paper_ids": retrieved_papers,
                "retrieved_evidence_ids": retrieved_evidence,
                "paper_recall": round(paper_recall, 4),
                "evidence_recall": round(evidence_recall, 4) if evidence_recall is not None else None,
                "reciprocal_rank": round(reciprocal_rank, 4),
            }
        )
    return {
        "queries": len(records),
        "top_k": top_k,
        "recall_at_k": _mean(recalls),
        "evidence_recall_at_k": _mean(evidence_recalls),
        "evidence_labeled_queries": len(evidence_recalls),
        "mrr": _mean(reciprocal_ranks),
        "per_query": per_query,
    }
