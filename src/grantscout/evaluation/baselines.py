from typing import Any

from grantscout.models.schemas import SearchResult
from grantscout.retrieval.store import CorpusStore


def single_pass_context(
    store: CorpusStore,
    query: str,
    top_k: int = 5,
    results: list[SearchResult] | None = None,
) -> dict[str, Any]:
    """Baseline A: one fixed retrieval pass and one context bundle."""
    results = results if results is not None else store.search(query, top_k=top_k)
    return {
        "baseline": "single_pass",
        "query": query,
        "evidence_ids": [result.evidence.id for result in results],
        "context": "\n\n".join(result.evidence.text for result in results),
    }


def fixed_rag(store: CorpusStore, query: str, top_k: int = 5) -> list[SearchResult]:
    """Baseline B: ordinary fixed Top-K lexical RAG."""
    return store.search(query, top_k=top_k)


def retrieval_metrics_for_results(
    results: list[SearchResult], relevant_papers: set[str], relevant_evidence: set[str]
) -> dict[str, float]:
    paper_ids = {result.paper.id for result in results}
    evidence_ids = {result.evidence.id for result in results}
    return {
        "paper_recall": len(paper_ids & relevant_papers) / max(len(relevant_papers), 1),
        "evidence_recall": len(evidence_ids & relevant_evidence) / max(len(relevant_evidence), 1),
    }
