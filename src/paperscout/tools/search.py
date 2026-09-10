from collections import defaultdict

from paperscout.models.schemas import PaperCandidate, SearchResult
from paperscout.retrieval.semantic import SemanticIndex
from paperscout.retrieval.reranker import CrossEncoderReranker
from paperscout.retrieval.store import CorpusStore


def search_papers(
    store: CorpusStore,
    query: str,
    top_k: int = 10,
    semantic_index: SemanticIndex | None = None,
    reranker: CrossEncoderReranker | None = None,
) -> list[PaperCandidate]:
    """Search evidence, then aggregate the strongest hits into paper candidates."""
    evidence_results = (
        semantic_index.search(query, top_k=max(top_k * 4, 20))
        if semantic_index is not None
        else store.search(query, top_k=max(top_k * 4, 20))
    )
    if reranker is not None:
        evidence_results = reranker.rerank(query, evidence_results, top_k=max(top_k * 4, 20))
    grouped: dict[str, list[SearchResult]] = defaultdict(list)
    for result in evidence_results:
        grouped[result.paper.id].append(result)
    candidates: list[PaperCandidate] = []
    for paper_results in grouped.values():
        paper_results.sort(key=lambda result: result.score, reverse=True)
        best = paper_results[0]
        matched_terms = sorted({term for result in paper_results[:3] for term in result.matched_terms})
        candidates.append(
            PaperCandidate(
                paper=best.paper,
                score=best.score,
                matched_terms=matched_terms,
                relevance_reason=(
                    f"Matched {len(matched_terms)} query terms in {len(paper_results)} evidence segment(s); "
                    f"best score {best.score:.3f}."
                ),
            )
        )
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates[:top_k]


def retrieve_evidence(
    store: CorpusStore,
    paper_id: str,
    query: str,
    top_k: int = 5,
    semantic_index: SemanticIndex | None = None,
    reranker: CrossEncoderReranker | None = None,
) -> list[SearchResult]:
    results = (
        semantic_index.search(query, top_k=top_k * 8)
        if semantic_index is not None
        else store.search(query, top_k=top_k * 4)
    )
    if reranker is not None:
        results = reranker.rerank(query, results, top_k=top_k * 8)
    return [result for result in results if result.evidence.paper_id == paper_id][:top_k]
