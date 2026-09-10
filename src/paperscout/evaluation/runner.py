from pathlib import Path
from typing import Any

from paperscout.config import Settings
from paperscout.evaluation.datasets import evaluation_records
from paperscout.evaluation.metrics import evaluate_queries
from paperscout.retrieval.semantic import SemanticIndex, SemanticIndexError
from paperscout.retrieval.store import CorpusStore


def run_evaluation(
    corpus: Path,
    queries: Path,
    top_k: int = 10,
    mode: str = "lexical",
    settings: Settings | None = None,
    semantic_index: SemanticIndex | None = None,
) -> dict[str, Any]:
    if mode not in {"lexical", "semantic"}:
        raise ValueError("Evaluation mode must be lexical or semantic")
    if mode == "semantic":
        configured = settings or Settings()
        if semantic_index is None:
            semantic_index = SemanticIndex(
                configured.vector_index_path, configured.embedding_model
            )
            semantic_index.load()
    with CorpusStore(corpus) as store:
        result = evaluate_queries(
            store,
            evaluation_records(queries),
            top_k=top_k,
            semantic_index=semantic_index,
        )
    result["mode"] = mode
    return result
