import os
from typing import Any

from grantscout.models.schemas import SearchResult


class RerankerError(RuntimeError):
    """Raised when the optional Cross-Encoder reranker cannot be loaded."""


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise RerankerError(
                "Reranking requires optional dependencies; run pip install -e '.[retrieval]'"
            ) from error
        resolved_device = os.getenv("GRANTSCOUT_RERANKER_DEVICE") or (None if self.device == "auto" else self.device)
        self._model = CrossEncoder(self.model_name, device=resolved_device)
        return self._model

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not results:
            return []
        pairs = [(query, f"{result.paper.title}\n{result.evidence.text}") for result in results]
        scores = self._load_model().predict(pairs)
        reranked = [
            result.model_copy(update={"score": max(float(score), 0.0)})
            for result, score in zip(results, scores, strict=True)
        ]
        reranked.sort(key=lambda result: result.score, reverse=True)
        return reranked[:top_k]
