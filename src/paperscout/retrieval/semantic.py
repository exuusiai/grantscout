import json
import os
from pathlib import Path
from typing import Any

from paperscout.models.schemas import SearchResult
from paperscout.retrieval.store import CorpusStore


class SemanticIndexError(RuntimeError):
    """Raised when optional semantic retrieval dependencies are unavailable."""


class SemanticIndex:
    """Persistent BGE embedding index with FAISS and NumPy fallbacks."""

    def __init__(self, index_path: Path, model_name: str, device: str = "auto") -> None:
        self.index_path = index_path
        self.metadata_path = index_path.with_suffix(index_path.suffix + ".json")
        self.model_name = model_name
        self.device = device
        self._model: Any = None
        self._index: Any = None
        self._metadata: list[dict[str, Any]] = []

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise SemanticIndexError(
                "Semantic retrieval requires optional dependencies; run pip install -e '.[retrieval]'"
            ) from error
        resolved_device = None if self.device == "auto" else self.device
        configured_device = os.getenv("PAPERSCOUT_EMBEDDING_DEVICE")
        if configured_device:
            resolved_device = configured_device
        elif self.device == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    capability = torch.cuda.get_device_capability()
                    architecture = f"sm_{capability[0]}{capability[1]}"
                    supported_architectures = torch.cuda.get_arch_list()
                    if supported_architectures and architecture not in supported_architectures:
                        resolved_device = "cpu"
            except (ImportError, RuntimeError):
                resolved_device = None
        self._model = SentenceTransformer(self.model_name, device=resolved_device)
        return self._model

    def build(self, store: CorpusStore, batch_size: int = 32) -> dict[str, Any]:
        records = store.list_evidence()
        if not records:
            raise SemanticIndexError("Cannot build a semantic index from an empty corpus")
        model = self._load_model()
        texts = [f"{paper.title}\n{evidence.text}" for paper, evidence in records]
        vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
        try:
            import faiss
        except ImportError:
            faiss = None
        if faiss is not None:
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, str(self.index_path))
            self._index = index
        else:
            try:
                import numpy as np
            except ImportError as error:
                raise SemanticIndexError("Semantic retrieval requires NumPy") from error
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(self.index_path) + ".npy", vectors)
            self._index = vectors
        self._metadata = [{"paper": paper.model_dump(mode="json"), "evidence": evidence.model_dump(mode="json")} for paper, evidence in records]
        self.metadata_path.write_text(json.dumps(self._metadata, ensure_ascii=False), encoding="utf-8")
        return {"items": len(records), "model": self.model_name, "index": str(self.index_path)}

    def load(self) -> None:
        if not self.metadata_path.exists():
            raise SemanticIndexError(f"Semantic index metadata does not exist: {self.metadata_path}")
        self._metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        try:
            import faiss

            if self.index_path.exists():
                self._index = faiss.read_index(str(self.index_path))
                return
        except ImportError:
            pass
        try:
            import numpy as np

            self._index = np.load(str(self.index_path) + ".npy")
        except (FileNotFoundError, ImportError) as error:
            raise SemanticIndexError(f"Semantic index data does not exist: {self.index_path}") from error

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        if self._index is None:
            self.load()
        query_vector = self._load_model().encode([query], normalize_embeddings=True)
        if hasattr(self._index, "search"):
            scores, indices = self._index.search(query_vector, top_k)
            ranked = zip(scores[0].tolist(), indices[0].tolist())
        else:
            scores = self._index @ query_vector[0]
            ranked = ((float(scores[index]), index) for index in scores.argsort()[::-1][:top_k])
        results: list[SearchResult] = []
        for score, index in ranked:
            if index < 0 or index >= len(self._metadata):
                continue
            metadata = self._metadata[index]
            from paperscout.models.schemas import EvidenceItem, Paper

            results.append(
                SearchResult(
                    paper=Paper.model_validate(metadata["paper"]),
                    evidence=EvidenceItem.model_validate(metadata["evidence"]),
                    score=max(float(score), 0.0),
                    matched_terms=[],
                )
            )
        return results
