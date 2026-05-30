"""Embedding cache — layer 1 of the three-layer cache.

Document and query embeddings are deterministic for a given (text, model) pair,
so they are computed once with batched inference and persisted as ``.npy``. On
subsequent runs they load from disk and the embedding model is never constructed.

The model name is part of every cache filename, so changing ``embedding.model``
naturally produces a cache miss and regenerates — no explicit invalidation logic.

On a fully-cached run the :class:`~sentence_transformers.SentenceTransformer`
object is never *constructed* (the heavy load of weights is skipped); the import
itself stays at module top per project convention.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

_BATCH_SIZE = 64


def _safe_model_name(model: str) -> str:
    """Filesystem-safe form of a model name (e.g. strip ``org/`` prefixes)."""
    return model.replace("/", "_")


def _embeddings_path(cache_dir: Path, dataset: str, model: str, kind: str) -> Path:
    """``{cache_dir}/embeddings/{dataset}_{model}_{kind}_embeddings.npy``."""
    name = f"{dataset}_{_safe_model_name(model)}_{kind}_embeddings.npy"
    return Path(cache_dir) / "embeddings" / name


class EmbeddingCache:
    """Loads embeddings from disk, computing and caching them on a miss.

    The :class:`sentence_transformers.SentenceTransformer` model is constructed
    lazily and only when something actually needs encoding, so a fully-cached run
    never instantiates it (and never loads its weights).
    """

    def __init__(self, cache_dir: str | Path, model_name: str, batch_size: int = _BATCH_SIZE):
        self.cache_dir = Path(cache_dir)
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None  # lazily constructed on first miss

    def _get_model(self):
        if self._model is None:
            # Constructed only on a cache miss, so a fully-cached run never loads
            # the model weights.
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Batched, normalized encoding of a list of texts."""
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # so inner product == cosine similarity
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def _load_or_compute(
        self, texts: list[str], dataset: str, kind: str
    ) -> np.ndarray:
        path = _embeddings_path(self.cache_dir, dataset, self.model_name, kind)
        if path.exists():
            return np.load(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        embeddings = self._encode(texts)
        np.save(path, embeddings)
        return embeddings

    def embed_documents(self, texts: list[str], dataset: str) -> np.ndarray:
        """Embeddings for the document corpus (cached per dataset/model)."""
        return self._load_or_compute(texts, dataset, kind="doc")

    def embed_queries(self, texts: list[str], dataset: str) -> np.ndarray:
        """Embeddings for the query set (cached per dataset/model)."""
        return self._load_or_compute(texts, dataset, kind="query")