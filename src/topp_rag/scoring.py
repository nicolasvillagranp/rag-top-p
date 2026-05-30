"""Similarity scorers — the seam that makes top-p similarity-agnostic.

Top-p selection operates on a score array; it does not care how the scores were
produced. A :class:`Scorer` encapsulates one similarity function and produces,
per query, the top ``M_max`` ``(scores, doc_ids)`` in descending score order.
Everything downstream (the retrievals cache, the sweep, the analysis) consumes
that output and never touches FAISS or cosine directly, so a new scorer drops in
without changing any other module.

* :class:`CosineScorer` — pure semantic similarity via FAISS inner product over
  normalized embeddings. Implemented and used now.
* :class:`HybridScorer` — ``alpha * BM25 + (1 - alpha) * cosine``. A documented
  seam: the interface and ``name`` exist, but ``score_all`` is not implemented.

Each scorer exposes a ``name`` that is woven into cache filenames so different
scorers never collide on disk.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import faiss
import numpy as np


@runtime_checkable
class Scorer(Protocol):
    """Produces per-query top-``M_max`` document scores from any similarity fn.

    Implementations must return scores in **descending** order per query.
    """

    name: str

    def fit(self, corpus_texts: list[str], corpus_embeddings: np.ndarray) -> None:
        """Index the corpus. Called once before :meth:`score_all`."""
        ...

    def score_all(
        self,
        query_texts: list[str],
        query_embeddings: np.ndarray,
        M_max: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-``M_max`` ``(scores[N, M_max], doc_ids[N, M_max])`` per query.

        Scores are in descending order along axis 1; ``doc_ids`` index into the
        corpus passed to :meth:`fit`.
        """
        ...


class CosineScorer:
    """Cosine similarity via a FAISS ``IndexFlatIP`` over normalized embeddings.

    Because embeddings are L2-normalized upstream (see
    :class:`~topp_rag.embeddings.EmbeddingCache`), inner product equals cosine
    similarity, and a flat IP index returns exact, descending-ordered results.
    """

    name = "cosine"

    def __init__(self) -> None:
        self._index = None

    def fit(self, corpus_texts: list[str], corpus_embeddings: np.ndarray) -> None:
        # corpus_texts is unused by cosine (scores come from embeddings) but is
        # part of the interface so text-based scorers (e.g. hybrid BM25) fit.
        embeddings = np.ascontiguousarray(corpus_embeddings, dtype=np.float32)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self._index = index

    def score_all(
        self,
        query_texts: list[str],
        query_embeddings: np.ndarray,
        M_max: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._index is None:
            raise RuntimeError("CosineScorer.fit must be called before score_all.")
        queries = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        # Single batched search over ALL queries at once — never a per-query loop.
        # FAISS returns scores already sorted in descending order per row.
        scores, doc_ids = self._index.search(queries, M_max)
        return scores, doc_ids


class HybridScorer:
    """``alpha * BM25 + (1 - alpha) * cosine`` — a documented seam.

    Not implemented yet. The interface and ``name`` are in place so that, when
    built, it slots in behind :class:`Scorer` with no change to the cache, sweep,
    or analysis. The score-combination (normalization) strategy is deliberately
    left open until then.

    Args:
        alpha: weight on the BM25 component, in ``(0, 1)``.
    """

    def __init__(self, alpha: float) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        self.alpha = alpha
        self.name = f"hybrid_a{alpha}"

    def fit(self, corpus_texts: list[str], corpus_embeddings: np.ndarray) -> None:
        raise NotImplementedError(
            "HybridScorer is a planned seam; only CosineScorer is implemented. "
            "BM25+cosine fusion and its normalization strategy are future work."
        )

    def score_all(
        self,
        query_texts: list[str],
        query_embeddings: np.ndarray,
        M_max: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "HybridScorer is a planned seam; only CosineScorer is implemented."
        )


def build_scorer(name: str, params: dict | None = None) -> Scorer:
    """Construct a scorer from its config name and params."""
    params = params or {}
    if name == "cosine":
        return CosineScorer()
    if name == "hybrid":
        return HybridScorer(alpha=params["alpha"])
    raise ValueError(f"Unknown scorer '{name}'. Expected 'cosine' or 'hybrid'.")
