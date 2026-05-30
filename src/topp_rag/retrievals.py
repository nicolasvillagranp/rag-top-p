"""Retrievals cache — layer 2 of the three-layer cache.

For each query we run the configured :class:`~topp_rag.scoring.Scorer` once to get
the top ``M_max`` documents, then persist the scores, document IDs, and per-document
token counts to parquet. Because ``M_max`` is the upper bound of the M grid, any
smaller M is just a slice ``arr[:M]`` of the cached row — the scorer (and FAISS)
never runs again during the sweep.

This module is scorer-agnostic: it depends only on the :class:`Scorer` interface
and never references FAISS or cosine directly. The scorer's ``name`` is woven into
the cache filename so different similarity functions never collide on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .embeddings import EmbeddingCache, _safe_model_name
from .scoring import Scorer
from .tokens import count_tokens_many


@dataclass(frozen=True)
class Retrievals:
    """In-memory view of the cached retrievals for one dataset.

    Each array is ``[N_queries, M_max]`` (or length ``N_queries`` for ids).
    ``scores`` are descending per row; ``doc_ids`` index the corpus; ``doc_tokens``
    is the token count of each retrieved document, aligned with ``doc_ids``.
    """

    query_ids: list[str]
    scores: np.ndarray  # float32 [N, M_max]
    doc_ids: np.ndarray  # int64   [N, M_max]
    doc_tokens: np.ndarray  # int64 [N, M_max]


def _retrievals_path(cache_dir: Path, dataset: str, model: str, scorer: str, M_max: int) -> Path:
    """``{cache_dir}/retrievals/{dataset}_{model}_{scorer}_top{M_max}_retrievals.parquet``."""
    name = f"{dataset}_{_safe_model_name(model)}_{scorer}_top{M_max}_retrievals.parquet"
    return Path(cache_dir) / "retrievals" / name


def _to_frame(r: Retrievals) -> pd.DataFrame:
    """Serialize a :class:`Retrievals` to a one-row-per-query DataFrame."""
    return pd.DataFrame(
        {
            "query_id": r.query_ids,
            "scores": list(r.scores),
            "doc_ids": list(r.doc_ids),
            "doc_tokens": list(r.doc_tokens),
        }
    )


def _from_frame(df: pd.DataFrame) -> Retrievals:
    """Deserialize a DataFrame back into a :class:`Retrievals`."""
    return Retrievals(
        query_ids=list(df["query_id"]),
        scores=np.array([np.asarray(x, dtype=np.float32) for x in df["scores"]]),
        doc_ids=np.array([np.asarray(x, dtype=np.int64) for x in df["doc_ids"]]),
        doc_tokens=np.array([np.asarray(x, dtype=np.int64) for x in df["doc_tokens"]]),
    )


class RetrievalCache:
    """Builds, persists, and loads the top-``M_max`` retrievals for a dataset."""

    def __init__(
        self,
        cache_dir: str | Path,
        embedding_cache: EmbeddingCache,
        scorer: Scorer,
    ):
        self.cache_dir = Path(cache_dir)
        self.embedding_cache = embedding_cache
        self.scorer = scorer

    def path_for(self, dataset: str, M_max: int) -> Path:
        return _retrievals_path(
            self.cache_dir,
            dataset,
            self.embedding_cache.model_name,
            self.scorer.name,
            M_max,
        )

    def _build(
        self,
        dataset: str,
        documents: list[dict],
        queries: list[dict],
        M_max: int,
    ) -> Retrievals:
        doc_texts = [d["text"] for d in documents]
        query_texts = [q["query"] for q in queries]

        doc_emb = self.embedding_cache.embed_documents(doc_texts, dataset)
        query_emb = self.embedding_cache.embed_queries(query_texts, dataset)

        self.scorer.fit(doc_texts, doc_emb)
        scores, doc_ids = self.scorer.score_all(query_texts, query_emb, M_max)
        scores = np.asarray(scores, dtype=np.float32)
        doc_ids = np.asarray(doc_ids, dtype=np.int64)

        # Per-document token counts, looked up by the retrieved doc IDs. Counting
        # the whole corpus once and indexing keeps token cost a single source of
        # truth (tokens.count_tokens_many), identical to what the harness sums.
        corpus_tokens = count_tokens_many(doc_texts)
        doc_tokens = corpus_tokens[doc_ids]

        return Retrievals(
            query_ids=[q["query_id"] for q in queries],
            scores=scores,
            doc_ids=doc_ids,
            doc_tokens=doc_tokens,
        )

    def get(
        self,
        dataset: str,
        documents: list[dict],
        queries: list[dict],
        M_max: int,
    ) -> Retrievals:
        """Load cached retrievals, or build and persist them on a miss."""
        path = self.path_for(dataset, M_max)
        if path.exists():
            return _from_frame(pd.read_parquet(path))

        retrievals = self._build(dataset, documents, queries, M_max)
        path.parent.mkdir(parents=True, exist_ok=True)
        _to_frame(retrievals).to_parquet(path, index=False)
        return retrievals
