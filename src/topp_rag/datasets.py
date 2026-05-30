"""Unified dataset loading.

Every dataset — the hand-crafted fixtures now, the four HuggingFace benchmarks
later — loads into one schema so the rest of the pipeline (embedding, scoring,
sweep, evaluation) is dataset-agnostic:

    LoadedDataset
      .documents  -> list[{doc_id: int, text: str}]
      .queries    -> list[{query_id: str, query: str, gold_doc_ids: list[int]}]

``gold_doc_ids`` index into ``documents``. Only the ``fixtures`` source is
implemented end-to-end; the real benchmarks are seams that raise a clear
``NotImplementedError`` until their loaders are written, so the harness, sweep,
and config already reference them by their final names.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from . import corpus

# Final benchmark names — referenced by config and the sweep before the real
# loaders exist, so nothing has to be renamed when they land.
HF_DATASETS = ("natural_questions", "triviaqa", "multihop_rag", "musique")


@dataclass(frozen=True)
class LoadedDataset:
    name: str
    documents: list[dict]
    queries: list[dict]


def _load_fixtures(fixtures_dir: str | Path) -> LoadedDataset:
    """Load the hand-crafted fixture corpus, materializing it if absent."""
    corpus.write_fixtures(fixtures_dir)
    return LoadedDataset(
        name="fixtures",
        documents=corpus.documents(),
        queries=corpus.queries(),
    )


def _sample_queries(queries: list[dict], max_queries: int | None, seed: int) -> list[dict]:
    """Deterministically sample up to ``max_queries`` queries.

    Sampling is seeded so a (dataset, max_queries, seed) triple always yields the
    same subset — a precondition for reproducible retrieval and sweep results.
    """
    if max_queries is None or max_queries >= len(queries):
        return queries
    rng = random.Random(seed)
    return rng.sample(queries, max_queries)


def load_dataset(
    name: str,
    fixtures_dir: str | Path,
    max_queries: int | None = None,
    seed: int = 42,
) -> LoadedDataset:
    """Load a dataset by name into the unified schema.

    Args:
        name: ``"fixtures"`` (implemented) or one of :data:`HF_DATASETS`.
        fixtures_dir: where the fixture JSON lives / is written.
        max_queries: cap on number of queries (deterministically sampled).
        seed: sampling seed.
    """
    if name == "fixtures":
        ds = _load_fixtures(fixtures_dir)
    elif name in HF_DATASETS:
        raise NotImplementedError(
            f"Real loader for '{name}' is not implemented yet. "
            f"Build everything against 'fixtures' first; the HuggingFace "
            f"loaders slot in here behind the same LoadedDataset schema."
        )
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Expected 'fixtures' or one of {HF_DATASETS}."
        )

    sampled = _sample_queries(ds.queries, max_queries, seed)
    return LoadedDataset(name=ds.name, documents=ds.documents, queries=sampled)