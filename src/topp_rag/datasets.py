"""Unified dataset loading.

Every dataset — the hand-crafted fixtures now, the four HuggingFace benchmarks
later — loads into one schema so the rest of the pipeline (embedding, scoring,
sweep, evaluation) is dataset-agnostic:

    LoadedDataset
      .documents  -> list[{doc_id: int, text: str}]
      .queries    -> list[{query_id: str, query: str, gold_doc_ids: list[int]}]

``gold_doc_ids`` index into ``documents``. The ``fixtures`` source plus MuSiQue,
MultiHop-RAG, and TriviaQA are implemented (see :mod:`topp_rag.loaders`). Natural
Questions remains a seam: its clean HF variants are answer-only, so there is no
candidate pool for top-p to select among.
"""

from __future__ import annotations

import random
from pathlib import Path

from . import corpus, loaders
from .schema import LoadedDataset

# Datasets with a real loader implemented (see topp_rag.loaders). Each maps a
# dataset name to a callable (max_queries, seed[, n_distractors]) -> LoadedDataset.
_HF_LOADERS = {
    # native-pool datasets (gold + distractors ship together)
    "musique": loaders.load_musique,
    "multihop_rag": loaders.load_multihop_rag,
    "triviaqa": loaders.load_triviaqa,
    "hotpotqa": loaders.load_hotpotqa,
    "2wikimultihopqa": loaders.load_2wikimultihopqa,
    # BEIR datasets (pool synthesized: gold from qrels + n_distractors)
    "scifact": loaders.load_scifact,
    "fever": loaders.load_fever,
    "nfcorpus": loaders.load_nfcorpus,
    "scidocs": loaders.load_scidocs,
    "fiqa": loaders.load_fiqa,
}

# Datasets whose candidate pool is synthesized from a corpus and therefore honor
# the ``n_distractors`` knob (others ship a native pool and ignore it).
_SYNTHETIC_POOL = ("scifact", "fever", "nfcorpus", "scidocs", "fiqa")

# Final benchmark names — referenced by config and the sweep. natural_questions
# stays a seam (answer-only on HF; no candidate pool).
HF_DATASETS = ("natural_questions", *_HF_LOADERS)


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
    The subset preserves the original order so downstream arrays stay stable.
    """
    if max_queries is None or max_queries >= len(queries):
        return queries
    rng = random.Random(seed)
    chosen = set(rng.sample(range(len(queries)), max_queries))
    return [q for i, q in enumerate(queries) if i in chosen]


def load_dataset(
    name: str,
    fixtures_dir: str | Path,
    max_queries: int | None = None,
    seed: int = 42,
    n_distractors: int = 49,
) -> LoadedDataset:
    """Load a dataset by name into the unified schema.

    Args:
        name: ``"fixtures"`` or one of :data:`HF_DATASETS`. ``natural_questions``
            is a seam (its clean HF variants are answer-only, no candidate pool).
        fixtures_dir: where the fixture JSON lives / is written.
        max_queries: cap on number of queries (deterministically sampled).
        seed: sampling seed.
        n_distractors: distractors per query for synthetic-pool datasets
            (SciFact); ignored by datasets that ship a native pool.

    The HF loaders sample queries internally and build the candidate-pool corpus
    from only the sampled queries' passages, so the corpus stays bounded.
    """
    if name == "fixtures":
        ds = _load_fixtures(fixtures_dir)
        return LoadedDataset(
            ds.name, ds.documents, _sample_queries(ds.queries, max_queries, seed)
        )

    if name in _HF_LOADERS:
        kwargs = {"max_queries": max_queries, "seed": seed}
        if name in _SYNTHETIC_POOL:
            kwargs["n_distractors"] = n_distractors
        return _HF_LOADERS[name](**kwargs)

    if name == "natural_questions":
        raise NotImplementedError(
            "natural_questions has no candidate pool: the clean HF variants "
            "(nq_open, sentence-transformers/natural-questions) are answer-only, "
            "so there is nothing for top-p to select among. Use musique, "
            "multihop_rag, or triviaqa, which ship per-query candidate pools."
        )

    raise ValueError(
        f"Unknown dataset '{name}'. Expected 'fixtures' or one of {HF_DATASETS}."
    )