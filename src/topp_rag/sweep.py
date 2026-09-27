"""Hyperparameter sweep — layer 3, entirely in memory over cached scores.

For each dataset we load the cached top-``M_max`` retrievals once, then iterate the
full ``p × τ × M × k`` grid evaluating every query with :func:`evaluate_query`. No
embedding model, scorer, or FAISS index is touched inside the loop — the sweep is
pure numpy over the cached arrays.

Results are serialized per (dataset, model, scorer) to parquet and the sweep is
**resumable**: completed grid cells are detected on reload and skipped.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config import Config
from .datasets import load_dataset
from .embeddings import EmbeddingCache, _safe_model_name
from .evaluate import evaluate_query
from .retrievals import RetrievalCache, Retrievals
from .scoring import build_scorer

# The grid-cell key: a configuration is uniquely identified by these columns.
_CONFIG_KEYS = ["dataset", "model", "scorer", "p", "tau", "M", "k"]


def _results_path(cache_dir: Path, dataset: str, model: str, scorer: str) -> Path:
    name = f"{dataset}_{_safe_model_name(model)}_{scorer}_sweep_results.parquet"
    return Path(cache_dir) / "results" / name


def _grid(config: Config) -> list[tuple]:
    """All ``(p, τ, M, k)`` combinations from the config."""
    return list(
        itertools.product(
            config.top_p.p_values,
            config.top_p.tau_values,
            config.retrieval.M_values,
            config.top_p.k_values,
        )
    )


def _gold_map(queries: list[dict]) -> dict[str, list[int]]:
    return {q["query_id"]: q["gold_doc_ids"] for q in queries}


def _sweep_dataset(
    config: Config,
    dataset: str,
    retrievals: Retrievals,
    gold: dict[str, list[int]],
    done_cells: set[tuple],
) -> list[dict]:
    """Evaluate every (config-cell × query) for one dataset, skipping done cells."""
    model = config.embedding.model
    scorer_name = config.scorer.name
    rows: list[dict] = []

    grid = _grid(config)
    progress = tqdm(grid, desc=f"sweep:{dataset}", unit="cfg")
    for p, tau, M, k in progress:
        cell = (dataset, model, scorer_name, p, tau, M, k)
        if cell in done_cells:
            continue
        for i, query_id in enumerate(retrievals.query_ids):
            rec = evaluate_query(
                dataset=dataset,
                query_id=query_id,
                gold_doc_ids=gold[query_id],
                scores=retrievals.scores[i],
                doc_ids=retrievals.doc_ids[i],
                doc_tokens=retrievals.doc_tokens[i],
                p=p,
                tau=tau,
                M=M,
                k=k,
            )
            rec["model"] = model
            rec["scorer"] = scorer_name
            # Guarantee check: top-p must never cost more tokens than top-k.
            assert rec["tokens_retrieved"] <= rec["tokens_top_k"], (
                f"top-p exceeded top-k tokens for {query_id} at {cell}"
            )
            rows.append(rec)
    return rows


def _load_existing(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _done_cells(existing: pd.DataFrame) -> set[tuple]:
    if existing.empty:
        return set()
    return set(map(tuple, existing[_CONFIG_KEYS].drop_duplicates().to_numpy()))


def fallback_rates(results: pd.DataFrame) -> pd.DataFrame:
    """Fallback rate per (dataset, model, scorer, p, τ, M, k) configuration."""
    return (
        results.groupby(_CONFIG_KEYS)["fallback_triggered"]
        .mean()
        .reset_index(name="fallback_rate")
    )


def run_sweep(config: Config) -> pd.DataFrame:
    """Run the full grid over all configured datasets; return the combined results.

    Resumable per (dataset, model, scorer): existing results are loaded and their
    completed grid cells skipped, then new cells are appended and re-serialized.
    """
    embedding_cache = EmbeddingCache(
        config.paths.cache_dir, config.embedding.model, config.embedding.batch_size
    )
    all_frames: list[pd.DataFrame] = []

    for dataset in config.datasets:
        ds = load_dataset(
            dataset,
            fixtures_dir=config.paths.fixtures_dir,
            max_queries=config.evaluation.max_queries,
            seed=config.seed,
            n_distractors=config.retrieval.n_distractors,
        )
        scorer = build_scorer(config.scorer.name, config.scorer.params)
        retrieval_cache = RetrievalCache(config.paths.cache_dir, embedding_cache, scorer)
        retrievals = retrieval_cache.get(
            ds.name, ds.documents, ds.queries, config.retrieval.M_max
        )

        path = _results_path(
            config.paths.cache_dir, ds.name, config.embedding.model, config.scorer.name
        )
        existing = _load_existing(path)
        new_rows = _sweep_dataset(
            config, ds.name, retrievals, _gold_map(ds.queries), _done_cells(existing)
        )

        combined = pd.concat(
            [existing, pd.DataFrame(new_rows)], ignore_index=True
        ) if new_rows else existing

        if new_rows:
            path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(path, index=False)
        all_frames.append(combined)

    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
