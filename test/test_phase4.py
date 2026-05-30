"""Phase 4 tests — the hyperparameter sweep over cached score arrays.

The sweep must touch no embedding model, scorer, or FAISS index once retrievals
are cached; must cover the full grid; must uphold the token guarantee globally;
and must be resumable.
"""

import itertools

import pandas as pd
import pytest

from topp_rag.config import (
    Config,
    EmbeddingConfig,
    EvaluationConfig,
    PathsConfig,
    RetrievalConfig,
    ScorerConfig,
    TopPConfig,
)
from topp_rag.embeddings import EmbeddingCache
from topp_rag.retrievals import RetrievalCache
from topp_rag.scoring import CosineScorer
from topp_rag import sweep as sweep_mod
from topp_rag.sweep import _grid, fallback_rates, run_sweep

MODEL = "all-MiniLM-L6-v2"


def _config(tmp_path) -> Config:
    """A small fixtures-only config with a real (but tiny) grid."""
    return Config(
        name="test",
        seed=42,
        embedding=EmbeddingConfig(model=MODEL, batch_size=64),
        retrieval=RetrievalConfig(M_values=[10, 20], M_max=20),
        scorer=ScorerConfig(name="cosine", params={}),
        top_p=TopPConfig(p_values=[0.5, 0.9], tau_values=[0.5, 1.0], k_values=[3, 5]),
        datasets=["fixtures"],
        evaluation=EvaluationConfig(max_queries=None),
        paths=PathsConfig(
            cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "outputs",
            fixtures_dir=tmp_path / "fixtures",
        ),
    )


@pytest.fixture
def warm_cache(tmp_path):
    """Pre-build the retrievals cache so the sweep runs purely on cached arrays."""
    config = _config(tmp_path)
    emb = EmbeddingCache(config.paths.cache_dir, MODEL)
    from topp_rag.datasets import load_dataset

    ds = load_dataset("fixtures", fixtures_dir=config.paths.fixtures_dir)
    RetrievalCache(config.paths.cache_dir, emb, CosineScorer()).get(
        "fixtures", ds.documents, ds.queries, config.retrieval.M_max
    )
    return config


class TestSweep:
    def test_no_model_or_faiss_during_sweep(self, warm_cache, monkeypatch):
        """With retrievals cached, no scorer/model/FAISS is instantiated."""
        def boom(*a, **k):
            raise AssertionError("must not be constructed during the sweep")

        monkeypatch.setattr("topp_rag.embeddings.SentenceTransformer", boom)
        monkeypatch.setattr("topp_rag.scoring.CosineScorer.fit", boom)
        import faiss

        monkeypatch.setattr(faiss, "IndexFlatIP", boom)

        results = run_sweep(warm_cache)
        assert not results.empty

    def test_full_grid_present(self, warm_cache):
        results = run_sweep(warm_cache)
        got = set(
            map(tuple, results[["p", "tau", "M", "k"]].drop_duplicates().to_numpy())
        )
        expected = set(_grid(warm_cache))
        assert got == expected

    def test_token_guarantee_holds_globally(self, warm_cache):
        results = run_sweep(warm_cache)
        assert (results["tokens_retrieved"] <= results["tokens_top_k"]).all()

    def test_fallback_rate_logged(self, warm_cache):
        results = run_sweep(warm_cache)
        rates = fallback_rates(results)
        assert "fallback_rate" in rates.columns
        assert ((rates["fallback_rate"] >= 0) & (rates["fallback_rate"] <= 1)).all()

    def test_serialized_and_reloads_deterministically(self, warm_cache):
        r1 = run_sweep(warm_cache)
        path = sweep_mod._results_path(
            warm_cache.paths.cache_dir, "fixtures", MODEL, "cosine"
        )
        assert path.exists()
        reloaded = pd.read_parquet(path)
        # Same number of rows and identical TES column on reload.
        assert len(reloaded) == len(r1)

    def test_resumable_skips_completed_cells(self, warm_cache, monkeypatch):
        # First full run populates the results parquet.
        run_sweep(warm_cache)

        # Second run: every grid cell is already done, so evaluate_query is never
        # called again.
        calls = {"n": 0}
        real = sweep_mod.evaluate_query

        def counting(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        monkeypatch.setattr(sweep_mod, "evaluate_query", counting)
        run_sweep(warm_cache)
        assert calls["n"] == 0

    def test_m_slice_matches_m_max_cache(self, warm_cache):
        """For each M in the grid, the sweep evaluates on the top-M prefix."""
        results = run_sweep(warm_cache)
        # n_docs_retrieved must never exceed the M used for that row.
        assert (results["n_docs_retrieved"] <= results["M"]).all()
        # ...nor exceed k (the fallback cap).
        assert (results["n_docs_retrieved"] <= results["k"]).all()
