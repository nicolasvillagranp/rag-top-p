"""Phase 5 tests — analysis data-shaping and deterministic plot generation."""

import numpy as np
import pandas as pd
import pytest

from topp_rag.analysis import (
    best_config_table,
    fallback_rate_vs_p,
    generate_all,
    tes_vs_p,
    token_savings,
)


def _synthetic_results() -> pd.DataFrame:
    """A controlled sweep-results frame for two datasets.

    Fallback rate is constructed to rise with p (the correct direction), and one
    (p, τ) is made clearly best per dataset for the summary-table test.
    """
    rows = []
    for dataset, base in [("alpha", 0.0), ("beta", 0.5)]:
        for p in [0.5, 0.7, 0.9]:
            for tau in [0.5, 1.0]:
                # More queries fall back as p grows.
                fb_prob = {0.5: 0.1, 0.7: 0.4, 0.9: 0.8}[p]
                for qi in range(10):
                    fell_back = qi < int(round(fb_prob * 10))
                    # Make (p=0.7, tau=0.5) the TES winner for both datasets.
                    tes = 2.0 if (p == 0.7 and tau == 0.5) else 1.0 + base
                    rows.append(
                        {
                            "dataset": dataset,
                            "model": "m",
                            "scorer": "cosine",
                            "p": p,
                            "tau": tau,
                            "M": 20,
                            "k": 3,
                            "n_docs_retrieved": 3 if fell_back else 1,
                            "tokens_retrieved": 30 if fell_back else 10,
                            "tokens_top_k": 30,
                            "fallback_triggered": fell_back,
                            "recall": 1.0,
                            "TES": tes,
                        }
                    )
    return pd.DataFrame(rows)


class TestShaping:
    def test_tes_vs_p_per_dataset(self):
        df = tes_vs_p(_synthetic_results())
        assert set(df["dataset"]) == {"alpha", "beta"}
        assert {"p", "mean_TES"} <= set(df.columns)

    def test_token_savings_non_negative(self):
        df = token_savings(_synthetic_results())
        # By the guarantee, savings are always >= 0.
        assert (df["token_savings_pct"] >= -1e-9).all()

    def test_fallback_rate_monotonic_non_decreasing(self):
        df = fallback_rate_vs_p(_synthetic_results())
        for _, sub in df.sort_values("p").groupby("dataset"):
            rates = sub["fallback_rate"].to_numpy()
            assert np.all(np.diff(rates) >= -1e-9), "fallback rate must not decrease in p"

    def test_best_config_picks_max_tes(self):
        table = best_config_table(_synthetic_results())
        # One row per dataset, and the winner is (p=0.7, tau=0.5).
        assert len(table) == 2
        for _, row in table.iterrows():
            assert (row["p"], row["tau"]) == (0.7, 0.5)


class TestRendering:
    def test_generate_all_writes_files_deterministically(self, tmp_path):
        results = _synthetic_results()
        out = tmp_path / "outputs"

        paths1 = generate_all(results, out)
        for p in paths1:
            assert p.exists()

        # The CSV summary table is byte-identical on a second run (deterministic).
        csv = next(p for p in paths1 if p.suffix == ".csv")
        first = csv.read_bytes()
        generate_all(results, out)
        assert csv.read_bytes() == first
