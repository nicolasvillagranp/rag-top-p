"""Analysis & plots — phase 5, deterministic from serialized sweep results.

Every figure and table is computed purely from the sweep parquet (no model, no
randomness), so they reproduce exactly. The data-shaping functions are separated
from the plotting so they can be unit-tested without rendering.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to files, never to a display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_CONFIG_KEYS = ["dataset", "model", "scorer", "p", "tau", "M", "k"]


# --- Data shaping (testable without rendering) ------------------------------


def tes_vs_p(results: pd.DataFrame) -> pd.DataFrame:
    """Mean TES as a function of p, per dataset (averaged over τ, M, k)."""
    return (
        results.groupby(["dataset", "p"])["TES"].mean().reset_index(name="mean_TES")
    )


def token_savings(results: pd.DataFrame) -> pd.DataFrame:
    """Per-row % token reduction vs the top-k baseline at the same k.

    ``100 * (1 - tokens_retrieved / tokens_top_k)``; non-negative by the guarantee.
    """
    out = results.copy()
    out["token_savings_pct"] = 100.0 * (
        1.0 - out["tokens_retrieved"] / out["tokens_top_k"]
    )
    return out


def fallback_rate_vs_p(results: pd.DataFrame) -> pd.DataFrame:
    """Mean fallback rate as a function of p, per dataset.

    Monotonically non-decreasing in ``p``: a larger probability-mass threshold
    selects a larger nucleus, so the ``|top-p| > k`` cap fires more often as
    p → 1.0.
    """
    return (
        results.groupby(["dataset", "p"])["fallback_triggered"]
        .mean()
        .reset_index(name="fallback_rate")
    )


def best_config_table(results: pd.DataFrame) -> pd.DataFrame:
    """Best ``(p, τ)`` configuration per dataset, ranked by mean TES.

    TES is averaged over M and k for each (dataset, p, τ); the top row per dataset
    is returned.
    """
    grouped = (
        results.groupby(["dataset", "p", "tau"])["TES"].mean().reset_index(name="mean_TES")
    )
    idx = grouped.groupby("dataset")["mean_TES"].idxmax()
    return grouped.loc[idx].sort_values("dataset").reset_index(drop=True)


# --- Rendering --------------------------------------------------------------


def _save(fig, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path


def plot_tes_vs_p(results: pd.DataFrame, output_dir: Path) -> Path:
    data = tes_vs_p(results)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(data=data, x="p", y="mean_TES", hue="dataset", marker="o", ax=ax)
    ax.axhline(1.0, ls="--", c="gray", lw=1, label="top-k break-even")
    ax.set(xlabel="p", ylabel="mean TES", title="TES vs p")
    return _save(fig, output_dir, "tes_vs_p.png")


def plot_token_savings(results: pd.DataFrame, output_dir: Path) -> Path:
    data = token_savings(results)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(data=data, x="dataset", y="token_savings_pct", ax=ax, cut=0)
    ax.set(xlabel="dataset", ylabel="% token reduction vs top-k", title="Token savings")
    return _save(fig, output_dir, "token_savings.png")


def plot_fallback_rate_vs_p(results: pd.DataFrame, output_dir: Path) -> Path:
    data = fallback_rate_vs_p(results)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(data=data, x="p", y="fallback_rate", hue="dataset", marker="o", ax=ax)
    ax.set(xlabel="p", ylabel="fallback rate", title="Fallback rate vs p")
    return _save(fig, output_dir, "fallback_rate_vs_p.png")


def write_summary_table(results: pd.DataFrame, output_dir: Path) -> Path:
    table = best_config_table(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "best_config.csv"
    table.to_csv(path, index=False)
    return path


def generate_all(results: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Produce every figure and the summary table; return the written paths."""
    output_dir = Path(output_dir)
    return [
        plot_tes_vs_p(results, output_dir),
        plot_token_savings(results, output_dir),
        plot_fallback_rate_vs_p(results, output_dir),
        write_summary_table(results, output_dir),
    ]
