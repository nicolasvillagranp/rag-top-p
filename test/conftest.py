"""Shared test fixtures for the topp_rag test suite."""

import numpy as np
import pytest


@pytest.fixture
def peaked_scores():
    """One dominant document — top-p should select very few."""
    return np.array([0.95, 0.30, 0.10, 0.08, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005])


@pytest.fixture
def flat_scores():
    """Uniformly relevant documents — top-p should select many."""
    return np.array([0.50, 0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41])


@pytest.fixture
def two_dominant_scores():
    """Two clear winners then a drop-off."""
    return np.array([0.90, 0.85, 0.20, 0.15, 0.10, 0.05, 0.04, 0.03, 0.02, 0.01])


@pytest.fixture
def single_score():
    """Edge case: only one document available."""
    return np.array([0.75])


# --- Phase 2+ fixtures: real fixture corpus + isolated cache dir -------------


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """An isolated cache directory so tests never touch the real data/cache."""
    return tmp_path / "cache"


@pytest.fixture
def fixtures_dir(tmp_path):
    """An isolated fixtures directory; the corpus is materialized on demand."""
    return tmp_path / "fixtures"


@pytest.fixture
def fixture_dataset(fixtures_dir):
    """The hand-crafted fixture dataset loaded into the unified schema."""
    from topp_rag.datasets import load_dataset

    return load_dataset("fixtures", fixtures_dir=fixtures_dir)
