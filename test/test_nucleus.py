"""Phase 1 tests — nucleus.py: softmax + top_p_select."""

import numpy as np
import pytest

from topp_rag.nucleus import softmax, top_p_select


class TestSoftmax:
    """Tests for temperature-scaled softmax."""

    def test_sums_to_one(self, peaked_scores, flat_scores):
        """Softmax output sums to 1.0 for any input array."""
        for scores in [peaked_scores, flat_scores]:
            for tau in [0.01, 0.1, 0.5, 1.0, 2.0, 10.0]:
                probs = softmax(scores, tau)
                assert np.isclose(probs.sum(), 1.0), f"Failed for tau={tau}"

    def test_low_temperature_concentrates_mass(self, peaked_scores):
        """τ → 0 concentrates all mass on the top document."""
        probs = softmax(peaked_scores, tau=0.01)
        # Top document should have nearly all mass
        assert probs[0] > 0.99

    def test_high_temperature_uniform(self, peaked_scores):
        """τ → ∞ produces a nearly uniform distribution."""
        probs = softmax(peaked_scores, tau=100.0)
        # All probabilities should be roughly equal
        expected_uniform = 1.0 / len(peaked_scores)
        assert np.allclose(probs, expected_uniform, atol=0.01)


class TestTopPSelect:
    """Tests for top-p document selection with k-fallback."""

    def test_cumsum_selects_smallest_prefix(self, peaked_scores):
        """Cumsum threshold correctly selects the smallest prefix with mass ≥ p."""
        # With peaked scores and moderate tau, top doc has most mass
        indices = top_p_select(peaked_scores, p=0.5, tau=0.5, k=10)
        # Should select few documents (peaked distribution)
        assert len(indices) <= 3
        # Verify the selected set actually has mass >= p
        probs = softmax(peaked_scores, tau=0.5)
        assert probs[indices].sum() >= 0.5

    def test_fallback_triggers_when_top_p_exceeds_k(self, flat_scores):
        """Fallback triggers correctly when |top-p| > k, returning exactly k."""
        # Flat scores with high p and small k → fallback
        indices = top_p_select(flat_scores, p=0.95, tau=1.0, k=3)
        assert len(indices) == 3

    def test_no_fallback_when_top_p_within_k(self, peaked_scores):
        """Fallback never triggers when |top-p| ≤ k."""
        # Peaked scores with moderate p → few docs selected, well under k=10
        indices = top_p_select(peaked_scores, p=0.5, tau=0.5, k=10)
        assert len(indices) < 10
        # Verify we're not artificially capping
        probs = softmax(peaked_scores, tau=0.5)
        cumsum = np.cumsum(probs)
        natural_n = int(np.searchsorted(cumsum, 0.5, side="left")) + 1
        assert len(indices) == natural_n

    def test_m_less_than_k(self):
        """Edge case: M < k — algorithm handles gracefully without error."""
        scores = np.array([0.8, 0.5, 0.3])  # M=3
        indices = top_p_select(scores, p=0.99, tau=1.0, k=10)  # k=10 > M=3
        # Should return at most M documents
        assert len(indices) <= 3

    def test_p_one_selects_all_capped_at_k(self):
        """Edge case: p = 1.0 — all M documents selected, then capped at k."""
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        indices = top_p_select(scores, p=1.0, tau=1.0, k=5)
        # p=1.0 wants all docs, but k=5 caps it
        assert len(indices) == 5

    def test_p_near_zero_selects_one(self, peaked_scores):
        """Edge case: p ≈ 0 — exactly 1 document returned."""
        # p very close to 0 — first doc's probability always exceeds it
        indices = top_p_select(peaked_scores, p=1e-10, tau=1.0, k=10)
        assert len(indices) == 1

    def test_indices_are_contiguous_from_zero(self, flat_scores):
        """Selected indices are always a contiguous prefix [0, 1, ..., n-1]."""
        indices = top_p_select(flat_scores, p=0.7, tau=1.0, k=8)
        expected = np.arange(len(indices))
        np.testing.assert_array_equal(indices, expected)
