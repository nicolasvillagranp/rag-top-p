"""Nucleus (top-p) document selection over similarity scores.

Pure numpy — zero external dependencies beyond numpy.
"""

import numpy as np


def softmax(scores: np.ndarray, tau: float) -> np.ndarray:
    """Temperature-scaled softmax over similarity scores.

    Args:
        scores: 1-D array of similarity scores (descending order expected).
        tau: Temperature parameter. Lower → peakier, higher → flatter.

    Returns:
        Probability distribution (sums to 1.0).
    """
    scaled = scores / tau
    # Numerical stability: subtract max before exp
    scaled = scaled - scaled.max()
    exp_scores = np.exp(scaled)
    return exp_scores / exp_scores.sum()


def top_p_select(scores: np.ndarray, p: float, tau: float, k: int) -> np.ndarray:
    """Select documents using nucleus (top-p) sampling with k-fallback.

    Greedily accumulates documents in ranked order until cumulative
    probability mass >= p, then caps at k if the selected set is too large.

    Args:
        scores: 1-D array of similarity scores (descending order expected).
        p: Probability mass threshold in (0, 1].
        tau: Softmax temperature.
        k: Maximum number of documents (fallback cap).

    Returns:
        Array of selected indices (0-based, into the scores array).
    """
    probs = softmax(scores, tau)
    cumsum = np.cumsum(probs)

    # Smallest prefix with cumulative mass >= p
    # np.searchsorted finds first index where cumsum >= p
    n_selected = int(np.searchsorted(cumsum, p, side="left")) + 1

    # Cap at k (the guarantee: never worse than top-k)
    n_selected = min(n_selected, k)

    # Cap at available documents
    n_selected = min(n_selected, len(scores))

    return np.arange(n_selected)
