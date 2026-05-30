"""Per-query evaluation — turns one (query, config) into the logged record.

Given the cached top-``M_max`` scores/ids/tokens for a single query and a top-p
configuration ``(p, τ, M, k)``, this computes the full set of per-query fields the
sweep logs (see README "Per-Query Logged Fields"): the top-p selection, the top-k
baseline at the same k, recall of each, token costs, fallback flag, and TES.

It operates purely on cached arrays — no embedding model, no scorer, no FAISS —
so it is the unit the in-memory sweep calls for every grid cell.
"""

from __future__ import annotations

import numpy as np

from .metrics import recall, tes
from .nucleus import top_p_select


def evaluate_query(
    *,
    dataset: str,
    query_id: str,
    gold_doc_ids: list[int],
    scores: np.ndarray,
    doc_ids: np.ndarray,
    doc_tokens: np.ndarray,
    p: float,
    tau: float,
    M: int,
    k: int,
) -> dict:
    """Evaluate one query under one ``(p, τ, M, k)`` configuration.

    Args:
        scores/doc_ids/doc_tokens: the cached top-``M_max`` arrays for this query
            (descending by score). Sliced to top-``M`` internally.
        gold_doc_ids: the query's gold document IDs.

    Returns:
        A dict with the per-query logged fields.
    """
    # Slice the candidate pool to top-M (a prefix of the M_max cache).
    s = np.asarray(scores[:M])
    ids = np.asarray(doc_ids[:M])
    toks = np.asarray(doc_tokens[:M])

    # Top-p selection (returns a contiguous prefix of indices into the pool).
    selected_idx = top_p_select(s, p=p, tau=tau, k=k)
    n_docs = len(selected_idx)
    selected_ids = ids[selected_idx]
    tokens_retrieved = int(toks[selected_idx].sum())

    # Top-k baseline at the same k: the first min(k, M) documents of the pool.
    k_eff = min(k, len(ids))
    topk_ids = ids[:k_eff]
    tokens_top_k = int(toks[:k_eff].sum())

    # Fallback fired when the top-p set was capped down to k by the guarantee.
    fallback_triggered = n_docs == k and n_docs < M

    q = recall(selected_ids.tolist(), gold_doc_ids)
    score = tes(q, tokens_retrieved, tokens_top_k)

    return {
        "dataset": dataset,
        "query_id": query_id,
        "p": p,
        "tau": tau,
        "M": M,
        "k": k,
        "n_docs_retrieved": n_docs,
        "tokens_retrieved": tokens_retrieved,
        "tokens_top_k": tokens_top_k,
        "fallback_triggered": fallback_triggered,
        "recall": q,
        "TES": score,
    }
