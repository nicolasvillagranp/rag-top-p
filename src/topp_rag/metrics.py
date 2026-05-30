"""Retrieval-quality and efficiency metrics.

* :func:`recall` — the ``Retrieval_Quality`` term: the fraction of a query's gold
  documents present in the retrieved set, ``|retrieved ∩ gold| / |gold| ∈ [0, 1]``.
  Proportional (not binary), so on multi-gold queries (MultiHop-RAG, MuSiQue)
  retrieving more of the gold set scores higher. On single-gold datasets it
  reduces to the binary hit@retrieved.
* :func:`tes` — the Token Efficiency Score: retrieval quality per token, normalized
  to the top-k baseline. ``TES > 1`` means better quality per token than top-k.
"""

from __future__ import annotations

from collections.abc import Iterable


def recall(retrieved_doc_ids: Iterable[int], gold_doc_ids: Iterable[int]) -> float:
    """Fraction of gold documents present in the retrieved set, in ``[0, 1]``.

    ``|retrieved ∩ gold| / |gold|``. With no gold documents recall is undefined;
    we return 0.0.
    """
    gold = set(gold_doc_ids)
    if not gold:
        return 0.0
    hits = len(gold & set(retrieved_doc_ids))
    return hits / len(gold)


def tes(retrieval_quality: float, tokens_retrieved: int, tokens_top_k: int) -> float:
    """Token Efficiency Score: ``quality / (tokens_retrieved / tokens_top_k)``.

    Equivalently ``quality * tokens_top_k / tokens_retrieved``. When the top-p set
    is identical to top-k (e.g. full fallback) the token costs match and TES equals
    the retrieval quality — so a perfectly-recalled fallback query scores exactly
    1.0, the break-even point.

    Edge cases:
      * ``tokens_top_k == 0`` — the baseline retrieved nothing; efficiency is
        undefined, return 0.0.
      * ``tokens_retrieved == 0`` — top-p retrieved nothing (quality is then 0
        too); return 0.0 rather than divide by zero.
    """
    if tokens_top_k == 0 or tokens_retrieved == 0:
        return 0.0
    return retrieval_quality * tokens_top_k / tokens_retrieved
