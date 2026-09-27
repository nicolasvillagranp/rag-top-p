"""Shared dataset schema.

Lives in its own module so both :mod:`topp_rag.datasets` (the dispatcher) and
:mod:`topp_rag.loaders` (the per-benchmark adapters) can import it without a
circular dependency.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoadedDataset:
    """One dataset in the unified schema.

    documents -> list[{doc_id: int, text: str}]   (doc_id is the corpus index)
    queries   -> list[{query_id: str, query: str, gold_doc_ids: list[int]}]
                 where gold_doc_ids index into ``documents``.
    """

    name: str
    documents: list[dict]
    queries: list[dict]
