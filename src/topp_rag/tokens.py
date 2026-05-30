"""Token counting — the single source of truth for token cost.

Both the retrievals cache (which precomputes per-document token counts) and the
evaluation harness (which sums tokens of the retrieved set) count tokens through
this module, so the two can never disagree.

Uses tiktoken's ``cl100k_base`` encoding — the tokenizer behind GPT-3.5/4 — as a
stable, model-agnostic proxy for LLM inference cost.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import tiktoken

_ENCODING_NAME = "cl100k_base"


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    """Lazily construct and cache the tiktoken encoding (it is not cheap)."""
    return tiktoken.get_encoding(_ENCODING_NAME)


def count_tokens(text: str) -> int:
    """Number of tokens in a single string."""
    return len(_encoding().encode(text))


def count_tokens_many(texts: list[str]) -> np.ndarray:
    """Token counts for a list of strings as an int array.

    Uses tiktoken's batched ``encode_ordinary_batch`` rather than a Python-level
    loop over ``encode``.
    """
    encoded = _encoding().encode_ordinary_batch(list(texts))
    return np.array([len(toks) for toks in encoded], dtype=np.int64)
