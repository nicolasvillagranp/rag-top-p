"""Phase 3 tests — dataset schema, metrics, and the per-query eval harness."""

import numpy as np
import pytest

from topp_rag.datasets import load_dataset
from topp_rag.evaluate import evaluate_query
from topp_rag.metrics import recall, tes


class TestDatasetSchema:
    def test_fixtures_unified_schema(self, fixtures_dir):
        ds = load_dataset("fixtures", fixtures_dir=fixtures_dir)
        assert {d["doc_id"] for d in ds.documents} == set(range(len(ds.documents)))
        for q in ds.queries:
            assert set(q) >= {"query_id", "query", "gold_doc_ids"}
            # gold ids resolve to real corpus documents.
            for gid in q["gold_doc_ids"]:
                assert 0 <= gid < len(ds.documents)

    def test_max_queries_is_deterministic(self, fixtures_dir):
        a = load_dataset("fixtures", fixtures_dir=fixtures_dir, max_queries=5, seed=7)
        b = load_dataset("fixtures", fixtures_dir=fixtures_dir, max_queries=5, seed=7)
        assert len(a.queries) == 5
        assert [q["query_id"] for q in a.queries] == [q["query_id"] for q in b.queries]

    def test_nq_remains_a_seam(self, fixtures_dir):
        # NQ has no candidate pool; the other HF datasets are implemented and
        # covered by test_loaders.py.
        with pytest.raises(NotImplementedError):
            load_dataset("natural_questions", fixtures_dir=fixtures_dir)


class TestRecall:
    def test_binary_when_single_gold(self):
        assert recall([1, 2, 3], [2]) == 1.0
        assert recall([1, 2, 3], [9]) == 0.0

    def test_proportional_when_multi_gold(self):
        assert recall([1, 2, 3], [2, 4, 6]) == pytest.approx(1 / 3)
        assert recall([1, 2, 3], [1, 2, 6]) == pytest.approx(2 / 3)
        assert recall([1, 2, 3], [1, 2, 3]) == 1.0

    def test_no_gold_returns_zero(self):
        assert recall([1, 2], []) == 0.0


class TestTES:
    def test_equals_one_on_identical_token_cost_and_full_recall(self):
        # Same token cost as baseline, perfect recall -> break-even TES = 1.0.
        assert tes(1.0, tokens_retrieved=100, tokens_top_k=100) == pytest.approx(1.0)

    def test_greater_than_one_when_fewer_tokens_same_recall(self):
        assert tes(1.0, tokens_retrieved=40, tokens_top_k=100) == pytest.approx(2.5)

    def test_zero_token_baseline_is_safe(self):
        assert tes(1.0, tokens_retrieved=10, tokens_top_k=0) == 0.0
        assert tes(1.0, tokens_retrieved=0, tokens_top_k=10) == 0.0


class TestEvaluateQuery:
    """The per-query harness on a synthetic, fully-controlled score row."""

    def _row(self):
        # 6 candidates, descending scores, known token costs and doc ids.
        scores = np.array([0.9, 0.8, 0.2, 0.15, 0.1, 0.05], dtype=np.float32)
        doc_ids = np.array([10, 11, 12, 13, 14, 15], dtype=np.int64)
        doc_tokens = np.array([5, 5, 5, 5, 5, 5], dtype=np.int64)
        return scores, doc_ids, doc_tokens

    def test_tes_one_when_fallback_to_topk(self):
        scores, doc_ids, doc_tokens = self._row()
        # Flat-ish high-p, low tau spreads mass; k forces fallback so the set
        # equals top-k -> identical tokens -> TES == recall == 1.0 (gold in topk).
        rec = evaluate_query(
            dataset="fixtures", query_id="q", gold_doc_ids=[10],
            scores=scores, doc_ids=doc_ids, doc_tokens=doc_tokens,
            p=1.0, tau=10.0, M=6, k=3,
        )
        assert rec["fallback_triggered"] is True
        assert rec["n_docs_retrieved"] == 3
        assert rec["tokens_retrieved"] == rec["tokens_top_k"]
        assert rec["recall"] == 1.0
        assert rec["TES"] == pytest.approx(1.0)

    def test_tes_above_one_when_top_p_prunes(self):
        scores, doc_ids, doc_tokens = self._row()
        # Peaked: low p + low tau -> 1 doc selected, gold is doc 10.
        rec = evaluate_query(
            dataset="fixtures", query_id="q", gold_doc_ids=[10],
            scores=scores, doc_ids=doc_ids, doc_tokens=doc_tokens,
            p=0.3, tau=0.3, M=6, k=3,
        )
        assert rec["n_docs_retrieved"] == 1
        assert rec["recall"] == 1.0
        # 1 doc (5 tokens) vs top-3 baseline (15 tokens) -> TES = 3.0.
        assert rec["tokens_top_k"] == 15
        assert rec["tokens_retrieved"] == 5
        assert rec["TES"] == pytest.approx(3.0)

    def test_top_m_slice_respected(self):
        scores, doc_ids, doc_tokens = self._row()
        # M=2 caps the candidate pool to the first two docs.
        rec = evaluate_query(
            dataset="fixtures", query_id="q", gold_doc_ids=[12],
            scores=scores, doc_ids=doc_ids, doc_tokens=doc_tokens,
            p=0.99, tau=10.0, M=2, k=5,
        )
        # doc 12 is outside the M=2 pool -> not retrievable.
        assert rec["recall"] == 0.0
        assert rec["n_docs_retrieved"] <= 2
