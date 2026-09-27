"""Phase 2 tests — embedding cache, scorer abstraction, retrievals cache.

Mirrors the README Phase 2 contract. The embedding model is mocked in cache/batch
tests for speed and determinism; the end-to-end retrieval tests use the real
``all-MiniLM-L6-v2`` model on the tiny fixture corpus (fast on CPU).
"""

import numpy as np
import pytest

from topp_rag.datasets import load_dataset
from topp_rag.embeddings import EmbeddingCache
from topp_rag.nucleus import top_p_select
from topp_rag.retrievals import RetrievalCache
from topp_rag.scoring import CosineScorer, HybridScorer, Scorer, build_scorer
from topp_rag.tokens import count_tokens_many

MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384  # all-MiniLM-L6-v2 output dimension


# --- Fake model / scorer used to assert behaviour without heavy compute ------


class FakeSentenceTransformer:
    """Records encode calls; returns deterministic normalized vectors."""

    instances = 0
    last_batch_size = None

    def __init__(self, model_name):
        type(self).instances += 1
        self.model_name = model_name

    def encode(self, texts, batch_size=32, show_progress_bar=False,
               normalize_embeddings=False, convert_to_numpy=True):
        type(self).last_batch_size = batch_size
        rng = np.random.default_rng(abs(hash(tuple(texts))) % (2**32))
        vecs = rng.standard_normal((len(texts), 8)).astype(np.float32)
        if normalize_embeddings:
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs


@pytest.fixture
def patched_st(monkeypatch):
    """Patch SentenceTransformer in the embeddings module with the fake."""
    FakeSentenceTransformer.instances = 0
    FakeSentenceTransformer.last_batch_size = None
    monkeypatch.setattr("topp_rag.embeddings.SentenceTransformer", FakeSentenceTransformer)
    return FakeSentenceTransformer


class TestEmbeddingCache:
    def test_batch_size_64(self, patched_st, tmp_cache_dir):
        cache = EmbeddingCache(tmp_cache_dir, MODEL)
        cache.embed_documents(["a", "b", "c"], dataset="fixtures")
        assert patched_st.last_batch_size == 64

    def test_written_then_loaded_without_reconstructing_model(self, patched_st, tmp_cache_dir):
        texts = ["a", "b", "c"]
        # First run: cache miss -> model constructed once.
        cache1 = EmbeddingCache(tmp_cache_dir, MODEL)
        emb1 = cache1.embed_documents(texts, dataset="fixtures")
        assert patched_st.instances == 1

        # Second run (fresh cache object): cache hit -> model never constructed.
        cache2 = EmbeddingCache(tmp_cache_dir, MODEL)
        emb2 = cache2.embed_documents(texts, dataset="fixtures")
        assert patched_st.instances == 1  # unchanged
        np.testing.assert_array_equal(emb1, emb2)

    def test_changing_model_regenerates_cache(self, patched_st, tmp_cache_dir):
        texts = ["a", "b"]
        EmbeddingCache(tmp_cache_dir, "model-A").embed_documents(texts, "fixtures")
        EmbeddingCache(tmp_cache_dir, "model-B").embed_documents(texts, "fixtures")
        # Two different model names -> two constructions, two cache files.
        assert patched_st.instances == 2
        files = list((tmp_cache_dir / "embeddings").glob("*_doc_embeddings.npy"))
        assert len(files) == 2


class TestCosineScorer:
    def test_search_called_once_as_matrix(self, mocker):
        scorer = CosineScorer()
        docs = np.random.default_rng(0).standard_normal((10, 8)).astype(np.float32)
        docs /= np.linalg.norm(docs, axis=1, keepdims=True)
        queries = docs[:4]
        scorer.fit(["d"] * 10, docs)

        spy = mocker.spy(scorer._index, "search")
        scorer.score_all(["q"] * 4, queries, M_max=5)
        assert spy.call_count == 1
        # Called with the full query matrix, not per-query.
        called_queries = spy.call_args[0][0]
        assert called_queries.shape == (4, 8)

    def test_scores_descending(self):
        scorer = CosineScorer()
        rng = np.random.default_rng(1)
        docs = rng.standard_normal((20, 8)).astype(np.float32)
        docs /= np.linalg.norm(docs, axis=1, keepdims=True)
        scorer.fit(["d"] * 20, docs)
        scores, _ = scorer.score_all(["q"] * 3, docs[:3], M_max=10)
        for row in scores:
            assert np.all(np.diff(row) <= 1e-6)

    def test_top_m_is_slice_of_m_max(self):
        scorer = CosineScorer()
        rng = np.random.default_rng(2)
        docs = rng.standard_normal((20, 8)).astype(np.float32)
        docs /= np.linalg.norm(docs, axis=1, keepdims=True)
        scorer.fit(["d"] * 20, docs)
        queries = docs[:5]
        s_max, ids_max = scorer.score_all(["q"] * 5, queries, M_max=15)
        s_small, ids_small = scorer.score_all(["q"] * 5, queries, M_max=5)
        np.testing.assert_array_equal(ids_max[:, :5], ids_small)
        np.testing.assert_allclose(s_max[:, :5], s_small, atol=1e-6)


class TestHybridSeam:
    def test_hybrid_is_a_scorer_but_unimplemented(self):
        h = HybridScorer(alpha=0.5)
        assert isinstance(h, Scorer)  # satisfies the runtime-checkable protocol
        assert h.name == "hybrid_a0.5"
        with pytest.raises(NotImplementedError):
            h.score_all(["q"], np.zeros((1, 8), dtype=np.float32), M_max=5)

    def test_alpha_must_be_in_open_unit_interval(self):
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError):
                HybridScorer(alpha=bad)

    def test_build_scorer_dispatch(self):
        assert isinstance(build_scorer("cosine"), CosineScorer)
        assert isinstance(build_scorer("hybrid", {"alpha": 0.3}), HybridScorer)
        with pytest.raises(ValueError):
            build_scorer("nope")


class FakeScorer:
    """A minimal Scorer with no FAISS/cosine coupling — returns fixed arrays."""

    name = "fake"

    def __init__(self, n_docs):
        self.n_docs = n_docs
        self.fitted = False

    def fit(self, corpus_texts, corpus_embeddings):
        self.fitted = True

    def score_all(self, query_texts, query_embeddings, M_max):
        n = len(query_texts)
        m = min(M_max, self.n_docs)
        scores = np.tile(np.linspace(1.0, 0.0, m, dtype=np.float32), (n, 1))
        doc_ids = np.tile(np.arange(m, dtype=np.int64), (n, 1))
        return scores, doc_ids


class TestRetrievalCache:
    def test_accepts_any_scorer_no_cosine_coupling(self, patched_st, tmp_cache_dir, fixture_dataset):
        """RetrievalCache works against an injected fake scorer (no FAISS)."""
        emb = EmbeddingCache(tmp_cache_dir, MODEL)
        n_docs = len(fixture_dataset.documents)
        cache = RetrievalCache(tmp_cache_dir, emb, FakeScorer(n_docs))
        r = cache.get("fixtures", fixture_dataset.documents, fixture_dataset.queries, M_max=10)
        assert r.scores.shape == (len(fixture_dataset.queries), 10)
        assert r.doc_ids.shape == (len(fixture_dataset.queries), 10)

    def test_cache_holds_exactly_m_max_and_tokens_consistent(self, tmp_cache_dir, fixture_dataset):
        emb = EmbeddingCache(tmp_cache_dir, MODEL)
        cache = RetrievalCache(tmp_cache_dir, emb, CosineScorer())
        M_max = 10
        r = cache.get("fixtures", fixture_dataset.documents, fixture_dataset.queries, M_max=M_max)
        n_q = len(fixture_dataset.queries)
        assert r.scores.shape == (n_q, M_max)

        # doc_tokens align with a fresh token count of the retrieved docs.
        corpus_tokens = count_tokens_many([d["text"] for d in fixture_dataset.documents])
        np.testing.assert_array_equal(r.doc_tokens, corpus_tokens[r.doc_ids])

    def test_scores_descending_per_query(self, tmp_cache_dir, fixture_dataset):
        emb = EmbeddingCache(tmp_cache_dir, MODEL)
        cache = RetrievalCache(tmp_cache_dir, emb, CosineScorer())
        r = cache.get("fixtures", fixture_dataset.documents, fixture_dataset.queries, M_max=10)
        for row in r.scores:
            assert np.all(np.diff(row) <= 1e-6)

    def test_persisted_and_reloaded_identically(self, tmp_cache_dir, fixture_dataset):
        emb = EmbeddingCache(tmp_cache_dir, MODEL)
        cache = RetrievalCache(tmp_cache_dir, emb, CosineScorer())
        args = ("fixtures", fixture_dataset.documents, fixture_dataset.queries, 10)
        r1 = cache.get(*args)
        assert cache.path_for("fixtures", 10).exists()
        r2 = cache.get(*args)  # second call loads from parquet
        np.testing.assert_allclose(r1.scores, r2.scores, atol=1e-6)
        np.testing.assert_array_equal(r1.doc_ids, r2.doc_ids)
        np.testing.assert_array_equal(r1.doc_tokens, r2.doc_tokens)

    def test_full_pipeline_returns_at_most_k(self, tmp_cache_dir, fixture_dataset):
        emb = EmbeddingCache(tmp_cache_dir, MODEL)
        cache = RetrievalCache(tmp_cache_dir, emb, CosineScorer())
        r = cache.get("fixtures", fixture_dataset.documents, fixture_dataset.queries, M_max=10)
        k = 5
        for row_scores in r.scores:
            selected = top_p_select(row_scores, p=0.9, tau=1.0, k=k)
            assert len(selected) <= k


class TestRetrievalRecallSmoke:
    """The gold document should rank highly for fixture queries (cosine works)."""

    def test_gold_in_top_5(self, tmp_cache_dir, fixture_dataset):
        emb = EmbeddingCache(tmp_cache_dir, MODEL)
        cache = RetrievalCache(tmp_cache_dir, emb, CosineScorer())
        r = cache.get("fixtures", fixture_dataset.documents, fixture_dataset.queries, M_max=10)
        hits = 0
        for q, retrieved in zip(fixture_dataset.queries, r.doc_ids):
            if set(q["gold_doc_ids"]) & set(retrieved[:5].tolist()):
                hits += 1
        # The fixtures are designed so cosine recalls the gold doc easily.
        assert hits / len(fixture_dataset.queries) >= 0.9
