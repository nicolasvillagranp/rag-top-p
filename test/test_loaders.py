"""Tests for the real benchmark loaders.

HuggingFace access is mocked with tiny fixtures shaped like the real datasets, so
these run offline and fast. A unified-schema conformance check is shared across
all three implemented loaders.
"""

import json

import pytest

from topp_rag import loaders
from topp_rag.datasets import load_dataset
from topp_rag.loaders import (
    CorpusBuilder,
    _answer_strings,
    _contains_answer,
    _truthy,
    load_2wikimultihopqa,
    load_fever,
    load_hotpotqa,
    load_multihop_rag,
    load_musique,
    load_scifact,
    load_triviaqa,
)
from topp_rag.schema import LoadedDataset


# --- pure helpers -----------------------------------------------------------


class TestHelpers:
    def test_corpus_builder_dedups_on_stripped_text(self):
        b = CorpusBuilder()
        assert b.intern("hello ") == 0
        assert b.intern("hello") == 0
        assert b.intern("world") == 1
        assert b.documents() == [
            {"doc_id": 0, "text": "hello"},
            {"doc_id": 1, "text": "world"},
        ]

    def test_truthy_handles_bool_and_string(self):
        assert _truthy(True) and _truthy("True") and _truthy("true")
        assert not _truthy(False) and not _truthy("False") and not _truthy("")

    def test_answer_strings_collects_value_and_aliases(self):
        ans = {"value": "Paris", "aliases": ["paris, france"], "normalized_aliases": []}
        assert _answer_strings(ans) == ["Paris", "paris, france"]
        assert _answer_strings("plain string") == ["plain string"]

    def test_contains_answer_is_word_boundary_aware(self):
        assert _contains_answer("the Eiffel Tower in Paris", ["Paris"])
        assert not _contains_answer("parisian cafe", ["Paris"])  # boundary


# --- schema conformance shared across loaders -------------------------------


def _assert_unified_schema(ds: LoadedDataset):
    doc_ids = {d["doc_id"] for d in ds.documents}
    assert doc_ids == set(range(len(ds.documents)))  # contiguous from 0
    for q in ds.queries:
        assert set(q) >= {"query_id", "query", "gold_doc_ids"}
        assert isinstance(q["query_id"], str)
        for gid in q["gold_doc_ids"]:
            assert gid in doc_ids  # gold resolves to a real corpus doc


# --- MuSiQue ----------------------------------------------------------------

_MUSIQUE_ROWS = [
    {
        "id": "2hop__1",
        "question": "Who is the spouse of the Green performer?",
        "paragraphs": [
            {"idx": 0, "title": "A", "paragraph_text": "Grant Green played jazz.", "is_supporting": True},
            {"idx": 1, "title": "B", "paragraph_text": "Distractor about cooking.", "is_supporting": False},
            {"idx": 2, "title": "C", "paragraph_text": "His spouse was Ann Green.", "is_supporting": "True"},
        ],
    },
]


class _FakeHFDataset(list):
    """A list that also supports the tiny HF Dataset API the loaders use."""

    def shuffle(self, seed):
        import random

        out = list(self)
        random.Random(seed).shuffle(out)
        return _FakeHFDataset(out)

    def select(self, indices):
        return _FakeHFDataset(self[i] for i in indices)


class TestMuSiQue:
    def test_supporting_paragraphs_are_gold(self, monkeypatch):
        monkeypatch.setattr(
            loaders, "_load_hf", lambda *a, **k: _FakeHFDataset(_MUSIQUE_ROWS)
        )
        ds = load_musique()
        _assert_unified_schema(ds)
        assert ds.name == "musique"
        q = ds.queries[0]
        # Two supporting paragraphs (bool True and string "True") -> 2 gold ids.
        assert len(q["gold_doc_ids"]) == 2
        gold_texts = {ds.documents[g]["text"] for g in q["gold_doc_ids"]}
        assert "Grant Green played jazz." in gold_texts
        assert "His spouse was Ann Green." in gold_texts


# --- MultiHop-RAG -----------------------------------------------------------

_MULTIHOP_CORPUS = [
    {"title": "Article X", "body": "Body of article X about FTX."},
    {"title": "Article Y", "body": "Body of article Y about markets."},
    {"title": "Article Z", "body": "Unrelated article Z."},
]
_MULTIHOP_QUERIES = [
    {
        "query": "Who faced the FTX trial?",
        "answer": "SBF",
        "evidence_list": [{"title": "Article X"}, {"title": "Article Y"}],
    },
]


class TestMultiHopRAG:
    def test_evidence_titles_map_to_corpus(self, monkeypatch, tmp_path):
        corpus_p = tmp_path / "corpus.json"
        queries_p = tmp_path / "MultiHopRAG.json"
        corpus_p.write_text(json.dumps(_MULTIHOP_CORPUS), encoding="utf-8")
        queries_p.write_text(json.dumps(_MULTIHOP_QUERIES), encoding="utf-8")

        def fake_download(repo, filename, repo_type):
            return str(corpus_p if filename == "corpus.json" else queries_p)

        monkeypatch.setattr(loaders, "hf_hub_download", fake_download)
        ds = load_multihop_rag()
        _assert_unified_schema(ds)
        assert ds.name == "multihop_rag"
        # The whole news corpus is kept (fixed shared pool).
        assert len(ds.documents) == 3
        # Gold = the two evidence articles X and Y (ids 0 and 1).
        assert ds.queries[0]["gold_doc_ids"] == [0, 1]


# --- TriviaQA ---------------------------------------------------------------

_TRIVIAQA_ROWS = [
    {
        "question_id": "tq_1",
        "question": "Who created the Chipmunks?",
        "answer": {"value": "Ross Bagdasarian", "aliases": ["Dave Seville"], "normalized_aliases": []},
        "entity_pages": {"wiki_context": ["Ross Bagdasarian created the Chipmunks."]},
        "search_results": {"search_context": ["A page mentioning Dave Seville.", "Irrelevant text."]},
    },
]


class TestTriviaQA:
    def test_distant_gold_is_answer_containing_passages(self, monkeypatch):
        monkeypatch.setattr(
            loaders, "_load_hf", lambda *a, **k: _FakeHFDataset(_TRIVIAQA_ROWS)
        )
        ds = load_triviaqa()
        _assert_unified_schema(ds)
        assert ds.name == "triviaqa"
        q = ds.queries[0]
        gold_texts = {ds.documents[g]["text"] for g in q["gold_doc_ids"]}
        # Both the wiki passage (answer value) and the search passage (alias) are gold.
        assert "Ross Bagdasarian created the Chipmunks." in gold_texts
        assert "A page mentioning Dave Seville." in gold_texts
        # The irrelevant passage is a distractor, not gold.
        assert "Irrelevant text." not in gold_texts


# --- HotpotQA ---------------------------------------------------------------

_HOTPOT_ROWS = [
    {
        "id": "hp_1",
        "question": "Which film director is older?",
        "answer": "Scott Derrickson",
        "supporting_facts": {"title": ["Scott Derrickson", "Ed Wood"], "sent_id": [0, 0]},
        "context": {
            "title": ["Scott Derrickson", "Ed Wood", "Distractor A", "Distractor B"],
            "sentences": [
                ["Scott Derrickson is a director.", " Born 1966."],
                ["Ed Wood was a filmmaker."],
                ["Unrelated paragraph about cooking."],
                ["Another unrelated paragraph."],
            ],
        },
    },
]


class TestHotpotQA:
    def test_supporting_titles_are_gold(self, monkeypatch):
        monkeypatch.setattr(
            loaders, "_load_hf", lambda *a, **k: _FakeHFDataset(_HOTPOT_ROWS)
        )
        ds = load_hotpotqa()
        _assert_unified_schema(ds)
        assert ds.name == "hotpotqa"
        q = ds.queries[0]
        # Two supporting paragraphs are gold; the two distractors are not.
        assert len(q["gold_doc_ids"]) == 2
        gold_texts = {ds.documents[g]["text"] for g in q["gold_doc_ids"]}
        assert "Scott Derrickson is a director. Born 1966." in gold_texts
        assert "Ed Wood was a filmmaker." in gold_texts

    def test_2wiki_uses_same_schema(self, monkeypatch):
        # 2WikiMultihopQA shares HotpotQA's fields, so the same fake rows work.
        monkeypatch.setattr(
            loaders, "_load_hf", lambda *a, **k: _FakeHFDataset(_HOTPOT_ROWS)
        )
        ds = load_2wikimultihopqa()
        _assert_unified_schema(ds)
        assert ds.name == "2wikimultihopqa"
        assert len(ds.queries[0]["gold_doc_ids"]) == 2


# --- SciFact ----------------------------------------------------------------

_SCIFACT_CORPUS = [{"_id": str(i), "title": "", "text": f"corpus passage {i}"} for i in range(10)]
_SCIFACT_QUERIES = [
    {"_id": "100", "title": "", "text": "claim one"},
    {"_id": "200", "title": "", "text": "claim two"},
]
_SCIFACT_QRELS = [
    {"query-id": 100, "corpus-id": 3, "score": 1},
    {"query-id": 200, "corpus-id": 7, "score": 1},
]


def _fake_scifact_load_hf(repo, name, split):
    if repo.endswith("-qrels"):
        return _SCIFACT_QRELS
    if name == "corpus":
        return _SCIFACT_CORPUS
    if name == "queries":
        return _SCIFACT_QUERIES
    raise AssertionError(f"unexpected load: {repo}, {name}, {split}")


class TestSciFact:
    def test_pool_is_gold_plus_n_distractors(self, monkeypatch):
        monkeypatch.setattr(loaders, "_load_hf", _fake_scifact_load_hf)
        ds = load_scifact(n_distractors=4)
        _assert_unified_schema(ds)
        assert ds.name == "scifact"
        assert len(ds.queries) == 2
        for q in ds.queries:
            assert len(q["gold_doc_ids"]) == 1  # single-gold per claim
        # Each query references gold + up to n_distractors docs; corpus is their
        # deduplicated union and stays small.
        assert len(ds.documents) <= 2 * (1 + 4)

    def test_n_distractors_capped_at_available(self, monkeypatch):
        monkeypatch.setattr(loaders, "_load_hf", _fake_scifact_load_hf)
        # Ask for far more distractors than the 9 non-gold corpus docs.
        ds = load_scifact(n_distractors=1000)
        # No crash; corpus is bounded by the actual corpus size.
        assert len(ds.documents) <= len(_SCIFACT_CORPUS)

    def test_gold_doc_contains_expected_passage(self, monkeypatch):
        monkeypatch.setattr(loaders, "_load_hf", _fake_scifact_load_hf)
        ds = load_scifact(n_distractors=2)
        by_qid = {q["query_id"]: q for q in ds.queries}
        gold_texts = {ds.documents[g]["text"] for g in by_qid["100"]["gold_doc_ids"]}
        assert "corpus passage 3" in gold_texts

    def test_beir_generalizes_to_other_datasets(self, monkeypatch):
        # The same BEIR loader serves FEVER/NFCorpus/etc — only the name differs.
        monkeypatch.setattr(loaders, "_load_hf", _fake_scifact_load_hf)
        ds = load_fever(n_distractors=3)
        _assert_unified_schema(ds)
        assert ds.name == "fever"
        assert all(len(q["gold_doc_ids"]) == 1 for q in ds.queries)


# --- dispatcher -------------------------------------------------------------


class TestDispatcher:
    def test_nq_remains_a_seam(self, tmp_path):
        with pytest.raises(NotImplementedError):
            load_dataset("natural_questions", fixtures_dir=tmp_path)

    def test_unknown_dataset_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError):
            load_dataset("not_a_dataset", fixtures_dir=tmp_path)

    def test_hf_dataset_routes_to_loader(self, monkeypatch, tmp_path):
        sentinel = LoadedDataset("musique", [{"doc_id": 0, "text": "x"}], [])
        monkeypatch.setitem(
            __import__("topp_rag.datasets", fromlist=["_HF_LOADERS"])._HF_LOADERS,
            "musique",
            lambda max_queries, seed: sentinel,
        )
        assert load_dataset("musique", fixtures_dir=tmp_path) is sentinel
