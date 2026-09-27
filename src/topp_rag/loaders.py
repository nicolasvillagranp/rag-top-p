"""Real benchmark loaders — HuggingFace datasets into the unified schema.

Each loader produces a :class:`~topp_rag.schema.LoadedDataset`:

    documents -> list[{doc_id: int, text: str}]
    queries   -> list[{query_id, query, gold_doc_ids: list[int]}]

Top-p is a *selection* method, so every dataset is cast into the same regime: a
per-query **candidate pool** (gold passages + distractors), unioned into one
deduplicated corpus with stable ``doc_id``s. ``gold_doc_ids`` index that corpus.

Implemented:

* **MuSiQue** — each question ships 20 paragraphs with an ``is_supporting`` flag;
  the supporting ones are gold, the rest are distractors. Native pool.
* **MultiHop-RAG** — a fixed 609-document news corpus; each query's evidence
  links to corpus documents by article title. The whole corpus is the pool.
* **TriviaQA** (``rc``) — each question ships Wikipedia + web passages; gold is
  *distant* — a passage is gold iff it contains the answer (or an alias) string.

Natural Questions is intentionally a seam: its clean HF variants are answer-only
(no candidate passages), so there is nothing for top-p to select among.

Downloads are lazy and cached by HuggingFace; missing/gated datasets raise a
clear error via :func:`_load_hf`.
"""

from __future__ import annotations

import json
import random
import re

from datasets import load_dataset as hf_load_dataset
from huggingface_hub import hf_hub_download

from .schema import LoadedDataset


class CorpusBuilder:
    """Interns passage texts into a deduplicated corpus with stable doc_ids.

    Identical passage text (after stripping) maps to a single ``doc_id`` no
    matter how many queries reference it, so the corpus is embedded once.
    """

    def __init__(self) -> None:
        self._id_by_text: dict[str, int] = {}
        self._texts: list[str] = []

    def intern(self, text: str) -> int:
        key = text.strip()
        doc_id = self._id_by_text.get(key)
        if doc_id is None:
            doc_id = len(self._texts)
            self._id_by_text[key] = doc_id
            self._texts.append(key)
        return doc_id

    def documents(self) -> list[dict]:
        return [{"doc_id": i, "text": t} for i, t in enumerate(self._texts)]


def _load_hf(repo: str, name: str | None, split: str):
    """Load an HF dataset split, turning failures into a clear, actionable error."""
    try:
        if name is not None:
            return hf_load_dataset(repo, name, split=split)
        return hf_load_dataset(repo, split=split)
    except Exception as e:  # noqa: BLE001 — re-raised with guidance
        raise RuntimeError(
            f"Failed to load HuggingFace dataset '{repo}'"
            f"{f' ({name})' if name else ''} split='{split}'. If it is gated, set "
            f"HF_TOKEN / run `huggingface-cli login`; otherwise check the repo id "
            f"and your network. Original error: {e!r}"
        ) from e


def _sample_rows(dataset, max_queries: int | None, seed: int):
    """Deterministically take up to ``max_queries`` rows from an HF Dataset.

    Building the corpus from only the sampled rows keeps embedding bounded — the
    candidate pool is exactly the passages of the queries we actually evaluate.
    """
    if max_queries is None or max_queries >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_queries))


def _sample_ids(ids: list, max_queries: int | None, seed: int) -> list:
    """Deterministically take up to ``max_queries`` items from a list of ids."""
    if max_queries is None or max_queries >= len(ids):
        return ids
    rng = random.Random(seed)
    chosen = set(rng.sample(range(len(ids)), max_queries))
    return [x for i, x in enumerate(ids) if i in chosen]


# --- MuSiQue ---------------------------------------------------------------

MUSIQUE_REPO = "dgslibisey/MuSiQue"


def load_musique(
    max_queries: int | None = None, seed: int = 42, split: str = "validation"
) -> LoadedDataset:
    rows = _sample_rows(_load_hf(MUSIQUE_REPO, None, split), max_queries, seed)
    builder = CorpusBuilder()
    queries: list[dict] = []

    for ex in rows:
        gold: list[int] = []
        for para in ex["paragraphs"]:
            doc_id = builder.intern(para["paragraph_text"])
            # is_supporting may arrive as bool or as the string "True"/"False".
            if _truthy(para["is_supporting"]):
                gold.append(doc_id)
        queries.append(
            {"query_id": str(ex["id"]), "query": ex["question"], "gold_doc_ids": gold}
        )

    return LoadedDataset("musique", builder.documents(), queries)


# --- MultiHop-RAG ----------------------------------------------------------

MULTIHOP_REPO = "yixuantt/MultiHopRAG"


def load_multihop_rag(max_queries: int | None = None, seed: int = 42) -> LoadedDataset:
    # The HF JSON does not auto-parse via `datasets`; read the raw files.
    corpus_path = hf_hub_download(MULTIHOP_REPO, "corpus.json", repo_type="dataset")
    queries_path = hf_hub_download(MULTIHOP_REPO, "MultiHopRAG.json", repo_type="dataset")
    with open(corpus_path, encoding="utf-8") as f:
        corpus_raw = json.load(f)
    with open(queries_path, encoding="utf-8") as f:
        queries_raw = json.load(f)

    # MultiHop-RAG ships ONE fixed news corpus shared by every query, so the whole
    # corpus is each query's candidate pool — we keep all docs and sample queries.
    documents = [
        {"doc_id": i, "text": d["body"]} for i, d in enumerate(corpus_raw)
    ]
    id_by_title = {d["title"]: i for i, d in enumerate(corpus_raw)}

    queries: list[dict] = []
    for qi, ex in enumerate(queries_raw):
        gold = sorted(
            {
                id_by_title[ev["title"]]
                for ev in ex.get("evidence_list", [])
                if ev.get("title") in id_by_title
            }
        )
        queries.append(
            {"query_id": f"mh_{qi}", "query": ex["query"], "gold_doc_ids": gold}
        )

    if max_queries is not None and max_queries < len(queries):
        rng = random.Random(seed)
        chosen = set(rng.sample(range(len(queries)), max_queries))
        queries = [q for i, q in enumerate(queries) if i in chosen]

    return LoadedDataset("multihop_rag", documents, queries)


# --- TriviaQA --------------------------------------------------------------

TRIVIAQA_REPO = "mandarjoshi/trivia_qa"


def load_triviaqa(
    max_queries: int | None = None, seed: int = 42, split: str = "validation"
) -> LoadedDataset:
    rows = _sample_rows(_load_hf(TRIVIAQA_REPO, "rc", split), max_queries, seed)
    builder = CorpusBuilder()
    queries: list[dict] = []

    for ex in rows:
        answer = ex["answer"]
        answer_strings = _answer_strings(answer)

        pool_texts: list[str] = []
        pool_texts.extend(ex["entity_pages"].get("wiki_context", []))
        pool_texts.extend(ex["search_results"].get("search_context", []))

        gold: list[int] = []
        for text in pool_texts:
            if not text.strip():
                continue
            doc_id = builder.intern(text)
            # Distant supervision: a passage is gold iff it contains the answer.
            if _contains_answer(text, answer_strings):
                gold.append(doc_id)

        # Skip degenerate questions with no usable passages.
        if not pool_texts:
            continue
        queries.append(
            {
                "query_id": str(ex["question_id"]),
                "query": ex["question"],
                "gold_doc_ids": sorted(set(gold)),
            }
        )

    return LoadedDataset("triviaqa", builder.documents(), queries)


# --- HotpotQA-schema datasets (distractor pools) ---------------------------
# HotpotQA and 2WikiMultihopQA share the exact same fields: a per-question
# `context` of titled paragraphs plus `supporting_facts` naming the gold titles.

# (name, repo, hf_config) — hf_config is None when the dataset has no config.
_HOTPOT_SCHEMA = {
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor"),
    "2wikimultihopqa": ("framolfese/2WikiMultihopQA", None),
}


def _load_hotpot_schema(
    name: str, max_queries: int | None, seed: int, split: str
) -> LoadedDataset:
    """Load a HotpotQA-schema dataset: native pool of titled context paragraphs.

    Each context paragraph is one document; gold paragraphs are those whose title
    appears in ``supporting_facts``. Native pool — ``n_distractors`` does not apply.
    """
    repo, hf_config = _HOTPOT_SCHEMA[name]
    rows = _sample_rows(_load_hf(repo, hf_config, split), max_queries, seed)
    builder = CorpusBuilder()
    queries: list[dict] = []

    for ex in rows:
        context = ex["context"]
        titles = context["title"]
        sentence_lists = context["sentences"]
        gold_titles = set(ex["supporting_facts"]["title"])

        gold: list[int] = []
        for title, sentences in zip(titles, sentence_lists):
            text = "".join(sentences)
            doc_id = builder.intern(text)
            if title in gold_titles:
                gold.append(doc_id)

        queries.append(
            {
                "query_id": str(ex["id"]),
                "query": ex["question"],
                "gold_doc_ids": sorted(set(gold)),
            }
        )

    return LoadedDataset(name, builder.documents(), queries)


def load_hotpotqa(
    max_queries: int | None = None, seed: int = 42, split: str = "validation"
) -> LoadedDataset:
    return _load_hotpot_schema("hotpotqa", max_queries, seed, split)


def load_2wikimultihopqa(
    max_queries: int | None = None, seed: int = 42, split: str = "validation"
) -> LoadedDataset:
    return _load_hotpot_schema("2wikimultihopqa", max_queries, seed, split)


# --- BEIR-schema datasets (synthesized distractor pools) -------------------
# BEIR datasets ship a corpus + queries + qrels (gold doc ids per query) but no
# candidate pool, so we synthesize one per query: gold doc(s) + n_distractors
# randomly sampled non-gold corpus docs. Several share this exact shape.

# name -> (corpus/queries repo, qrels split). qrels repo is "{repo}-qrels".
_BEIR_DATASETS = {
    "scifact": ("BeIR/scifact", "test"),
    "fever": ("BeIR/fever", "test"),
    "nfcorpus": ("BeIR/nfcorpus", "test"),
    "scidocs": ("BeIR/scidocs", "test"),
    "fiqa": ("BeIR/fiqa", "test"),
}


def _load_beir(
    name: str,
    max_queries: int | None,
    seed: int,
    n_distractors: int,
    split: str | None,
) -> LoadedDataset:
    """Load a BEIR dataset as per-query candidate pools.

    The pool for each query is its gold doc(s) plus ``n_distractors`` randomly
    sampled non-gold corpus docs (capped at the number available). The corpus is
    the deduplicated union of every pool.
    """
    repo, default_split = _BEIR_DATASETS[name]
    qrels_split = split or default_split

    corpus_rows = _load_hf(repo, "corpus", "corpus")
    query_rows = _load_hf(repo, "queries", "queries")
    qrels = _load_hf(f"{repo}-qrels", None, qrels_split)

    # corpus_id (str) -> passage text.
    text_by_corpus_id = {str(r["_id"]): r["text"] for r in corpus_rows}
    all_corpus_ids = list(text_by_corpus_id)

    # query_id (str) -> its gold corpus_ids (score > 0).
    gold_by_query: dict[str, list[str]] = {}
    for r in qrels:
        if int(r["score"]) > 0:
            gold_by_query.setdefault(str(r["query-id"]), []).append(str(r["corpus-id"]))

    # Only queries that have at least one gold doc are usable.
    query_text = {str(r["_id"]): r["text"] for r in query_rows}
    usable_qids = [qid for qid in gold_by_query if qid in query_text]
    usable_qids = _sample_ids(usable_qids, max_queries, seed)

    builder = CorpusBuilder()
    queries: list[dict] = []
    rng = random.Random(seed)

    for qid in usable_qids:
        gold_ids = [g for g in gold_by_query[qid] if g in text_by_corpus_id]
        if not gold_ids:
            continue
        gold_set = set(gold_ids)
        candidates = [cid for cid in all_corpus_ids if cid not in gold_set]
        n = min(n_distractors, len(candidates))
        distractor_ids = rng.sample(candidates, n)

        gold: list[int] = []
        for cid in gold_ids + distractor_ids:
            doc_id = builder.intern(text_by_corpus_id[cid])
            if cid in gold_set:
                gold.append(doc_id)

        queries.append(
            {"query_id": qid, "query": query_text[qid], "gold_doc_ids": sorted(set(gold))}
        )

    return LoadedDataset(name, builder.documents(), queries)


def load_scifact(max_queries=None, seed=42, n_distractors=49, split=None) -> LoadedDataset:
    return _load_beir("scifact", max_queries, seed, n_distractors, split)


def load_fever(max_queries=None, seed=42, n_distractors=49, split=None) -> LoadedDataset:
    return _load_beir("fever", max_queries, seed, n_distractors, split)


def load_nfcorpus(max_queries=None, seed=42, n_distractors=49, split=None) -> LoadedDataset:
    return _load_beir("nfcorpus", max_queries, seed, n_distractors, split)


def load_scidocs(max_queries=None, seed=42, n_distractors=49, split=None) -> LoadedDataset:
    return _load_beir("scidocs", max_queries, seed, n_distractors, split)


def load_fiqa(max_queries=None, seed=42, n_distractors=49, split=None) -> LoadedDataset:
    return _load_beir("fiqa", max_queries, seed, n_distractors, split)


# --- helpers ---------------------------------------------------------------


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _answer_strings(answer: dict) -> list[str]:
    """All surface forms of a TriviaQA answer (value + aliases)."""
    strings: list[str] = []
    if isinstance(answer, dict):
        if answer.get("value"):
            strings.append(answer["value"])
        strings.extend(answer.get("aliases", []) or [])
        strings.extend(answer.get("normalized_aliases", []) or [])
    elif isinstance(answer, str):
        strings.append(answer)
    return [s for s in strings if s]


def _contains_answer(text: str, answer_strings: list[str]) -> bool:
    """Case-insensitive whole-word-ish containment of any answer surface form."""
    haystack = text.lower()
    for ans in answer_strings:
        needle = ans.lower().strip()
        if needle and re.search(r"\b" + re.escape(needle) + r"\b", haystack):
            return True
    return False
