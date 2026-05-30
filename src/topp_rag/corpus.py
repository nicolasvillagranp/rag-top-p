"""Hand-crafted fixture corpus with known ground truth.

A small (24-document) corpus across a few topical clusters, paired with queries
whose gold documents are known. This is the deterministic substrate for all
Phase 2-4 tests: embeddings of real text are reproducible, retrieval ordering is
stable, and we know exactly which document should be recalled for each query.

The clusters are designed to exercise both ends of the top-p behaviour:

* **Peaked** queries (e.g. a precise factoid) have one dominant matching
  document and several loosely-related ones -> top-p should prune aggressively.
* **Flat** queries (e.g. a broad topic with several near-equivalent documents)
  have several similarly-relevant documents -> top-p approaches the k cap.

The data is defined inline below and written to ``{fixtures_dir}`` as JSON so it
loads through the same interface as the (future) real datasets.
"""

from __future__ import annotations

import json
from pathlib import Path

# --- Documents -------------------------------------------------------------
# Topical clusters: space, programming, cooking, weather/climate.
# doc_id is the document's stable identifier (its index here).
_DOCUMENTS: list[str] = [
    # 0-5  space / astronomy
    "The Moon is Earth's only natural satellite and orbits the planet roughly every 27 days.",
    "Mars, the fourth planet from the Sun, is often called the Red Planet due to iron oxide on its surface.",
    "Jupiter is the largest planet in the Solar System, a gas giant with a famous Great Red Spot storm.",
    "The Sun is a yellow dwarf star at the center of the Solar System, providing light and heat to Earth.",
    "A solar eclipse occurs when the Moon passes between the Earth and the Sun, blocking sunlight.",
    "Astronauts aboard the International Space Station orbit the Earth about sixteen times each day.",
    # 6-11  programming / software
    "Python is a high-level programming language known for its readable syntax and large ecosystem.",
    "A hash table stores key-value pairs and offers average constant-time lookup by hashing keys.",
    "Git is a distributed version control system that tracks changes in source code over time.",
    "Recursion is a technique where a function calls itself to solve smaller instances of a problem.",
    "Compiled languages translate source code into machine code ahead of time for faster execution.",
    "A binary search algorithm finds a target in a sorted array by repeatedly halving the range.",
    # 12-17  cooking / food
    "To make a basic tomato sauce, simmer crushed tomatoes with garlic, olive oil, and basil.",
    "Bread rises because yeast ferments sugars and releases carbon dioxide gas into the dough.",
    "Searing meat over high heat creates a browned crust through the Maillard reaction.",
    "A good vinaigrette balances acid from vinegar with oil, usually in a one-to-three ratio.",
    "Risotto is cooked by gradually adding warm stock to arborio rice while stirring constantly.",
    "Caramelizing onions slowly over low heat brings out their natural sweetness over time.",
    # 18-23  weather / climate
    "A hurricane is a large rotating storm system that forms over warm tropical ocean waters.",
    "Rain forms when water vapor in clouds condenses into droplets heavy enough to fall.",
    "The greenhouse effect traps heat in the atmosphere through gases like carbon dioxide.",
    "Snow forms when atmospheric water vapor freezes into ice crystals before reaching the ground.",
    "Lightning is a sudden electrostatic discharge during a thunderstorm between clouds and ground.",
    "Climate refers to long-term weather patterns in a region, averaged over decades.",
]

# --- Queries ---------------------------------------------------------------
# gold_doc_ids lists the document(s) that genuinely answer the query.
_QUERIES: list[dict] = [
    {"query_id": "q0", "query": "How long does the Moon take to orbit the Earth?", "gold_doc_ids": [0]},
    {"query_id": "q1", "query": "Why is Mars called the Red Planet?", "gold_doc_ids": [1]},
    {"query_id": "q2", "query": "What is the largest planet in the Solar System?", "gold_doc_ids": [2]},
    {"query_id": "q3", "query": "What causes a solar eclipse?", "gold_doc_ids": [4]},
    {"query_id": "q4", "query": "What is Python used for in programming?", "gold_doc_ids": [6]},
    {"query_id": "q5", "query": "How does a hash table achieve fast lookups?", "gold_doc_ids": [7]},
    {"query_id": "q6", "query": "What does Git do for source code?", "gold_doc_ids": [8]},
    {"query_id": "q7", "query": "How do you make a simple tomato sauce?", "gold_doc_ids": [12]},
    {"query_id": "q8", "query": "Why does bread dough rise?", "gold_doc_ids": [13]},
    {"query_id": "q9", "query": "How is risotto cooked?", "gold_doc_ids": [16]},
    {"query_id": "q10", "query": "How does a hurricane form?", "gold_doc_ids": [18]},
    {"query_id": "q11", "query": "What is the greenhouse effect?", "gold_doc_ids": [20]},
]


def documents() -> list[dict]:
    """The fixture documents as ``{doc_id, text}`` records."""
    return [{"doc_id": i, "text": text} for i, text in enumerate(_DOCUMENTS)]


def queries() -> list[dict]:
    """The fixture queries as ``{query_id, query, gold_doc_ids}`` records."""
    return [dict(q) for q in _QUERIES]


def write_fixtures(fixtures_dir: str | Path) -> tuple[Path, Path]:
    """Materialize the corpus to ``{fixtures_dir}`` as JSON.

    Returns the (documents_path, queries_path) pair. Idempotent: rewriting
    produces byte-identical files, so it is safe to call on every run.
    """
    fixtures_dir = Path(fixtures_dir)
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    docs_path = fixtures_dir / "fixtures_documents.json"
    queries_path = fixtures_dir / "fixtures_queries.json"
    docs_path.write_text(json.dumps(documents(), indent=2), encoding="utf-8")
    queries_path.write_text(json.dumps(queries(), indent=2), encoding="utf-8")
    return docs_path, queries_path