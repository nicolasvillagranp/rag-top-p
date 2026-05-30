# Top-P RAG Retrieval

A retrieval method for Retrieval-Augmented Generation (RAG) that adaptively selects the number of documents per query using nucleus sampling over softmax-normalized similarity scores — provably never worse than top-k, and strictly more token-efficient on peaked query distributions.

---

## Motivation

Standard RAG systems retrieve a fixed number of documents (`top-k`) regardless of query complexity. This is wasteful: a simple factoid query may need one document, while a multi-hop reasoning query may need five. Fixing k forces a global compromise — too many tokens for easy queries, too few for hard ones.

Top-P RAG retrieval adapts the retrieved set size to the shape of the similarity score distribution on a per-query basis, directly inspired by nucleus sampling (Holtzman et al., 2020). The key guarantee: **Top-P is provably never worse than top-k** — the fallback mechanism ensures that if the adaptive selection would return more than k documents, top-k is used instead.

---

## The Algorithm

**Inputs**
- `q` — query string
- Vector store (FAISS + sentence-transformers)
- `M` — candidate pool size (M >> k, e.g. 20–50)
- `p` — probability mass threshold ∈ (0, 1)
- `k` — maximum documents to return (fallback cap)
- `τ` — softmax temperature (controls distribution sharpness)

**Steps**

1. Retrieve the top `M` candidate documents ranked by cosine similarity, yielding scores `s_1 ≥ s_2 ≥ ... ≥ s_M`

2. Apply temperature-scaled softmax to convert raw scores into a probability distribution:

```
p_i = exp(s_i / τ) / Σ_j exp(s_j / τ)
```

3. Greedily accumulate documents in ranked order until cumulative probability mass ≥ `p`. The selected set is the smallest prefix satisfying this threshold.

4. If `|selected| > k`, return top-k instead.

**Output:** between 1 and k documents, adaptive to the query's score distribution shape.

**Intuition:** A peaked distribution (one dominant document) saturates `p` early → few documents returned. A flat distribution (many equally relevant documents) requires more documents to saturate `p` → approaches k. The fallback ensures the method is always at least as good as top-k.

---

## Evaluation Metric

We adopt and extend the **Token Efficiency Score (TES)** introduced by Xu et al. (2025):

```
TES = Retrieval_Quality / (tokens_retrieved / tokens_top_k_baseline)
```

Where:
- `Retrieval_Quality` ∈ [0, 1] — proportional recall: the fraction of the query's gold documents/passages present in the retrieved set (`|retrieved ∩ gold| / |gold|`). On single-gold datasets this reduces to a binary hit; on multi-hop datasets it rewards retrieving more of the gold set
- Denominator — token cost of this method normalized to the top-k baseline
- `TES > 1` — better quality per token than top-k
- `TES = 1` — identical efficiency to top-k (fallback activated or same set returned)
- `TES < 1` — worse than top-k (under-retrieval)

Unlike Xu et al., who use document count as the cost proxy, we use **token count**, which directly determines computational and financial cost in LLM inference. We also normalize against a top-k baseline, making TES interpretable as relative efficiency across datasets and configurations.

---

## Baselines

- **Top-k (primary):** fixed k, standard cosine similarity retrieval
- **Top-k sweep:** k ∈ {1, 2, 3, 5, 10} — maps the quality-cost frontier that top-p navigates adaptively

---

## Benchmarks

| Dataset | Query type | Expected behaviour |
|---|---|---|
| **Natural Questions (NQ)** | Single-hop, factoid | Peaked distributions → aggressive pruning, high token savings |
| **TriviaQA** | Single-hop, factoid | Similar to NQ, robustness check across domains |
| **MultiHop-RAG** | Multi-hop | Flatter distributions → k fallback activates more frequently |
| **MuSiQue** | Hard multi-hop | Stress test for under-retrieval edge cases |

---

## Hyperparameter Grid

| Parameter | Values |
|---|---|
| `p` | 0.5, 0.6, 0.7, 0.8, 0.9, 0.95 |
| `τ` | 0.1, 0.5, 1.0, 2.0 |
| `M` | 20, 50 |
| `k` | 3, 5, 10 |

---

## Implementation Stack

- **Vector store:** FAISS (local, no external dependencies)
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2` for speed, `bge-large-en` for quality)
- **Datasets:** HuggingFace `datasets`
- **Evaluation:** HuggingFace `evaluate` (recall, precision)
- **Orchestration:** plain Python — no RAG framework, keeping the pipeline transparent and fully reproducible

---

## Caching & Performance Architecture

The hyperparameter grid has many combinations (p × τ × M × k). Naively re-running FAISS and the embedding model for each configuration would make the sweep take days. The following architecture reduces it to seconds.

**Principle: separate what changes from what does not.**

FAISS and the embedding model are deterministic — for a given query and corpus, the top-M results and their raw cosine similarity scores never change across the hyperparameter grid. Only the top-p selection logic (which is pure numpy on a score array) changes. Therefore:

```
Embed once → Index once → Retrieve once per query → Sweep entirely in memory
```

**Three-layer cache, all persisted to disk:**

```
cache/
├── embeddings/
│   ├── {dataset}_{model}_doc_embeddings.npy     # all document embeddings
│   └── {dataset}_{model}_query_embeddings.npy   # all query embeddings
├── retrievals/
│   └── {dataset}_{model}_top{M}_retrievals.parquet  # scores + doc_ids, one row per query
└── results/
    └── {dataset}_{model}_sweep_results.parquet   # full per-query sweep output
```

**Layer 1 — Embeddings cache:** embed all documents and queries once using batched inference (`batch_size=64`). Save as numpy arrays. On subsequent runs, load from disk and skip embedding entirely.

**Layer 2 — Retrievals cache:** run FAISS once per query against the full document index, saving the top-`M_max` scores and document IDs (where `M_max = max(M_grid) = 50`). Since 50 is the upper bound, any smaller M in the grid is just a slice of this array — FAISS never runs again after this.

**Layer 3 — Sweep results:** the full (p × τ × M × k) grid operates purely on the cached score arrays in memory. No embedding model, no FAISS, no disk I/O inside the sweep loop.

**Batching requirements:**
- Document embedding: `batch_size=64`, progress bar per dataset
- Query embedding: `batch_size=64`
- FAISS retrieval: batched matrix search (`index.search(query_matrix, M_max)`) over all queries at once — do not loop query by query

---

## Implementation Phases

### Phase 1 — Core Algorithm

Implement the top-p retrieval function in complete isolation from any vector store or dataset. Input is a raw numpy array of similarity scores; output is a list of selected indices.

**Tests to pass:**
- Softmax output sums to 1.0 for any input array
- Temperature τ → 0 concentrates all mass on the top document; τ → ∞ produces a uniform distribution
- Cumsum threshold correctly selects the smallest prefix with mass ≥ p
- Fallback triggers correctly when |top-p| > k, returning exactly k documents
- Fallback never triggers when |top-p| ≤ k
- Edge case: M < k — algorithm handles gracefully without error
- Edge case: p = 1.0 — all M documents selected, then capped at k
- Edge case: p = 0.0 — exactly 1 document returned

---

### Phase 2 — Vector Store Integration & Embedding Cache

Connect the Phase 1 algorithm to FAISS and sentence-transformers. Implement the full caching pipeline. Build a small hand-crafted corpus (20–30 documents) with known ground truth to enable deterministic testing.

**Tests to pass:**
- Document and query embeddings are generated in batches of 64 (verify batch calls, not sequential loops)
- Embedding cache is written to disk on first run and loaded from disk on second run — embedding model is never called twice for the same dataset/model pair
- FAISS search is called once as a batched matrix operation over all queries, not in a per-query loop
- Retrieved scores are in strictly descending order per query
- Retrievals cache stores exactly M_max=50 candidates per query with correct doc IDs and scores
- For any M ≤ M_max in the grid, the top-M slice of the cached retrievals exactly matches a fresh FAISS query for M
- Full pipeline (load embeddings → load retrievals → top-p) returns ≤ k documents per query
- Token count per document is precomputed and stored alongside the retrievals cache
- Cache is invalidated and regenerated when the embedding model name changes

---

### Phase 3 — Dataset & Evaluation Harness

Load all four benchmarks, implement retrieval-side evaluation metrics, and validate the harness on a small slice (100 queries per dataset) using top-k (k=5) as a sanity check.

**Tests to pass:**
- All four datasets load and parse correctly into a unified schema: `{query_id, query, gold_doc_ids[]}`
- Recall@retrieved is 1.0 when the gold document is always injected into the retrieved set (smoke test)
- Recall@retrieved is 0.0 when a deliberately wrong retriever is used (sanity check)
- TES = 1.0 exactly when top-p always falls back to top-k (token costs are identical)
- TES > 1.0 on a synthetic case where top-p returns fewer tokens with identical recall
- Token counts per document are consistent between the retrievals cache and the evaluation harness
- Results are reproducible across runs given fixed cached embeddings and retrievals

---

### Phase 4 — Hyperparameter Sweep

Run the full grid (p × τ × M × k) across all four datasets using only cached score arrays — no embedding or FAISS calls inside the sweep loop. Each configuration is evaluated on the full dataset or a fixed random sample (1000 queries per dataset).

**Tests to pass:**
- No embedding model or FAISS index is instantiated during the sweep (verified by mocking them to raise if called)
- No configuration is missing from the results grid (all p × τ × M × k combinations present)
- For any M in the grid, the sweep correctly slices the top-M scores from the M_max=50 cache
- Top-p never returns more tokens than top-k baseline for any single query (the guarantee holds globally, zero exceptions)
- Fallback rate is logged per configuration and dataset
- Results are serialized to disk as parquet and reload deterministically
- Sweep is resumable — completed (dataset, model, config) combinations are skipped on re-run
- Full sweep over all datasets and configurations completes in under 5 minutes on CPU

---

### Phase 5 — Analysis & Plots

Produce all figures and summary tables from serialized sweep results.

**Tests to pass:**
- All plots reproduce exactly from serialized results (deterministic, no randomness)
- TES vs p curves are generated per dataset
- Token savings distribution (% reduction vs top-k baseline) is plotted per dataset
- Fallback rate vs p is plotted — must be monotonically non-decreasing as p → 1.0 (a larger probability-mass threshold selects a larger nucleus, which hits the k cap more often)
- Summary table: best (p, τ) configuration per dataset ranked by TES

---

## Per-Query Logged Fields

Every query in the sweep logs the following:

| Field | Description |
|---|---|
| `dataset` | Benchmark name |
| `query_id` | Unique query identifier |
| `p`, `τ`, `M`, `k` | Hyperparameter configuration |
| `n_docs_retrieved` | Number of documents returned by top-p |
| `tokens_retrieved` | Total token count of returned documents |
| `tokens_top_k` | Token count of top-k baseline (same k) |
| `fallback_triggered` | Boolean — whether k cap was applied |
| `recall` | Proportional recall: fraction of gold passages in retrieved set |
| `TES` | Token Efficiency Score |

---

## References

- Holtzman et al. (2020). *The Curious Case of Neural Text Degeneration.* ICLR. — nucleus sampling
- Xu et al. (2025). *Cluster-based Adaptive Retrieval: Dynamic Context Selection for RAG Applications.* arXiv:2511.14769 — TES metric
- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.