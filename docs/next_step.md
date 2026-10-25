# Next Steps: Integrating Semantic & Hybrid Retrieval

This document outlines the step-by-step roadmap to integrate **Bonus 1 (Semantic Embeddings)** and **Bonus 2 (Hybrid Retrieval)** into the project without modifying or breaking any mandatory baseline behavior or time constraints.

---

## Table of Contents
1. [Architectural Isolation Strategy](#1-architectural-isolation-strategy)
2. [Step 1: Indexer Updates (`indexer.py`)](#2-step-1-indexer-updates-indexerpy)
3. [Step 2: Retriever Architecture & Index Helpers (`retriever.py`)](#3-step-2-retriever-architecture--index-helpers-retrieverpy)
4. [Step 3: Reciprocal Rank Fusion (RRF) Implementation (Bonus 2: Hybrid Search)](#4-step-3-reciprocal-rank-fusion-rrf-implementation-bonus-2-hybrid-search)
5. [Step 4: CLI Integration (`__main__.py`)](#5-step-4-cli-integration-__main__py)
6. [Step 5: Verification & Recall Comparison](#6-step-5-verification--recall-comparison)

---

## 1. Architectural Isolation Strategy

### The 5-Minute Indexing Constraint
* **The Mandatory Rule:** Chapter VII strictly enforces indexing $\le 5$ minutes on CPU.
* **The Reality:** BM25 indexes the ~12,500 chunks in **15 seconds**. Computing neural embeddings for 12,500 chunks on pure CPU takes **8–10 minutes**.
* **The Solution:** Semantic embedding computation must be **opt-in** via an optional CLI parameter (`--semantic`, default: `False`). The standard evaluation command (`uv run python -m src index`) will always remain pure BM25, guaranteeing evaluation compliance.

---

## 2. Step 1: Indexer Updates (`indexer.py`)

### What Needs to be Done:
1. **Import `Embedder`** in `indexer.py`.
2. **Add a semantic generation method** (`build_semantic_index`):
   - Extracts all chunk texts: `texts = [c["text"] for c in self.all_chunks]`.
   - *Note on file types:* Embeddings are computed for **both Python (`.py`) and Markdown (`.md`) chunks** equally. Semantic vectors capture function logic as well as conceptual explanations.
   - Computes vectors via `Embedder.encode(texts, batch_size=32)`.
   - Saves the tensor matrix to `data/processed/vector_index.pt` using `torch.save()`.
3. **Add `--semantic` argument to `Indexer.build(semantic: bool = False)`**:
   - If `semantic=False`: builds and saves only BM25 (`index.pkl`).
   - If `semantic=True`: builds BM25 and additionally generates and saves `vector_index.pt`.

---

## 3. Step 2: Retriever Architecture & Index Helpers (`retriever.py`)

### The Architecture Challenge:
Currently, `retrieve()` directly instantiates and returns `list[MinimalSource]`. If we want to combine BM25 and Semantic results using RRF, we need the raw ranked candidate positions **before** they are converted into `MinimalSource` objects.

### The Solution: Internal Integer Index Helpers
Decouple candidate ranking (integer chunk indices) from object serialization (`MinimalSource`).

```
                              [ User Query ]
                                    │
               ┌────────────────────┴────────────────────┐
               │                                         │
        mode="bm25"                              mode="semantic"
               │                                         │
               ▼                                         ▼
     _get_bm25_indices()                     _get_semantic_indices()
    [chunk_42, chunk_105]                      [chunk_105, chunk_99]
               │                                         │
               └────────────────────┬────────────────────┘
                                    │ (if mode="hybrid")
                                    ▼
                              _rrf_fusion()
                                    │
                                    ▼
                         [ Top-K Chunk Indices ]
                                    │
                                    ▼
                        _to_minimal_source()
                                    │
                                    ▼
                        list[MinimalSource]
```

#### Detailed Breakdown of Methods in `Retriever`:

1. **`_get_bm25_indices(query: str, k: int) -> list[int]`**:
   - Evaluates BM25 scores across chunks.
   - Returns the sorted list of raw integer chunk indices: `[42, 105, 312, ...]`.

2. **`_get_semantic_indices(query: str, k: int) -> list[int]`**:
   - Encodes query: `query_vec = self.embedder.encode([query])`.
   - Computes Cosine similarity via dot product: `scores = torch.matmul(self.vector_index, query_vec.T).squeeze(-1)`.
   - Extracts top-$k$ indices: `torch.topk(scores, k=k).indices.tolist()`.
   - Returns sorted integer chunk indices: `[105, 99, 42, ...]`.

3. **`_to_minimal_source(chunk_idx: int) -> MinimalSource`**:
   - Helper that maps a single integer chunk index to its `MinimalSource` model:
     ```python
     chunk = self.chunks[chunk_idx]
     return MinimalSource(
         file_path=chunk["file_path"],
         first_character_index=chunk["first_character_index"],
         last_character_index=chunk["last_character_index"],
     )
     ```

4. **Public `retrieve(query: str, k: int = 10, mode: str = "bm25") -> list[MinimalSource]`**:
   - Dispatches to the appropriate helper based on `mode`:
     ```python
     if mode == "bm25":
         indices = self._get_bm25_indices(query, k)
     elif mode == "semantic":
         indices = self._get_semantic_indices(query, k)
     elif mode == "hybrid":
         indices = self._rrf_fusion(query, k)
     else:
         raise ValueError(f"Unknown retrieval mode: {mode}")

     return [self._to_minimal_source(idx) for idx in indices]
     ```
   - **Contract Guaranteed:** Externally, `retrieve()` always returns `list[MinimalSource]`.

---

## 4. Step 3: Reciprocal Rank Fusion (RRF) Implementation (Bonus 2: Hybrid Search)

### Why Raw Scores Cannot be Added Directly:
* **BM25 Scores:** Unbounded positive numbers $[0, \infty)$ (e.g., $18.4$, $32.1$).
* **Semantic Scores:** Cosine similarities bounded to $[-1.0, 1.0]$ (e.g., $0.85$).
* Direct addition is meaningless due to incompatible scale distributions.

### How RRF Operates on Integer Indices:
RRF evaluates each chunk solely based on its ordinal placement (1st, 2nd, 3rd) in each candidate ranking:

$$\text{RRF\_Score}(d) = \sum_{m \in \{\text{BM25}, \text{Semantic}\}} \frac{1}{60 + \text{rank}_m(d)}$$

#### Concrete Implementation of `_rrf_fusion`:
```python
def _rrf_fusion(self, query: str, k: int, rrf_k: int = 60) -> list[int]:
    # Retrieve top candidates from both engines (retrieve more candidates, e.g., 2 * k)
    bm25_indices = self._get_bm25_indices(query, k=k * 2)
    semantic_indices = self._get_semantic_indices(query, k=k * 2)

    rrf_scores: dict[int, float] = defaultdict(float)

    # 1. Accumulate points from BM25 ranking
    for rank, idx in enumerate(bm25_indices):
        rrf_scores[idx] += 1.0 / (rrf_k + rank + 1)

    # 2. Accumulate points from Semantic ranking
    for rank, idx in enumerate(semantic_indices):
        rrf_scores[idx] += 1.0 / (rrf_k + rank + 1)

    # 3. Sort chunk indices descending by combined RRF score
    sorted_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    return sorted_indices[:k]
```

* **Consensus Chunks:** Chunks ranked highly by both BM25 and Semantic search receive two reciprocal bonuses and jump to the top.
* **Complementary Chunks:** Chunks discovered only by one system (e.g., exact identifier or conceptual synonym) remain preserved in the top results.

---

## 5. Step 4: CLI Integration (`__main__.py`)

### Required CLI Changes:
1. **Update `index` command**:
   ```bash
   uv run python -m src index --max_chunk_size 2000 --semantic True
   ```
2. **Update `search` and `search_dataset` commands**:
   Add `--mode` argument (default: `"bm25"`):
   ```bash
   # Mandatory Lexical search (default):
   uv run python -m src search --query "..." --k 10 --mode bm25

   # Bonus 1: Semantic search:
   uv run python -m src search --query "..." --k 10 --mode semantic

   # Bonus 2: Hybrid search (BM25 + Semantic via RRF):
   uv run python -m src search --query "..." --k 10 --mode hybrid
   ```

---

## 6. Step 5: Verification & Recall Comparison

Once implemented, evaluate all three retrieval modes against public datasets using:
```bash
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```
Compare Recall@5 across:
- **BM25 only** (Baseline: ~81.0% docs / ~60.6% code)
- **Semantic only** (Captures synonyms, intent, conceptual questions)
- **Hybrid (RRF)** (Combines exact identifier matching with conceptual synonyms for maximum overall Recall)
