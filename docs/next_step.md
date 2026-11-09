# Next Steps: Integrating Semantic & Hybrid Retrieval

This document outlines the step-by-step roadmap and current implementation status for integrating **Bonus 1 (Semantic Embeddings)** and **Bonus 2 (Hybrid Retrieval)** into the project without modifying or breaking any mandatory baseline behavior or time constraints.

---

## Progress Dashboard

| Component / Step | Target File | Status | Notes / Current State |
| :--- | :--- | :---: | :--- |
| **Arch. Isolation Strategy** | `__main__.py`, `indexer.py` | ✅ **Completed** | `--semantic` is opt-in (default: `False`). Indexing baseline strictly $\le 5$ min. |
| **Step 1: Indexer Updates** | `src/rag_against_the_machine/indexer.py` | ✅ **Completed** | Generates `vectors.npy` via `Embedder.embedding_encoder()`. Exports TSVs for TF Projector. |
| **Step 2: Retriever Architecture** | `src/rag_against_the_machine/retriever.py` | 🟡 **Action Needed** | Internal rankers (`_get_bm25_indices`, `_get_semantic_indices`, `_to_minimal_source`) are complete. **Needs `mode` parameter** in `retrieve()` and `search_dataset()`. |
| **Step 3: RRF Fusion** | `src/rag_against_the_machine/retriever.py` | ✅ **Completed** | Cormack formula ($k=60$) implemented with candidate pooling `min(max(k*2, 50), N)`. |
| **Step 4: CLI Integration** | `src/rag_against_the_machine/__main__.py` | 🟡 **Action Needed** | `index --semantic` is done. **Needs `--mode` parameter** exposed in `search` and `search_dataset`. |
| **Step 5: Verification & Recall** | Evaluation Suite | ⏳ **Pending Execution** | Run benchmark matrix across BM25, Semantic, and Hybrid modes on public datasets. |

---

## Table of Contents
1. [Architectural Isolation Strategy](#1-architectural-isolation-strategy)
2. [Step 1: Indexer Updates (`indexer.py`)](#2-step-1-indexer-updates-indexerpy) — *(Completed)*
3. [Step 2: Retriever Architecture & Index Helpers (`retriever.py`)](#3-step-2-retriever-architecture--index-helpers-retrieverpy) — *(Action Required)*
4. [Step 3: Reciprocal Rank Fusion (`retriever.py`)](#4-step-3-reciprocal-rank-fusion-rrf-implementation-bonus-2-hybrid-search) — *(Completed)*
5. [Step 4: CLI Integration (`__main__.py`)](#5-step-4-cli-integration-__main__py) — *(Action Required)*
6. [Step 5: Verification & Recall Comparison Matrix](#6-step-5-verification--recall-comparison-matrix) — *(Pending)*
7. [Immediate Action Checklist](#7-immediate-action-checklist)

---

## 1. Architectural Isolation Strategy

### The 5-Minute Indexing Constraint
* **The Mandatory Rule:** Chapter VII strictly enforces indexing time $\le 5$ minutes on CPU.
* **The Reality:** BM25 indexes the ~12,500 chunks in **~15 seconds**. Computing neural embeddings for 12,500 chunks on pure CPU takes **8–10 minutes**.
* **The Implementation:**
  - Semantic embedding computation is strictly **opt-in** via `--semantic True`.
  - Default execution: `uv run python -m src index` builds only the lexical BM25 index (`index.pkl`), guaranteeing evaluation compliance under the 5-minute limit.

---

## 2. Step 1: Indexer Updates (`indexer.py`) — `[COMPLETED]`

### Implementation Summary:
1. **`Embedder` Integration:** Imported and initialized in `Indexer`.
2. **Batch Text Extraction (`text_extraction`):**
   - Efficient list comprehension extracting chunk text slices across all Python (`.py`) and Markdown (`.md`) chunks:
     ```python
     def text_extraction(self) -> list[str]:
         return [chunk["text"] for chunk in self.all_chunks]
     ```
3. **Semantic Embedding Encoding & Storage (`save` & `build`):**
   - Generated using `self.embedder.embedding_encoder(texts=...)` in batches of 32 with L2 normalization.
   - Saved as a NumPy array: `data/processed/vectors.npy`.
   - **Bonus Visualization:** Automatically outputs `vectors.tsv` and `metadata.tsv` to `data/processed/tf_projector/` for 3D PCA/t-SNE visualization via TensorFlow Embedding Projector.
4. **Conditional Flag:** `Indexer.build(semantic: bool = False)` builds only BM25 unless `semantic=True`.

---

## 3. Step 2: Retriever Architecture & Index Helpers (`retriever.py`) — `[ACTION REQUIRED]`

### Architecture Workflow:

```
                              [ User Query ]
                                    │
               ┌────────────────────┼────────────────────┐
               │ (mode="bm25")      │ (mode="semantic")  │ (mode="hybrid")
               ▼                    ▼                    │
      _get_bm25_indices()   _get_semantic_indices()      │
     [chunk_42, chunk_105]    [chunk_105, chunk_99]      │
               │                    │                    │
               │                    │           ┌────────┴────────┐
               │                    │           ▼                 ▼
               │                    │   _get_bm25_indices  _get_semantic_indices
               │                    │           └────────┬────────┘
               │                    │                    ▼
               │                    │               _rrf_fusion()
               │                    │                    │
               └────────────────────┼────────────────────┘
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

### Current Status & Changes Needed:

#### 1. Internal Helpers (Already Implemented):
* `_get_bm25_indices(query: str, k: int) -> list[int]`: Returns sorted integer indices from CustomBM25 scores.
* `_get_semantic_indices(query: str, k: int) -> list[int]`: Encodes the query, computes Cosine similarity via dot product against `self.vectors`, and extracts top-$k$ indices using `torch.topk`.
* `_to_minimal_source(chunk_idx: int) -> MinimalSource`: Converts an internal integer index to a `MinimalSource` object.

#### 2. Public `retrieve()` Method (Requires Modification):
Currently, `retrieve()` in `retriever.py` does not accept the `mode` parameter; it automatically executes RRF if vectors exist, preventing independent evaluation of BM25 or pure Semantic retrieval.

**Target Implementation:**
```python
def retrieve(
    self,
    query: str,
    k: int = 10,
    mode: str = "bm25",
) -> list[MinimalSource]:
    """
    Retrieve top-k sources using the specified retrieval strategy.
    
    Modes:
      - 'bm25': Pure lexical BM25 retrieval.
      - 'semantic': Pure dense semantic retrieval (requires vectors.npy).
      - 'hybrid': Reciprocal Rank Fusion combining BM25 and Semantic.
    """
    if not query or not query.strip() or k <= 0:
        return []

    mode = mode.lower()
    if mode == "bm25":
        indices = self._get_bm25_indices(query, k)
    elif mode == "semantic":
        if self.vectors is None:
            raise ValueError(
                "Semantic mode requested but vectors.npy is not loaded. "
                "Run 'index --semantic True' first."
            )
        indices = self._get_semantic_indices(query, k)
    elif mode == "hybrid":
        if self.vectors is not None:
            indices = self._rrf_fusion(query, k)
        else:
            print("Warning: Semantic vectors not found. Falling back to BM25.")
            indices = self._get_bm25_indices(query, k)
    else:
        raise ValueError(
            f"Unknown retrieval mode: '{mode}'. Choose from 'bm25', 'semantic', 'hybrid'."
        )

    return [self._to_minimal_source(idx) for idx in indices]
```

#### 3. `search_dataset()` Method (Requires Modification):
Update signature and call site to pass `mode`:
```python
def search_dataset(
    self,
    dataset_path: str,
    k: int = 10,
    mode: str = "bm25",
    save_directory: str = "data/output/search_results/UnansweredQuestions",
) -> None:
    ...
    for item in tqdm(questions):
        sources = self.retrieve(item["question"], k=k, mode=mode)
    ...
```

---

## 4. Step 3: Reciprocal Rank Fusion (RRF) Implementation (`retriever.py`) — `[COMPLETED]`

### Implementation Summary:
* **Mathematical Formula:** Standard Cormack reciprocal rank with smoothing constant $k_0 = 60$:
  $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{BM25}, \text{Semantic}\}} \frac{1}{60 + \text{rank}_m(d)}$$
* **Candidate Pool Optimization:**
  Retrieves a dynamically sized candidate pool: `candidate_count = min(max(k * 2, 50), len(self.chunks))`.
* **Reciprocal Scoring:** Accumulates points across ranks $1 \dots N$ for both lexical and semantic rankings. Chunks ranked by both models receive a dual-bonus score boost.

---

## 5. Step 4: CLI Integration (`__main__.py`) — `[ACTION REQUIRED]`

### CLI Updates Required:

1. **`index` Command** — *(Already Implemented)*
   ```bash
   # Standard BM25 (Default):
   uv run python -m src index --max_chunk_size 2000

   # Bonus: BM25 + Semantic Vectors:
   uv run python -m src index --max_chunk_size 2000 --semantic True
   ```

2. **`search` Command** — *(Needs `--mode` argument)*
   Update `search` in `__main__.py`:
   ```python
   def search(
       self,
       query: str,
       k: int = 10,
       mode: str = "bm25",
       processed_dir: str = "data/processed",
   ) -> None:
       ...
       results = retriever.retrieve(query, k=k, mode=mode)
   ```
   CLI Usage:
   ```bash
   uv run python -m src search --query "How is attention computed in vLLM?" --k 10 --mode bm25
   uv run python -m src search --query "How is attention computed in vLLM?" --k 10 --mode semantic
   uv run python -m src search --query "How is attention computed in vLLM?" --k 10 --mode hybrid
   ```

3. **`search_dataset` Command** — *(Needs `--mode` argument)*
   Update `search_dataset` in `__main__.py`:
   ```python
   def search_dataset(
       self,
       dataset_path: str,
       k: int = 10,
       mode: str = "bm25",
       save_directory: str = "data/output/search_results/UnansweredQuestions",
       processed_dir: str = "data/processed",
   ) -> None:
       ...
       retriever.search_dataset(
           dataset_path=dataset_path,
           k=k,
           mode=mode,
           save_directory=save_directory,
       )
   ```

---

## 6. Step 5: Verification & Recall Comparison Matrix — `[PENDING]`

### Benchmarking Protocol

Run the evaluation pipeline across all three modes for both documentation and code datasets:

#### 1. Generate Search Results
```bash
# Documentation Dataset (dataset_docs_public.json)
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --mode bm25 --k 10 --save_directory data/output/search_results_bm25/UnansweredQuestions
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --mode semantic --k 10 --save_directory data/output/search_results_semantic/UnansweredQuestions
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --mode hybrid --k 10 --save_directory data/output/search_results_hybrid/UnansweredQuestions

# Code Dataset (dataset_code_public.json)
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --mode bm25 --k 10 --save_directory data/output/search_results_bm25/UnansweredQuestions
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --mode semantic --k 10 --save_directory data/output/search_results_semantic/UnansweredQuestions
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --mode hybrid --k 10 --save_directory data/output/search_results_hybrid/UnansweredQuestions
```

#### 2. Evaluate Recall@k
```bash
# Example evaluation command:
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results_hybrid/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

#### 3. Target Benchmark Comparison Table

| Dataset | Metric | BM25 (Baseline) | Semantic Only | Hybrid (RRF) | Delta (Hybrid vs BM25) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Documentation** | **Recall@1** | TBD | TBD | TBD | TBD |
| | **Recall@3** | TBD | TBD | TBD | TBD |
| | **Recall@5** | ~81.0% | TBD | TBD | TBD |
| | **Recall@10** | TBD | TBD | TBD | TBD |
| **Code** | **Recall@1** | TBD | TBD | TBD | TBD |
| | **Recall@3** | TBD | TBD | TBD | TBD |
| | **Recall@5** | ~60.6% | TBD | TBD | TBD |
| | **Recall@10** | TBD | TBD | TBD | TBD |

---

## 7. Immediate Action Checklist

- [ ] **Action 1 (`retriever.py`)**: Add `mode: str = "bm25"` parameter to `Retriever.retrieve()` and validate modes (`"bm25"`, `"semantic"`, `"hybrid"`).
- [ ] **Action 2 (`retriever.py`)**: Add `mode: str = "bm25"` to `Retriever.search_dataset()` and forward it to `self.retrieve()`.
- [ ] **Action 3 (`__main__.py`)**: Expose `--mode` parameter in `RAGCLI.search()` and `RAGCLI.search_dataset()`.
- [ ] **Action 4 (Benchmark)**: Run dataset generation and record final Recall metrics in the comparison table above.
