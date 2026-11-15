# Software Specifications: RAG against the machine

## 1. Overview
This document describes the architecture, data contracts, and implementation of the **Retrieval-Augmented Generation (RAG)** system built for the `vLLM` codebase. Retrieval supports BM25, semantic embeddings, and hybrid Reciprocal Rank Fusion (RRF). Indexing reuses chunks and embeddings for unchanged source files.

For theoretical foundations, mathematical derivations (BM25, Attention, Mean Pooling), and Information Retrieval (IR) principles, refer to the [Technical Knowledge Base](knowledge_base.md).

---

## 2. System Architecture

The system decouples data preparation from real-time querying into two discrete pipelines:
1. **The ETL / Indexing Pipeline** (Offline Batch Processing)
2. **The Inference / Serving Pipeline** (Online Real-Time Serving)

```
[Raw Repository: data/raw/]
        │
        ▼ (ETL / Indexing Pipeline - Offline)
 1. File Discovery & Extraction (.py, .md, .txt)
 2. Domain-Specific Chunking (< 2000 chars)
 3. BM25 Indexing and optional Semantic Embeddings
 4. Persistent Storage (index.pkl and vectors.npy)
        │
════════╪═════════════════════════════════════════════════════════
        │
        ▼ (Inference / Query Pipeline - Online)
 1. User Query / Dataset JSON
 2. BM25, Semantic, or Hybrid RRF Retrieval
 3. Augmentation (Prompt Assembly with Context Slicing)
 4. LLM Generation (Qwen/Qwen3-0.6B Inference)
 5. Pydantic Output Validation & Export (data/output/)
```

---

### 2.1. The ETL / Indexing Pipeline (Offline Batch Processing)
Executed via the `index` CLI command:

#### Stage 1: Extraction & Filtering
* **Input Directory:** `data/raw/vllm-0.10.1/`.
* **Target Files:** Python (`.py`), Markdown (`.md`), and plain text documentation (e.g., `CMakeLists.txt`). Non-text and binary artifacts are excluded.
* **Character Span Tracking:** Character offsets (`first_character_index`, `last_character_index`) are calculated strictly relative to the original source file.

#### Stage 2: Domain-Specific Chunking
* **Markdown Chunking:** Uses `# ` and `## ` headings as preferred boundaries, subject to the configured maximum size.
* **Text Chunking:** Uses newline boundaries, subject to the configured maximum size.
* **Python Chunking:** Uses `class `, `def `, and `async def ` markers as preferred boundaries; it does not parse Python into an AST.
* **Hard Constraint:** Every chunk is strictly bounded to `--max_chunk_size` (default: 2,000 characters). Sources longer than 2,000 characters are rejected by the grading suite.

#### Stage 3: Lexical and Optional Semantic Index Construction
* Computes document frequencies ($DF$), inverse document frequencies ($IDF$), chunk lengths ($|D|$), and average corpus chunk length ($avgdl$).
* Rebuilds global BM25 statistics from all current chunks on each indexing run.
* When semantic indexing is requested, encodes chunks with `all-MiniLM-L6-v2`. On later runs, vectors for unchanged files can be reused.
* `tqdm` progress bars report long-running file processing, tokenization, and embedding work. The embedding bar counts only chunks that need new vectors.

#### Stage 4: Incremental Source Processing
* Calculates a SHA-256 content hash for each supported source file.
* Reuses chunks and, when their saved artifacts validate, embeddings for unchanged files; re-chunks and re-embeds added or changed files.
* Files removed from the raw corpus are omitted from the new chunk and vector collections.

#### Stage 5: Binary Persistence and Consistency Checks
* **Target Directory:** `data/processed/`.
* `index.pkl` stores chunks, the BM25 index, source-file hashes, and signatures for the vector artifacts.
* `vectors.npy`, when present, stores a matrix with one 384-dimensional row per chunk, in the same order as `all_chunks`.
* Chunk signatures and a SHA-256 hash of vector contents detect mismatched index and vector files before semantic or hybrid retrieval reuses them.
* **Execution Limit:** Must index the complete corpus in **$\le 5$ minutes**.

---

### 2.2. The Inference / Query Pipeline (Online Real-Time Serving)
Executed via the `search`, `search_dataset`, `answer`, and `answer_dataset` CLI commands:

#### Stage 1: Query Preprocessing
* Ingests a single query string or batch queries from a JSON dataset.
* Applies the identical tokenization and normalization used during indexing.

#### Stage 2: Retrieval & Top-$k$ Ranking
* Loads and validates `data/processed/index.pkl`; semantic and hybrid modes also validate and load `vectors.npy`.
* Supports BM25 lexical ranking, semantic ranking by cosine similarity of normalized embeddings, and hybrid ranking with Reciprocal Rank Fusion.
* Sorts and selects the Top-$k$ results, wrapping them in `MinimalSource` objects.
* **Throughput Requirement:** Evaluates 200 questions in **$\le 90$ seconds**.
* **Accuracy Targets:** Recall@5 $\ge 80\%$ on documentation, $\ge 50\%$ on code.

#### Stage 3: Context Slicing & Prompt Augmentation
* Reads source files from disk on-demand and slices the exact character range `file_content[start:end]`.
* Assembles a structured prompt injecting the Top-$3$ most confident sources into the model context window.

#### Stage 4: Local LLM Generation
* Feeds the context prompt to `Qwen/Qwen3-0.6B`.
* Runs in `torch.inference_mode()` with `bfloat16`, `use_cache=True`, and reasoning bypass (`<think>\n</think>\n`).

#### Stage 5: Structured Serialization
* Validates output structures via Pydantic v2.
* Exports formatted JSON files into `data/output/`.

---

## 3. Lexical Engine Selection: BM25
The system implements a custom **Okapi BM25** engine rather than TF-IDF.

| Feature | TF-IDF | BM25 (Implemented) | Rationale |
| :--- | :--- | :--- | :--- |
| **Term Frequency** | Linear growth | Saturated curve ($k_1 = 1.5$) | Prevents code keyword repetition from dominating ranking |
| **Length Normalization** | Weak / None | Explicit penalty ($b = 0.75$) | Balances concise functions against large files |
| **Recall@5 Target** | Lower on code | $\ge 80\%$ docs / $\ge 50\%$ code | Consistently reaches required thresholds |

*(See [Technical Knowledge Base](knowledge_base.md#3-lexical-retrieval-tf-idf-vs-bm25-deep-dive) for complete mathematical proofs).*

## 4. Semantic and Hybrid Retrieval

* Semantic embeddings use `sentence-transformers/all-MiniLM-L6-v2`, producing one normalized 384-dimensional vector per input text (corpus chunks and queries).
* Similarity is calculated with a dot product between normalized vectors, equivalent to cosine similarity.
* Hybrid mode combines BM25 and semantic rankings with RRF (`rrf_k=60`); it combines ranks rather than comparing the two systems' raw scores.

---

## 5. Codebase Module Architecture

```
src/
├── __init__.py           # Package namespace definition
├── __main__.py           # Python Fire CLI dispatcher & subcommands
├── model.py              # Pydantic v2 data schemas
├── chunker.py            # Python, Markdown, and text chunking
├── indexer.py            # Incremental chunk/index builder and persistence
├── retriever.py          # BM25, semantic, hybrid, and dataset search
├── generator.py          # Qwen/Qwen3-0.6B loader, prompt assembly & inference
└── embedding.py          # all-MiniLM-L6-v2 vector embedder
```

---

## 6. Software Design Patterns & Conventions

### 6.1. Factory Method Pattern (`@classmethod load`)
* **Problem:** Instantiating `Retriever` requires pre-computed indices that only exist on disk in `data/processed/index.pkl`.
* **Pattern:** Implemented `@classmethod def load(cls, processed_dir: str) -> "Retriever"`.
* **Benefit:** Encapsulates deserialization logic away from constructor calls, eliminating empty or broken instance states.

### 6.2. CLI-to-Service Method Mapping
* Core engine methods mirror CLI commands directly:
  * CLI `index` $\longrightarrow$ `Indexer.build()`
  * CLI `search` $\longrightarrow$ `Retriever.retrieve()`
  * CLI `search_dataset` $\longrightarrow$ `Retriever.search_dataset()`
  * CLI `answer` $\longrightarrow$ `Generator.generate_answer()`
  * CLI `answer_dataset` $\longrightarrow$ `Generator.generate_answer_dataset()`

### 6.3. Cross-Platform Path Normalization
* All internal file paths are converted to POSIX format (`Path(p).as_posix()`).
* Ensures forward slashes (`/`) match the Linux evaluation moulinette even when executing on Windows.

### 6.4. Strict Pydantic v2 Contracts
* All data exchanged across pipeline stages conforms to Pydantic schemas:
  * `MinimalSource` (`file_path`, `first_character_index`, `last_character_index`)
  * `MinimalSearchResults` (`question_id`, `question`, `retrieved_sources`)
  * `StudentSearchResults` (`search_results`, `k`)
  * `MinimalAnswer` (`question_id`, `question`, `retrieved_sources`, `answer`)
  * `StudentSearchResultsAndAnswer` (`search_results`, `k`)

### 6.5. Incremental Index and Artifact Consistency
* Source-file SHA-256 hashes identify unchanged, changed, added, and removed files.
* Chunks and vectors are grouped by source path so unchanged file data can be reused while changed and new files are processed.
* BM25 statistics are rebuilt over the final combined corpus because its statistics depend on all chunks.
* The saved chunk signature and vector-content hash let the retriever reject stale or mismatched semantic artifacts.

---

## 7. Challenges & Engineering Solutions

### 7.1. Index Preservation During Score Ranking
* **Issue:** `sorted(scores)` strips original chunk indices.
* **Fix:** Enumerate chunk scores into `(index, score)` tuples and sort using `key=lambda x: x[1], reverse=True`.

### 7.2. Missing Non-Markdown Documentation Files
* **Issue:** 3 reference questions targeted `CMakeLists.txt`, which was initially excluded by `.py`/`.md` filtering.
* **Fix:** Added `.txt` handling (including `CMakeLists.txt`) to file discovery and routed it to text chunking.

### 7.3. Dynamic Dataset Scoping
* **Issue:** Dataset filenames collide if written to a single folder.
* **Fix:** Preserved filename identity dynamically via `Path(dataset_path).name` into user-configured `--save_directory`.

### 7.4. Truncated Model Answers & Thinking Overhead
* **Issue:** Default `Qwen3-0.6B` reasoning generated excessive thinking tokens, exhausting `max_new_tokens` and truncating final answers.
* **Fix:** Injected empty `<think>\n</think>\n` tokens into the prompt to bypass reasoning deliberation, and expanded `max_new_tokens` to 180. Resulted in 0/100 truncations and an 80% reduction in inference latency.
