# Software Specifications: RAG against the machine

## 1. Overview
This document specifies the software architecture, engineering design patterns, data contracts, and implementation constraints of the **Retrieval-Augmented Generation (RAG)** system built for the `vLLM` codebase.

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
 3. Lexical Indexing (BM25 Inverted Index)
 4. Persistent Storage (data/processed/index.pkl)
        │
════════╪═════════════════════════════════════════════════════════
        │
        ▼ (Inference / Query Pipeline - Online)
 1. User Query / Dataset JSON
 2. BM25 Retrieval (Top-K Chunks with exact file character spans)
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
* **Markdown Chunking:** Splits along semantic headers (`#`, `##`, `###`) and horizontal breaks. Chunks smaller than 200 characters are merged forward to prevent orphan header chunks.
* **Python Chunking:** Splits along top-level structural boundaries (`class `, `def `, `async def `).
* **Hard Constraint:** Every chunk is strictly bounded to `--max_chunk_size` (default: 2,000 characters). Sources longer than 2,000 characters are rejected by the grading suite.

#### Stage 3: Inverted Index Construction
* Computes document frequencies ($DF$), inverse document frequencies ($IDF$), chunk lengths ($|D|$), and average corpus chunk length ($avgdl$).
* Constructs the in-memory inverted term frequency index.

#### Stage 4: Binary Persistence
* **Target Directory:** `data/processed/`.
* Serializes the inverted index, vocabulary tables, BM25 statistical parameters, and chunk metadata into `data/processed/index.pkl`.
* **Execution Limit:** Must index the complete corpus in **$\le 5$ minutes**.

---

### 2.2. The Inference / Query Pipeline (Online Real-Time Serving)
Executed via the `search`, `search_dataset`, `answer`, and `answer_dataset` CLI commands:

#### Stage 1: Query Preprocessing
* Ingests a single query string or batch queries from a JSON dataset.
* Applies the identical tokenization and normalization used during indexing.

#### Stage 2: Retrieval & Top-$k$ Ranking
* Loads `data/processed/index.pkl` via lazy deserialization.
* Evaluates query terms across candidate chunks using BM25 scoring.
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

---

## 4. Codebase Module Architecture

```
src/
├── __init__.py           # Package namespace definition
├── __main__.py           # Python Fire CLI dispatcher & subcommands
├── data_models.py        # Pydantic v2 data schemas
├── chunker.py            # AST/Regex Python & Markdown chunking engine
├── indexer.py            # BM25 statistics engine & index builder
├── retriever.py          # Query scoring, Top-K ranking, and dataset search
├── generator.py          # Qwen/Qwen3-0.6B loader, prompt assembly & inference
└── embedding.py          # Bonus 1: all-MiniLM-L6-v2 vector embedder
```

---

## 5. Software Design Patterns & Conventions

### 5.1. Factory Method Pattern (`@classmethod load`)
* **Problem:** Instantiating `Retriever` requires pre-computed indices that only exist on disk in `data/processed/index.pkl`.
* **Pattern:** Implemented `@classmethod def load(cls, processed_dir: str) -> "Retriever"`.
* **Benefit:** Encapsulates deserialization logic away from constructor calls, eliminating empty or broken instance states.

### 5.2. API Symmetrical Aliasing
* Core engine methods mirror CLI commands directly:
  * CLI `index` $\longrightarrow$ `Indexer.build()`
  * CLI `search` $\longrightarrow$ `Retriever.search()`
  * CLI `search_dataset` $\longrightarrow$ `Retriever.search_dataset()`
  * CLI `answer` $\longrightarrow$ `Generator.answer()`
  * CLI `answer_dataset` $\longrightarrow$ `Generator.answer_dataset()`

### 5.3. Cross-Platform Path Normalization
* All internal file paths are converted to POSIX format (`Path(p).as_posix()`).
* Ensures forward slashes (`/`) match the Linux evaluation moulinette even when executing on Windows.

### 5.4. Strict Pydantic v2 Contracts
* All data exchanged across pipeline stages conforms to Pydantic schemas:
  * `MinimalSource` (`file_path`, `first_character_index`, `last_character_index`)
  * `MinimalSearchResults` (`question_id`, `question`, `retrieved_sources`)
  * `StudentSearchResults` (`search_results`, `k`)
  * `MinimalAnswer` (`question_id`, `question`, `retrieved_sources`, `answer`)
  * `StudentSearchResultsAndAnswer` (`search_results`, `k`)

---

## 6. Challenges & Engineering Solutions

### 6.1. Index Preservation During Score Ranking
* **Issue:** `sorted(scores)` strips original chunk indices.
* **Fix:** Enumerate chunk scores into `(index, score)` tuples and sort using `key=lambda x: x[1], reverse=True`.

### 6.2. Missing Non-Markdown Documentation Files
* **Issue:** 3 reference questions targeted `CMakeLists.txt`, which was initially excluded by `.py`/`.md` filtering.
* **Fix:** Added `.txt` handling (including `CMakeLists.txt`) to file discovery and routed it to text chunking.

### 6.3. Dynamic Dataset Scoping
* **Issue:** Dataset filenames collide if written to a single folder.
* **Fix:** Preserved filename identity dynamically via `Path(dataset_path).name` into user-configured `--save_directory`.

### 6.4. Truncated Model Answers & Thinking Overhead
* **Issue:** Default `Qwen3-0.6B` reasoning generated excessive thinking tokens, exhausting `max_new_tokens` and truncating final answers.
* **Fix:** Injected empty `<think>\n</think>\n` tokens into the prompt to bypass reasoning deliberation, and expanded `max_new_tokens` to 180. Resulted in 0/100 truncations and an 80% reduction in inference latency.