*This project has been created as part of the 42 curriculum by gpecelli.*

# RAG Against the Machine

A lightweight, local **Retrieval-Augmented Generation (RAG)** system designed to ingest, chunk, index, and query the complex `vLLM` codebase (`vllm-0.10.1`). Built without heavy external retrieval frameworks, the project implements a custom **BM25 lexical search engine** from scratch and pairs it with a local Small Language Model (`Qwen/Qwen3-0.6B`) to deliver answers grounded strictly in retrieved source code and technical documentation.

---

## Table of Contents
- [1. Description](#1-description)
- [2. System Architecture](#2-system-architecture)
- [3. Chunking Strategy](#3-chunking-strategy)
- [4. Retrieval Method (BM25)](#4-retrieval-method-bm25)
- [5. Performance Targets & Analysis](#5-performance-targets--analysis)
- [6. Design Decisions](#6-design-decisions)
- [7. Challenges Faced & Solutions](#7-challenges-faced--solutions)
- [8. Instructions & Installation](#8-instructions--installation)
- [9. Example Usage](#9-example-usage)
- [10. Resources & AI Usage Disclosure](#10-resources--ai-usage-disclosure)

---

## 1. Description

Large Language Models often suffer from hallucinations when reasoning over rapidly evolving or highly technical codebases. **RAG against the machine** addresses this challenge through a transparent, high-performance RAG pipeline that:
1. **Parses & Chunks** raw source files (`.py`) and documentation (`.md`) while strictly preserving character offsets (`first_character_index`, `last_character_index`).
2. **Indexes** the entire corpus offline using a custom, probabilistic BM25 implementation.
3. **Retrieves** the top-$k$ most relevant context snippets in milliseconds.
4. **Augments** prompts with retrieved context slices and generates grounded natural language answers using `Qwen/Qwen3-0.6B`.
5. **Evaluates** retrieval accuracy against ground-truth datasets using the `Recall@k` metric.

---

## 2. System Architecture

The system decouples data preparation from real-time querying into two distinct pipelines:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   ETL / INDEXING PIPELINE (Offline)                    │
│                                                                        │
│  data/raw/vllm-0.10.1/                                                 │
│        │                                                               │
│        ├──> File Discovery (.py, .md filtering)                        │
│        ├──> Structural Chunking (< 2000 chars, span tracking)          │
│        ├──> Term Tokenization (lowercased term frequencies)            │
│        ├──> Global BM25 Table Construction (IDF, avgdl, |D|)           │
│        └──> Binary Persistence ──> data/processed/index.pkl            │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
═════════════════════════════════════╪════════════════════════════════════
                                     │
┌────────────────────────────────────▼───────────────────────────────────┐
│                  INFERENCE / SERVING PIPELINE (Online)                 │
│                                                                        │
│  User Query / Public Dataset JSON                                      │
│        │                                                               │
│        ├──> Query Tokenization                                         │
│        ├──> BM25 Scoring & Top-K Ranking                               │
│        ├──> Provenance Formatting (MinimalSource objects)              │
│        ├──> Context Extraction (file slicing [start:end])              │
│        ├──> Local LLM Inference (Qwen/Qwen3-0.6B)                      │
│        └──> JSON Export ──> data/output/ (Pydantic validation)         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Chunking Strategy

Standard naive fixed-character chunking destroys syntactic context in code and headers in documentation. This project implements a **domain-aware chunking strategy**:

* **Python Source Files (`.py`):** Split along structural code definitions (`class `, `def `, `async def `) and top-level docstrings, ensuring functions and classes remain cohesive search units.
* **Markdown Documentation (`.md`):** Split along hierarchical semantic headers (`#`, `##`, `###`), keeping explanations grouped with their respective section headings.
* **Hard Size Constraint:** Every chunk is strictly bounded to $\le 2000$ characters (`--max_chunk_size`, default: 2000). If a function or section exceeds 2000 characters, it is recursively subdivided without crossing word boundaries.
* **Exact Span Tracking:** Each chunk stores its origin and absolute character offsets:
  * `file_path`: Relative POSIX path (e.g., `data/raw/vllm-0.10.1/vllm/engine.py`).
  * `first_character_index`: Start character offset within the source file.
  * `last_character_index`: End character offset within the source file.

---

## 4. Retrieval Method (BM25)

Instead of relying on standard TF-IDF (which suffers from term-frequency over-rewarding and lacks document length penalties), we implemented the **Best Matching 25 (BM25)** probabilistic ranking algorithm from scratch.

### The Mathematical Formulation

For a search query $Q$ with terms $q_1, \dots, q_m$ and a candidate chunk $D$:

$$\text{Score}(D, Q) = \sum_{i=1}^{m} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Where:
* **$f(q_i, D)$:** Term frequency of word $q_i$ inside chunk $D$.
* **$|D|$ / $\text{avgdl}$:** Length of chunk $D$ divided by average chunk length across the entire corpus.
* **$k_1 = 1.5$ (Term Saturation Knob):** Prevents repeated keywords from continuously inflating scores. Finding an identifier 10 times yields diminishing returns compared to finding it 2 times.
* **$b = 0.75$ (Length Normalization Knob):** Penalizes long, verbose chunks while favoring concise, dense code snippets.
* **$\text{IDF}(q_i)$ (Robertson Smoothing):**
  $$\text{IDF}(q_i) = \ln \left( 1 + \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} \right)$$
  Penalizes ubiquitous keywords (`import`, `self`, `return`) while heavily weighting rare domain terms (`LoraConfig`, `CacheEngine`).

---

## 5. Performance Targets & Analysis

The following BM25 measurements were collected in the current development
workspace. Recall was measured with the project's `evaluate` command against
the answered public datasets. Timing includes command startup; the 200-query
search timing also includes loading the index and writing the result JSON. The
throughput batch combined the 100 documentation and 99 code questions, then
repeated one query to reach 200.

| Benchmark Metric | Mandatory Threshold | Measured Result |
| :--- | :--- | :--- |
| **Documentation Recall@5** | $\ge 80\%$ | **84.0%** (100 questions) |
| **Code Recall@5** | $\ge 50\%$ | **80.8%** (99 questions) |
| **Indexing Throughput** | $\le 5$ minutes | **5.4 seconds** (1,969 files; 13,349 chunks) |
| **Search Throughput** | $\le 90$s for 200 questions | **8.8 seconds** (200-question batch) |
| **Chunk Context Length** | $\le 2000$ characters | Enforced by the configurable chunk-size limit |

---

## 6. Design Decisions

1. **Custom BM25 from First Principles:**
   * Avoided external libraries (`rank-bm25`, `scikit-learn`) to ensure zero black-box behavior, exact token alignment between indexing and querying, and strict memory efficiency.
2. **Factory Method Pattern (`@classmethod load`):**
   * Decoupled index loading from instance state. Calling `retriever = Retriever.load(processed_dir)` safely deserializes `index.pkl` in one call without requiring empty dummy instances.
3. **API Symmetry (`search = retrieve`):**
   * Named core methods to mirror CLI commands 1-to-1 (`index` $\rightarrow$ `Indexer.build()`, `search` $\rightarrow$ `Retriever.search()`, `search_dataset` $\rightarrow$ `Retriever.search_dataset()`), eliminating mental friction.
4. **Pydantic v2 Contract Enforcement:**
   * Used Pydantic models (`MinimalSource`, `MinimalSearchResults`, `StudentSearchResults`) with native `.model_dump_json(indent=2)` for rapid, type-checked JSON output.
5. **Cross-Platform Compatibility:**
   * Normalized all paths via `Path.as_posix()` to prevent Windows backslash (`\`) discrepancies when evaluated on Linux moulinette environments.

---

## 7. Challenges Faced & Solutions

* **Preserving Indices During Top-$k$ Ranking:**
  * *Challenge:* Sorting raw score arrays (`sorted(scores)`) loses track of which chunk produced each score.
  * *Solution:* Combined `enumerate(scores)` with a lambda key (`key=lambda item: item[1], reverse=True`) to sort `(index, score)` tuples while keeping chunk indexing intact.
* **Separation of Provenance vs. Context:**
  * *Challenge:* `MinimalSource` stores character offsets, but LLM generation requires the actual text string.
  * *Solution:* Built an augmentation step that reads source files on demand and slices `file_content[first_char:last_char]` to assemble context prompts without bloating metadata JSONs.
* **Dynamic Dataset Scoping:**
  * *Challenge:* Multiple datasets (`dataset_docs_public.json`, `dataset_code_public.json`) must be evaluated without overwriting outputs.
  * *Solution:* Used `Path(dataset_path).name` to dynamically mirror input filenames inside the target `--save_directory`.

---

## 8. Instructions & Installation

### Prerequisites
Ensure you have Python and uv installed, then set up the environment:
* Python 3.10+
* [uv](https://docs.astral.sh/uv/) package manager

### Installation
```bash
# Clone the repository
git clone <repository_url>
cd RAG_against_the_machine

# Install dependencies with uv
make install
# or directly: uv sync
```
#### If you are on a 42 Network machine USER -> ($USER | (whoami) | user_name)

```bash
rm -rf .venv && mkdir -p /goinfre/$USER/{venvs/rag_against_the_machine,.cache,tmp,uv,huggingface}
```

```bash
export UV_PROJECT_ENVIRONMENT=/goinfre/$USER/venvs/rag_against_the_machine UV_LINK_MODE=copy XDG_CACHE_HOME=/goinfre/$USER/.cache TMPDIR=/goinfre/$USER/tmp UV_CACHE_DIR=/goinfre/$USER/uv HF_HOME=/goinfre/$USER/huggingface
```

### Data Preparation
```bash
# Extract vLLM raw repository and evaluation datasets
make data
```

### Quality Assurance (Linting)
```bash
# Run flake8 and mypy in strict typing mode
make lint
make lint-strict
```

---

## 9. Example Usage

All operations are exposed via `uv run python -m src <command>`:

### 1. Ingest & Index the Corpus
```bash
uv run python -m src index --max_chunk_size 2000
```

### 2. Search a Single Query
```bash
uv run python -m src search --query "What HTTP endpoint is used to load a LoRA adapter?" --k 5
```

### 3. Search an Entire Question Dataset
```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

### 4. Evaluate Search Results with the Moulinette
```bash
./moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 --max_context_length 2000
```

### 5. Generate Answers for a Dataset
```bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

---

## 10. Resources & AI Usage Disclosure

### References
* **Robertson et al. (1994):** *Okapi at TREC-3* — Foundation for the BM25 probabilistic retrieval formula.
* **vLLM Documentation:** Architecture and configuration reference for the raw repository.
* **Pydantic v2 Documentation:** Fast serialization and contract validation in Python.
* **[Python `pickle` Documentation](https://docs.python.org/3/library/pickle.html):** Python's object serialization module, used to save and reload the processed BM25 index in `data/processed/index.pkl`.
* **Python Fire:** Google library for building CLI interfaces automatically from classes.
* **Hugging Face `transformers`:** Library used for loading and running `Qwen/Qwen3-0.6B` (`AutoTokenizer`, `AutoModelForCausalLM`). The core autoregressive generation mechanism lives in [`transformers/generation/utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py):
  * `GenerationMixin.generate()` entrypoint: Line **2568**
  * Core autoregressive token loop (`_sample`): Line **3104**, with the `while self._has_unfinished_sequences(...)` iterative step loop at Line **3211**.

### AI Usage Disclosure
In accordance with the 42 curriculum guidelines, Artificial Intelligence (LLM pair-programming assistant) was used as follows:
* **Algorithmic Verification:** Assisting with mathematical derivation and review of the Robertson BM25 IDF smoothing formula.
* **Code Review & Type Checking:** Checking flake8 compliance, PEP 519 path conventions, and strict mypy type hints.
* **Architecture Discussion:** Discussing trade-offs between TF-IDF vs. BM25, factory pattern design for deserialization, and cross-platform path normalization.
* **Code Implementation:** All final code, chunking logic, data models, and index structures were designed, implemented, and reviewed directly by the student.
