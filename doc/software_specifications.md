# Software Specifications: RAG against the machine

## 1. Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system designed to answer technical questions about the `vLLM` codebase.

The system relies on a **Standard RAG (or Naive RAG)** architecture: it processes raw text into an easily searchable format, retrieves the most relevant pieces of text based on a user's question, and uses a local Large Language Model (`Qwen/Qwen3-0.6B`) to generate a human-readable answer grounded in the retrieved sources.

---

## 2. System Architecture

The architecture is strictly divided into two decoupled flows:
1. **The ETL / Indexing Pipeline** (Offline Batch Processing)
2. **The Inference / Query Pipeline** (Online Real-Time Serving)

```
[Raw Repository: data/raw/]
        │
        ▼ (ETL / Indexing Pipeline - Offline)
 1. File Discovery & Extraction (.py, .md)
 2. Domain-Specific Chunking (< 2000 chars)
 3. Lexical Indexing (Custom BM25)
 4. Persistent Storage (data/processed/)
        │
════════╪═════════════════════════════════════════════════════════
        │
        ▼ (Inference / Query Pipeline - Online)
 1. User Query / Evaluation Dataset
 2. BM25 Retrieval (Top-K Chunks with exact file character spans)
 3. Augmentation (Prompt Assembly with Context)
 4. LLM Generation (Qwen/Qwen3-0.6B Inference)
 5. Pydantic Output Validation & Export (data/output/)
```

---

### 2.1. The ETL / Indexing Pipeline (Offline Batch Processing)
This pipeline extracts knowledge from the raw repository and prepares an optimized search index. It is executed via the `index` CLI command.

#### Stage 1: Extraction & Filtering
* **Input:** `data/raw/vllm-0.10.1/` directory.
* **Process:** Recursively traverses the file tree while applying strict filtering. Non-informational files, build artifacts, images, and configuration files (`.yaml`, `.toml`, `CMakeLists.txt`, `Dockerfile`) are ignored.
* **Target Files:** Only Python (`.py`) source files and Markdown (`.md`) documentation are kept.
* **Span Tracking:** For each file, exact character offsets are tracked relative to the original file content so that downstream character ranges (`first_character_index`, `last_character_index`) match verbatim.

#### Stage 2: Domain-Specific Chunking
Because Python code and Markdown documentation follow completely different syntactical structures, two distinct chunking strategies are implemented:
* **Markdown Chunking:** Splits text by semantic boundaries (headers `#`, `##`, `###`, horizontal rules, and paragraph breaks). This keeps documentation topics together.
* **Python Code Chunking:** Splits code along logical structural boundaries (classes, functions, methods, module docstrings). 
* **Hard Constraint:** Every chunk is strictly constrained to `--max_chunk_size` (default: 2,000 characters). The grader rejects any retrieved source exceeding 2,000 characters.

#### Stage 3: Lexical Index Construction
* Every chunk is tokenized into normalized terms (lowercasing, symbol stripping while preserving code identifiers).
* Global vocabulary metrics are pre-calculated: Document Frequencies ($DF$), Inverse Document Frequencies ($IDF$), individual chunk lengths ($|D|$), and the average corpus chunk length ($avgdl$).
* A custom inverted index / vocabulary map is constructed.

#### Stage 4: Persistence
* **Output Destination:** `data/processed/`.
* The inverted index, vocabulary tables, BM25 statistical parameters, and chunk metadata (file paths and character spans) are serialized to disk.
* **Performance Requirement:** The entire indexing process must complete in **less than 5 minutes**.

---

### 2.2. The Inference / Query Pipeline (Online Real-Time Serving)
This pipeline handles user interaction, retrieval scoring, context injection, and answer generation. It is executed via the `search`, `search_dataset`, `answer`, and `answer_dataset` CLI commands.

#### Stage 1: Query Preprocessing
* Receives a natural language question (either from an interactive CLI query or batch JSON datasets).
* The query undergoes the same tokenization and normalization pipeline as the indexed documents to ensure mathematical compatibility.

#### Stage 2: Lexical Retrieval (BM25 Scoring)
* Loads the pre-computed BM25 index from `data/processed/`.
* Evaluates the query against every chunk using the BM25 formula, penalizing length outliers and saturated terms.
* Ranks chunks by score and selects the **Top-$k$** candidates.
* Formats each result into a `MinimalSource` structure (`file_path`, `first_character_index`, `last_character_index`).
* **Performance Requirement:** Batch retrieval must process 200 questions in **under 90 seconds**, reaching at least **80% Recall@5 on docs** and **50% Recall@5 on code**.

#### Stage 3: Augmentation & Prompt Assembly
* Extracts the raw text snippets corresponding to the Top-$k$ `MinimalSource`s.
* Formats a grounded prompt instructing the LLM to answer the question using **only** the provided snippets, mitigating hallucinations.

#### Stage 4: Generation (Local LLM Inference)
* Feeds the prompt to the local `Qwen/Qwen3-0.6B` model within its context window budget.
* The model synthesizes a concise, grounded explanation.

#### Stage 5: Structured Serialization
* Wraps the results into strictly validated Pydantic models:
  * `StudentSearchResults` for search-only operations.
  * `StudentSearchResultsAndAnswer` for full Q&A operations.
* Serializes valid JSON output into `data/output/`.

---

## 3. Lexical Search: TF-IDF vs. BM25

The project requires implementing at least one classic lexical retrieval method: **TF-IDF** or **BM25**.

### 3.1. Theoretical Comparison

| Feature | TF-IDF (Term Frequency - Inverse Document Frequency) | BM25 (Best Matching 25) |
| :--- | :--- | :--- |
| **Model Type** | Vector Space Model (heuristic) | Probabilistic Relevance Framework |
| **Term Frequency (TF) Scaling** | **Linear or Logarithmic:** Score grows continuously as a term is repeated in a document. | **Saturated:** Non-linear curve controlled by parameter $k_1$. Repeated terms reach a ceiling. |
| **Document Length Normalization** | **Weak / Fixed:** Long documents with many terms naturally accumulate higher scores. | **Explicit & Tunable:** Controlled by parameter $b$. Normalizes against average corpus document length ($avgdl$). |
| **Susceptibility to Keyword Stuffing** | **High:** Repetitive identifiers dominate scores. | **Low:** Diminishing returns prevent repeated words from dominating. |
| **Tuning Parameters** | None (standard formula). | $k_1 \in [1.2, 2.0]$ (saturation) and $b \approx 0.75$ (length penalty). |

---

### 3.2. Detailed Mathematical Breakdown

#### 1. Term Frequency Saturation
* **TF-IDF:** If a chunk contains the keyword `"tensor"` 10 times, its term frequency score is roughly 10 times higher than a chunk with 1 mention. In codebases, boilerplate repetition (e.g., loops, variable reassignment) causes TF-IDF to falsely rank repetitive utility code over concise definitions.
* **BM25:** Uses the saturation formula:
  $$\text{TF}_{\text{BM25}} = \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \text{length\_penalty}}$$
  As $f(q_i, D)$ increases, the score asymptotes toward $k_1 + 1$. Finding a term 15 times is barely more valuable than finding it 3 times.

#### 2. Document Length Normalization
* **TF-IDF:** Has no native awareness of whether a chunk is unusually short or unusually long relative to the rest of the corpus.
* **BM25:** Uses parameter $b$ to penalize chunks that are longer than the average document length ($avgdl$):
  $$\text{length\_penalty} = 1 - b + b \cdot \frac{|D|}{avgdl}$$
  * If a 150-character chunk contains all search terms, BM25 recognizes it as dense, highly relevant context.
  * If a 2,000-character chunk contains the same terms dispersed across irrelevant code, BM25 discounts its score.

---

### 3.3. Justification: Why BM25 was Chosen for this Project

1. **High Variance in Code & Doc Chunk Sizes:** In the vLLM repository, documentation sections and Python functions vary significantly in length. BM25’s tunable length normalization ($b = 0.75$) prevents massive files from drowning out concise, accurate snippets.
2. **Resistance to Identifier Repetition:** Python code frequently repeats keywords (e.g., `self`, `kwargs`, `model`, `engine`, `config`). TF-IDF over-rewards repetitive boilerplate. BM25’s term saturation ($k_1 = 1.5$) prevents keyword spam from biasing retrieval.
3. **Meeting Mandatory Recall Thresholds:** Meeting **$\ge 80\%$ Recall@5 on documentation** and **$\ge 50\%$ Recall@5 on code** requires high precision in the top 5 results. BM25 consistently outperforms standard TF-IDF across information retrieval benchmarks on technical corpora.

---

## 4. Codebase Module Architecture

```
src/
├── __init__.py         # Package initialization
├── __main__.py         # Python Fire CLI entrypoint
├── models.py           # Pydantic data schemas (MinimalSource, StudentSearchResults, etc.)
├── chunker.py          # Dual chunking engine (Python AST/regex & Markdown splitters)
├── indexer.py          # Custom BM25 engine & offline index persistence
├── retriever.py        # Top-K query retrieval & character span extraction
└── generator.py        # Qwen/Qwen3-0.6B model loader, prompt builder, inference engine
```