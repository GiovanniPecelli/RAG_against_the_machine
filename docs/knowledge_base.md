# Technical Knowledge Base: RAG Against the Machine

A comprehensive theoretical and conceptual guide covering the mathematical, algorithmic, and engineering principles behind Retrieval-Augmented Generation (RAG), Information Retrieval (IR), Dense Embeddings, and Local Language Models.

---

## Table of Contents
1. [Foundations of Retrieval-Augmented Generation (RAG)](#1-foundations-of-retrieval-augmented-generation-rag)
2. [Information Retrieval (IR): Sparse vs. Dense](#2-information-retrieval-ir-sparse-vs-dense)
3. [Lexical Retrieval: TF-IDF vs. BM25 Deep Dive](#3-lexical-retrieval-tf-idf-vs-bm25-deep-dive)
4. [Semantic Retrieval: Vector Embeddings & Transformers](#4-semantic-retrieval-vector-embeddings--transformers)
5. [Hybrid Retrieval & Fusion Algorithms](#5-hybrid-retrieval--fusion-algorithms)
6. [Chunking Strategies for Technical Codebases](#6-chunking-strategies-for-technical-codebases)
7. [Evaluation Metrics & Character-Level IoU](#7-evaluation-metrics--character-level-iou)
8. [Autoregressive Generation & Local Inference Optimization](#8-autoregressive-generation--local-inference-optimization)

---

## 1. Foundations of Retrieval-Augmented Generation (RAG)

### The Knowledge Boundary of Language Models
A Large Language Model (LLM) possesses parameterized knowledge frozen at the time its pre-training finishes. It cannot natively know:
1. Private internal repositories (e.g., proprietary corporate code).
2. Code versions, commits, or bug fixes made after its training cutoff date.
3. Exact identifier locations, line numbers, or specific file offsets.

When questioned about unknown or detailed technical entities, autoregressive models default to plausible-sounding hallucinations rather than stating uncertainty.

### Fine-Tuning vs. Context Injection (RAG)
| Dimension | Fine-Tuning | Retrieval-Augmented Generation (RAG) |
| :--- | :--- | :--- |
| **Knowledge Update** | Requires re-training or adapter updates (hours/days). | Immediate upon index update (seconds/minutes). |
| **Traceability / Provenance** | Opaque; neural weights cannot cite file names or spans. | Transparent; every answer is bound to exact source spans. |
| **Cost & Compute** | High GPU compute cost. | Runs on lightweight hardware (even CPU-only). |
| **Hallucination Rate** | Moderate to High. | Low (heavily constrained by prompt context). |

### The 4 Stages of the RAG Lifecycle
```
[ Corpus ] ──> 1. Indexing (ETL) ──> [ Inverted / Vector Index ]
                                                │
[ User Query ] ──> 2. Retrieval ────────────────┘
                         │
                         ▼
             [ Relevant Snippets ]
                         │
                         ▼
                 3. Augmentation (Prompt Engineering)
                         │
                         ▼
                 4. Generation (Local LLM Inference) ──> [ Grounded Answer ]
```

1. **Indexing (ETL):** Discovers documents, segments them into semantically cohesive chunks, and computes searchable representations (sparse vocabulary matrices or dense embeddings).
2. **Retrieval:** Evaluates incoming user queries against the index to return the top-$k$ most relevant candidate chunks.
3. **Augmentation:** Extracts raw text slices for retrieved chunks and constructs a strict prompt boundary instructing the model to rely only on the provided evidence.
4. **Generation:** An autoregressive model synthesizes the answer conditioned on the retrieved context.

---

## 2. Information Retrieval (IR): Sparse vs. Dense

Modern Information Retrieval is divided into two fundamental paradigms:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      Information Retrieval (IR)                        │
└──────────────────┬──────────────────────────────────┬──────────────────┘
                   │                                  │
         ┌─────────▼────────┐                ┌────────▼─────────┐
         │ Sparse / Lexical │                │  Dense / Semantic│
         └─────────┬────────┘                └────────┬─────────┘
                   │                                  │
    • Exact keyword matching          • Conceptual / semantic similarity
    • High-dimensional, sparse vectors• Low-dimensional, dense vectors (e.g., 384)
    • Inverted Index (TF-IDF, BM25)   • Vector Index (Cosine, Dot Product)
    • Excels at: code identifiers,    • Excels at: synonyms, intent,
      exact error codes, function names paraphrases, high-level queries
```

### The Inverted Index Structure
In sparse retrieval, documents are indexed using an **Inverted Index**: a mapping from unique vocabulary terms to the list of document IDs (and positions) where they appear:

```
"engine"      ──> [ Doc 1 (tf: 3), Doc 14 (tf: 1), Doc 98 (tf: 5) ]
"cache"       ──> [ Doc 1 (tf: 2), Doc 23 (tf: 4) ]
"loraconfig"  ──> [ Doc 42 (tf: 8) ]
```

When a query arrives, retrieval requires looking up only the query's terms in the inverted index, skipping the vast majority of unrelated documents.

---

## 3. Lexical Retrieval: TF-IDF vs. BM25 Deep Dive

### 3.1. Standard TF-IDF Formula

```
                   f(t, D)            /       N           \
Score_TFIDF(t, D) = ───────  ×  ln    │ ───────────────── │
                     |D|              \ |{d ∈ D : t ∈ d}  /
                   └──┬──┘             └────────┬────────┘
                   Term Freq (TF)       Inverse Doc Freq (IDF)
```

$$\text{Score}_{\text{TF-IDF}}(t, D) = \frac{f(t, D)}{|D|} \times \ln\left(\frac{N}{|\{d \in \mathcal{D} : t \in d\}|}\right)$$

* **$f(t, D)$**: Number of times term $t$ appears in chunk $D$.
* **$|D|$**: Total number of words in chunk $D$.
* **$N$**: Total number of chunks in the corpus.
* **$|\{d \in \mathcal{D} : t \in d\}|$**: Number of chunks containing term $t$.

#### Why TF-IDF Underperforms in Technical Codebases:
1. **Linear TF Scaling:** Repeating a variable 20 times in a function produces a score 20 times higher than a function mentioning it once.
2. **Weak Length Normalization:** Long utility files easily accumulate higher scores than concise function definitions.

---

### 3.2. Okapi BM25 Formula

```
                     m                                    f(q_i, D) · (k₁ + 1)
Score_BM25(D, Q) =  ∑   IDF(q_i)  ×  ─────────────────────────────────────────────────────────
                    i=1               f(q_i, D)  +  k₁ · ( 1 - b  +  b · ( |D| / avgdl ) )
                       └────┬───┘    └───────────────────────────┬───────────────────────────┘
                         Weight                       Saturated TF with Length Penalty
```

$$\text{Score}_{\text{BM25}}(D, Q) = \sum_{i=1}^{m} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

---

### 3.3. Robertson IDF with $+0.5$ Smoothing

```
              /      N  -  n(q_i)  +  0.5 \
IDF(q_i) = ln │ 1 + ────────────────────── │
              \          n(q_i)  +  0.5   /
```

$$\text{IDF}(q_i) = \ln \left( 1 + \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} \right)$$

* **$N$**: Total number of chunks in corpus.
* **$n(q_i)$**: Number of chunks containing query term $q_i$.
* **$+0.5$ Smoothing**: Ensures IDF stays positive ($\text{IDF} > 0$) even when a common word appears in $> 50\%$ of chunks, preventing division by zero.

---

### 3.4. BM25 Tuning Knobs ($k_1$ and $b$)

```
1. Term Saturation Knob (k₁ = 1.5):
   Controls the saturation ceiling. As term frequency f(q_i, D) grows toward infinity,
   the score asymptotes to (k₁ + 1) = 2.5:

   f = 1  ──> Score ≈ 1.00
   f = 3  ──> Score ≈ 1.80
   f = 10 ──> Score ≈ 2.25
   f = 50 ──> Score ≈ 2.45   (Diminishing returns prevent keyword spamming)

2. Length Normalization Knob (b = 0.75):
   Penalty Factor = 1 - b + b · ( |D| / avgdl )

   |D| = avgdl  ──> Factor = 1.00 (Neutral)
   |D| > avgdl  ──> Factor > 1.00 (Penalizes verbose/bloated files)
   |D| < avgdl  ──> Factor < 1.00 (Boosts compact, dense snippets)
```

---

## 4. Semantic Retrieval: Vector Embeddings & Transformers

### 4.1. The `all-MiniLM-L6-v2` Architecture
* **Parameters:** 22.7 Million.
* **Transformer Layers:** 6.
* **Hidden Dimension:** 384.
* **Attention Heads:** 12.

---

### 4.2. Scaled Dot-Product Self-Attention Formula

```
                        /    Q · Kᵀ   \
Attention(Q, K, V) = softmax │ ─────────── │  ×  V
                        \    √d_k     /
                             └───┬───┘
                            Scaling
```

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

* **$Q$ (Queries), $K$ (Keys), $V$ (Values)**: Linear projections of token embeddings ($X \in \mathbb{R}^{N \times 384}$).
* **$Q K^T \in \mathbb{R}^{N \times N}$**: Dot-product relevance between every pair of words.
* **$\sqrt{d_k}$ ($d_k = 32$)**: Scaling factor that prevents gradients from vanishing during softmax.
* **$\text{softmax}(\cdot)$**: Normalizes pairwise scores into row probabilities summing to $1.0$.

---

### 4.3. Mean Pooling with Attention Mask

```
              N
              ∑  ( mask_i  ×  token_vector_i )
             i=1
Chunk/Text Vector u = ───────────────────────────────
                                  N
                                  ∑  mask_i
                                 i=1
```

$$\vec{u} = \frac{\sum_{i=1}^N m_i \cdot \vec{t}_i}{\sum_{i=1}^N m_i}$$

* **$\vec{t}_i \in \mathbb{R}^{384}$**: Output embedding for token $i$.
* **$m_i \in \{0, 1\}$**: Attention mask bit ($1$ for a real token, $0$ for padding).
* **Result $\vec{u} \in \mathbb{R}^{384}$**: One vector representing the complete input text, usually one chunk, without padding positions affecting its mean.

For a batch of $B$ texts padded to a sequence length $L$, the transformer output has shape $[B, L, 384]$. Mean pooling operates over the token dimension $L$ independently for each text, producing $[B, 384]$: one vector per input chunk, not one vector for the whole batch.

---

### 4.4. $L_2$ Normalization Formula & PyTorch Implementation

```
                         u
Normalized Vector v = ─────────
                      || u ||₂

                          u
                    = ────────────────────────
                       ______________________
                      /  384
                     √   ∑    ( u_k )²
                        k=1
```

$$\vec{v} = \frac{\vec{u}}{\|\vec{u}\|_2} = \frac{\vec{u}}{\sqrt{\sum_{k=1}^{384} u_k^2}}$$

#### The Intuition: Direction vs. Length
* Think of an embedding vector as an arrow in 384-dimensional space.
* **The Direction** represents the semantic concept (e.g., "cache configuration").
* **The Length** reflects arbitrary magnitude introduced by word count and pooling weights.
* Dividing the vector by its Euclidean length leaves its **direction completely unchanged**, while fixing its length to exactly **$1.0$ meter**.

#### PyTorch Code Breakdown: `torch.nn.functional.normalize(embeddings, p=2, dim=1)`
1. **`p=2` (Euclidean / $L_2$ Norm)**:
   Applies the generalized Pythagorean theorem across all coordinates:
   $$\|\vec{u}\|_2 = \sqrt{u_1^2 + u_2^2 + \dots + u_{384}^2}$$
   $p=2$ specifies the power of 2 (sum of squares under a square root).
2. **`dim=1` (Row-wise computation)**:
   Given a batch tensor of shape `[batch_size, 384]`, `dim=1` tells PyTorch to compute the Euclidean norm horizontally across the 384 feature columns for each sentence independently, preventing cross-contamination between batch rows.

---

### 4.5. Why $L_2$ Normalization Accelerates Retrieval (Cosine Similarity $\to$ Dot Product)

#### The General Cosine Similarity Formula:
$$\cos(\theta) = \frac{\vec{q} \cdot \vec{c}}{\|\vec{q}\|_2 \times \|\vec{c}\|_2}$$

* **Without Prior Normalization:**
  Evaluating 1 query against 12,500 indexed chunks requires computing **12,500 square roots and floating-point divisions** at query time, creating a severe CPU latency bottleneck.
* **With Pre-Normalized Vectors ($\|\vec{q}\|_2 = 1.0$ and $\|\vec{c}\|_2 = 1.0$):**
  $$\cos(\theta) = \frac{\vec{q} \cdot \vec{c}}{1.0 \times 1.0} = \vec{q} \cdot \vec{c} = \sum_{k=1}^{384} q_k c_k$$
  All square roots and divisions vanish completely. The entire search operation collapses into a single parallelized dot-product matrix multiplication:

$$\vec{S} = C \cdot \vec{q}$$

                  384
cos(θ) = q · c =  ∑   ( q_k  ×  c_k )
                 k=1
```

$$\cos(\theta) = \frac{\vec{q} \cdot \vec{c}}{\|\vec{q}\|_2 \|\vec{c}\|_2} = \vec{q} \cdot \vec{c} = \sum_{k=1}^{384} q_k c_k$$

#### Parallel Corpus Matrix Search:
Searching $M$ indexed chunks with a query vector $\vec{q}$ simplifies to one matrix-vector multiplication:

$$\vec{S} = C \cdot \vec{q}$$

```
   [   — chunk 1 (384) —   ]                 [ score 1 ]
   [   — chunk 2 (384) —   ]     [ query ]   [ score 2 ]
   [          ...          ]  ×  [ (384) ] = [   ...   ]
   [   — chunk M (384) —   ]                 [ score M ]
         Matrix C (M × 384)         Vector q     Scores (M)
```

---

## 5. Hybrid Retrieval & Fusion Algorithms

### Reciprocal Rank Fusion (RRF) Formula

```
                 1                        1
RRF_Score(d) = ──────────────  +  ─────────────────
               k_rrf + r_BM25(d)     k_rrf + r_Dense(d)
```

$$\text{RRF\_Score}(d) = \sum_{m \in \mathcal{M}} \frac{1}{k_{\text{rrf}} + r_m(d)}$$

* **$\mathcal{M} = \{\text{BM25}, \text{Dense}\}$**: The search engines combined.
* **$r_m(d) \in \{1, 2, \dots, K\}$**: Rank position of chunk $d$ in system $m$ (1st place = 1, 2nd place = 2).
* **$k_{\text{rrf}} = 60$**: Smoothing constant. Prevents a single system from dominating the combined ranking.
* **Advantage**: Score-free combination. Does not require calibrating BM25 scores $[0, \infty)$ against Cosine scores $[-1, 1]$.

### RRF in This Project
The hybrid retriever gathers up to `min(max(2 * k, 50), number_of_chunks)` candidates from each ranking, adds their reciprocal-rank contributions, and returns the top `k` combined chunk indices. It uses `k_rrf = 60`.

---

## 6. Chunking Strategies for Technical Codebases

```
Raw File
   │
   ├── If Python (.py) ──> Marker-Based Splitter
   │                        • Splits at: 'class ', 'def ', 'async def '
   │                        • Preserves: method signatures & docstrings
   │
   ├── If Markdown (.md) ──> Header-Based Splitter
   │                        • Uses preferred markers: '# ', '## '
   └── If Text (.txt) ────> Line-Based Splitter
                            • Uses newline boundaries
```

All three strategies also split content to respect the configured maximum chunk size, which defaults to 2,000 characters. Preferred markers start a new chunk only after the current chunk has reached 1,000 characters; this avoids splitting too early at every heading or code declaration. Oversized individual lines are split into smaller pieces.

---

## 7. Evaluation Metrics & Character-Level IoU

### 7.1. Recall@k Formula

```
               | Top-k Retrieved Sources  ∩  Ground-Truth Sources |
Recall@k  =   ───────────────────────────────────────────────────────
                             | Ground-Truth Sources |
```

$$\text{Recall}@k = \frac{|\text{Retrieved}_k \cap \text{GroundTruth}|}{|\text{GroundTruth}|}$$

---

### 7.2. Character-Level Intersection over Union (IoU)

For ground-truth range $A = [A_{\text{start}}, A_{\text{end}}]$ and candidate chunk $B = [B_{\text{start}}, B_{\text{end}}]$:

```
          Length of Overlap (Intersection)
IoU  =  ────────────────────────────────────
          Length of Total Span (Union)

           max( 0,  min(A_end, B_end) - max(A_start, B_start) )
     =  ──────────────────────────────────────────────────────────
           max(A_end, B_end)  -  min(A_start, B_start)
```

$$\text{IoU}(A, B) = \frac{\text{length}(A \cap B)}{\text{length}(A \cup B)} = \frac{\max(0, \min(A_{\text{end}}, B_{\text{end}}) - \max(A_{\text{start}}, B_{\text{start}}))}{\max(A_{\text{end}}, B_{\text{end}}) - \min(A_{\text{start}}, B_{\text{start}})}$$

* **Recall Criterion:** If candidate chunk $B$ is in the **same file** as ground truth $A$ and **$\text{IoU}(A, B) \ge 0.05$**, the source counts as successfully retrieved.

---

## 8. Autoregressive Generation & Local Inference Optimization

### 8.1. Autoregressive Probability Chain Rule

$$P(w_1, w_2, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_1, w_2, \dots, w_{t-1})$$

Text is generated token-by-token. Each iteration samples next token $w_t$ based on the full prefix history.

### 8.2. CPU Inference Optimizations
1. **Bypassing Reasoning Deliberation:** Inverting `<think>\n</think>\n` directly into prompt templates stops Qwen3 from generating internal monologues, speeding up generation by $80\%$.
2. **KV Caching (`use_cache=True`):** Avoids recomputing attention states for previous tokens; turns $O(T^2)$ into $O(T)$.
3. **Brain Floating Point (`bfloat16`):** Cuts memory bandwidth in half and leverages hardware AVX vector units.
4. **Context Budgeting:** Feeding Top-$3$ sources instead of Top-$10$ eliminates context pollution and prevents token exhaustion.
