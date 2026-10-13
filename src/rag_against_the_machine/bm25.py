"""
BM25 (Best Matching 25) Retrieval Algorithm
============================================

MATHEMATICAL FOUNDATION:
------------------------
BM25 is a probabilistic ranking algorithm used to calculate the relevance
score of a document (chunk D) given a search query (Q = [q_1, q_2, ..., q_m]).

1. The Scoring Formula:
   For a query Q and a document D:
   
                                           f(q_i, D) * (k1 + 1)
Score(D, Q) = SUM [ IDF(q_i) * --------------------------------------------- ]
             i                  f(q_i, D) + k1 * (1 - b + b * (|D| / avgdl))

   Where:
   - q_i        : The i-th term (word) in the query.
   - f(q_i, D)  : Term Frequency (TF) — how many times q_i appears in chunk D.
   - |D|        : Document length — total number of words in chunk D.
   - avgdl      : Average Document Length across the entire corpus:
                  avgdl = (sum of all |D|) / N
   - N          : Total number of documents (chunks) in the corpus.
   - k1         : Term frequency saturation parameter (default: 1.5).
                  Controls how quickly additional occurrences of a term stop 
                  increasing the score. Prevents keyword repetition from dominating.
   - b          : Document length normalization parameter (default: 0.75).
                  Controls how much longer documents are penalized.
                  - If b = 1.0: Full length penalty (favors short, dense snippets).
                  - If b = 0.0: No length penalty (document length is ignored).

2. Inverse Document Frequency (IDF) Formula:
   Measures the rarity of a term across all documents:

                         N - n(q_i) + 0.5
      IDF(q_i) = ln( 1 + ---------------- )
                           n(q_i) + 0.5

   Where:
   - n(q_i)     : Document Frequency — number of chunks containing the word q_i.
   - N          : Total number of chunks.
   
   Behavior:
   - If a word appears in almost all chunks (e.g. 'the', 'import'):
     n(q_i) ≈ N  -->  IDF(q_i) ≈ 0  (word is penalized, negligible score).
   - If a word appears in very few chunks (e.g. 'LoraConfig', 'cache_engine'):
     n(q_i) is small  -->  IDF(q_i) is high (word receives massive score multiplier).

3. Pipeline Lifecycle:
   - Indexing Phase (Offline):
     Pre-calculate N, doc_len, avgdl, and IDF(word) for the entire vocabulary.
   - Retrieval Phase (Online):
     Compute Score(D, Q) for each chunk and rank the Top-k results.
"""

import math
from collections import Counter


class CustomBM25:
    def __init__(
        self,
        corpus: list[list[str]],
        k1: float = 1.5,            # Knob k1: Rapresent the term-saturation
        b: float = 0.75             # Knob b: Penality for the chunk's length
    ):
        """
        HYPERPARAMETERS ("Tuning Knobs"):
        In Machine Learning, parameters like k1 and b are called 'Hyperparameters'.
        The metaphor 'tuning knob' describes adjustable settings used to calibrate
        the behavior and sensitivity of the ranking algorithm:

        - k1 (default: 1.5): Term Frequency Saturation Knob.
          Controls the ceiling for word repetition, ensuring diminishing returns
          so that repeated words do not dominate the score.

        - b (default: 0.75): Length Normalization Penalty Knob.
          Penalizes chunks that are longer than the average chunk length (avgdl),
          favoring concise and focused snippets.
        """
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_len                # The word count of each chunk
        self.avgdl                  # Avarage word count across al chunk
        self.idf                    # Rarity score of every word in the corpus
