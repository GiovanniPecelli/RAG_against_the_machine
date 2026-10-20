r"""
BM25 (Best Matching 25) Retrieval Algorithm
============================================

MATHEMATICAL FOUNDATION:
------------------------
BM25 is a probabilistic ranking algorithm used to calculate the relevance
score of a document (chunk D) given a search query (Q = [q_1, q_2, ..., q_m]).

1. The Scoring Formula:
   For a query Q and a document D:
   
                  /                        f(q_i, D) * (k1 + 1)              \
Score(D, Q) = SUM | IDF(q_i) * --------------------------------------------- |
               i  \             f(q_i, D) + k1 * (1 - b + b * (|D| / avgdl)) /

   Where:
   - i          : Mean that the result is a SUM of every word's in the query
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

                    /        N - n(q_i) + 0.5   \
      IDF(q_i) = ln | 1 + --------------------- |
                    \         n(q_i) + 0.5      /

   Where:
   - n(q_i)     : Document Frequency — number of chunks containing the word.
   - N          : Total number of chunks.
   - "+ 0.5"    : simply an anti-crash safety airbag avoiding division by 0.
   
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

from collections import Counter
import math


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

        # The number of word for each chunk -> one chunk a time
        self.doc_len: list[int] = [len(doc) for doc in corpus]

        # Avarage word count across al chunk
        self.avgdl = (
            sum(self.doc_len) / self.corpus_size
        ) if self.corpus_size > 0 else 0.0

        self.doc_freqs, self.idf = self._calculate_idf(corpus)
    
    def _calculate_idf(
            self, corpus: list[list[str]]
    ) -> tuple[list[dict[str, int]], dict[str, float]]:
        """
        Calculate the rarity score of every word in the corpus, called
        Inverse Document Frequency (IDF)

        doc_frequencies: For each chunk -> count_word occurrence
            [
                chunk 0 {"word_1": 2, "word_2": 1}
                chunk 1 {"word_2": 1, "word_3": 1}
            ]
        word_doc_counts: How many chunk have that word inside
            {
                "word_1": 1, mean: only 1 chunk have the word_1
                "word_2": 2,
            }
        """
        doc_frequencies: list[dict] = []
        word_doc_counts = Counter()
        for chunk in corpus:
            # How many times words are in that chunk
            freq: dict = Counter(chunk)
            doc_frequencies.append(freq)
            for word in freq.keys():
                word_doc_counts[word] += 1
        
        idf = {}
        for word, count in word_doc_counts.items():
            numerator = self.corpus_size - count + 0.5
            denominator = count + 0.5
            idf[word] = math.log(1.0 + (numerator / denominator))
        
        return doc_frequencies, idf

    def get_score(self, query: list[str]) -> list[float]:
        """
        Calculate the score for each chunk for the current query
        scores: list
            - index: rapresents the chunk
            - value: rapresents the score of that chunk for the query
        BM25:
            - penality: check the chunk length to avoid spamming
            - tf: Term frequency
        """
        scores: list = [0.0] * self.corpus_size

        for index, chunk_freq in enumerate(self.doc_freqs):
            score = 0.0

            for word in query:
                if word not in chunk_freq:
                    continue
                
                # How many times the word appear in the current chunk
                freq = chunk_freq[word]

                # BM25 formula
                penality = 1.0 - self.b + self.b * (
                    self.doc_len[index] / self.avgdl
                )
                tf = (freq * (self.k1 + 1)) / (freq + self.k1 * penality)
                
                # Score Compounding:
                # If multiple words from the query appear in this chunk,
                # their individual scores are accumulated together into
                # the total chunk score.
                score += self.idf.get(word, 0.0) * tf

            scores[index] = score

        return scores

if __name__ == "__main__":
    tokenized_corpus = [
        ["class", "engine", "run"],
        ["engine", "cache", "engine"],
        ["run", "fast", "cache"],
    ]

    bm25 = CustomBM25(tokenized_corpus)
    print("=== INITIALIZING BM25 ===")
    print(f"Total chunks (N): {bm25.corpus_size}")
    print(f"Doc lengths: {bm25.doc_len}")
    print(f"Average length (avgdl): {bm25.avgdl:.2f}")
    print("\nIDF Rarity Scores:")
    for word, rarity in bm25.idf.items():
        print(f"  - '{word}': {rarity:.4f}")

    print("\n" + "=" * 50)

    # 1. Test Query 1: Single word "engine"
    query_1 = ["engine"]
    scores_1 = bm25.get_score(query_1)
    print(f"Query: {query_1}")
    print(f"Scores for all chunks: {[round(s, 4) for s in scores_1]}")
    winner_1 = scores_1.index(max(scores_1))
    print(f"Best match: Chunk {winner_1} (score: {scores_1[winner_1]:.4f})")

    print("\n" + "=" * 50)

    # 2. Test Query 2: Multi-word "fast" and "cache"
    query_2 = ["fast", "cache"]
    scores_2 = bm25.get_score(query_2)
    print(f"Query: {query_2}")
    print(f"Scores for all chunks: {[round(s, 4) for s in scores_2]}")
    winner_2 = scores_2.index(max(scores_2))
    print(f"Best match: Chunk {winner_2} (score: {scores_2[winner_2]:.4f})")