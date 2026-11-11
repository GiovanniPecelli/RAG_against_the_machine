"""
What is Pickle in Python?
=========================
'pickle' is Python's built-in module for binary object serialization.

- Serialization (pickle.dump):
  Converts complex in-memory Python objects (such as chunk dictionaries and
  the CustomBM25 instance with precomputed IDF tables and term frequencies)
  into a byte stream that can be stored on disk (.pkl file).

- Deserialization (pickle.load):
  Reads the byte stream from disk and instantly reconstructs the original
  Python objects in RAM with their exact state preserved.

Why use Pickle in this RAG pipeline?
It decouples offline indexing from online retrieval. We index the codebase
once (~3,200 files) and save the resulting objects to data/processed/index.pkl.
The Retriever then loads this precomputed binary index in milliseconds,
eliminating the need to re-parse and re-index the raw files on every query.
"""

from collections import defaultdict
from pathlib import Path
from typing import Self
from tqdm import tqdm
import numpy as np
import pickle
import torch
import json

from .tokenizer import Tokenizer
from .embedding import Embedder
from .bm25 import CustomBM25
from .model import (
    MinimalSource,
    MinimalSearchResults,
    StudentSearchResults,
)


class Retriever:
    def __init__(
        self,
        chunks: list[dict],
        bm25: CustomBM25,
        vectors: torch.Tensor | None = None
    ) -> None:
        self.chunks = chunks
        self.bm25 = bm25
        self.vectors = vectors
        self.tokenizer = Tokenizer()
        self.embedder: Embedder | None = None

    @classmethod
    def load(
        cls,
        processed_dir: str = "data/processed",
        mode: str = "bm25"
    ) -> Self | None:
        """Load the chunks and bm25 data form index.pkl file"""
        valid_modes = {"bm25", "semantic", "hybrid"}
        if mode not in valid_modes:
            raise ValueError(
                f"Unknown mode: {mode}. Choose from: {', '.join(valid_modes)}"
            )

        index_path = Path(processed_dir) / "index.pkl"
        if not index_path.exists():
            print(
                f"Error: Index file not found at '{index_path}'. "
                "Run 'index' first."
            )
            return None

        with open(index_path, "rb") as f:
            data = pickle.load(f)

        vector_file = Path(processed_dir) / "vectors.npy"
        vectors: torch.Tensor | None = None

        if mode in ("semantic", "hybrid"):
            if not vector_file.exists():
                raise FileNotFoundError(
                    f"Semantic vectors not found: {vector_file}. "
                     "Run index with semantic=True first."
                )

            np_vectors = np.load(vector_file)
            vectors = torch.from_numpy(np_vectors).float()

        return cls(
            chunks=data["all_chunks"],
            bm25=data["bm25"],
            vectors=vectors,
        )

    def _get_bm25_indices(self, query: str, k: int = 10) -> list[int]:
        """Indexes chunks with bm25
        return:
            sorted index
        """
        tokenized_query = self.tokenizer.tokenize(query)

        # Retrieve use bm25.get_score who need the QUERY
        # scores is a float list where every chunk have a score
        # that rapresent the relationship with the current query
        scores = self.bm25.get_score(tokenized_query)
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda item: item[1], reverse=True)

        # need only the INDICES -> "chunk idx", "float score for chunk"
        return [idx for idx, _ in indexed_scores[:k]]

    def _get_semantic_indices(self, query: str, k: int = 10) -> list[int]:
        """Indexes chunks with semantic embeddings"""
        if self.vectors is None:
            return []

        if self.embedder is None:
            self.embedder = Embedder()

        query_vec = self.embedder.embedding_encoder(texts=[query])

        # Calculate cosine similarity via Dot_Product
        # (both are already L2-normalized)
        # 1) self.vectors: Document matrix of shape (N, D)
        # 2) query_vec.T: where T(Transpose): Flips the query from (1, D)
        #    to (D, 1) to enable matmul
        # 3) .squeeze(-1): Removes the trailing dimension of size 1,
        #    turning (N, 1) into a flat 1D tensor of shape (N,)
        #    N: values that rapresent similarity of each chunks with the query
        scores = torch.matmul(
            self.vectors,
            query_vec.T
        ).squeeze(-1)

        # torch.topk(...): is an optimized function provide by PyTorch
        # that scans each score calculated by the previous line and 
        # exctracts only the top_k values (the nearest -> higher values)
        top_k = min(k, len(self.chunks))
        return torch.topk(scores, k=top_k).indices.tolist()

    def _rrf_fusion(self, query: str, k: int, rrf_k: int = 60) -> list[int]:
        """
        Combine the ranking BM25 and Semantic with a reciprocal rankfusion
        args:
            rrf_k=60: is the standard damping constant established by Cormack.
            It prevents top-ranked items (rank 1) from overwhelmingly domina-
            ting items ranked closely behind.
            (rrf_k = costante di smorzamento standard)
        Main formula:
                                    1                     1
            RRF_Score(d) = ────────────────── + ───────────────────────
                            60 + rank_BM25(d)    60 + rank_Semantic(d)
        """
        candidate_count = min(max(k * 2, 50), len(self.chunks))

        # get the indices for every RAG strategy algorithms
        bm25_ranks = self._get_bm25_indices(query, k=candidate_count)
        semantic_ranks = self._get_semantic_indices(query, k=candidate_count)

        # Avoid a crash due to a KeyError by assigning a default value of 0.0
        # if chunk_idx is not present
        rrf_scores: dict[int, float] = defaultdict(float)

        # rank is the position of the chunk based on the previous rag system
        # scoring. if the same chank_idx will be already present in rrf_scores
        # the chunk gets an additional score boost based on its rank
        for rank, chunk_idx in enumerate(bm25_ranks, start=1):
            rrf_scores[chunk_idx] += 1.0 / (rrf_k + rank)
        for rank, chunk_idx in enumerate(semantic_ranks, start=1):
            rrf_scores[chunk_idx] += 1.0 / (rrf_k + rank)

        sorted_indices = sorted(
            rrf_scores.keys(),
            key=lambda idx: rrf_scores[idx],
            reverse=True,
        )
        return sorted_indices[:k]

    def _to_minimal_source(self, chunk_idx: int) -> MinimalSource:
        chunk = self.chunks[chunk_idx]
        return MinimalSource(
            file_path=chunk["file_path"],
            first_character_index=chunk["first_character_index"],
            last_character_index=chunk["last_character_index"],
        )

    def retrieve(
        self,
        query: str,
        k: int = 10,
        mode: str = "bm25"
    ) -> list[MinimalSource]:
        """
        based in the mode choice the correct indices retriever.
        check if vectors.npy and index.pkl exists and the string passed
        as mode.
        """
        if not query or not query.strip() or k <= 0:
            return []

        valid_modes = {"bm25", "semantic", "hybrid"}
        if mode not in valid_modes:
            raise ValueError(
                f"Unknown mode: {mode}. Choose from: {', '.join(valid_modes)}"
            )

        if mode in ("bm25"):
            indices = self._get_bm25_indices(query, k)
        elif mode in ("semantic"):
            if self.vectors is None:
                raise ValueError(
                    "Semantic vectors are missing. Run "
                    "'index --semantic=True' before using semantic mode."
                )
            indices = self._get_semantic_indices(query, k)
        elif mode in ("hybrid"):
            if self.vectors is None:
                raise ValueError(
                    "Semantic vectors are missing. Run "
                    "'index --semantic=True' before using hybrid mode."
                )
            indices = self._rrf_fusion(query, k)

        return [self._to_minimal_source(idx) for idx in indices]
    
    # Method ALIAS
    search = retrieve

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results/UnansweredQuestions",
        mode: str = "bm25"
    ) -> None:
        if k <= 0:
            print("Error: k must be greater than 0.")
            return

        ds_path = Path(dataset_path)
        if not ds_path.exists():
            print(f"Error: Dataset not found at '{dataset_path}'")
            return

        try:
            with open(ds_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Malformed JSON in dataset '{dataset_path}'")
            return
        except Exception as e:
            print(f"Error reading dataset '{dataset_path}': {e}")
            return

        if not isinstance(data, dict) or "rag_questions" not in data:
            print(f"Error: Invalid dataset format in '{dataset_path}'")
            return

        # Python automatically cast it to a list of dict
        # {"question_id": "value_id",
        #  "question": "query_str"}
        questions: list[dict] = data["rag_questions"]
        if not isinstance(questions, list):
            print(f"Error: 'rag_questions' must be a list")
            return
        if not questions:
            print(f"Warning: '{dataset_path}' contains zero questions.")
            return

        all_results: list[MinimalSearchResults] = []
        for item in tqdm(questions):
            if (
                not isinstance(item, dict)
                or "question_id" not in item
                or "question" not in item
            ):
                print(f"Warning: skipping malformed question record: {item}.")
                continue

            q_id = item["question_id"]
            q_text = item["question"]

            if not isinstance(q_id, str) or not q_id.strip():
                print(
                    "Warning: skipping record "
                    f"with invalid question_id: {item}."
                )
                continue

            if not isinstance(q_text, str):
                print(
                    "Warning: skipping record "
                    f"with invalid question: {item}."
                )
                continue
            
            if not q_text.strip():
                sources = []
            else:
                sources = self.retrieve(q_text, k=k, mode=mode)

            item_result = MinimalSearchResults(
                question_id=q_id,
                question=q_text,
                retrieved_sources=sources
            )

            all_results.append(item_result)

        # WRAP in the StudentSearchResults model
        output_data = StudentSearchResults(
            search_results=all_results,
            k=k,
        )

        # ====================================================================
        #                   Saving Output in save_directory
        # ====================================================================
        try:
            save_dir = Path(save_directory)
            save_dir.mkdir(parents=True, exist_ok=True)
            output_file = save_dir / ds_path.name

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output_data.model_dump_json(indent=2))
            print(f"Saved student_search_results to {output_file}")
        except Exception as e:
            print(f"Error saving output to '{save_directory}': {e}")