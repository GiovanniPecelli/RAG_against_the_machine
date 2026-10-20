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

from pathlib import Path
from typing import Self
from tqdm import tqdm
import json
import pickle

from .tokenizer import Tokenizer
from .bm25 import CustomBM25
from .model import (
    MinimalSource,
    MinimalSearchResults,
    StudentSearchResults,
)


class Retriever:
    def __init__(self, chunks: list[dict], bm25: CustomBM25) -> None:
        self.chunks = chunks
        self.bm25 = bm25
        self.tokenizer = Tokenizer()

    @classmethod
    def load(cls, processed_dir: str = "data/processed") -> Self | None:
        """Load the chunks and bm25 data form index.pkl file"""
        index_path = Path(processed_dir) / "index.pkl"
        if not index_path.exists():
            print(
                f"Error: Index file not found at '{index_path}'. "
                "Run 'index' first."
            )
            return None

        with open(index_path, "rb") as f:
            data = pickle.load(f)
        return cls(
            chunks=data["chunk"],
            bm25=data["bm25"]
        )

    def retrieve(self, query: str, k: int = 5) -> list[MinimalSource]:
        if not query or not query.strip() or k <= 0:
            return []

        tokenized_query = self.tokenizer.tokenize(query)

        # Retrieve use bm25.get_score who need the QUERY
        scores = self.bm25.get_score(tokenized_query)
        indexed_scores = list(enumerate(scores))
        sorted_scores = sorted(
            indexed_scores,
            key=lambda item: item[1],
            reverse=True
        )
        top_k_scores = sorted_scores[:k]

        results: list[MinimalSource] = []
        for index, score in top_k_scores:
            chunk = self.chunks[index]
            source = MinimalSource(
                file_path=chunk["file_path"],
                first_character_index=chunk["first_character_index"],
                last_character_index=chunk["last_character_index"],
            )
            results.append(source)

        return results
    
    # Method ALIAS
    search = retrieve

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 5,
        save_directory: str = "data/output/search_results/UnansweredQuestions"
    ) -> None:
        if k <= 0:
            print("Error: k must be greater than 0.")
            return

        ds_path = Path(dataset_path)
        if not ds_path.exists():
            print(f"Error: Dataset not found at '{dataset_path}'")
            return

        try:
            with open(ds_path, "r") as f:
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

        all_results: list[MinimalSearchResults] = []
        for item in tqdm(questions):
            sources = self.retrieve(item["question"], k=k)

            item_result = MinimalSearchResults(
                question_id=item["question_id"],
                question=item["question"],
                retrieved_sources=sources
            )

            all_results.append(item_result)

        # WRAP in the StudentSearchResults model
        output_data = StudentSearchResults(
            search_results=all_results,
            k=k,
        )

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        # ds_path.name take the last name in the Path
        output_file = save_dir / ds_path.name

        with open(output_file, "w") as f:
            f.write(output_data.model_dump_json(indent=2))

        print(f"Saved student_search_results to {output_file}")