from pathlib import Path
import json
import pickle

from tqdm import tqdm

from .tokenizer import Tokenizer
from .bm25 import CustomBM25
from .model import (
    MinimalSource,
    MinimalSearchResults,
    StudentSearchResults,
)


class Retriever:
    def __init__(self, chunks: list[dict], bm25: CustomBM25):
        self.chunks = chunks
        self.bm25 = bm25
        self.tokenizer = Tokenizer()

    @classmethod
    def load(cls, processed_dir: str = "data/processed"):
        """Load the chunks and bm25 data form index.pkl file"""
        index_path = Path(processed_dir) / "index.pkl"
        if not index_path.exists():
            raise FileNotFoundError(f"Index.pkl not found at '{index_path}'")

        with open(index_path, "rb") as f:
            data = pickle.load(f)
        return cls(
            chunks=data["chunk"],
            bm25=data["bm25"]
        )

    def retrieve(self, query: str, k: int = 5) -> list[MinimalSource]:
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
        ds_path = Path(dataset_path)
        if not ds_path.exists():
            raise FileNotFoundError(f"dataset not found at '{dataset_path}'")

        with open(ds_path, "r") as f:
            data = json.load(f)

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