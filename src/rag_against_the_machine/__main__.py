from .model import MinimalAnswer, StudentSearchResultsAndAnswer
from .retriever import Retriever
from .generator import Generator
from .indexer import Indexer

from pathlib import Path
import fire

from rag_against_the_machine import retriever


class RAGCLI:
    def index(
        self,
        max_chunk_size: int = 2000,
        processed_dir: str = "data/processed",
        raw_dir: str = "data/raw/vllm-0.10.1",
    ) -> None:
        """ingest the file and build the BM25 index."""
        if max_chunk_size <= 0:
            print("Error: max_chunk_index must be greater than 0")
            return

        raw_path = Path(raw_dir)
        if not raw_path.exists():
            print(f"Error: '{raw_dir}' does not exist.")
            return

        print(
            f"Indexing started in '{raw_dir}' "
            f"(chunk max: {max_chunk_size})..."
        )
        indexer = Indexer(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            max_chunk_size=max_chunk_size,
        )
        indexer.build()

    def search(
        self,
        query: str,
        k: int = 5,
        processed_dir: str = "data/processed",
    ) -> None:
        """Search for i top-k snipped for a single question."""
        if not query or not query.strip():
            print("Error: The query can't be empty")
            return

        if k <= 0:
            print("Error: k must be greater than 0.")
            return

        print(f"Serach for: '{query}' (top {k})...")

        # Load retriever and get the result ["search" is an alias]
        retriever = Retriever.load(processed_dir)
        if not retriever:
            return

        results = retriever.search(query, k=k)

        # Results is a list[MinimalSource]
        for i, source in enumerate(results, start=1):
            print(
                f"[{i}] {source.file_path} "
                f"(chars {source.first_character_index} "
                f"to {source.last_character_index})"
            )

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 5,
        save_directory: str = (
            "data/output/search_results/UnansweredQuestions"
        ),
        processed_dir: str = "data/processed",
    ) -> None:
        """Call the retriever to search for the entire dataset"""

        retriever = Retriever.load(processed_dir)
        if not retriever:
            return

        retriever.search_dataset(
            dataset_path=dataset_path,
            k=k,
            save_directory=save_directory
        )

    def answer(
            self,
            query: str,
            k: int = 5,
            processed_dir: str = "data/processed",
            model_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        """Answer a single query using the retrieved context."""
        if not query or not query.strip():
            print("Error: The query can't be empty.")
            return

        if k <= 0:
            print("Error: k must be greater than 0.")
            return

        retriever = Retriever.load(processed_dir)
        if not retriever:
            return

        sources = retriever.retrieve(query, k)

        generator = Generator(model_name=model_name)
        answer_text = generator.generate_answer(query, sources)
        

if __name__ == "__main__":
    fire.Fire(RAGCLI)