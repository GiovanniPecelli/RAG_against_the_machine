import fire
from src.indexer import Indexer


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
        # TODO
        # Chiamare Indexer(raw_dir, processed_dir, max_chunk_size).build()

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
        # TODO
        # chiama Retriever.search()

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 5,
        save_directory: str = (
            "data/output/search_results/UnansweredQuestions"
        ),
        processed_dir: str = "data/processed",
    ) -> None:
        """Call the retriever"""
        pass
        #retriever = Retriever.load(processed_dir)
        #retriever.search_dataset(
        #    dataset_path=dataset_path,
        #    k=k,
        #    save_directory=save_directory
        #)

if __name__ == "__main__":
    fire.Fire(RAGCLI)