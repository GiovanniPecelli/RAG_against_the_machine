from pathlib import Path
from tqdm import tqdm
from .chunker import Chunker
import sys


class Indexer:
    def __init__(self,
        raw_dir: str = "data/raw/vllm-0.10.1",
        processed_dir: str = "data/processed",
        max_chunk_size: int = 2000
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.max_chunk_size = max_chunk_size
        
        # This list will hold all chunks from all files
        self.all_chunks = []

    def get_target_files(self) -> list[Path]:
        """Finds all .py and .md files in the raw folder."""
        if not self.raw_dir.exists():
            print(f"Error: Directory {self.raw_dir} does not exist.")
            return []

        # rglob("*") searches recursively through all subfolders
        return [
            file for file in self.raw_dir.rglob("*")
            if file.is_file() and file.suffix in [".py", ".md"]
        ]

    def collect_chunks(self):
        """
        Finds all files and chunks them one by one.
            files: contain a list of path -> file
            this function send them to the chunker
        """
        files = self.get_target_files()
        chunker = Chunker(max_chunk_size=self.max_chunk_size)

        for file_path in tqdm(files):
            file_chunks = chunker.chunk_file(
                file_path, max_chunk_size=self.max_chunk_size
            )
            self.all_chunks.extend(file_chunks)