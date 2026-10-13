from .tokenizer import Tokenizer
from .chunker import Chunker
from pathlib import Path
from tqdm import tqdm
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
        # CHUNK DATA STRUCTURE
        # chunk({
        #     "file_path": normalized_path,
        #     "text": current_text,
        #     "first_character_index": chunk_start_index,
        #     "last_character_index": chunk_start_index + len(current_text),
        # })
        self.all_chunks = []
        
        self.chunker = Chunker(max_chunk_size=self.max_chunk_size)
        self.tokenizer = Tokenizer()

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

        for file_path in tqdm(files):
            file_chunks = self.chunker.chunk_file(file_path)
            self.all_chunks.extend(file_chunks)

    def tokenize_chunks(self) -> list[list[str]]:
        """Tokenizes the text of every collected chunk."""
        tokenized_corpus = []

        for chunk in tqdm(self.all_chunks):
            tokens = self.tokenizer.tokenize(chunk["text"])
            tokenized_corpus.append(tokens)

        return tokenized_corpus

    def build(self) -> None:
        # Collecting and chunking files
        self.collect_chunks()

        # Tokenizing chunks
        tokenized_corpus = self.tokenize_chunks()

        # Building the bm25 algorithm
        # TODO

        # Saving index to disk
        # TODO