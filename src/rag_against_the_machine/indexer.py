from .tokenizer import Tokenizer
from .embedding import Embedder
from .chunker import Chunker
from .bm25 import CustomBM25
from pathlib import Path
from tqdm import tqdm
import pickle
import torch


class Indexer:
    def __init__(self,
        raw_dir: str = "data/raw/vllm-0.10.1",
        processed_dir: str = "data/processed",
        max_chunk_size: int = 2000
    ) -> None:
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
        self.all_chunks: list[dict] = []
        self.vectors: torch.Tensor | None = None
        
        self.chunker = Chunker(max_chunk_size=self.max_chunk_size)
        self.tokenizer = Tokenizer()
        self.bm25: CustomBM25 | None = None
        self.embedder = Embedder()

    def get_target_files(self) -> list[Path]:
        """Finds all .py and .md files in the raw folder."""
        if not self.raw_dir.exists():
            print(f"Error: Directory {self.raw_dir} does not exist.")
            return []

        # rglob("*") searches recursively through all subfolders
        files = [
            file for file in self.raw_dir.rglob("*")
            if file.is_file() and file.suffix in [".py", ".md", ".txt"]
        ]
        return sorted(files)

    def collect_chunks(self) -> None:
        """
        Finds all files and chunks them one by one.
            files: contain a list of path -> file
            this function send them to the chunker
        """
        files = self.get_target_files()

        print("\nStart of the file chunk ...")
        for file_path in tqdm(files):
            file_chunks: list[dict] = self.chunker.chunk_file(file_path)
            self.all_chunks.extend(file_chunks)

    def tokenize_chunks(self) -> list[list[str]]:
        """Tokenizes the text of every collected chunk."""
        tokenized_corpus = []

        print("\nTokenizing the chunks ...")
        for chunk in tqdm(self.all_chunks):
            tokens = self.tokenizer.tokenize(chunk["text"])
            tokenized_corpus.append(tokens)

        return tokenized_corpus

    def text_extraction(self) -> list[str]:
        """
        text extraction from every processed chunks.
        This text is necessary to the semantic embedding algorithm
        """
        extracted_chunk_text: list[str] = []

        current_path = None
        current_content: str = ""
        
        if self.all_chunks is None:
            print("[ERROR] text extraction not possible -> check indexer.py")
            return []

        for chunk in tqdm(self.all_chunks):
            current_path = self.all_chunks["file_path"]

            if chunk["file_path"] != current_path:
                with open (chunk["file_path"], "r", encoding="utf-8") as f:
                    current_content = f.read()
            current_path = chunk["file_path"]
        
            text_slice = current_content[
                chunk["first_character_index"] : chunk["last_character_index"]
            ]
            extracted_chunk_text.append(text_slice)

            return extracted_chunk_text
    
    # TODO ADD: save the vectors var in a .pt file
    def save(self, semantic: bool = False) -> None:
        """
        Saves: 
        -   chunks 
        -   BM25 index
        -   if semantic == True -> semantic embedding file.pt
        to disk in data/processed/."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # "/" in pathlib join the payh with the "file_name"
        index_file = self.processed_dir / "index.pkl"

        data = {
            "chunk": self.all_chunks,
            "bm25": self.bm25,
        }
        with open(index_file, "wb") as f:
            pickle.dump(data, f)
        print(f"\nIndex successfully saved to: {index_file}")

    def build(self, semantic: bool = False) -> None:
        # Collecting and chunking files
        self.collect_chunks()

        # Tokenizing chunks
        tokenized_corpus = self.tokenize_chunks()

        # Building the bm25 algorithm
        self.bm25 = CustomBM25(tokenized_corpus)

        if semantic:
            extracted_chunk_text = self.text_extraction()
            vectors: torch.Tensor = self.embedder.embedding_encoder(
                texts=extracted_chunk_text
            )

        # Saving index to disk
        self.save(semantic)