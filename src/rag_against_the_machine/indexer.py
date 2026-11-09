from .tokenizer import Tokenizer
from .embedding import Embedder
from .chunker import Chunker
from .bm25 import CustomBM25

from pathlib import Path
from tqdm import tqdm

import numpy as np
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
        text extraction from every chunks.
        This text is necessary to the semantic embedding algorithm
        """
        return [chunk["text"] for chunk in self.all_chunks]

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
            "all_chunks": self.all_chunks,
            "bm25": self.bm25,
        }
        with open(index_file, "wb") as f:
            pickle.dump(data, f)
        print(f"\nIndex successfully saved to: {index_file}")

        if semantic and self.vectors is not None:
            vector_file = self.processed_dir / "vectors.npy"
            
            # change the tensor on the cpu before conversion
            np.save(vector_file, self.vectors.cpu().numpy())
            print(f"Semantic vectors successfully saved to: {vector_file}")

            # ================================================================
            #          Saving vectors as TSV (tab_separated_value)
            # ================================================================
            projector_dir = self.processed_dir / "tf_projector"
            projector_dir.mkdir(parents=True, exist_ok=True)

            vectors_tsv = projector_dir / "vectors.tsv"
            meta_tsv = projector_dir / "metadata.tsv"

            with open(vectors_tsv, "w", encoding="utf-8") as f:
                vectors_list = self.vectors.cpu().tolist()
                for vector in vectors_list:
                    f.write("\t".join(str(coord) for coord in vector) + "\n")

            with open(meta_tsv, "w", encoding="utf-8") as f:
                f.write("FilePath\tFileType\tStartIndex\tChunkSize\n")
                for chunk in self.all_chunks:
                    path = chunk["file_path"]
                    start = chunk["first_character_index"]
                    if Path(path).suffix:
                        file_type = Path(path).suffix
                    else: 
                        file_type = "unknown"

                    # "\" - ignore the line separation
                    size = chunk["last_character_index"] \
                            - chunk["first_character_index"]
                    f.write(f"{path}\t{file_type}\t{start}\t{size}\n")

            print(f"TensorFlow Projector TSV files saved to: {projector_dir}")

    def build(self, semantic: bool = False) -> None:
        # Collecting and chunking files
        self.collect_chunks()

        # Tokenizing chunks
        tokenized_corpus = self.tokenize_chunks()

        # Building the bm25 algorithm
        self.bm25 = CustomBM25(tokenized_corpus)

        if semantic:
            extracted_chunk_text = self.text_extraction()
            self.vectors = self.embedder.embedding_encoder(
                texts=extracted_chunk_text
            )

        # Saving index to disk
        self.save(semantic)