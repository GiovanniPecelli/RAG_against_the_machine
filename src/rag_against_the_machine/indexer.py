from .tokenizer import Tokenizer
from .embedding import Embedder
from .chunker import Chunker
from .bm25 import CustomBM25

from pathlib import Path
from typing import Any
from tqdm import tqdm

import numpy as np
import hashlib
import pickle
import torch
import json


class Indexer:
    """Build and persist lexical and optional semantic corpus indices."""

    def __init__(
        self,
        raw_dir: str = "data/raw/vllm-0.10.1",
        processed_dir: str = "data/processed",
        max_chunk_size: int = 2000,
    ) -> None:
        """Configure source, output, and chunking paths and settings.

        Args:
            raw_dir: Directory containing source files to index.
            processed_dir: Directory where index artifacts are stored.
            max_chunk_size: Maximum character length of each chunk.

        Returns:
            None.
        """
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
        self.all_chunks: list[dict[str, Any]] = []
        self.file_hashes: dict[str, str] = {}
        self.vectors: torch.Tensor | None = None

        self.chunker = Chunker(max_chunk_size=self.max_chunk_size)
        self.tokenizer = Tokenizer()
        self.bm25: CustomBM25 | None = None

    def get_target_files(self) -> list[Path]:
        """Find supported source files below the raw data directory.

        Returns:
            Sorted paths for Python, Markdown, and text files.
        """
        if not self.raw_dir.exists():
            print(f"Error: Directory {self.raw_dir} does not exist.")
            return []

        # rglob("*") searches recursively through all subfolders
        files = [
            file for file in self.raw_dir.rglob("*")
            if file.is_file() and file.suffix in [".py", ".md", ".txt"]
        ]
        return sorted(files)

    def get_file_hashes(self) -> dict[str, str]:
        """
        Return a SHA-256 hash for every target file.
        The returned dictionary maps each file path to its content hash.

        Returns:
            A mapping from POSIX-style source paths to SHA-256 hex digests.
        """
        file_hashes: dict[str, str] = {}

        for file_path in self.get_target_files():
            file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            file_hashes[file_path.as_posix()] = file_hash

        return file_hashes

    def group_chunks_by_file(
        self,
        chunks: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Group indexed chunks by their source file path.

        Args:
            chunks: Chunk records to group.

        Returns:
            A mapping from each file path to its ordered chunk records.
        """
        chunks_by_file: dict[str, list[dict[str, Any]]] = {}

        for chunk in chunks:
            file_path = chunk["file_path"]
            chunks_by_file.setdefault(file_path, []).append(chunk)

        return chunks_by_file

    @staticmethod
    def get_chunks_signature(chunks: list[dict[str, Any]]) -> str:
        """Hash an ordered list of chunks for artifact consistency checks.

        Args:
            chunks: Ordered chunk records to serialize and hash.

        Returns:
            SHA-256 hex digest of the serialized chunk list.
        """
        serialized_chunks = json.dumps(
            chunks,
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(
            serialized_chunks.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def get_vectors_hash(vectors: np.ndarray) -> str:
        """Hash the raw contents of a NumPy embedding matrix.

        Args:
            vectors: Embedding matrix whose bytes should be hashed.

        Returns:
            SHA-256 hex digest of the matrix bytes.
        """
        return hashlib.sha256(vectors.tobytes()).hexdigest()

    def collect_chunks(
        self,
        unchanged_files: set[str],
        old_chunks_by_file: dict[str, list[dict[str, Any]]],
    ) -> None:
        """
        Finds all files and chunks them one by one.
            files: contain a list of path -> file
            this function send them to the chunker

        Args:
            unchanged_files: Paths whose saved chunks can be reused.
            old_chunks_by_file: Previously indexed chunks grouped by path.

        Returns:
            None. The collected records are stored in ``self.all_chunks``.
        """
        files = self.get_target_files()

        print("\nStart of the file chunk ...")
        for file_path in tqdm(files):
            path = file_path.as_posix()

            # file_chunks is a list of dict
            if path in unchanged_files:
                file_chunks = old_chunks_by_file.get(path, [])
            else:
                # if the file is changed -> send to chunk_file
                file_chunks = self.chunker.chunk_file(file_path)

            self.all_chunks.extend(file_chunks)

    def tokenize_chunks(self) -> list[list[str]]:
        """Tokenize each collected chunk and its source path.

        Returns:
            Token lists aligned by position with ``self.all_chunks``.
        """
        tokenized_corpus = []

        print("\nTokenizing the chunks ...")
        for chunk in tqdm(self.all_chunks):
            text_tokens = self.tokenizer.tokenize(chunk["text"])
            path_tokens = self.tokenizer.tokenize(chunk["file_path"])
            tokenized_corpus.append(path_tokens + text_tokens)
        return tokenized_corpus

    def text_extraction(self) -> list[str]:
        """
        text extraction from every chunks.
        This text is necessary to the semantic embedding algorithm

        Returns:
            Chunk text strings in the same order as ``self.all_chunks``.
        """
        return [chunk["text"] for chunk in self.all_chunks]

    def load_previous_index(self) -> dict[str, Any] | None:
        """Load the previously saved index data, if available.

        Returns:
            The saved index dictionary, or ``None`` if it cannot be loaded.
        """
        index_path = self.processed_dir / "index.pkl"
        if not index_path.is_file():
            return None

        try:
            with open(index_path, "rb") as f:
                previous_data: Any = pickle.load(f)
        except (
            OSError,
            pickle.PickleError,
            EOFError,
            AttributeError,
            ImportError,
            IndexError,
        ) as e:
            print(
                f"Warning: could not load previous index '{index_path}': {e}"
            )
            return None

        if not isinstance(previous_data, dict):
            print(
                f"Warning: previous index '{index_path}' "
                "has an invalid format."
            )
            return None

        return previous_data

    def save(self, semantic: bool = False) -> None:
        """
        Saves:
        -   chunks
        -   BM25 index
        -   if semantic == True -> semantic embedding file.pt
        to disk in data/processed/.

        Args:
            semantic: Whether semantic indexing was requested.

        Returns:
            None.
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # "/" in pathlib join the payh with the "file_name"
        index_file = self.processed_dir / "index.pkl"

        data = {
            "all_chunks": self.all_chunks,
            "bm25": self.bm25,
            "file_hashes": self.file_hashes,
            "vector_chunks_signature": (
                self.get_chunks_signature(self.all_chunks)
                if self.vectors is not None
                else None
            ),
            "vectors_hash": (
                self.get_vectors_hash(self.vectors.cpu().numpy())
                if self.vectors is not None
                else None
            ),
        }
        with open(index_file, "wb") as f:
            pickle.dump(data, f)
        print(f"\nIndex successfully saved to: {index_file}")

        # vectors.npy stores the embedding matrix: one row per chunk
        # and 384 columns per row
        if self.vectors is not None:
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
                    size = (
                        chunk["last_character_index"]
                        - chunk["first_character_index"]
                    )
                    f.write(f"{path}\t{file_type}\t{start}\t{size}\n")

            print(f"TensorFlow Projector TSV files saved to: {projector_dir}")

    def build(self, semantic: bool = False) -> None:
        """Build and save the corpus index and optional semantic vectors.

        Args:
            semantic: Whether to create semantic vectors for the corpus.

        Returns:
            None.
        """
        # previous_data contain "bm25", "all_chunks" and file_hashes
        previous_data = self.load_previous_index()
        previous_hashes: dict[str, str] = {}

        if previous_data is not None:
            stored_hashes = previous_data.get("file_hashes")
            if isinstance(stored_hashes, dict):
                previous_hashes = stored_hashes

        # hash calculation
        self.file_hashes = self.get_file_hashes()

        current_paths = set(self.file_hashes)
        previous_paths = set(previous_hashes)

        unchanged_files = {
            path for path in current_paths.intersection(previous_paths)
            if self.file_hashes[path] == previous_hashes[path]
        }

        # saving the previous index chunks
        previous_chunks: list[dict[str, Any]] = []

        if previous_data is not None:
            stored_chunks = previous_data.get("all_chunks")
            if isinstance(stored_chunks, list):
                previous_chunks = stored_chunks

        old_chunks_by_file = self.group_chunks_by_file(previous_chunks)

        stored_vector_signature = None
        if previous_data is not None:
            stored_vector_signature = previous_data.get(
                "vector_chunks_signature"
            )

        previous_chunks_signature = self.get_chunks_signature(previous_chunks)
        vectors_match_previous_chunks = (
            stored_vector_signature == previous_chunks_signature
        )

        # The complete vector matrix from the previous index, aligned by row
        # with previous_chunks.
        previous_vectors: torch.Tensor | None = None
        vector_file = self.processed_dir / "vectors.npy"

        if vector_file.is_file() and vectors_match_previous_chunks:
            # loaded_vectors_array is a numpy array - 2D matrix (chunk, dim)
            loaded_vectors_array = np.load(vector_file)
            loaded_vectors_hash = self.get_vectors_hash(loaded_vectors_array)
            # .get() return None if the key miss
            stored_vectors_hash = (
                previous_data.get("vectors_hash")
                if previous_data is not None
                else None
            )
            if loaded_vectors_hash == stored_vectors_hash:
                previous_vectors = torch.from_numpy(
                    loaded_vectors_array
                ).float()

        # This dictionary groups the previous index's chunk vectors by file.
        # Unlike previous_vectors, it lets the build reuse vectors one file
        # at a time.
        previous_vectors_by_file: dict[str, list[torch.Tensor]] = {}

        if previous_vectors is not None:
            if len(previous_vectors) != len(previous_chunks):
                print(
                    "Warning: previous vectors do not match previous chunks."
                )
                previous_vectors = None
            else:
                # zip() pairs items from two sequences by position. Here:
                # - previous_chunks[0] pairs with previous_vectors[0]
                # - previous_chunks[1] pairs with previous_vectors[1]
                for chunk, vector in zip(previous_chunks, previous_vectors):
                    path = chunk["file_path"]
                    # Store each previous chunk vector under its source file.
                    previous_vectors_by_file.setdefault(path, []).append(
                        vector
                    )

        # Collecting and chunking files
        self.collect_chunks(unchanged_files, old_chunks_by_file)
        current_chunks_by_file: dict[str, list[dict[str, Any]]] = (
            self.group_chunks_by_file(self.all_chunks)
        )

        # Tokenizing chunks
        tokenized_corpus: list[list[str]] = self.tokenize_chunks()

        # Building the bm25 algorithm
        self.bm25 = CustomBM25(tokenized_corpus)

        if semantic or vector_file.is_file():
            # vector_batches is a list of torch.Tensor where a every element
            # of the list rapresent a file -> and the tensor, a matrix which
            # contains a 2D matrix [N, D]
            #    N is the chunks number: one row for every chunk in the file
            #    D: fixed dimension
            vector_batches: list[torch.Tensor] = []
            embedder: Embedder | None = None
            # Count chunks needing new embeddings; reused vectors are excluded
            chunks_to_embed: int = 0

            for path, file_chunks in current_chunks_by_file.items():
                # These are vectors for this path from the previous index.
                previous_file_vectors = previous_vectors_by_file.get(path)
                can_reuse = (
                    path in unchanged_files
                    and previous_file_vectors is not None
                    and len(previous_file_vectors) == len(file_chunks)
                )
                if not can_reuse:
                    chunks_to_embed += len(file_chunks)

            if chunks_to_embed > 0:
                print(f"Generating {chunks_to_embed} embeddings chunks...")
            with tqdm(
                total=chunks_to_embed,
                unit="chunk",
                disable=chunks_to_embed == 0
            ) as progress_bar:
                for path, file_chunks in current_chunks_by_file.items():
                    # Look up this file's vectors in the previous index,
                    # if present.
                    previous_file_vectors = previous_vectors_by_file.get(path)

                    if (
                        path in unchanged_files
                        and previous_file_vectors is not None
                        and len(previous_file_vectors) == len(file_chunks)
                    ):
                        # file_vectors are the vectors used in the new index
                        # here they are reused from the previous index.
                        file_vectors = torch.stack(previous_file_vectors)
                    else:
                        if embedder is None:
                            embedder = Embedder()

                        # if the file is changed -> call embedding_encoder
                        file_texts = [chunk["text"] for chunk in file_chunks]
                        file_vectors = embedder.embedding_encoder(
                            texts=file_texts,
                            progress_bar=progress_bar
                        )

                    vector_batches.append(file_vectors)

            if vector_batches:
                # Combine the per-file vector matrices row by row, keeping
                # their order aligned with the chunks in self.all_chunks.
                self.vectors = torch.cat(vector_batches, dim=0)

        # Saving index to disk
        self.save(semantic)
