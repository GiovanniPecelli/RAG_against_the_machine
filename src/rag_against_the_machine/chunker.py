from pathlib import Path


class Chunker:
    def __init__(self, max_chunk_size: int = 2000, overlap_size: int = 0):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size

    def chunk_file(self, file_path: Path) -> list[dict]:
        """
        Reads the file and delegates to the right strategy.
        Cross-platform path normalization:
            On Windows, Path objects produce backslashes ('\'),
            whereas POSIX/Linux systems use forward slashes ('/').
            with reference paths (e.g., 'data/raw/vllm-0.10.1/docs/...').
            Using .as_posix() guarantees standard forward slashes.
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []

        normalized_path: str = file_path.as_posix()

        # a chunks list for a single file
        file_chunks: list[dict] = []

        if file_path.suffix == ".md":
            file_chunks = self.chunk_Markdown(content, normalized_path)
        elif file_path.suffix == ".txt":
            file_chunks = self.chunk_text(content, normalized_path)
        elif file_path.suffix == ".py":
            file_chunks = self.chunk_python(content, normalized_path)

        return file_chunks

    def chunk_Markdown(
        self,
        content: str,
        normalized_path: str
    ) -> list[dict]:
        """Split Markdown files by headers (#) and paragraphs."""
        return self._slice_files(
            content,
            normalized_path,
            keyworks=("# ","## ",)
        )

    def chunk_text(
        self,
        content: str,
        normalized_path: str
    ) -> list[dict]:
        """Split Generic Text and config files by (\n) or max chunk size"""
        return self._slice_files(
            content,
            normalized_path,
            keyworks=("\n",)
        )

    def chunk_python(
        self,
        content: str,
        normalized_path: str
    ) -> list[dict]:
        """Split Python files by headers (class, def and async def)"""
        return self._slice_files(
            content,
            normalized_path,
            keyworks=("class ", "def ", "async def ")
        )

    def _slice_files(
        self,
        content: str,
        normalized_path: str,
        keyworks: tuple
    ) -> list[dict]:
        """Split Python by classes and functions."""
        chunks: list[dict] = []
        current_lines: list[tuple[int, str]] = []
        current_len = 0
        current_index: int = 0

        lines = content.splitlines(keepends=True)
        for line in lines:
            line_len = len(line)
            start_line_index = current_index

            # ================================================================
            #            SPECIAL CASE: IF LINE > MAX_CHUNK_SIZE
            # ================================================================
            if line_len > self.max_chunk_size:
                if current_lines:
                    chunk_text = "".join(l for _, l in current_lines)
                    chunks.append({
                        "file_path": normalized_path,
                        "text": chunk_text,
                        "first_character_index": current_lines[0][0],
                        "last_character_index": (
                            current_lines[0][0] + len(chunk_text)
                        ),
                    })
                    current_lines = []
                    current_len = 0

                for sub_start in range(0, line_len, self.max_chunk_size):
                    sub_text = line[
                        sub_start : sub_start + self.max_chunk_size
                    ]
                    sub_idx = start_line_index + sub_start
                    chunks.append({
                        "file_path": normalized_path,
                        "text": sub_text,
                        "first_character_index": sub_idx,
                        "last_character_index": sub_idx + len(sub_text),
                    })
                current_index += line_len
                continue

            # ================================================================
            #        NORMAL CASE: a single line length < MAX_CHUNK_SIZE
            # ================================================================
            is_new_header = line.lstrip().startswith(keyworks)
            would_overflow = (
                (current_len + line_len) > self.max_chunk_size
            )
            should_split = (
                (
                    is_new_header and current_len >= 1000
                ) or would_overflow)

            # NEW CHUNK -> if there is an header or would overflow
            if should_split and current_lines:
                chunk_text = "".join(l for _, l in current_lines)
                chunks.append({
                    "file_path": normalized_path,
                    "text": chunk_text,
                    "first_character_index": current_lines[0][0],
                    "last_character_index": (
                        current_lines[0][0] + len(chunk_text)
                    ),
                })

                # SLIDING WINDOW: save the last lines to create chunks overlap
                overlap_lines = []
                acc_overlap = 0
                for start_idx, prev_line in reversed(current_lines):
                    if acc_overlap + len(prev_line) > self.overlap_size:
                        break

                    # Restoring sequential sorting
                    overlap_lines.insert(0, (start_idx, prev_line))
                    acc_overlap += len(prev_line)

                current_lines = overlap_lines
                current_len = acc_overlap

            current_lines.append((start_line_index, line))
            current_len += line_len
            current_index += line_len

        if current_lines:
            chunk_text = "".join(l for _, l in current_lines)
            chunks.append({
                "file_path": normalized_path,
                "text": chunk_text,
                "first_character_index": current_lines[0][0],
                "last_character_index": current_lines[0][0] + len(chunk_text),
            })

        return chunks


if __name__ == "__main__":
    chunker = Chunker(max_chunk_size=200)

    md_file = Path("tests/data/sample.md")
    if md_file.exists():
        print("--- Testing Markdown ---")
        full_text = md_file.read_text(encoding="utf-8")
        chunks = chunker.chunk_file(md_file)
        for chunk in chunks:
            start = chunk['first_character_index']
            end = chunk['last_character_index']

            is_valid_chunk = full_text[start:end] == chunk['text']
            assert is_valid_chunk, "Error in the chunks index"
            print(
                ("-" * 50), "\n"
                f"{chunk['first_character_index']}:"
                f"{chunk['last_character_index']}\n"
                f"{chunk['text']}\n"
            )

    py_file = Path("tests/data/sample.py")
    if py_file.exists():
        print("--- Testing Python ---")
        full_text = py_file.read_text(encoding="utf-8")
        chunks = chunker.chunk_file(py_file)
        for chunk in chunks:
            start = chunk['first_character_index']
            end = chunk['last_character_index']

            is_valid_chunk = full_text[start:end] == chunk['text']
            assert is_valid_chunk, "Error in the chunks index"
            print(
                ("-" * 50), "\n"
                f"{chunk['first_character_index']}:"
                f"{chunk['last_character_index']}\n"
                f"{chunk['text']}\n"
            )