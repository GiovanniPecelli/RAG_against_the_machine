from pathlib import Path


class Chunker:
    def __init__(self, max_chunk_size: int = 2000):
        self.max_chunk_size = max_chunk_size

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

        if file_path.suffix == ".md":
            return self.chunk_Markdown(content, normalized_path)
        elif file_path.suffix == ".txt":
            return self.chunk_text(content, normalized_path)
        elif file_path.suffix == ".py":
            return self.chunk_python(content, normalized_path)

        return []

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
        chunk: list[dict] = []
        current_text = ""
        chunk_start_index = 0

        lines = content.splitlines(keepends=True)
        current_index: int = 0
        for line in lines:
            start_line_index = current_index

            # ================================================================
            #            SPECIAL CASE: IF LINE > MAX_CHUNK_SIZE
            # ================================================================
            if len(line) > self.max_chunk_size:
                if current_text:
                    chunk.append({
                        "file_path": normalized_path,
                        "text": current_text,
                        "first_character_index": chunk_start_index,
                        "last_character_index": chunk_start_index + len(
                            current_text
                        ),
                    })
                    current_text = ""

                for sub_start in range(0, len(line), self.max_chunk_size):
                    sub_text = line[
                        sub_start : sub_start + self.max_chunk_size
                    ]
                    sub_first_idx = current_index + sub_start

                    if len(sub_text) == self.max_chunk_size:
                        chunk.append({
                            "file_path": normalized_path,
                            "text": sub_text,
                            "first_character_index": sub_first_idx,
                            "last_character_index": sub_first_idx + len(
                                sub_text
                            ),
                        })
                        chunk_start_index = sub_first_idx + len(sub_text)
                    else:
                        # IF the residual text is less then max_chunk_size
                        current_text = sub_text
                        chunk_start_index = sub_first_idx

                current_index += len(line)
                continue

            # ================================================================
            #        NORMAL CASE: a single line length < MAX_CHUNK_SIZE
            # ================================================================
            is_new_header = line.lstrip().startswith(keyworks)
            would_overflow = (
                (len(current_text) + len(line)) > self.max_chunk_size
            )

            should_split = (
                (is_new_header and len(current_text) >= 200) or would_overflow
            )

            # NEW CHUNK -> if there is an header or would overflow
            if should_split and current_text:
                chunk.append({
                    "file_path": normalized_path,
                    "text": current_text,
                    "first_character_index": chunk_start_index,
                    "last_character_index": chunk_start_index + len(
                        current_text
                    ),
                })
                # Reset the variable for the next chunk
                current_text = ""
                chunk_start_index = start_line_index

            current_text += line
            current_index += len(line)

        if current_text:
            chunk.append({
                "file_path": normalized_path,
                "text": current_text,
                "first_character_index": chunk_start_index,
                "last_character_index": chunk_start_index + len(current_text),
            })

        return chunk


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