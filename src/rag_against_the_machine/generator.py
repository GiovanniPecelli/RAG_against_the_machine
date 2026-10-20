from .model import MinimalSource


class Generator:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """Loads tokenizer and Qwen/Qwen3-0.6B model onto available device."""
        pass

    def generate_answer(
            self,
            query: str,
            sources: list[MinimalSource]
    ) -> str:
        """Extracts context from sources, builds prompt, and generates answer."""
        pass