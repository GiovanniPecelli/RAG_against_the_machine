import re


class Tokenizer:
    """Simple tokenizer that normalizes text and splits code identifiers."""
    def __init__(self, min_length: int = 1):
        self.min_length = min_length

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []

        # Normalize CamelCase (es. ModelRunner = Model Runner)
        # r"\1 \2" add a spece between: first_char" "second_char
        normalized = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

        # Normalize snake_case (es. model_runner = model runner)
        normalized = normalized.replace("_", " ")

        # Extract every words and get away all the unusefull simbols
        # -  Uniform the text
        # -  find-all based on the rule -> the Regex -> r"\w+"
        # -     r: read as a raw
        # -     \w: each letters and numbers
        # -     +: one or more together
        tokens = re.findall(r"\w+", normalized.lower())
        return [t for t in tokens if len(t) >= self.min_length]