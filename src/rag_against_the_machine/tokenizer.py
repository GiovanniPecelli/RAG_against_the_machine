import re


class Tokenizer:
    """Simple tokenizer that normalizes text and splits code identifiers."""
    def __init__(self, min_length: int = 1):
        self.min_length = min_length

        self.stop_words = {
            "the", "a", "an", "is", "in", "to", "of", "and", "or",
            "for", "it", "with", "as", "at", "by", "from", "on"
        }

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []

        raw_tokens = re.findall(r"\w+", text)
        result_tokens: list[str] = []

        for token in raw_tokens:
            lowered = token.lower()

            if (len(lowered) >= self.min_length
                    and lowered not in self.stop_words):
                result_tokens.append(lowered)

            has_snake = "_" in token
            has_camel = bool(re.search(r"([a-z])([A-Z])", token))

            if has_snake or has_camel:
                # Normalize CamelCase (es. ModelRunner = Model Runner)
                # r"\1 \2" add a spece between: first_char" "second_char
                normalized = re.sub(r"([a-z])([A-Z])", r"\1 \2", token)

                # Normalize snake_case (es. model_runner = model runner)
                normalized = normalized.replace("_", " ")

                # Extract every words and get away all the unusefull simbols
                # -  Uniform the text
                # -  find-all based on the rule -> the Regex -> r"\w+"
                # -     r: read as a raw
                # -     \w: each letters and numbers
                # -     +: one or more together
                sub_tokens = re.findall(r"\w+", normalized.lower())

                for sub in sub_tokens:
                    if len(sub) >= self.min_length and sub not in self.stop_words:
                        result_tokens.append(sub)

        return result_tokens