import re


class Tokenizer:
    def __init__(self, min_length: int = 1):
        self.min_length = min_length

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []

        # Extract every words and get away all the unusefull simbols
        # -  Uniform the text
        # -  find-all based on the rule -> the Regex -> r"\w+"
        # -     r: read as a raw
        # -     \w: each letters and numbers
        # -     +: one or more together
        tokens = re.findall(r"\w+", text.lower())
        return [t for t in tokens if len(t) >= self.min_length]