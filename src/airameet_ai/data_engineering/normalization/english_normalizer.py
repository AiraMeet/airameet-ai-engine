"""English preservation behavior from AI-DS-02."""


def normalize_english(text: str) -> str:
    """Preserve English spelling, technical tokens, and punctuation."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return text
