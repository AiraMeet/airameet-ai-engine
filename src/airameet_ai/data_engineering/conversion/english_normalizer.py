"""English preservation contract from AI-DS-02."""


def normalize_english(text: str) -> str:
    """Return English text unchanged, as specified by the notebook."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return text
