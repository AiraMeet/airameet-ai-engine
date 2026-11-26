"""Mixed Arabic-English preservation behavior from AI-DS-02."""


def normalize_code_switching(text: str) -> str:
    """Preserve mixed text; the notebook applies no extra mixed-text rule."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return text
