"""Unicode and general text processing extracted from AI-DS-02."""

import unicodedata


def normalize_unicode(text: str) -> str:
    """Apply the notebook's Unicode, control character, and whitespace rules."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = unicodedata.normalize("NFC", text)
    text = (
        text.replace("\ufeff", "")
        .replace("\t", " ")
        .replace("\n", " ")
        .replace("\r", " ")
    )
    text = "".join(char for char in text if unicodedata.category(char) != "Cc")
    return unicodedata.normalize("NFC", " ".join(text.split()))
