"""Unicode cleanup and ordered text normalization interface."""

from __future__ import annotations

import unicodedata

from .arabic_normalizer import normalize_arabic_orthography, remove_arabic_diacritics
from .code_switching_normalizer import normalize_code_switching
from .english_normalizer import normalize_english
from .punctuation_normalizer import normalize_punctuation


def normalize_unicode(text: str) -> str:
    """Apply the notebook's NFC, control character, and whitespace rules."""
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


def normalize_text(text: str) -> str:
    """Run normalization stages in their original notebook order."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = normalize_unicode(text)
    text = remove_arabic_diacritics(text)
    text = normalize_arabic_orthography(text)
    text = normalize_english(text)
    text = normalize_code_switching(text)
    return normalize_punctuation(text)


class TextProcessor:
    """Reusable interface for applying the AI-DS-02 text transformations."""

    def process(self, text: str) -> str:
        """Normalize one string while retaining notebook input validation."""
        return normalize_text(text)
