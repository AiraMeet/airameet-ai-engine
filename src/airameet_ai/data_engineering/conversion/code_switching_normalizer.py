"""Code-switching text normalization extracted from AI-DS-02."""

from .arabic_normalizer import normalize_arabic_orthography, remove_arabic_diacritics
from .punctuation_normalizer import normalize_punctuation
from .text_processor import normalize_unicode


def normalize_text(text: str) -> str:
    """Apply notebook transformations in their original order."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = normalize_unicode(text)
    text = remove_arabic_diacritics(text)
    text = normalize_arabic_orthography(text)
    return normalize_punctuation(text)
