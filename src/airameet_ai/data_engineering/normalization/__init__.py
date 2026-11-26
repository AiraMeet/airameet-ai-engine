"""Reusable Arabic and English code-switching text normalization."""

from .arabic_normalizer import normalize_arabic_orthography, remove_arabic_diacritics
from .code_switching_normalizer import normalize_code_switching
from .english_normalizer import normalize_english
from .punctuation_normalizer import normalize_punctuation
from .text_processor import TextProcessor, normalize_text, normalize_unicode

__all__ = [
    "TextProcessor",
    "normalize_arabic_orthography",
    "normalize_code_switching",
    "normalize_english",
    "normalize_punctuation",
    "normalize_text",
    "normalize_unicode",
    "remove_arabic_diacritics",
]
