"""Text profiling helpers based on the AI-DS-02 exploratory notebook cells."""

import re
from collections import Counter

import pandas as pd


def profile_text(frame: pd.DataFrame) -> dict[str, object]:
    """Return the notebook's language, number, multiline, and character counts."""
    if "text" not in frame.columns:
        raise ValueError("Required column 'text' was not found")
    texts = frame["text"].dropna().astype(str)
    arabic_pattern = re.compile(r"[\u0600-\u06FF]")
    english_pattern = re.compile(r"[A-Za-z]")
    number_pattern = re.compile(r"\d")
    english_words: Counter[str] = Counter()
    special_chars: Counter[str] = Counter()
    arabic_count = english_count = mixed_count = numeric_count = multiline_count = 0
    for text in texts:
        has_arabic = bool(arabic_pattern.search(text))
        has_english = bool(english_pattern.search(text))
        arabic_count += has_arabic
        english_count += has_english
        mixed_count += has_arabic and has_english
        numeric_count += bool(number_pattern.search(text))
        multiline_count += "\n" in text
        english_words.update(re.findall(r"[A-Za-z]+", text))
        special_chars.update(
            char for char in text if not char.isalnum() and not char.isspace()
        )
    return {
        "total_texts": len(texts),
        "arabic_texts": arabic_count,
        "english_containing_texts": english_count,
        "mixed_arabic_english": mixed_count,
        "texts_containing_numbers": numeric_count,
        "multiline_texts": multiline_count,
        "top_english_words": english_words.most_common(20),
        "top_special_characters": special_chars.most_common(20),
    }


def preview_normalization(frame: pd.DataFrame, sample_size: int = 20) -> pd.DataFrame:
    """Return the notebook's before/after/changed preview as a DataFrame."""
    if "text" not in frame.columns:
        raise ValueError("Required column 'text' was not found")
    from airameet_ai.data_engineering.normalization import normalize_text

    sample = frame[["text"]].head(sample_size).copy()
    sample["normalized_text"] = sample["text"].map(
        lambda value: normalize_text(value) if isinstance(value, str) else value
    )
    sample["changed"] = sample["text"] != sample["normalized_text"]
    return sample
