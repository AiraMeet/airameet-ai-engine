import pytest

from airameet_ai.data_engineering.normalization.english_normalizer import (
    normalize_english,
)


@pytest.mark.parametrize("text", ["PyTorch", "don't", "GPT-6", "Hello, World!"])
def test_english_terms_are_preserved(text: str) -> None:
    assert normalize_english(text) == text


def test_english_normalizer_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        normalize_english(None)
