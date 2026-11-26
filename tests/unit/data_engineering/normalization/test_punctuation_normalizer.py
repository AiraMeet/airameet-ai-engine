import pytest

from airameet_ai.data_engineering.normalization.punctuation_normalizer import (
    normalize_punctuation,
)


@pytest.mark.parametrize(
    "source, expected",
    [('"(hello)"', "hello"), ("Hello, world!", "Hello, world!"), ("  a   b ", "a b")],
)
def test_punctuation_and_whitespace(source: str, expected: str) -> None:
    assert normalize_punctuation(source) == expected


def test_punctuation_normalizer_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        normalize_punctuation(None)
