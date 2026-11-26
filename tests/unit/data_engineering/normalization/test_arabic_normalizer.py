import pytest

from airameet_ai.data_engineering.normalization.arabic_normalizer import (
    normalize_arabic_orthography,
    remove_arabic_diacritics,
)


@pytest.mark.parametrize("source, expected", [("أَحْمَد", "احمد"), ("ىةـ", "يه")])
def test_arabic_stages(source: str, expected: str) -> None:
    assert normalize_arabic_orthography(remove_arabic_diacritics(source)) == expected


@pytest.mark.parametrize(
    "function", [normalize_arabic_orthography, remove_arabic_diacritics]
)
def test_arabic_stages_reject_non_string(function) -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        function(None)
