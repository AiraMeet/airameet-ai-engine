import pytest

from airameet_ai.data_engineering.normalization.code_switching_normalizer import (
    normalize_code_switching,
)


def test_mixed_text_is_preserved_by_code_switch_stage() -> None:
    assert normalize_code_switching("Hello يا أحمد GPT-6") == "Hello يا أحمد GPT-6"


def test_code_switch_normalizer_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        normalize_code_switching(None)
