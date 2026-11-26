import pandas as pd
import pytest

from airameet_ai.data_engineering.inspection import preview_normalization, profile_text


def test_profile_text_matches_notebook_metrics() -> None:
    result = profile_text(pd.DataFrame({"text": ["أهلا GPT-6\n", "123", None]}))
    assert result["total_texts"] == 2
    assert result["arabic_texts"] == 1
    assert result["english_containing_texts"] == 1
    assert result["mixed_arabic_english"] == 1
    assert result["texts_containing_numbers"] == 2
    assert result["multiline_texts"] == 1


def test_preview_normalization_preserves_null_values() -> None:
    result = preview_normalization(pd.DataFrame({"text": ["أَحْمَد", None]}))
    assert result["normalized_text"].iloc[0] == "احمد"
    assert pd.isna(result["normalized_text"].iloc[1])
    # The notebook's direct inequality preview reports a missing value as changed.
    assert result["changed"].tolist() == [True, True]


@pytest.mark.parametrize("function", [profile_text, preview_normalization])
def test_text_inspection_requires_text_column(function) -> None:
    with pytest.raises(ValueError, match="Required column 'text'"):
        function(pd.DataFrame({"other": ["value"]}))
