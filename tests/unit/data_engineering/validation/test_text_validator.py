import pandas as pd
import pytest

from airameet_ai.data_engineering.validation.text_validator import (
    validate_cleaned_dataset,
    validate_unified_metadata,
)


def test_cleaned_dataset_preserves_non_text_columns_and_allows_missing_text() -> None:
    source = pd.DataFrame({"id": [1, 2], "text": ["أَحْمَد", None]})
    cleaned = pd.DataFrame({"id": [1, 2], "text": ["احمد", None]})
    validate_cleaned_dataset(source, cleaned)


@pytest.mark.parametrize(
    "cleaned",
    [
        pd.DataFrame({"text": ["ok"]}),
        pd.DataFrame({"id": [2], "text": ["احمد"]}),
    ],
)
def test_cleaned_dataset_rejects_schema_or_row_changes(cleaned: pd.DataFrame) -> None:
    source = pd.DataFrame({"id": [1, 2], "text": ["احمد", "ok"]})
    with pytest.raises(ValueError):
        validate_cleaned_dataset(source, cleaned)


def test_unified_metadata_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        validate_unified_metadata(pd.DataFrame({"text": []}))
