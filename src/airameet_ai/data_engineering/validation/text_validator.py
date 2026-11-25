"""Validation for normalized AI-DS-02 metadata."""

import pandas as pd

from airameet_ai.data_engineering.conversion import normalize_text


def validate_cleaned_dataset(original: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    """Ensure row/schema/metadata preservation and normalization idempotency."""
    if len(original) != len(cleaned):
        raise ValueError("Row count changed during text processing")
    if list(original.columns) != list(cleaned.columns):
        raise ValueError("Columns changed during text processing")
    if "text" not in original.columns:
        raise ValueError("Required column 'text' was not found")
    for column in original.columns:
        if column != "text":
            pd.testing.assert_series_equal(original[column], cleaned[column])
    again = cleaned["text"].map(
        lambda value: normalize_text(value) if isinstance(value, str) else value
    )
    pd.testing.assert_series_equal(cleaned["text"], again)


def validate_unified_metadata(frame: pd.DataFrame) -> None:
    """Validate the notebook's common metadata schema and timing constraints."""
    required = [
        "utterance_id",
        "audio_path",
        "text",
        "split",
        "speaker_id",
        "recording_id",
        "start",
        "end",
        "duration",
        "source_dataset",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Unified metadata is missing columns: {missing}")
    if frame["utterance_id"].duplicated().any():
        raise ValueError("Unified metadata contains duplicate utterance IDs")
    for column in ("audio_path", "text", "start", "end", "duration"):
        if frame[column].isna().any():
            raise ValueError(f"Unified metadata contains missing {column} values")
    if (frame["end"] < frame["start"]).any() or (frame["duration"] < 0).any():
        raise ValueError("Unified metadata contains invalid timestamps")
