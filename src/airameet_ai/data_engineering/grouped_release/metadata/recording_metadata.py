"""Metadata loading, recording identity validation, and group construction."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from airameet_ai.data_engineering.inspection import load_metadata

LOGGER = logging.getLogger(__name__)
SOURCE_FILES = {
    "Erzen": "Erzen_cleaned_metadata.csv",
    "Arabic-English": "Arabic_English_cleaned_metadata.csv",
}


def load_cleaned_sources(directory: str | Path) -> dict[str, tuple[Path, pd.DataFrame]]:
    """Load available AI-DS-02 metadata exports, without changing source rows."""
    root = Path(directory)
    sources: dict[str, tuple[Path, pd.DataFrame]] = {}
    for source, filename in SOURCE_FILES.items():
        path = root / filename
        if path.is_file():
            frame = load_metadata(path)
            sources[source] = (path, frame)
            LOGGER.info("Loaded %s metadata: %d records", source, len(frame))
    if not sources:
        raise FileNotFoundError(
            f"No cleaned AI-DS-02 metadata files found in {root}; expected "
            f"{', '.join(SOURCE_FILES.values())}"
        )
    return sources


def is_missing(value: Any) -> bool:
    """Return whether a scalar metadata value is missing or blank."""
    if value is None or bool(pd.isna(value)):
        return True
    return isinstance(value, str) and not value.strip()


def recording_group(record: dict[str, Any]) -> tuple[str, str]:
    """Return globally unique group identity or reject missing identifiers."""
    source = record.get("source_dataset")
    recording = record.get("recording_id")
    if is_missing(source) or is_missing(recording):
        raise ValueError("recording group requires non-empty source_dataset and recording_id")
    return str(source).strip(), str(recording).strip()


def group_records(
    records: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Preserve every eligible utterance under its source/recording group."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[recording_group(record)].append(record)
    return dict(sorted(grouped.items()))
