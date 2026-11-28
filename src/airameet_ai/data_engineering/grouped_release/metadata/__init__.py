"""Metadata loading and source-aware identity normalization."""

from .metadata_corrector import plain_value, prepare_source_records
from .recording_metadata import (
    group_records,
    is_missing,
    load_cleaned_sources,
    recording_group,
)

__all__ = [
    "group_records",
    "is_missing",
    "load_cleaned_sources",
    "plain_value",
    "prepare_source_records",
    "recording_group",
]
