"""
Dataset package for AiraMeet.

Reusable dataset inspection, validation,
and audio conversion functions.
"""

from .audio_converter import (
    convert_audio_file,
    convert_audio_batch,
)

from .metadata_validator import (
    inspect_metadata,
    metadata_quality_summary,
    validate_audio_paths,
    wada_snr,
    validate_metadata_schema,
)

__all__ = [
    "inspect_metadata",
    "metadata_quality_summary",
    "validate_audio_paths",
    "wada_snr",
    "validate_metadata_schema",
    "convert_audio_file",
    "convert_audio_batch",
]