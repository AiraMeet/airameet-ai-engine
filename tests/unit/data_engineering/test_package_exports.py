"""Unit tests for data_engineering package public API."""

from __future__ import annotations

import airameet_ai.data_engineering as pkg


class TestPackageExports:
    """Verify the public interface of data_engineering."""

    def test_all_declared_exports_defined(self) -> None:
        expected = {
            "DataEngineeringPipeline",
            "PipelineResults",
            "TextNormalizationPipeline",
            "audit_audio_files",
            "convert_audio_batch",
            "convert_audio_file",
            "inspect_metadata",
            "load_metadata",
            "metadata_quality_summary",
            "validate_audio_paths",
            "validate_metadata_schema",
            "wada_snr",
        }
        assert set(pkg.__all__) == expected

    def test_exported_symbols_are_available(self) -> None:
        for name in pkg.__all__:
            assert hasattr(pkg, name), f"Missing export: {name}"
