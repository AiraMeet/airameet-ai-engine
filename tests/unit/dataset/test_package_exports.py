"""Unit tests for dataset package public API and exports."""

from __future__ import annotations

import airameet_ai.dataset as dataset_pkg


class TestPackageExports:
    """Tests verifying dataset package public interface."""

    def test_package_imports_successfully(self) -> None:
        assert dataset_pkg is not None

    def test_all_declared_exports_defined(self) -> None:
        assert hasattr(dataset_pkg, "__all__")
        expected_exports = {
            "inspect_metadata",
            "metadata_quality_summary",
            "validate_audio_paths",
            "wada_snr",
            "validate_metadata_schema",
            "convert_audio_file",
            "convert_audio_batch",
        }
        actual_exports = set(dataset_pkg.__all__)
        assert actual_exports == expected_exports

    def test_all_exported_symbols_are_callable(self) -> None:
        for symbol_name in dataset_pkg.__all__:
            symbol = getattr(dataset_pkg, symbol_name)
            assert callable(symbol), f"Exported symbol '{symbol_name}' is not callable"
