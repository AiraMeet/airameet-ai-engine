"""Unit tests for metadata_validator module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from airameet_ai.dataset.metadata_validator import (
    inspect_metadata,
    metadata_quality_summary,
    validate_audio_paths,
    validate_metadata_schema,
    wada_snr,
)


class TestInspectMetadata:
    """Tests for inspect_metadata function."""

    def test_inspect_valid_metadata(self, sample_metadata_df: pd.DataFrame) -> None:
        result = inspect_metadata(sample_metadata_df, "test_dataset")

        assert result["dataset"] == "test_dataset"
        assert result["rows"] == 3
        assert result["columns"] == sample_metadata_df.columns.tolist()
        assert isinstance(result["data_types"], dict)
        assert isinstance(result["missing_values"], dict)
        assert all(count == 0 for count in result["missing_values"].values())

    def test_inspect_empty_metadata(self) -> None:
        df = pd.DataFrame(columns=["meeting", "speaker", "text"])
        result = inspect_metadata(df, "empty_dataset")

        assert result["dataset"] == "empty_dataset"
        assert result["rows"] == 0
        assert result["columns"] == ["meeting", "speaker", "text"]
        assert result["missing_values"] == {"meeting": 0, "speaker": 0, "text": 0}

    def test_inspect_metadata_with_missing_values(self) -> None:
        df = pd.DataFrame(
            {
                "col_a": [1, None, 3],
                "col_b": [None, None, "c"],
            }
        )
        result = inspect_metadata(df, "null_dataset")

        assert result["rows"] == 3
        assert result["missing_values"] == {"col_a": 1, "col_b": 2}


class TestMetadataQualitySummary:
    """Tests for metadata_quality_summary function."""

    def test_clean_dataset_summary(self, sample_metadata_df: pd.DataFrame) -> None:
        summary = metadata_quality_summary(sample_metadata_df, "clean_data")

        assert summary["Dataset"] == "clean_data"
        assert summary["Rows"] == 3
        assert summary["Missing Values"] == 0
        assert summary["Duplicate Rows"] == 0
        assert summary["Duplicate Audio Paths"] == 0
        assert summary["start Invalid"] == 0
        assert summary["start Negative"] == 0
        assert summary["end Invalid"] == 0
        assert summary["end Negative"] == 0
        assert summary["duration Invalid"] == 0
        assert summary["duration Negative"] == 0
        assert summary["Duration Mismatches"] == 0

    def test_detects_duplicates(self) -> None:
        df = pd.DataFrame(
            {
                "meeting": ["M1", "M1"],
                "audio_path": ["a.wav", "a.wav"],
                "start": [0.0, 0.0],
                "end": [1.0, 1.0],
                "duration": [1.0, 1.0],
            }
        )
        summary = metadata_quality_summary(df, "dup_data")

        assert summary["Rows"] == 2
        assert summary["Duplicate Rows"] == 1
        assert summary["Duplicate Audio Paths"] == 1

    def test_missing_audio_path_column(self) -> None:
        df = pd.DataFrame({"start": [0.0], "end": [1.0], "duration": [1.0]})
        summary = metadata_quality_summary(df, "no_audio_col")

        assert summary["Duplicate Audio Paths"] is None

    def test_detects_invalid_and_negative_timestamps(self) -> None:
        df = pd.DataFrame(
            {
                "start": [0.0, -1.0, "invalid_time"],
                "end": [5.0, 4.0, np.nan],
                "duration": [5.0, -2.0, np.inf],
            }
        )
        summary = metadata_quality_summary(df, "bad_timestamps")

        assert summary["start Invalid"] == 1
        assert summary["start Negative"] == 1
        assert summary["end Invalid"] == 1
        assert summary["duration Invalid"] == 1
        assert summary["duration Negative"] == 1

    def test_duration_mismatch_detection(self) -> None:
        df = pd.DataFrame(
            {
                # Row 0: exact match
                # Row 1: mismatch (|10 - 0 - 5| = 5 > 0.05)
                # Row 2: within tolerance (|5.04 - 0 - 5.0| = 0.04 <= 0.05)
                "start": [0.0, 0.0, 0.0],
                "end": [5.0, 10.0, 5.04],
                "duration": [5.0, 5.0, 5.0],
            }
        )
        summary = metadata_quality_summary(df, "mismatch_data")

        assert summary["Duration Mismatches"] == 1

    def test_partial_timestamp_columns(self) -> None:
        df = pd.DataFrame({"start": [0.0, 1.0], "text": ["hi", "there"]})
        summary = metadata_quality_summary(df, "partial_cols")

        assert "start Invalid" in summary
        assert "end Invalid" not in summary
        assert "Duration Mismatches" not in summary


class TestValidateAudioPaths:
    """Tests for validate_audio_paths function."""

    def test_validate_found_audio(
        self,
        sample_metadata_df: pd.DataFrame,
        synthetic_dataset_dir: Path,
    ) -> None:
        result = validate_audio_paths(sample_metadata_df, synthetic_dataset_dir)

        assert "resolved_audio_path" in result.columns
        assert "audio_exists" in result.columns
        assert "path_status" in result.columns
        assert len(result) == 3
        assert (result["path_status"] == "found").all()
        assert bool(result["audio_exists"].all()) is True

    def test_validate_missing_file_not_found(
        self,
        synthetic_dataset_dir: Path,
    ) -> None:
        df = pd.DataFrame({"audio_path": ["audio/non_existent.wav"]})
        result = validate_audio_paths(df, synthetic_dataset_dir)

        assert result.loc[0, "path_status"] == "file_not_found"
        assert bool(result.loc[0, "audio_exists"]) is False
        assert result.loc[0, "resolved_audio_path"] is not None

    def test_validate_missing_metadata_path(
        self,
        synthetic_dataset_dir: Path,
    ) -> None:
        df = pd.DataFrame({"audio_path": [None, "", "   "]})
        result = validate_audio_paths(df, synthetic_dataset_dir)

        assert (result["path_status"] == "missing_metadata_path").all()
        assert not bool(result["audio_exists"].any())
        assert result["resolved_audio_path"].isna().all()

    def test_validate_absolute_path_rejected(
        self,
        tmp_path: Path,
        synthetic_dataset_dir: Path,
    ) -> None:
        abs_file = tmp_path / "outside.wav"
        abs_file.touch()

        df = pd.DataFrame({"audio_path": [str(abs_file.resolve())]})
        result = validate_audio_paths(df, synthetic_dataset_dir)

        assert result.loc[0, "path_status"] == "invalid_path"
        assert bool(result.loc[0, "audio_exists"]) is False
        assert result.loc[0, "resolved_audio_path"] is None

    def test_validate_path_traversal_rejected(
        self,
        synthetic_dataset_dir: Path,
    ) -> None:
        df = pd.DataFrame({"audio_path": ["../../etc/shadow"]})
        result = validate_audio_paths(df, synthetic_dataset_dir)

        assert result.loc[0, "path_status"] == "invalid_path"
        assert bool(result.loc[0, "audio_exists"]) is False

    def test_missing_path_column_raises_error(
        self,
        synthetic_dataset_dir: Path,
    ) -> None:
        df = pd.DataFrame({"wrong_column": ["audio/test.wav"]})
        with pytest.raises(
            ValueError, match="Required column 'audio_path' was not found"
        ):
            validate_audio_paths(df, synthetic_dataset_dir)

    def test_nonexistent_dataset_root_raises_error(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"audio_path": ["audio/test.wav"]})
        non_existent = tmp_path / "does_not_exist"
        with pytest.raises(NotADirectoryError):
            validate_audio_paths(df, non_existent)

    def test_file_as_dataset_root_raises_error(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"audio_path": ["audio/test.wav"]})
        file_root = tmp_path / "file.txt"
        file_root.touch()
        with pytest.raises(NotADirectoryError):
            validate_audio_paths(df, file_root)

    def test_custom_path_column(
        self,
        synthetic_dataset_dir: Path,
    ) -> None:
        df = pd.DataFrame({"custom_audio_col": ["audio/mtg1_01.wav"]})
        result = validate_audio_paths(
            df, synthetic_dataset_dir, path_column="custom_audio_col"
        )
        assert result.loc[0, "path_status"] == "found"
        assert bool(result.loc[0, "audio_exists"]) is True


class TestWadaSnr:
    """Tests for wada_snr function."""

    def test_valid_mono_waveform(self) -> None:
        # Generate 16 kHz sine wave
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        snr = wada_snr(audio)
        assert isinstance(snr, float)
        assert -20.0 <= snr <= 101.0

    def test_valid_stereo_waveform(self) -> None:
        t = np.linspace(0, 0.5, 8000, endpoint=False)
        ch1 = 0.5 * np.sin(2 * np.pi * 440 * t)
        ch2 = 0.4 * np.sin(2 * np.pi * 880 * t)
        stereo_audio = np.stack([ch1, ch2], axis=1)

        snr = wada_snr(stereo_audio)
        assert isinstance(snr, float)
        assert -20.0 <= snr <= 101.0

    def test_empty_audio_raises_error(self) -> None:
        empty = np.array([])
        with pytest.raises(ValueError, match="Audio signal is empty"):
            wada_snr(empty)

    def test_nan_audio_raises_error(self) -> None:
        audio = np.array([0.1, np.nan, 0.3])
        with pytest.raises(ValueError, match="Audio contains NaN or Inf"):
            wada_snr(audio)

    def test_inf_audio_raises_error(self) -> None:
        audio = np.array([0.1, np.inf, 0.3])
        with pytest.raises(ValueError, match="Audio contains NaN or Inf"):
            wada_snr(audio)

    def test_silent_audio_raises_error(self) -> None:
        silent = np.zeros(16000)
        with pytest.raises(ValueError, match="Cannot estimate SNR for silent audio"):
            wada_snr(silent)

    def test_invalid_dimensions_raises_error(self) -> None:
        audio_3d = np.ones((2, 2, 2))
        with pytest.raises(ValueError, match="Audio must be mono or stereo"):
            wada_snr(audio_3d)

    def test_extreme_low_snr_boundary(self) -> None:
        # Uniform noise with very low amplitude variance
        np.random.seed(42)
        noise = np.random.uniform(0.999, 1.0, 16000)
        snr = wada_snr(noise)
        assert snr == -20.0

    def test_extreme_high_snr_boundary(self) -> None:
        # Spiky signal where one sample is peak and remainder is near zero
        impulse = np.zeros(16000)
        impulse[0] = 1.0
        snr = wada_snr(impulse)
        assert snr == 100.0


class TestValidateMetadataSchema:
    """Tests for validate_metadata_schema function."""

    def test_valid_schema(self, sample_metadata_df: pd.DataFrame) -> None:
        report = validate_metadata_schema(sample_metadata_df, "test_valid")

        assert report["dataset"] == "test_valid"
        assert report["total_records"] == 3
        assert report["schema_valid"] is True
        assert report["missing_columns"] == []
        assert report["empty_audio_paths"] == 0
        assert report["empty_text"] == 0
        assert report["invalid_timestamps"] == 0
        assert report["negative_timestamps"] == 0
        assert report["end_before_start"] == 0
        assert report["duration_mismatches"] == 0
        assert report["duplicate_audio_paths"] == 0

    def test_missing_required_columns(self) -> None:
        df = pd.DataFrame({"meeting": ["M1"], "speaker": ["S1"]})
        report = validate_metadata_schema(df, "incomplete")

        assert report["schema_valid"] is False
        assert len(report["missing_columns"]) > 0
        assert "audio_path" in report["missing_columns"]
        assert "text" in report["missing_columns"]
        # When columns are missing, detailed metrics are not computed
        assert "empty_audio_paths" not in report

    def test_custom_required_columns(self) -> None:
        df = pd.DataFrame({"custom_col_1": [1], "custom_col_2": ["a"]})
        report = validate_metadata_schema(
            df, "custom", required_columns=["custom_col_1", "custom_col_2"]
        )

        assert report["schema_valid"] is True
        assert report["missing_columns"] == []

    def test_detects_empty_paths_and_text(self) -> None:
        df = pd.DataFrame(
            {
                "meeting": ["M1", "M2"],
                "speaker": ["S1", "S2"],
                "audio_path": ["", "   "],
                "text": ["Hello", "   "],
                "start": [0.0, 1.0],
                "end": [1.0, 2.0],
                "duration": [1.0, 1.0],
            }
        )
        report = validate_metadata_schema(df, "empty_strings")

        assert report["empty_audio_paths"] == 2
        assert report["empty_text"] == 1

    def test_detects_timing_anomalies(self) -> None:
        df = pd.DataFrame(
            {
                "meeting": ["M1", "M2", "M3"],
                "speaker": ["S1", "S2", "S3"],
                "audio_path": ["a1.wav", "a2.wav", "a1.wav"],
                "text": ["t1", "t2", "t3"],
                "start": [5.0, -1.0, "bad"],
                "end": [2.0, 3.0, 10.0],  # row 0: end < start
                "duration": [3.0, 4.0, 5.0],  # row 1: negative start
            }
        )
        report = validate_metadata_schema(df, "timing_issues")

        assert report["end_before_start"] >= 1
        assert report["negative_timestamps"] >= 1
        assert report["invalid_timestamps"] >= 1
        assert report["duplicate_audio_paths"] == 1
