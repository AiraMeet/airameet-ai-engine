"""Unit tests for audio_auditor module."""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pandas as pd
import pytest

from airameet_ai.data_engineering.quality.audio_auditor import audit_audio_files


def _make_wav(
    path: Path,
    sample_rate: int = 16000,
    duration: float = 0.5,
    channels: int = 1,
    freq: float = 440.0,
    amplitude: float = 0.5,
) -> Path:
    """Helper: create a synthetic 16-bit PCM WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(sample_rate * duration)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(num_samples):
            val = int(32767.0 * amplitude * math.sin(2.0 * math.pi * freq * (i / sample_rate)))
            for _ in range(channels):
                frames.extend(struct.pack("<h", val))
        wf.writeframes(frames)
    return path


class TestAuditAudioFiles:
    """Tests for the audit_audio_files function."""

    def test_valid_mono_audio_is_readable(self, tmp_path: Path) -> None:
        _make_wav(tmp_path / "audio" / "valid.wav", duration=1.0)
        df = pd.DataFrame({"audio_path": ["audio/valid.wav"]})

        result = audit_audio_files(df, tmp_path)

        assert len(result) == 1
        assert bool(result.loc[0, "readable"]) is True
        assert result.loc[0, "error"] is None

    def test_valid_stereo_audio_is_readable(self, tmp_path: Path) -> None:
        _make_wav(
            tmp_path / "audio" / "stereo.wav", channels=2, duration=0.5
        )
        df = pd.DataFrame({"audio_path": ["audio/stereo.wav"]})

        result = audit_audio_files(df, tmp_path)

        assert bool(result.loc[0, "readable"]) is True
        assert result.loc[0, "channels"] == 2

    def test_records_sample_rate(self, tmp_path: Path) -> None:
        _make_wav(tmp_path / "audio" / "file.wav", sample_rate=16000, duration=0.5)
        df = pd.DataFrame({"audio_path": ["audio/file.wav"]})

        result = audit_audio_files(df, tmp_path)

        assert result.loc[0, "sample_rate"] == 16000

    def test_records_duration(self, tmp_path: Path) -> None:
        _make_wav(tmp_path / "audio" / "file.wav", duration=1.0)
        df = pd.DataFrame({"audio_path": ["audio/file.wav"]})

        result = audit_audio_files(df, tmp_path)

        assert result.loc[0, "duration"] == pytest.approx(1.0, abs=0.05)

    def test_missing_file_records_error_gracefully(self, tmp_path: Path) -> None:
        (tmp_path / "audio").mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"audio_path": ["audio/nonexistent.wav"]})

        result = audit_audio_files(df, tmp_path)

        assert len(result) == 1
        assert bool(result.loc[0, "readable"]) is False
        assert result.loc[0, "error"] is not None

    def test_none_audio_path_records_error(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"audio_path": [None]})

        result = audit_audio_files(df, tmp_path)

        assert bool(result.loc[0, "readable"]) is False
        assert result.loc[0, "error"] is not None

    def test_snr_is_computed_for_valid_audio(self, tmp_path: Path) -> None:
        _make_wav(tmp_path / "audio" / "file.wav", duration=1.0, amplitude=0.5)
        df = pd.DataFrame({"audio_path": ["audio/file.wav"]})

        result = audit_audio_files(df, tmp_path)

        assert result.loc[0, "snr_status"] == "success"
        assert result.loc[0, "snr_db"] is not None
        assert isinstance(result.loc[0, "snr_db"], float)

    def test_peak_amplitude_is_recorded(self, tmp_path: Path) -> None:
        _make_wav(tmp_path / "audio" / "file.wav", duration=0.5)
        df = pd.DataFrame({"audio_path": ["audio/file.wav"]})

        result = audit_audio_files(df, tmp_path)

        assert result.loc[0, "peak_amplitude"] is not None
        assert 0.0 < float(result.loc[0, "peak_amplitude"]) <= 1.0

    def test_clipping_detection_counts_samples_above_099(self, tmp_path: Path) -> None:
        # Amplitude near 1.0 to trigger clipping
        _make_wav(tmp_path / "audio" / "clipped.wav", duration=0.5, amplitude=0.999)
        df = pd.DataFrame({"audio_path": ["audio/clipped.wav"]})

        result = audit_audio_files(df, tmp_path)

        assert bool(result.loc[0, "readable"]) is True
        assert result.loc[0, "clipping_samples_gt_099"] is not None

    def test_path_traversal_is_rejected(self, tmp_path: Path) -> None:
        # Create a file outside the dataset root
        outside_file = tmp_path.parent / "outside.wav"
        _make_wav(outside_file, duration=0.1)
        df = pd.DataFrame({"audio_path": ["../../outside.wav"]})

        result = audit_audio_files(df, tmp_path)

        assert bool(result.loc[0, "readable"]) is False
        assert result.loc[0, "error"] is not None

    def test_multiple_files_processed_independently(self, tmp_path: Path) -> None:
        _make_wav(tmp_path / "audio" / "a.wav", duration=0.5)
        df = pd.DataFrame({"audio_path": ["audio/a.wav", "audio/missing.wav"]})

        result = audit_audio_files(df, tmp_path)

        assert len(result) == 2
        assert bool(result.loc[0, "readable"]) is True
        assert bool(result.loc[1, "readable"]) is False

    def test_resolved_audio_paths_override_relative_paths(self, tmp_path: Path) -> None:
        raw_root = tmp_path / "raw"
        processed_root = tmp_path / "processed"
        # Create audio only in processed, not in raw
        processed_file = _make_wav(
            processed_root / "audio" / "file.wav", duration=0.5
        )
        raw_root.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame({"audio_path": ["audio/file.wav"]})
        resolved_paths = pd.Series([str(processed_file)])

        result = audit_audio_files(
            df,
            raw_root,
            audio_paths=resolved_paths,
            allowed_roots=(processed_root,),
        )

        assert bool(result.loc[0, "readable"]) is True

    def test_total_samples_is_recorded(self, tmp_path: Path) -> None:
        sample_rate = 16000
        duration = 0.5
        _make_wav(
            tmp_path / "audio" / "file.wav", sample_rate=sample_rate, duration=duration
        )
        df = pd.DataFrame({"audio_path": ["audio/file.wav"]})

        result = audit_audio_files(df, tmp_path)

        expected_samples = int(sample_rate * duration)
        actual = int(result.loc[0, "total_samples"])
        assert abs(actual - expected_samples) <= 10  # allow small rounding

    def test_result_is_dataframe(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"audio_path": ["audio/nonexistent.wav"]})

        result = audit_audio_files(df, tmp_path)

        assert isinstance(result, pd.DataFrame)

    def test_result_has_expected_columns(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"audio_path": ["audio/nonexistent.wav"]})

        result = audit_audio_files(df, tmp_path)

        expected_columns = {
            "audio_path",
            "source_path",
            "sample_rate",
            "channels",
            "duration",
            "readable",
            "peak_amplitude",
            "snr_db",
            "snr_status",
            "error",
        }
        assert expected_columns.issubset(set(result.columns))

    def test_empty_dataframe_returns_empty_result(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"audio_path": []})

        result = audit_audio_files(df, tmp_path)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


    def test_snr_success_on_tonal_audio(self, tmp_path: Path) -> None:
        """Longer tonal audio should yield a successful WADA-SNR estimate."""
        _make_wav(tmp_path / "audio" / "tone.wav", duration=1.0, amplitude=0.4)
        df = pd.DataFrame({"audio_path": ["audio/tone.wav"]})

        result = audit_audio_files(df, tmp_path)

        assert bool(result.loc[0, "readable"]) is True
        assert result.loc[0, "snr_status"] == "success"
        assert result.loc[0, "snr_db"] is not None

