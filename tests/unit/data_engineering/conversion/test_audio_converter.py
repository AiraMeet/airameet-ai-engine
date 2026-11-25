"""Unit tests for audio_converter module."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from airameet_ai.data_engineering.conversion.audio_converter import (
    convert_audio_batch,
    convert_audio_file,
)


class TestConvertAudioFile:
    """Tests for convert_audio_file function."""

    def test_convert_valid_mono_wav(
        self,
        tmp_path: Path,
        make_wav_file: Callable[..., Path],
    ) -> None:
        source = make_wav_file(tmp_path / "in.wav", sample_rate=16000, duration=0.1)
        output = tmp_path / "out" / "processed.wav"

        result = convert_audio_file(source, output, target_sample_rate=16000)

        assert result["status"] == "success"
        assert result["error"] is None
        assert result["source_path"] == str(source)
        assert result["output_path"] == str(output)
        assert output.exists()

        info = sf.info(str(output))
        assert info.samplerate == 16000
        assert info.channels == 1
        assert info.subtype == "PCM_16"
        assert info.format == "WAV"

    def test_convert_stereo_to_mono(
        self,
        tmp_path: Path,
        make_wav_file: Callable[..., Path],
    ) -> None:
        source = make_wav_file(
            tmp_path / "stereo.wav", sample_rate=16000, channels=2, duration=0.1
        )
        output = tmp_path / "mono.wav"

        result = convert_audio_file(source, output)

        assert result["status"] == "success"
        assert output.exists()
        info = sf.info(str(output))
        assert info.channels == 1

    def test_resample_audio(
        self,
        tmp_path: Path,
        make_wav_file: Callable[..., Path],
    ) -> None:
        # 8000 Hz source resampled to 16000 Hz
        source = make_wav_file(
            tmp_path / "8k.wav", sample_rate=8000, channels=1, duration=0.1
        )
        output = tmp_path / "16k.wav"

        result = convert_audio_file(source, output, target_sample_rate=16000)

        assert result["status"] == "success"
        assert output.exists()
        info = sf.info(str(output))
        assert info.samplerate == 16000

    def test_creates_parent_directories(
        self,
        tmp_path: Path,
        make_wav_file: Callable[..., Path],
    ) -> None:
        source = make_wav_file(tmp_path / "source.wav", sample_rate=16000)
        output = tmp_path / "deep" / "nested" / "dir" / "out.wav"

        result = convert_audio_file(source, output)

        assert result["status"] == "success"
        assert output.exists()

    def test_empty_audio_file_fails(self, tmp_path: Path) -> None:
        empty_wav = tmp_path / "empty.wav"
        # Write 0-byte file
        empty_wav.touch()
        output = tmp_path / "out.wav"

        result = convert_audio_file(empty_wav, output)

        assert result["status"] == "failed"
        assert result["error"] is not None
        assert not output.exists()

    def test_nonexistent_source_fails(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "not_there.wav"
        output = tmp_path / "out.wav"

        result = convert_audio_file(nonexistent, output)

        assert result["status"] == "failed"
        assert result["error"] is not None
        assert not output.exists()

    def test_corrupt_audio_fails(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "corrupt.wav"
        corrupt.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt garbage not real audio")
        output = tmp_path / "out.wav"

        result = convert_audio_file(corrupt, output)

        assert result["status"] == "failed"
        assert result["error"] is not None
        assert not output.exists()

    def test_nan_or_inf_audio_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dummy_source = tmp_path / "dummy.wav"
        dummy_source.touch()
        output = tmp_path / "out.wav"

        # Mock sf.read to return NaN values
        def mock_read(*args, **kwargs):
            return np.array([[np.nan], [0.5]]), 16000

        monkeypatch.setattr(sf, "read", mock_read)

        result = convert_audio_file(dummy_source, output)

        assert result["status"] == "failed"
        assert "Audio contains NaN or Inf." in result["error"]

    def test_empty_audio_array_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dummy_source = tmp_path / "dummy.wav"
        dummy_source.touch()
        output = tmp_path / "out.wav"

        monkeypatch.setattr(sf, "read", lambda *args, **kwargs: (np.array([]), 16000))

        result = convert_audio_file(dummy_source, output)

        assert result["status"] == "failed"
        assert "Audio file is empty." in result["error"]


class TestConvertAudioBatch:
    """Tests for convert_audio_batch function."""

    def test_batch_success(
        self,
        tmp_path: Path,
        make_wav_file: Callable[..., Path],
    ) -> None:
        source_dir = tmp_path / "src_dir"
        output_dir = tmp_path / "out_dir"

        make_wav_file(source_dir / "a.wav", duration=0.05)
        make_wav_file(source_dir / "sub" / "b.wav", duration=0.05)

        df = pd.DataFrame({"audio_path": ["a.wav", "sub/b.wav"]})

        results = convert_audio_batch(df, source_dir, output_dir)

        assert len(results) == 2
        assert all(r["status"] == "success" for r in results)
        assert (output_dir / "a.wav").exists()
        assert (output_dir / "sub" / "b.wav").exists()

    def test_batch_skip_existing_when_not_overwrite(
        self,
        tmp_path: Path,
        make_wav_file: Callable[..., Path],
    ) -> None:
        source_dir = tmp_path / "src_dir"
        output_dir = tmp_path / "out_dir"

        make_wav_file(source_dir / "a.wav", duration=0.05)
        # Pre-create output file
        existing_out = output_dir / "a.wav"
        existing_out.parent.mkdir(parents=True, exist_ok=True)
        existing_out.write_text("existing content")

        df = pd.DataFrame({"audio_path": ["a.wav"]})

        results = convert_audio_batch(df, source_dir, output_dir, overwrite=False)

        assert len(results) == 1
        assert results[0]["status"] == "skipped"
        assert results[0]["error"] is None
        # Verify content was preserved
        assert existing_out.read_text() == "existing content"

    def test_batch_overwrites_when_overwrite_true(
        self,
        tmp_path: Path,
        make_wav_file: Callable[..., Path],
    ) -> None:
        source_dir = tmp_path / "src_dir"
        output_dir = tmp_path / "out_dir"

        make_wav_file(source_dir / "a.wav", duration=0.05)
        existing_out = output_dir / "a.wav"
        existing_out.parent.mkdir(parents=True, exist_ok=True)
        existing_out.write_text("old content")

        df = pd.DataFrame({"audio_path": ["a.wav"]})

        results = convert_audio_batch(df, source_dir, output_dir, overwrite=True)

        assert len(results) == 1
        assert results[0]["status"] == "success"
        info = sf.info(str(existing_out))
        assert info.samplerate == 16000

    def test_batch_path_traversal_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        source_dir = tmp_path / "src_dir"
        output_dir = tmp_path / "out_dir"
        source_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame({"audio_path": ["../outside.wav"]})

        results = convert_audio_batch(df, source_dir, output_dir)

        assert len(results) == 1
        assert results[0]["status"] == "failed"
        assert "Invalid path outside source directory" in results[0]["error"]

    def test_batch_handles_individual_failures_gracefully(
        self,
        tmp_path: Path,
        make_wav_file: Callable[..., Path],
    ) -> None:
        source_dir = tmp_path / "src_dir"
        output_dir = tmp_path / "out_dir"

        make_wav_file(source_dir / "valid.wav", duration=0.05)
        # invalid.wav does not exist

        df = pd.DataFrame({"audio_path": ["valid.wav", "missing.wav"]})

        results = convert_audio_batch(df, source_dir, output_dir)

        assert len(results) == 2
        assert results[0]["status"] == "success"
        assert results[1]["status"] == "failed"
        assert (output_dir / "valid.wav").exists()
