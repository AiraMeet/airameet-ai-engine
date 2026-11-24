"""Shared fixtures for AiraMeet AI Engine test suite."""

from __future__ import annotations

import math
import struct
import wave
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def make_wav_file() -> Callable[..., Path]:
    """Factory fixture to create tiny synthetic 16-bit PCM WAV files."""

    def _create_wav(
        path: Path | str,
        sample_rate: int = 16000,
        duration: float = 0.1,
        channels: int = 1,
        freq: float = 440.0,
    ) -> Path:
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        num_samples = int(sample_rate * duration)

        with wave.open(str(target_path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(sample_rate)

            frames = bytearray()
            for i in range(num_samples):
                val = int(
                    32767.0 * 0.5 * math.sin(2.0 * math.pi * freq * (i / sample_rate))
                )
                for _ in range(channels):
                    frames.extend(struct.pack("<h", val))

            wf.writeframes(frames)

        return target_path

    return _create_wav


@pytest.fixture
def sample_metadata_df() -> pd.DataFrame:
    """Fixture providing a valid baseline metadata DataFrame."""
    return pd.DataFrame(
        {
            "meeting": ["MTG_001", "MTG_001", "MTG_002"],
            "speaker": ["SPK_A", "SPK_B", "SPK_A"],
            "audio_path": [
                "audio/mtg1_01.wav",
                "audio/mtg1_02.wav",
                "audio/mtg2_01.wav",
            ],
            "text": [
                "Hello everyone welcome",
                "Let us review the roadmap",
                "Decisions approved for release",
            ],
            "start": [0.0, 5.0, 10.0],
            "end": [4.5, 9.2, 14.5],
            "duration": [4.5, 4.2, 4.5],
        }
    )


@pytest.fixture
def synthetic_dataset_dir(tmp_path: Path, make_wav_file: Callable[..., Path]) -> Path:
    """Fixture providing a mock dataset directory with synthetic audio files."""
    dataset_dir = tmp_path / "test_dataset"
    audio_dir = dataset_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    make_wav_file(audio_dir / "mtg1_01.wav", sample_rate=16000, duration=0.05)
    make_wav_file(audio_dir / "mtg1_02.wav", sample_rate=16000, duration=0.05)
    make_wav_file(audio_dir / "mtg2_01.wav", sample_rate=16000, duration=0.05)

    return dataset_dir
