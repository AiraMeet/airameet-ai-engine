"""Unit tests for WADA-SNR estimation."""

from __future__ import annotations

import numpy as np
import pytest

from airameet_ai.data_engineering.quality.snr import wada_snr


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

