"""
Module: audio_converter

Purpose:
    Convert audio files to the AiraMeet standard:
    WAV, Mono, 16 kHz, PCM 16-bit.
"""

from pathlib import Path
import math

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def convert_audio_file(
    source_path: str | Path,
    output_path: str | Path,
    target_sample_rate: int = 16000,
) -> dict:
    """
    Convert one audio file without modifying the original.

    Returns a dictionary describing the conversion result.
    """

    source_path = Path(source_path)
    output_path = Path(output_path)

    try:
        audio, sample_rate = sf.read(
            str(source_path),
            dtype="float32",
            always_2d=True,
        )

        if audio.size == 0:
            raise ValueError("Audio file is empty.")

        if not np.isfinite(audio).all():
            raise ValueError("Audio contains NaN or Inf.")

        # Convert stereo or multi-channel audio to mono.
        audio = audio.mean(axis=1)

        # Resample only when the sample rate is different.
        if sample_rate != target_sample_rate:
            divisor = math.gcd(sample_rate, target_sample_rate)

            audio = resample_poly(
                audio,
                target_sample_rate // divisor,
                sample_rate // divisor,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as WAV, mono, 16 kHz, PCM 16-bit.
        sf.write(
            str(output_path),
            audio,
            target_sample_rate,
            format="WAV",
            subtype="PCM_16",
        )

        return {
            "source_path": str(source_path),
            "output_path": str(output_path),
            "status": "success",
            "error": None,
        }

    except Exception as error:
        return {
            "source_path": str(source_path),
            "output_path": str(output_path),
            "status": "failed",
            "error": str(error),
        }
def convert_audio_batch(
    audio_df,
    source_root: str | Path,
    output_root: str | Path,
    path_column: str = "audio_path",
    overwrite: bool = False,
) -> list[dict]:
    """
    Convert multiple audio files while preserving relative paths.

    Existing output files are skipped unless overwrite=True.
    Each file is processed independently.
    """

    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()

    results = []

    for relative_path in audio_df[path_column]:

        relative_path = Path(str(relative_path))

        source_path = (source_root / relative_path).resolve()
        output_path = (output_root / relative_path).resolve()

        if not source_path.is_relative_to(source_root):
            results.append({
                "source_path": str(source_path),
                "output_path": str(output_path),
                "status": "failed",
                "error": "Invalid path outside source directory.",
            })
            continue

        if output_path.exists() and not overwrite:
            results.append({
                "source_path": str(source_path),
                "output_path": str(output_path),
                "status": "skipped",
                "error": None,
            })
            continue

        result = convert_audio_file(
            source_path=source_path,
            output_path=output_path,
        )

        results.append(result)

    return results