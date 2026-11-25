"""Full-file audio inspection used by dataset quality audits."""

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from airameet_ai.data_engineering.quality.snr import wada_snr


def audit_audio_files(
    metadata: pd.DataFrame,
    audio_root: str | Path,
    *,
    audio_paths: pd.Series | None = None,
    allowed_roots: tuple[str | Path, ...] = (),
) -> pd.DataFrame:
    """Inspect audio properties, signal integrity, clipping, and WADA-SNR.

    ``audio_paths`` can provide resolved paths (for example, preferring a
    standardized processed copy while retaining ``audio_path`` in the report).
    Errors are recorded per file so one corrupt recording does not hide later
    audit results.
    """
    root = Path(audio_root).resolve()
    permitted_roots = (root, *(Path(path).resolve() for path in allowed_roots))
    paths = audio_paths if audio_paths is not None else metadata["audio_path"]
    results: list[dict[str, object]] = []

    for relative_path, raw_path in zip(metadata["audio_path"], paths, strict=True):
        result: dict[str, object] = {
            "audio_path": relative_path,
            "source_path": None,
            "sample_rate": None,
            "channels": None,
            "format": None,
            "subtype": None,
            "duration": None,
            "readable": False,
            "has_nan_inf": None,
            "peak_amplitude": None,
            "is_silent": None,
            "clipping_samples_gt_099": None,
            "clipping_percentage_gt_099": None,
            "clipped_samples": None,
            "total_samples": None,
            "clipped_percentage": None,
            "snr_db": None,
            "snr_status": "failed",
            "error": None,
        }

        try:
            if pd.isna(raw_path):
                raise FileNotFoundError("No audio path resolved for metadata row")
            candidate = Path(str(raw_path))
            audio_path = (
                candidate.resolve()
                if candidate.is_absolute()
                else (root / candidate).resolve()
            )
            if not any(audio_path.is_relative_to(path) for path in permitted_roots):
                raise ValueError("Audio path points outside the dataset root")
            result["source_path"] = str(audio_path)

            info = sf.info(audio_path)
            result.update(
                sample_rate=info.samplerate,
                channels=info.channels,
                format=info.format,
                subtype=info.subtype,
                duration=info.duration,
            )

            audio, _ = sf.read(audio_path, dtype="float32")
            finite = bool(np.isfinite(audio).all())
            result["readable"] = True
            result["has_nan_inf"] = not finite
            result["total_samples"] = int(audio.size)

            if finite and audio.size:
                peak = float(np.max(np.abs(audio)))
                clipping_samples_gt_099 = int(np.sum(np.abs(audio) > 0.99))
                clipped_samples = int(np.sum(np.abs(audio) >= 0.999))
                result.update(
                    peak_amplitude=peak,
                    is_silent=peak < 1e-6,
                    clipping_samples_gt_099=clipping_samples_gt_099,
                    clipping_percentage_gt_099=(
                        clipping_samples_gt_099 / audio.size * 100
                    ),
                    clipped_samples=clipped_samples,
                    clipped_percentage=clipped_samples / audio.size * 100,
                )
                mono_audio = audio.mean(axis=1) if audio.ndim == 2 else audio
                try:
                    result["snr_db"] = wada_snr(mono_audio)
                    result["snr_status"] = "success"
                except ValueError as error:
                    result["error"] = str(error)
            elif audio.size == 0:
                result["error"] = "Audio file is empty"

        except (OSError, RuntimeError, ValueError) as error:
            result["error"] = str(error)

        results.append(result)

    return pd.DataFrame(results)
