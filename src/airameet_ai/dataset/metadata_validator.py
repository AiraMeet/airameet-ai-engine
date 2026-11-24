"""
Module: metadata_validator

Purpose:
    Reusable metadata inspection and validation functions
    for the AiraMeet ASR datasets.

Functions:
    inspect_metadata:
        Inspects dataset structure, columns, data types,
        and missing values.

    metadata_quality_summary:
        Generates a quality report including missing values,
        duplicates, invalid timestamps, and duration mismatches.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def inspect_metadata(df: pd.DataFrame, dataset_name: str) -> dict:
    """
    Inspect the structure and basic quality of metadata.

    Args:
        df: Dataset metadata as a Pandas DataFrame.
        dataset_name: Name of the dataset.

    Returns:
        Dictionary containing metadata inspection results.
    """
    return {
        "dataset": dataset_name,
        "rows": len(df),
        "columns": df.columns.tolist(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
    }


def metadata_quality_summary(df: pd.DataFrame, dataset_name: str) -> dict:
    """
    Generate a detailed metadata quality summary.

    Args:
        df: Dataset metadata as a Pandas DataFrame.
        dataset_name: Name of the dataset.

    Returns:
        Dictionary containing metadata quality metrics.
    """
    summary = {
        "Dataset": dataset_name,
        "Rows": len(df),
        "Missing Values": int(df.isna().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum()),
        "Duplicate Audio Paths": (
            int(df["audio_path"].duplicated().sum())
            if "audio_path" in df.columns
            else None
        ),
    }

    for column in ["start", "end", "duration"]:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")

            summary[f"{column} Invalid"] = int(
                (values.isna() | ~np.isfinite(values)).sum()
            )

            summary[f"{column} Negative"] = int((values < 0).sum())

    if all(column in df.columns for column in ["start", "end", "duration"]):
        start = pd.to_numeric(df["start"], errors="coerce")
        end = pd.to_numeric(df["end"], errors="coerce")
        duration = pd.to_numeric(df["duration"], errors="coerce")

        valid = np.isfinite(start) & np.isfinite(end) & np.isfinite(duration)

        mismatches = ((end - start - duration).abs() > 0.05) & valid

        summary["Duration Mismatches"] = int(mismatches.sum())

    return summary


def validate_audio_paths(
    df: pd.DataFrame,
    dataset_root: str | Path,
    path_column: str = "audio_path",
) -> pd.DataFrame:
    """
    Validate audio file paths against the actual dataset directory.

    Args:
        df:
            Dataset metadata as a Pandas DataFrame.

        dataset_root:
            Root directory containing the dataset audio folder.
            Relative paths in the metadata are resolved from here.

        path_column:
            Name of the column containing audio file paths.

    Returns:
        A copy of the metadata with three additional columns:
            - resolved_audio_path
            - audio_exists
            - path_status

    Path statuses:
        found:
            Audio file exists.

        missing_metadata_path:
            Path is empty or missing in metadata.

        invalid_path:
            Path is invalid or points outside the dataset root.

        file_not_found:
            Path is valid but the audio file does not exist.
    """

    if path_column not in df.columns:
        raise ValueError(f"Required column '{path_column}' was not found.")

    root = Path(dataset_root).resolve()

    if not root.exists() or not root.is_dir():
        raise NotADirectoryError(
            f"Dataset root does not exist or is not a directory: {root}"
        )

    results = []

    for raw_path in df[path_column]:
        # Check for missing or empty metadata paths
        if pd.isna(raw_path) or not str(raw_path).strip():
            results.append(
                {
                    "resolved_audio_path": None,
                    "audio_exists": False,
                    "path_status": "missing_metadata_path",
                }
            )
            continue

        try:
            relative_path = Path(str(raw_path).strip())

            # Absolute paths are not accepted in this dataset pipeline
            if relative_path.is_absolute():
                raise ValueError("Absolute paths are not allowed.")

            resolved_path = (root / relative_path).resolve()

            # Prevent paths from escaping the dataset directory
            if not resolved_path.is_relative_to(root):
                raise ValueError("Path points outside the dataset root.")

            if resolved_path.is_file():
                results.append(
                    {
                        "resolved_audio_path": str(resolved_path),
                        "audio_exists": True,
                        "path_status": "found",
                    }
                )
            else:
                results.append(
                    {
                        "resolved_audio_path": str(resolved_path),
                        "audio_exists": False,
                        "path_status": "file_not_found",
                    }
                )

        except (ValueError, OSError, RuntimeError):
            results.append(
                {
                    "resolved_audio_path": None,
                    "audio_exists": False,
                    "path_status": "invalid_path",
                }
            )

    # Preserve the original metadata and add validation results
    validated_df = df.copy()

    path_results = pd.DataFrame(results, index=df.index)

    return pd.concat([validated_df, path_results], axis=1)


def wada_snr(audio: np.ndarray) -> float:
    """
    Estimate speech SNR in dB using WADA-SNR.

    Args:
        audio: Mono audio waveform as a NumPy array.

    Returns:
        Estimated SNR in decibels.

    Raises:
        ValueError: If audio is empty, invalid, or silent.
    """

    eps = 1e-10

    # Convert input to a NumPy array
    audio = np.asarray(audio, dtype=np.float64)

    if audio.size == 0:
        raise ValueError("Audio signal is empty.")

    if not np.isfinite(audio).all():
        raise ValueError("Audio contains NaN or Inf values.")

    # Convert stereo to mono if necessary
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if audio.ndim != 1:
        raise ValueError("Audio must be mono or stereo.")

    # Reject silent audio
    peak = np.max(np.abs(audio))

    if peak < eps:
        raise ValueError("Cannot estimate SNR for silent audio.")

    # Normalize waveform for numerical stability
    audio = audio / peak

    abs_audio = np.maximum(np.abs(audio), eps)

    # Calculate waveform amplitude statistics
    v1 = max(abs_audio.mean(), eps)
    v2 = np.log(abs_audio).mean()
    v3 = np.log(v1) - v2

    # WADA-SNR reference lookup table
    db_vals = np.arange(-20, 101)

    g_vals = np.array(
        [
            0.40974774,
            0.40986926,
            0.40998566,
            0.40969089,
            0.40986186,
            0.40999006,
            0.41027138,
            0.41052627,
            0.41101024,
            0.41143264,
            0.41231718,
            0.41337272,
            0.41526426,
            0.4178192,
            0.42077252,
            0.42452799,
            0.42918886,
            0.43510373,
            0.44234195,
            0.45161485,
            0.46221153,
            0.47491647,
            0.48883809,
            0.50509236,
            0.52353709,
            0.54372088,
            0.56532427,
            0.58847532,
            0.61346212,
            0.63954496,
            0.66750818,
            0.69583724,
            0.72454762,
            0.75414799,
            0.78323148,
            0.81240985,
            0.84219775,
            0.87166406,
            0.90030504,
            0.92880418,
            0.95655449,
            0.9835349,
            1.01047155,
            1.0362095,
            1.06136425,
            1.08579312,
            1.1094819,
            1.13277995,
            1.15472826,
            1.17627308,
            1.19703503,
            1.21671694,
            1.23535898,
            1.25364313,
            1.27103891,
            1.28718029,
            1.30302865,
            1.31839527,
            1.33294817,
            1.34700935,
            1.3605727,
            1.37345513,
            1.38577122,
            1.39733504,
            1.40856397,
            1.41959619,
            1.42983624,
            1.43958467,
            1.44902176,
            1.45804831,
            1.46669568,
            1.47486938,
            1.48269965,
            1.49034339,
            1.49748214,
            1.50435106,
            1.51076426,
            1.51698915,
            1.5229097,
            1.528578,
            1.53389835,
            1.5391211,
            1.5439065,
            1.54858517,
            1.55310776,
            1.55744391,
            1.56164927,
            1.56566348,
            1.56938671,
            1.57307767,
            1.57654764,
            1.57980083,
            1.58304129,
            1.58602496,
            1.58880681,
            1.59162477,
            1.5941969,
            1.59693155,
            1.599446,
            1.60185011,
            1.60408668,
            1.60627134,
            1.60826199,
            1.61004547,
            1.61192472,
            1.61369656,
            1.61534074,
            1.61688905,
            1.61838916,
            1.61985374,
            1.62135878,
            1.62268119,
            1.62390423,
            1.62513143,
            1.62632463,
            1.6274027,
            1.62842767,
            1.62945532,
            1.6303307,
            1.63128026,
            1.63204102,
        ]
    )

    # Find the closest reference interval
    indices = np.where(g_vals < v3)[0]

    if len(indices) == 0:
        return float(db_vals[0])

    idx = indices[-1]

    if idx == len(db_vals) - 1:
        return float(db_vals[-1])

    # Linear interpolation
    snr = db_vals[idx] + ((v3 - g_vals[idx]) / (g_vals[idx + 1] - g_vals[idx])) * (
        db_vals[idx + 1] - db_vals[idx]
    )

    return float(snr)


def validate_metadata_schema(
    df: pd.DataFrame,
    dataset_name: str,
    required_columns: list[str] | None = None,
) -> dict:
    """
    Validate required metadata columns and basic field integrity.
    """

    if required_columns is None:
        required_columns = [
            "meeting",
            "speaker",
            "audio_path",
            "text",
            "start",
            "end",
            "duration",
        ]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    report = {
        "dataset": dataset_name,
        "total_records": len(df),
        "required_columns": required_columns,
        "missing_columns": missing_columns,
        "schema_valid": len(missing_columns) == 0,
    }

    if missing_columns:
        return report

    report["missing_values"] = {
        column: int(df[column].isna().sum()) for column in required_columns
    }

    report["empty_audio_paths"] = (
        int(df["audio_path"].astype("string").str.strip().eq("").sum())
        if "audio_path" in df.columns
        else 0
    )

    report["empty_text"] = (
        int(df["text"].astype("string").str.strip().eq("").sum())
        if "text" in df.columns
        else 0
    )

    if all(col in df.columns for col in ["start", "end", "duration"]):
        start = pd.to_numeric(df["start"], errors="coerce")
        end = pd.to_numeric(df["end"], errors="coerce")
        duration = pd.to_numeric(df["duration"], errors="coerce")

        valid_times = np.isfinite(start) & np.isfinite(end) & np.isfinite(duration)

        report["invalid_timestamps"] = int((~valid_times).sum())
        report["negative_timestamps"] = int(
            ((start < 0) | (end < 0) | (duration < 0)).sum()
        )
        report["end_before_start"] = int((end < start).sum())
        report["duration_mismatches"] = int(
            (((end - start - duration).abs() > 0.05) & valid_times).sum()
        )
    else:
        report["invalid_timestamps"] = 0
        report["negative_timestamps"] = 0
        report["end_before_start"] = 0
        report["duration_mismatches"] = 0

    report["duplicate_audio_paths"] = (
        int(df["audio_path"].duplicated().sum()) if "audio_path" in df.columns else 0
    )

    return report
