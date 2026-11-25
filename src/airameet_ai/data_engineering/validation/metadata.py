"""Reusable metadata inspection and validation for ASR datasets."""

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
