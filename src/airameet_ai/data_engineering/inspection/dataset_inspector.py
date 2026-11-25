"""Input helpers for the dataset engineering workflow."""

from pathlib import Path

import pandas as pd


def load_metadata(
    path: str | Path, *, sheet_name: str | int = "Metadata"
) -> pd.DataFrame:
    """Load CSV or Excel metadata without changing the source file."""
    metadata_path = Path(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    suffix = metadata_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(metadata_path, encoding="utf-8-sig")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(metadata_path, sheet_name=sheet_name)
    raise ValueError(f"Unsupported metadata format: {metadata_path.suffix}")
