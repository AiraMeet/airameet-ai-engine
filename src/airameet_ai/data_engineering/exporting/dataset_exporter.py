"""UTF-8 metadata export utilities."""

from pathlib import Path

import pandas as pd


def export_metadata(frame: pd.DataFrame, path: str | Path) -> Path:
    """Export CSV in the notebook's UTF-8 BOM encoding and verify it exists."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8-sig")
    if not destination.is_file() or destination.stat().st_size == 0:
        raise OSError(f"Metadata export failed: {destination}")
    return destination
