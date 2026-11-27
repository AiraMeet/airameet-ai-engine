"""Recalculate AI-DS-03 split statistics from the release JSONL manifests.

Run from the project root with ``python scripts/generate_stats.py``. Duration
and vocabulary calculations intentionally match the AI-DS-03 notebook. Release
scope metadata already recorded in ``reports/dataset_stats.json`` is retained,
because source blockers and validation exclusions cannot be recovered from
the manifests alone.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataset.manifest_builder import NORMALIZED_RECORD_FIELDS


SPLIT_FILES = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "test": "test.jsonl",
}
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"
STATS_PATH = REPO_ROOT / "reports" / "dataset_stats.json"


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records and check that each row follows the manifest schema."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path.relative_to(REPO_ROOT)}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a JSON object in {path.relative_to(REPO_ROOT)}:{line_number}"
                )
            missing = set(NORMALIZED_RECORD_FIELDS).difference(row)
            if missing:
                raise ValueError(
                    f"Missing normalized fields in {path.relative_to(REPO_ROOT)}: "
                    f"{', '.join(sorted(missing))}"
                )
            rows.append(row)
    return rows


def duration_stats(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Return duration statistics using the notebook's linear quantiles."""
    values = sorted(float(row["duration"]) for row in rows)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Manifest durations must be finite positive seconds")
    if not values:
        return {"count": 0, "hours": 0.0}

    def quantile(q: float) -> float:
        index = (len(values) - 1) * q
        low, high = math.floor(index), math.ceil(index)
        return (
            values[low]
            if low == high
            else values[low] * (high - index) + values[high] * (index - low)
        )

    return {
        "count": len(values),
        "total_seconds": sum(values),
        "hours": sum(values) / 3600,
        "min": values[0],
        "q25": quantile(0.25),
        "median": quantile(0.5),
        "q75": quantile(0.75),
        "q90": quantile(0.9),
        "q95": quantile(0.95),
        "max": values[-1],
        "mean": sum(values) / len(values),
    }


def split_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate the same utterance, group, duration, and vocabulary metrics as the notebook."""
    for row in rows:
        if not isinstance(row["text"], str):
            raise ValueError("Manifest transcript text must be a string")
        for field in ("utterance_id", "recording_id", "source_dataset"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"Manifest field {field!r} must be a non-empty string")
    tokens = [token for row in rows for token in row["text"].split()]
    return {
        "utterances": len(rows),
        "groups": len(
            {(row["source_dataset"], row["recording_id"]) for row in rows}
        ),
        "duration": duration_stats(rows),
        "vocabulary": {
            "token_count": len(tokens),
            "unique_token_count": len(set(tokens)),
            "tokenization": "whitespace; case-sensitive; punctuation retained",
        },
        "sources": dict(sorted(Counter(row["source_dataset"] for row in rows).items())),
        "speakers": dict(
            sorted(
                Counter(row["speaker_id"] or "<missing>" for row in rows).items()
            )
        ),
    }


def add_actual_ratios(
    splits: dict[str, list[dict[str, Any]]],
    statistics: dict[str, dict[str, Any]],
) -> None:
    """Attach actual row and composite-group fractions across all manifests."""
    total_rows = sum(map(len, splits.values()))
    all_groups = {
        (row["source_dataset"], row["recording_id"])
        for rows in splits.values()
        for row in rows
    }
    total_groups = len(all_groups)
    if not total_rows or not total_groups:
        raise ValueError("The three manifests contain no records or recording groups")

    for name, rows in splits.items():
        groups = {
            (row["source_dataset"], row["recording_id"]) for row in rows
        }
        statistics[name]["utterance_ratio"] = len(rows) / total_rows
        statistics[name]["group_ratio"] = len(groups) / total_groups


def assert_no_leakage(splits: dict[str, list[dict[str, Any]]]) -> None:
    """Reject duplicate utterances or composite recording groups across splits."""
    group_sets = {
        name: {(row["source_dataset"], row["recording_id"]) for row in rows}
        for name, rows in splits.items()
    }
    id_sets = {
        name: {row["utterance_id"] for row in rows} for name, rows in splits.items()
    }
    names = list(splits)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            if not group_sets[left].isdisjoint(group_sets[right]):
                raise ValueError(f"Composite recording group leakage: {left}/{right}")
            if not id_sets[left].isdisjoint(id_sets[right]):
                raise ValueError(f"Utterance ID leakage: {left}/{right}")


def main() -> None:
    """Recompute split metrics and update the existing report deterministically."""
    missing_paths = [
        path
        for path in [*(MANIFEST_DIR / name for name in SPLIT_FILES.values()), STATS_PATH]
        if not path.is_file()
    ]
    if missing_paths:
        missing = ", ".join(str(path.relative_to(REPO_ROOT)) for path in missing_paths)
        raise FileNotFoundError(f"Required AI-DS-03 input/report file(s) missing: {missing}")

    splits = {
        split_name: load_manifest(MANIFEST_DIR / filename)
        for split_name, filename in SPLIT_FILES.items()
    }
    assert_no_leakage(splits)

    statistics = {name: split_statistics(rows) for name, rows in splits.items()}
    add_actual_ratios(splits, statistics)

    report = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    report["splits"] = statistics
    report_bytes = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    STATS_PATH.write_bytes(report_bytes)

    print(f"Updated {STATS_PATH.relative_to(REPO_ROOT)} from the three JSONL manifests.")
    for name, values in statistics.items():
        print(
            f"{name}: {values['utterances']} utterances, {values['groups']} groups, "
            f"{values['duration']['hours']:.6f} hours, "
            f"{values['vocabulary']['token_count']} tokens, "
            f"{values['vocabulary']['unique_token_count']} unique tokens"
        )


if __name__ == "__main__":
    main()
