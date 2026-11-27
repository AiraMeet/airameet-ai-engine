"""Statistics derived from the final manifest partitions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..metadata.recording_metadata import recording_group


def _duration_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values = sorted(float(row["duration"]) for row in rows)
    if not values:
        return {"count": 0, "total_seconds": 0.0, "hours": 0.0}
    return {
        "count": len(values),
        "total_seconds": float(sum(values)),
        "hours": float(sum(values) / 3600),
        "min": values[0],
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.5)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.9)),
        "q95": float(np.quantile(values, 0.95)),
        "max": values[-1],
        "mean": float(sum(values) / len(values)),
    }


def _split_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    sources = Counter(str(row["source_dataset"]) for row in rows)
    speakers = Counter(
        str(row.get("speaker_id") or "<missing>") for row in rows
    )
    tokens = [token for row in rows for token in str(row["text"]).split()]
    return {
        "utterances": len(rows),
        "recording_groups": len({recording_group(row) for row in rows}),
        "duration": _duration_summary(rows),
        "vocabulary": {
            "token_count": len(tokens),
            "unique_token_count": len(set(tokens)),
            "tokenization": "whitespace; case-sensitive; punctuation retained",
        },
        "source_dataset_distribution": dict(sorted(sources.items())),
        "speaker_statistics": dict(sorted(speakers.items())),
    }


def build_statistics(
    partitions: Mapping[str, Sequence[dict[str, Any]]],
    *,
    rejected_by_source: Mapping[str, Mapping[str, int]],
    excluded_audio_by_source: Mapping[str, int],
) -> dict[str, Any]:
    """Calculate release-wide and per-split metrics from written records."""
    summaries = {name: _split_summary(rows) for name, rows in partitions.items()}
    total_utterances = sum(item["utterances"] for item in summaries.values())
    total_groups = sum(item["recording_groups"] for item in summaries.values())
    proportions = {
        name: {
            "utterance_proportion": (
                item["utterances"] / total_utterances if total_utterances else 0.0
            ),
            "recording_group_proportion": (
                item["recording_groups"] / total_groups if total_groups else 0.0
            ),
        }
        for name, item in summaries.items()
    }
    overall_rows = [row for rows in partitions.values() for row in rows]
    missing = {
        field: sum(
            1
            for row in overall_rows
            if row.get(field) is None or (isinstance(row.get(field), str) and not row[field].strip())
        )
        for field in ("utterance_id", "audio_path", "text", "recording_id", "duration", "speaker_id")
    }
    return {
        "total_utterances": total_utterances,
        "total_recording_groups": total_groups,
        "splits": {
            name: {**summary, **proportions[name]}
            for name, summary in summaries.items()
        },
        "duration": _duration_summary(overall_rows),
        "vocabulary": _split_summary(overall_rows)["vocabulary"],
        "source_dataset_distribution": dict(
            sorted(Counter(str(row["source_dataset"]) for row in overall_rows).items())
        ),
        "speaker_statistics": dict(
            sorted(Counter(str(row.get("speaker_id") or "<missing>") for row in overall_rows).items())
        ),
        "missing_or_invalid_metadata": {
            "missing_field_counts_in_release": missing,
            "rejected_rows_by_source_and_reason": {
                source: dict(sorted(counts.items()))
                for source, counts in sorted(rejected_by_source.items())
            },
            "accepted_rows_excluded_for_unresolved_audio_by_source": dict(
                sorted(excluded_audio_by_source.items())
            ),
        },
    }
