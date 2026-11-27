"""Input validation and normalization for the AI-DS-03 manifest builder.

The normalized record schema is ``utterance_id``, ``audio_path``, ``text``,
``recording_id``, ``speaker_id``, ``start``, ``end``, ``duration``, and
``source_dataset``. Identifier, path, and group fields are trimmed strings;
transcript text is preserved as supplied. Timing values are finite floats in
seconds. ``speaker_id`` and the paired ``start``/``end`` values may be null
when unavailable. No identifiers are generated. Grouped splitting by source
and recording ID does not inspect audio files or generate release artifacts.

Use :func:`validate_and_normalize_records` to validate a DataFrame, one row
mapping, or an iterable of row mappings. Its result contains normalized accepted records and
row-referenced rejections with stable reason counts.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

NORMALIZED_RECORD_FIELDS = (
    "utterance_id",
    "audio_path",
    "text",
    "recording_id",
    "speaker_id",
    "start",
    "end",
    "duration",
    "source_dataset",
)

REQUIRED_RECORD_FIELDS = (
    "utterance_id",
    "audio_path",
    "text",
    "recording_id",
    "duration",
    "source_dataset",
)

PLACEHOLDER_GROUP_IDS = frozenset({"train", "dev", "test", "unknown", ""})
DURATION_MATCH_TOLERANCE_SECONDS = 0.05
SPLIT_RANDOM_SEED = 42
TEST_GROUP_RATIO = 0.10
VALIDATION_OF_REMAINDER_RATIO = 1 / 9


@dataclass(frozen=True)
class RejectedRow:
    """A rejected input row and diagnostics identifying its source row.

    ``source_row`` retains the DataFrame index label (or iterable position).
    ``source_row_position`` disambiguates duplicate DataFrame index labels.
    """

    source_row: Any
    reasons: tuple[str, ...]
    utterance_id: str | None = None
    source_row_position: int | None = None


@dataclass(frozen=True)
class ValidationResult:
    """Accepted rows and rejection diagnostics.

    ``rejection_counts`` counts occurrences of each reason, not rejected rows;
    a row with multiple reasons contributes once to each applicable count.
    The number of rejected rows is ``len(rejected_rows)``.
    """

    accepted_records: list[dict[str, Any]]
    rejected_rows: list[RejectedRow]
    rejection_counts: dict[str, int]

    @property
    def total_rows(self) -> int:
        """Return the number of input rows represented in this result."""
        return len(self.accepted_records) + len(self.rejected_rows)


@dataclass(frozen=True)
class GroupSplitResult:
    """Grouped partitions and their actual utterance/group counts and ratios.

    Ratios are fractions of the complete input. ``GroupShuffleSplit`` targets
    group counts, so utterance ratios can differ from the requested targets.
    """

    train_records: list[dict[str, Any]]
    validation_records: list[dict[str, Any]]
    test_records: list[dict[str, Any]]
    utterance_counts: dict[str, int]
    group_counts: dict[str, int]
    utterance_ratios: dict[str, float]
    group_ratios: dict[str, float]


def _rows_with_references(
    records: pd.DataFrame | Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> list[tuple[Any, int, Mapping[str, Any]]]:
    """Materialize rows with their original label and unique input position."""
    if isinstance(records, pd.DataFrame):
        return [
            (index, position, row)
            for position, (index, row) in enumerate(
                zip(
                    records.index.tolist(),
                    records.to_dict(orient="records"),
                    strict=True,
                )
            )
        ]

    if isinstance(records, Mapping):
        return [(0, 0, records)]

    result: list[tuple[Any, int, Mapping[str, Any]]] = []
    for position, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise TypeError(f"row at position {position} is not a mapping")
        result.append((position, position, row))
    return result


def _normalized_string(
    row: Mapping[str, Any], field: str, reasons: list[str], *, required: bool
) -> str | None:
    """Validate a string field and return its whitespace-trimmed value."""
    if field not in row:
        if required:
            reasons.append(f"missing_field:{field}")
        return None
    value = row[field]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        if required:
            reasons.append(f"empty_field:{field}")
        return None
    if not isinstance(value, str):
        reasons.append(f"invalid_type:{field}")
        return None
    normalized = value.strip()
    if not normalized and required:
        reasons.append(f"empty_field:{field}")
        return None
    return normalized or None


def _finite_number(
    row: Mapping[str, Any], field: str, reasons: list[str], *, required: bool
) -> float | None:
    """Parse supported numeric values explicitly as finite floats.

    Only ``numbers.Real`` values and strings are supported. ``Decimal`` is not
    a ``Real`` and is intentionally rejected rather than implicitly broadening
    the accepted input types.
    """
    if field not in row:
        if required:
            reasons.append(f"missing_field:{field}")
        return None
    value = row[field]
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            reasons.append(f"empty_field:{field}")
        return None
    if isinstance(value, bool) or not isinstance(value, (Real, str)):
        reasons.append(f"invalid_number:{field}")
        return None
    try:
        number = float(value.strip() if isinstance(value, str) else value)
    except (OverflowError, ValueError):
        reasons.append(f"invalid_number:{field}")
        return None
    if not math.isfinite(number):
        reasons.append(f"non_finite:{field}")
        return None
    return number


def validate_and_normalize_records(
    records: pd.DataFrame | Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> ValidationResult:
    """Validate source-neutral records and normalize valid values.

    Required inputs are ``utterance_id``, ``audio_path``, ``text``,
    ``recording_id``, ``duration``, and ``source_dataset``. ``speaker_id`` may
    be absent or null. ``start`` and ``end`` may both be absent/null; if one
    is supplied, both must be finite numeric values with ``end >= start``.
    Numeric strings are parsed explicitly; invalid and non-finite values are
    rejected. Durations must be strictly positive. Every row sharing a
    duplicate normalized utterance ID is rejected.

    DataFrame index labels are retained as ``source_row`` references. For
    other iterables, the zero-based input position is used only for
    diagnostics and is never substituted for an utterance ID.
    """
    source_rows = _rows_with_references(records)
    ids_by_row: list[str | None] = []
    id_counts: Counter[str] = Counter()

    for _, _, row in source_rows:
        raw_id = row.get("utterance_id")
        normalized_id = raw_id.strip() if isinstance(raw_id, str) else None
        if normalized_id:
            id_counts[normalized_id] += 1
        ids_by_row.append(normalized_id or None)

    accepted: list[dict[str, Any]] = []
    rejected: list[RejectedRow] = []
    counts: Counter[str] = Counter()

    for (source_row, source_row_position, row), candidate_id in zip(
        source_rows, ids_by_row, strict=True
    ):
        reasons: list[str] = []
        utterance_id = _normalized_string(
            row, "utterance_id", reasons, required=True
        )
        audio_path = _normalized_string(row, "audio_path", reasons, required=True)
        text = row.get("text")
        if "text" not in row:
            reasons.append("missing_field:text")
            text = None
        elif not isinstance(text, str):
            reasons.append("invalid_type:text")
            text = None
        elif not text.strip():
            reasons.append("empty_field:text")

        recording_id = _normalized_string(
            row, "recording_id", reasons, required=True
        )
        if recording_id is not None and recording_id.casefold() in PLACEHOLDER_GROUP_IDS:
            reasons.append("placeholder_group_id:recording_id")

        source_dataset = _normalized_string(
            row, "source_dataset", reasons, required=True
        )
        speaker_is_missing = (
            "speaker_id" not in row or row["speaker_id"] is None
        )
        if speaker_is_missing:
            speaker_id = None
        else:
            speaker_id = _normalized_string(
                row, "speaker_id", reasons, required=False
            )

        duration = _finite_number(row, "duration", reasons, required=True)
        if duration is not None and duration <= 0:
            reasons.append("non_positive_duration:duration")

        start_present = "start" in row and row["start"] is not None
        end_present = "end" in row and row["end"] is not None
        start: float | None = None
        end: float | None = None
        if start_present != end_present:
            reasons.append("incomplete_time_pair:start,end")
        elif start_present and end_present:
            start = _finite_number(row, "start", reasons, required=True)
            end = _finite_number(row, "end", reasons, required=True)
            if start is not None and start < 0:
                reasons.append("negative_time:start")
            if end is not None and end < 0:
                reasons.append("negative_time:end")
            if start is not None and end is not None and end < start:
                reasons.append("invalid_time_order:start,end")

            # Existing metadata validation uses a 0.05-second tolerance for
            # duration mismatches; this also absorbs floating-point rounding.
            expected_duration = (
                end - start if start is not None and end is not None else None
            )
            if (
                start is not None
                and end is not None
                and start >= 0
                and end >= start
                and duration is not None
                and expected_duration is not None
                and (
                    duration
                    < expected_duration - DURATION_MATCH_TOLERANCE_SECONDS
                    or duration
                    > expected_duration + DURATION_MATCH_TOLERANCE_SECONDS
                )
            ):
                reasons.append("duration_mismatch")

        if candidate_id is not None and id_counts[candidate_id] > 1:
            reasons.append("duplicate_utterance_id:utterance_id")

        if reasons:
            stable_reasons = tuple(dict.fromkeys(reasons))
            rejected.append(
                RejectedRow(
                    source_row=source_row,
                    reasons=stable_reasons,
                    utterance_id=utterance_id,
                    source_row_position=source_row_position,
                )
            )
            counts.update(stable_reasons)
            continue

        accepted.append(
            {
                "utterance_id": utterance_id,
                "audio_path": audio_path,
                "text": text,
                "recording_id": recording_id,
                "speaker_id": speaker_id,
                "start": start,
                "end": end,
                "duration": duration,
                "source_dataset": source_dataset,
            }
        )

    return ValidationResult(
        accepted_records=accepted,
        rejected_rows=rejected,
        rejection_counts=dict(sorted(counts.items())),
    )


def split_grouped_records(
    records: pd.DataFrame | Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> GroupSplitResult:
    """Split normalized records by ``(source_dataset, recording_id)`` groups.

    Test groups are selected with a 10% holdout. Validation is selected as
    ``1/9`` of the remaining groups, targeting 10% of the original group
    count. Both grouped shuffles use seed 42. Returned rows retain their input
    order within each partition; actual row and group ratios are reported.

    At least three distinct composite groups are required. Missing, blank, or
    placeholder group components and missing or duplicate utterance IDs raise
    ``ValueError``. This function does not create identifiers or split rows
    within a group.
    """
    source_rows = _rows_with_references(records)
    rows = [row for _, _, row in source_rows]
    if not rows:
        raise ValueError("At least three recording groups are required")

    group_key_to_label: dict[tuple[str, str], int] = {}
    group_labels: list[int] = []
    utterance_ids: list[str] = []
    seen_utterance_ids: set[str] = set()

    for position, row in enumerate(rows):
        source_value = row.get("source_dataset")
        recording_value = row.get("recording_id")
        group_components = (
            ("source_dataset", source_value),
            ("recording_id", recording_value),
        )
        normalized_components: list[str] = []
        for field, value in group_components:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"row {position}: {field} is missing or blank; "
                    "a genuine composite group key is required"
                )
            normalized = value.strip()
            if normalized.casefold() in PLACEHOLDER_GROUP_IDS:
                raise ValueError(
                    f"row {position}: {field} contains placeholder value "
                    f"{normalized!r}"
                )
            normalized_components.append(normalized)

        group_key = (normalized_components[0], normalized_components[1])
        if group_key not in group_key_to_label:
            # First-seen labels are deterministic for a fixed input order and
            # are one-dimensional integers, as required by sklearn.
            group_key_to_label[group_key] = len(group_key_to_label)
        group_labels.append(group_key_to_label[group_key])

        utterance_id = row.get("utterance_id")
        if not isinstance(utterance_id, str) or not utterance_id.strip():
            raise ValueError(
                f"row {position}: utterance_id is missing or blank"
            )
        utterance_id = utterance_id.strip()
        if utterance_id in seen_utterance_ids:
            raise ValueError(
                f"row {position}: duplicate utterance_id {utterance_id!r}"
            )
        seen_utterance_ids.add(utterance_id)
        utterance_ids.append(utterance_id)

    total_groups = len(group_key_to_label)
    if total_groups < 3:
        raise ValueError(
            f"At least three recording groups are required; found {total_groups}"
        )

    groups = np.asarray(group_labels, dtype=np.int64)
    all_indices = list(range(len(rows)))
    train_validation_indices, test_indices = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_GROUP_RATIO,
            random_state=SPLIT_RANDOM_SEED,
        ).split(all_indices, groups=groups)
    )

    remaining_groups = groups[train_validation_indices]
    train_local_indices, validation_local_indices = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=VALIDATION_OF_REMAINDER_RATIO,
            random_state=SPLIT_RANDOM_SEED,
        ).split(train_validation_indices, groups=remaining_groups)
    )

    partition_indices = {
        "train": sorted(train_validation_indices[train_local_indices].tolist()),
        "validation": sorted(
            train_validation_indices[validation_local_indices].tolist()
        ),
        "test": sorted(test_indices.tolist()),
    }
    assert all(partition_indices.values()), "Grouped splitting produced an empty split"
    partitions = {
        name: [rows[index] for index in indices]
        for name, indices in partition_indices.items()
    }
    partition_groups = {
        name: {
            (str(rows[index]["source_dataset"]).strip(),
             str(rows[index]["recording_id"]).strip())
            for index in indices
        }
        for name, indices in partition_indices.items()
    }

    split_names = ("train", "validation", "test")
    for left_index, left_name in enumerate(split_names):
        for right_name in split_names[left_index + 1 :]:
            assert partition_groups[left_name].isdisjoint(
                partition_groups[right_name]
            ), f"Composite recording group leaked between {left_name} and {right_name}"
            left_ids = {row["utterance_id"].strip() for row in partitions[left_name]}
            right_ids = {
                row["utterance_id"].strip() for row in partitions[right_name]
            }
            assert left_ids.isdisjoint(right_ids), (
                f"Utterance IDs overlap between {left_name} and {right_name}"
            )

    output_ids = [
        row["utterance_id"].strip()
        for name in split_names
        for row in partitions[name]
    ]
    assert Counter(output_ids) == Counter(utterance_ids), (
        "Grouped split lost or duplicated input utterance records"
    )

    utterance_counts = {
        name: len(partitions[name]) for name in split_names
    }
    group_counts = {
        name: len(partition_groups[name]) for name in split_names
    }
    utterance_ratios = {
        name: utterance_counts[name] / len(rows) for name in split_names
    }
    group_ratios = {
        name: group_counts[name] / total_groups for name in split_names
    }

    return GroupSplitResult(
        train_records=partitions["train"],
        validation_records=partitions["validation"],
        test_records=partitions["test"],
        utterance_counts=utterance_counts,
        group_counts=group_counts,
        utterance_ratios=utterance_ratios,
        group_ratios=group_ratios,
    )


__all__ = [
    "NORMALIZED_RECORD_FIELDS",
    "PLACEHOLDER_GROUP_IDS",
    "DURATION_MATCH_TOLERANCE_SECONDS",
    "SPLIT_RANDOM_SEED",
    "GroupSplitResult",
    "RejectedRow",
    "REQUIRED_RECORD_FIELDS",
    "ValidationResult",
    "validate_and_normalize_records",
    "split_grouped_records",
]
