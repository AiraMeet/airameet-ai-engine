"""Tests for AI-DS-03 input validation and normalization."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from dataset.manifest_builder import (
    GroupSplitResult,
    NORMALIZED_RECORD_FIELDS,
    RejectedRow,
    split_grouped_records,
    validate_and_normalize_records,
)


@pytest.fixture
def valid_row() -> dict[str, object]:
    return {
        "utterance_id": "utt-001",
        "audio_path": "audio/session/utt-001.wav",
        "text": "  hello world  ",
        "recording_id": "session-01",
        "speaker_id": "speaker-01",
        "start": "1.25",
        "end": 3.75,
        "duration": "2.5",
        "source_dataset": "Example",
    }


def make_grouped_records(
    group_specs: list[tuple[str, str, int]],
) -> list[dict[str, object]]:
    """Build normalized synthetic rows from (source, recording, size) specs."""
    rows: list[dict[str, object]] = []
    for source_dataset, recording_id, size in group_specs:
        for utterance_number in range(size):
            rows.append(
                {
                    "utterance_id": (
                        f"{source_dataset}-{recording_id}-{utterance_number}"
                    ),
                    "audio_path": f"audio/{source_dataset}/{recording_id}/{utterance_number}.wav",
                    "text": "synthetic transcript",
                    "recording_id": recording_id,
                    "speaker_id": None,
                    "start": 0.0,
                    "end": 1.0,
                    "duration": 1.0,
                    "source_dataset": source_dataset,
                }
            )
    return rows


def test_valid_mapping_is_normalized_without_changing_transcript(
    valid_row: dict[str, object],
) -> None:
    result = validate_and_normalize_records([valid_row])

    assert result.accepted_records == [
        {
            "utterance_id": "utt-001",
            "audio_path": "audio/session/utt-001.wav",
            "text": "  hello world  ",
            "recording_id": "session-01",
            "speaker_id": "speaker-01",
            "start": 1.25,
            "end": 3.75,
            "duration": 2.5,
            "source_dataset": "Example",
        }
    ]
    assert tuple(result.accepted_records[0]) == NORMALIZED_RECORD_FIELDS
    assert result.rejected_rows == []
    assert result.rejection_counts == {}


def test_single_mapping_is_accepted_as_one_row(
    valid_row: dict[str, object],
) -> None:
    result = validate_and_normalize_records(valid_row)

    assert result.total_rows == 1
    assert result.accepted_records[0]["utterance_id"] == "utt-001"


def test_dataframe_index_is_preserved_as_diagnostic_reference(
    valid_row: dict[str, object],
) -> None:
    invalid = deepcopy(valid_row)
    invalid["text"] = "   "
    invalid["utterance_id"] = "utt-002"
    frame = pd.DataFrame([valid_row, invalid], index=["source-a", "source-b"])

    result = validate_and_normalize_records(frame)

    assert result.accepted_records[0]["utterance_id"] == "utt-001"
    assert result.rejected_rows == [
        RejectedRow(
            source_row="source-b",
            reasons=("empty_field:text",),
            utterance_id="utt-002",
            source_row_position=1,
        )
    ]


@pytest.mark.parametrize(
    ("field", "expected_reason"),
    [
        ("utterance_id", "missing_field:utterance_id"),
        ("audio_path", "missing_field:audio_path"),
        ("text", "missing_field:text"),
        ("recording_id", "missing_field:recording_id"),
        ("duration", "missing_field:duration"),
        ("source_dataset", "missing_field:source_dataset"),
    ],
)
def test_missing_required_fields_are_reported(
    valid_row: dict[str, object], field: str, expected_reason: str
) -> None:
    del valid_row[field]

    result = validate_and_normalize_records([valid_row])

    assert result.accepted_records == []
    assert expected_reason in result.rejected_rows[0].reasons
    assert result.rejection_counts[expected_reason] == 1
    assert result.rejected_rows[0].source_row == 0


@pytest.mark.parametrize("group_id", ["train", " DEV ", "Test", "unknown"])
def test_placeholder_group_ids_are_rejected_case_insensitively(
    valid_row: dict[str, object], group_id: str
) -> None:
    valid_row["recording_id"] = group_id

    result = validate_and_normalize_records([valid_row])

    assert result.accepted_records == []
    assert result.rejection_counts == {"placeholder_group_id:recording_id": 1}


def test_blank_group_id_is_rejected(
    valid_row: dict[str, object],
) -> None:
    valid_row["recording_id"] = "   "

    result = validate_and_normalize_records([valid_row])

    assert result.accepted_records == []
    assert result.rejection_counts == {"empty_field:recording_id": 1}


@pytest.mark.parametrize(
    ("duration", "expected_reason"),
    [
        ("not-a-number", "invalid_number:duration"),
        (float("nan"), "non_finite:duration"),
        (float("inf"), "non_finite:duration"),
        (-0.1, "non_positive_duration:duration"),
        (0, "non_positive_duration:duration"),
    ],
)
def test_invalid_durations_are_rejected(
    valid_row: dict[str, object], duration: object, expected_reason: str
) -> None:
    valid_row["duration"] = duration

    result = validate_and_normalize_records([valid_row])

    assert result.accepted_records == []
    assert expected_reason in result.rejected_rows[0].reasons


@pytest.mark.parametrize(
    ("duration", "expected_reason"),
    [
        (True, "invalid_number:duration"),
        ("   ", "empty_field:duration"),
        ("NaN", "non_finite:duration"),
        ("Infinity", "non_finite:duration"),
        ("-inf", "non_finite:duration"),
        (Decimal("2.5"), "invalid_number:duration"),
    ],
)
def test_numeric_types_and_invalid_numeric_strings(
    valid_row: dict[str, object], duration: object, expected_reason: str
) -> None:
    valid_row["duration"] = duration

    result = validate_and_normalize_records([valid_row])

    assert result.accepted_records == []
    assert result.rejection_counts == {expected_reason: 1}


@pytest.mark.parametrize("duration", ["2.5e0", " 2.5 "])
def test_explicit_numeric_string_formats_are_accepted(
    valid_row: dict[str, object], duration: str
) -> None:
    valid_row["duration"] = duration

    result = validate_and_normalize_records([valid_row])

    assert len(result.accepted_records) == 1
    assert result.accepted_records[0]["duration"] == 2.5


@pytest.mark.parametrize(
    ("start", "end", "duration"),
    [(0.0, 2.0, 2.049), (0.0, 2.0, 2.05), (0.0, 1.0, 1.05)],
)
def test_duration_difference_at_or_within_tolerance_is_accepted(
    valid_row: dict[str, object], start: float, end: float, duration: float
) -> None:
    valid_row.update(start=start, end=end, duration=duration)

    result = validate_and_normalize_records([valid_row])

    assert len(result.accepted_records) == 1
    assert result.rejected_rows == []


@pytest.mark.parametrize(
    ("start", "end", "duration"), [(0.0, 2.0, 2.051), (0.0, 1.0, 1.051)]
)
def test_duration_difference_over_tolerance_is_rejected(
    valid_row: dict[str, object], start: float, end: float, duration: float
) -> None:
    valid_row.update(start=start, end=end, duration=duration)

    result = validate_and_normalize_records([valid_row])

    assert result.rejection_counts == {"duration_mismatch": 1}


def test_representative_current_metadata_timings_remain_valid(
    valid_row: dict[str, object],
) -> None:
    representative_rows = [
        {**valid_row, "start": 13.84, "end": 18.35, "duration": 4.51},
        {**valid_row, "utterance_id": "utt-002", "start": 0, "end": 2.91, "duration": 2.91},
    ]

    result = validate_and_normalize_records(representative_rows)

    assert len(result.accepted_records) == 2
    assert result.rejected_rows == []


def test_invalid_timestamps_do_not_add_duration_mismatch_reason(
    valid_row: dict[str, object],
) -> None:
    valid_row.update(start=3.0, end=2.0, duration=1.0)

    result = validate_and_normalize_records([valid_row])

    assert result.rejection_counts == {"invalid_time_order:start,end": 1}


def test_timestamp_values_absent_are_allowed_without_consistency_check(
    valid_row: dict[str, object],
) -> None:
    del valid_row["start"]
    del valid_row["end"]
    valid_row["duration"] = 99

    result = validate_and_normalize_records([valid_row])

    assert len(result.accepted_records) == 1
    assert result.accepted_records[0]["start"] is None
    assert result.accepted_records[0]["end"] is None


def test_present_nan_timestamps_are_rejected_not_treated_as_absent(
    valid_row: dict[str, object],
) -> None:
    valid_row.update(start=float("nan"), end=float("nan"))

    result = validate_and_normalize_records([valid_row])

    assert result.rejection_counts == {"non_finite:end": 1, "non_finite:start": 1}


@pytest.mark.parametrize(
    ("start", "end", "expected_reason"),
    [
        (4, 3, "invalid_time_order:start,end"),
        (float("nan"), 3, "non_finite:start"),
        (1, float("inf"), "non_finite:end"),
        (-1, 3, "negative_time:start"),
        (None, 3, "incomplete_time_pair:start,end"),
    ],
)
def test_invalid_time_values_or_order_are_rejected(
    valid_row: dict[str, object],
    start: object,
    end: object,
    expected_reason: str,
) -> None:
    valid_row["start"] = start
    valid_row["end"] = end

    result = validate_and_normalize_records([valid_row])

    assert result.accepted_records == []
    assert expected_reason in result.rejected_rows[0].reasons


@pytest.mark.parametrize(
    ("field", "value", "expected_reason"),
    [
        ("text", "   ", "empty_field:text"),
        ("audio_path", "  ", "empty_field:audio_path"),
    ],
)
def test_empty_text_and_audio_paths_are_rejected(
    valid_row: dict[str, object], field: str, value: str, expected_reason: str
) -> None:
    valid_row[field] = value

    result = validate_and_normalize_records([valid_row])

    assert result.accepted_records == []
    assert result.rejection_counts == {expected_reason: 1}


def test_duplicate_ids_reject_every_occurrence_with_counts(
    valid_row: dict[str, object],
) -> None:
    second = deepcopy(valid_row)
    second["audio_path"] = "audio/session/second.wav"

    result = validate_and_normalize_records([valid_row, second])

    assert result.accepted_records == []
    assert [row.source_row for row in result.rejected_rows] == [0, 1]
    assert all(
        "duplicate_utterance_id:utterance_id" in row.reasons
        for row in result.rejected_rows
    )
    assert result.rejection_counts["duplicate_utterance_id:utterance_id"] == 2


def test_three_occurrences_of_an_id_are_all_rejected(
    valid_row: dict[str, object],
) -> None:
    rows = [deepcopy(valid_row) for _ in range(3)]
    for index, row in enumerate(rows):
        row["audio_path"] = f"audio/session/{index}.wav"

    result = validate_and_normalize_records(rows)

    assert result.accepted_records == []
    assert len(result.rejected_rows) == 3
    assert result.rejection_counts["duplicate_utterance_id:utterance_id"] == 3


def test_ids_matching_after_trimming_are_duplicates(
    valid_row: dict[str, object],
) -> None:
    second = deepcopy(valid_row)
    second["utterance_id"] = " utt-001 "
    second["audio_path"] = "audio/session/second.wav"

    result = validate_and_normalize_records([valid_row, second])

    assert result.accepted_records == []
    assert [row.utterance_id for row in result.rejected_rows] == [
        "utt-001",
        "utt-001",
    ]


def test_missing_and_blank_ids_do_not_count_as_duplicates(
    valid_row: dict[str, object],
) -> None:
    missing = deepcopy(valid_row)
    del missing["utterance_id"]
    blank = deepcopy(valid_row)
    blank["utterance_id"] = "   "

    result = validate_and_normalize_records([missing, blank])

    assert result.rejection_counts == {
        "empty_field:utterance_id": 1,
        "missing_field:utterance_id": 1,
    }
    assert [row.reasons for row in result.rejected_rows] == [
        ("missing_field:utterance_id",),
        ("empty_field:utterance_id",),
    ]


def test_absent_optional_speaker_and_times_are_normalized_to_null(
    valid_row: dict[str, object],
) -> None:
    del valid_row["speaker_id"]
    del valid_row["start"]
    del valid_row["end"]

    result = validate_and_normalize_records([valid_row])

    assert result.accepted_records[0]["speaker_id"] is None
    assert result.accepted_records[0]["start"] is None
    assert result.accepted_records[0]["end"] is None


def test_non_string_speaker_id_is_rejected(
    valid_row: dict[str, object],
) -> None:
    valid_row["speaker_id"] = 7

    result = validate_and_normalize_records([valid_row])

    assert result.rejection_counts == {"invalid_type:speaker_id": 1}


def test_reason_order_and_counts_are_stable_for_multiple_errors(
    valid_row: dict[str, object],
) -> None:
    valid_row.update(
        audio_path=" ",
        text=" ",
        recording_id="test",
        start=3.0,
        end=1.0,
    )

    result = validate_and_normalize_records([valid_row])

    expected_reasons = (
        "empty_field:audio_path",
        "empty_field:text",
        "placeholder_group_id:recording_id",
        "invalid_time_order:start,end",
    )
    assert result.rejected_rows[0].reasons == expected_reasons
    assert result.rejection_counts == {reason: 1 for reason in sorted(expected_reasons)}
    assert len(result.rejected_rows) == 1


def test_duplicate_dataframe_index_labels_have_unique_row_positions(
    valid_row: dict[str, object],
) -> None:
    first = deepcopy(valid_row)
    second = deepcopy(valid_row)
    first.update(utterance_id="utt-001", text=" ")
    second.update(utterance_id="utt-002", text=" ")
    frame = pd.DataFrame([first, second], index=["duplicate", "duplicate"])

    result = validate_and_normalize_records(frame)

    assert [row.source_row for row in result.rejected_rows] == [
        "duplicate",
        "duplicate",
    ]
    assert [row.source_row_position for row in result.rejected_rows] == [0, 1]


def test_iterable_results_and_reason_counts_are_deterministic(
    valid_row: dict[str, object],
) -> None:
    invalid = deepcopy(valid_row)
    invalid["recording_id"] = "unknown"
    invalid["utterance_id"] = "utt-002"
    rows = [valid_row, invalid]

    first = validate_and_normalize_records(rows)
    second = validate_and_normalize_records(rows)

    assert first == second
    assert first.total_rows == 2
    assert first.rejection_counts == {"placeholder_group_id:recording_id": 1}


def test_numpy_non_finite_number_is_rejected(
    valid_row: dict[str, object],
) -> None:
    valid_row["duration"] = np.float64(np.inf)

    result = validate_and_normalize_records([valid_row])

    assert result.rejection_counts == {"non_finite:duration": 1}


def test_grouped_split_is_deterministic_with_seed_42() -> None:
    records = make_grouped_records(
        [(f"source-{index % 2}", f"recording-{index}", 2) for index in range(20)]
    )

    first = split_grouped_records(records)
    second = split_grouped_records(records)

    assert isinstance(first, GroupSplitResult)
    assert first == second


def test_grouped_split_prevents_composite_group_leakage() -> None:
    records = make_grouped_records(
        [(f"source-{index % 2}", f"recording-{index}", 2) for index in range(20)]
    )

    result = split_grouped_records(records)
    partitions = {
        "train": result.train_records,
        "validation": result.validation_records,
        "test": result.test_records,
    }
    groups_by_split = {
        name: {(row["source_dataset"], row["recording_id"]) for row in rows}
        for name, rows in partitions.items()
    }

    assert groups_by_split["train"].isdisjoint(groups_by_split["validation"])
    assert groups_by_split["train"].isdisjoint(groups_by_split["test"])
    assert groups_by_split["validation"].isdisjoint(groups_by_split["test"])
    assert set.union(*groups_by_split.values()) == {
        (row["source_dataset"], row["recording_id"]) for row in records
    }


def test_grouped_split_preserves_every_utterance_once_and_input_order() -> None:
    records = make_grouped_records(
        [(f"source-{index % 2}", f"recording-{index}", 3) for index in range(18)]
    )
    original_positions = {
        row["utterance_id"]: index for index, row in enumerate(records)
    }

    result = split_grouped_records(records)
    split_rows = (
        result.train_records,
        result.validation_records,
        result.test_records,
    )
    split_ids = [row["utterance_id"] for rows in split_rows for row in rows]

    assert Counter(split_ids) == Counter(row["utterance_id"] for row in records)
    for rows in split_rows:
        positions = [original_positions[row["utterance_id"]] for row in rows]
        assert positions == sorted(positions)


def test_grouped_split_reports_actual_counts_and_ratios() -> None:
    records = make_grouped_records(
        [(f"source-{index % 2}", f"recording-{index}", 2) for index in range(20)]
    )
    result = split_grouped_records(records)
    partitions = {
        "train": result.train_records,
        "validation": result.validation_records,
        "test": result.test_records,
    }
    unique_groups = {
        (row["source_dataset"], row["recording_id"]) for row in records
    }

    for name, rows in partitions.items():
        actual_groups = {
            (row["source_dataset"], row["recording_id"]) for row in rows
        }
        assert result.utterance_counts[name] == len(rows)
        assert result.group_counts[name] == len(actual_groups)
        assert result.utterance_ratios[name] == pytest.approx(len(rows) / len(records))
        assert result.group_ratios[name] == pytest.approx(
            len(actual_groups) / len(unique_groups)
        )
    assert sum(result.utterance_counts.values()) == len(records)
    assert sum(result.group_counts.values()) == len(unique_groups)


def test_grouped_split_reports_row_skew_for_unequal_group_sizes() -> None:
    specs = [("source-a", "large-session", 1000)] + [
        (f"source-{index % 2}", f"small-session-{index}", 1)
        for index in range(11)
    ]
    result = split_grouped_records(make_grouped_records(specs))

    assert sum(result.group_counts.values()) == 12
    assert sum(result.utterance_counts.values()) == 1011
    assert sum(result.group_ratios.values()) == pytest.approx(1.0)
    assert sum(result.utterance_ratios.values()) == pytest.approx(1.0)
    assert max(result.utterance_ratios.values()) > 0.8


def test_grouped_split_rejects_fewer_than_three_groups() -> None:
    records = make_grouped_records(
        [("source-a", "recording-1", 1), ("source-a", "recording-2", 1)]
    )

    with pytest.raises(ValueError, match="At least three recording groups"):
        split_grouped_records(records)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("recording_id", None),
        ("recording_id", "  "),
        ("recording_id", " TRAIN "),
        ("recording_id", "dev"),
        ("recording_id", "Test"),
        ("recording_id", "unknown"),
        ("source_dataset", None),
        ("source_dataset", " "),
        ("source_dataset", "unknown"),
    ],
)
def test_grouped_split_rejects_missing_blank_and_placeholder_components(
    field: str, value: object
) -> None:
    records = make_grouped_records(
        [("source-a", f"recording-{index}", 1) for index in range(3)]
    )
    records[0][field] = value

    with pytest.raises(ValueError, match=field):
        split_grouped_records(records)


def test_grouped_split_rejects_duplicate_utterance_ids() -> None:
    records = make_grouped_records(
        [("source-a", f"recording-{index}", 1) for index in range(3)]
    )
    records[1]["utterance_id"] = records[0]["utterance_id"]

    with pytest.raises(ValueError, match="duplicate utterance_id"):
        split_grouped_records(records)


def test_same_recording_id_from_different_sources_is_not_merged() -> None:
    records = make_grouped_records(
        [
            ("source-a", "shared-recording-id", 1),
            ("source-b", "shared-recording-id", 1),
            ("source-a", "recording-a2", 1),
            ("source-b", "recording-b2", 1),
            ("source-a", "recording-a3", 1),
            ("source-b", "recording-b3", 1),
        ]
    )

    result = split_grouped_records(records)
    split_rows = (
        result.train_records,
        result.validation_records,
        result.test_records,
    )
    split_groups = {
        (row["source_dataset"], row["recording_id"])
        for rows in split_rows
        for row in rows
    }

    assert sum(result.group_counts.values()) == 6
    assert ("source-a", "shared-recording-id") in split_groups
    assert ("source-b", "shared-recording-id") in split_groups
