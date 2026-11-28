"""Determinism, disjointness, and proportion semantics for grouped splits."""

from copy import deepcopy

import pytest

from airameet_ai.data_engineering.grouped_release.splitting.split_generator import (
    SEED_V11,
    split_records,
    validate_partition_integrity,
)


def test_seed_42_produces_repeatable_assignment(grouped_records_factory) -> None:
    records = grouped_records_factory()
    first = split_records(deepcopy(records), seed=42)
    second = split_records(deepcopy(records), seed=42)

    def assignment(splits):
        return {
            row["utterance_id"]: split
            for split, rows in splits.items()
            for row in rows
        }

    assert SEED_V11 == 42
    assert assignment(first) == assignment(second)


def test_recording_session_and_exact_hash_groups_never_leak(grouped_records_factory) -> None:
    records = grouped_records_factory()
    # Join two independent identity keys only through exact byte identity.
    records[0]["source_audio_sha256"] = "cross-source-duplicate"
    records[-1]["source_audio_sha256"] = "cross-source-duplicate"
    partitions = split_records(records)
    integrity = validate_partition_integrity(records, partitions)

    assert integrity["session_leaks"] == 0
    assert integrity["exact_duplicate_leaks"] == 0
    assert integrity["identity_group_leaks"] == 0
    assert integrity["eligible_utterances"] == len(records)
    locations = {
        row["utterance_id"]: split
        for split, rows in partitions.items()
        for row in rows
    }
    assert locations[records[0]["utterance_id"]] == locations[records[-1]["utterance_id"]]
    by_session = {}
    for row in records:
        if row["session_id"]:
            by_session.setdefault(row["session_id"], set()).add(locations[row["utterance_id"]])
    assert all(len(splits) == 1 for splits in by_session.values())


def test_row_and_group_proportions_are_reported_separately(grouped_records_factory) -> None:
    records = grouped_records_factory(group_sizes=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120))
    partitions = split_records(records)
    integrity = validate_partition_integrity(records, partitions)

    assert set(integrity["partition_group_proportions"]) == {"train", "validation", "test"}
    assert set(integrity["partition_utterance_proportions"]) == {"train", "validation", "test"}
    assert sum(integrity["partition_group_proportions"].values()) == pytest.approx(1.0)
    assert sum(integrity["partition_utterance_proportions"].values()) == pytest.approx(1.0)
    assert integrity["partition_group_proportions"] != integrity["partition_utterance_proportions"]


def test_speaker_independence_is_not_claimed_or_enforced(grouped_records_factory) -> None:
    partitions = split_records(grouped_records_factory(speaker="shared-speaker"))
    report = validate_partition_integrity(
        [row for rows in partitions.values() for row in rows], partitions
    )

    assert report["speaker_independent"] is False
    assert any(report["speaker_overlaps"].values())


@pytest.mark.parametrize("records", [[], [{"utterance_id": "one"}, {"utterance_id": "two"}]])
def test_empty_or_too_small_input_is_rejected(records) -> None:
    with pytest.raises(ValueError, match="At least"):
        split_records(records)


def test_missing_recording_identity_is_not_accepted(grouped_records_factory) -> None:
    records = grouped_records_factory()
    del records[0]["split_group_key"]
    with pytest.raises(KeyError):
        split_records(records)


def test_duplicate_ids_or_missing_rows_fail_integrity(grouped_records_factory) -> None:
    records = grouped_records_factory()
    partitions = split_records(records)
    partitions["train"].append(dict(partitions["train"][0]))
    with pytest.raises(AssertionError, match="Duplicate utterance IDs"):
        validate_partition_integrity(records, partitions)

    partitions = split_records(grouped_records_factory())
    partitions["test"].pop()
    with pytest.raises(AssertionError, match="coverage"):
        validate_partition_integrity(records, partitions)
