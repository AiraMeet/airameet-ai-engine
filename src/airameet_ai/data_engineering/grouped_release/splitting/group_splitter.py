"""Deterministic group-level train/validation/test partitioning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sklearn.model_selection import GroupShuffleSplit

from ..metadata.recording_metadata import group_records, recording_group

SEED = 42
TARGET_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}


def split_grouped_records(
    records: Sequence[dict[str, Any]], *, seed: int = SEED
) -> dict[str, list[dict[str, Any]]]:
    """Split unique recording groups; ratios describe target group counts.

    The original notebook's imported helper is unavailable. This explicit
    recipe selects 10% of unique groups for test, then 1/9 of the remainder
    for validation, leaving approximately 80% for training. GroupShuffleSplit
    rounds group counts, so small datasets and uneven recording sizes can
    produce different group and utterance proportions.
    """
    grouped = group_records(list(records))
    groups = list(grouped)
    if len(groups) < 3:
        raise ValueError(
            f"At least three eligible recording groups are required; found {len(groups)}"
        )

    indices = list(range(len(groups)))
    first = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    remaining_indices, test_indices = next(
        first.split(indices, groups=indices)
    )
    second = GroupShuffleSplit(n_splits=1, test_size=1 / 9, random_state=seed)
    train_local, validation_local = next(
        second.split(remaining_indices, groups=remaining_indices)
    )
    index_sets = {
        "train": {remaining_indices[int(i)] for i in train_local},
        "validation": {remaining_indices[int(i)] for i in validation_local},
        "test": {int(i) for i in test_indices},
    }
    partitions = {
        name: [record for i in sorted(group_ids) for record in grouped[groups[i]]]
        for name, group_ids in index_sets.items()
    }
    verify_partitions(records, partitions)
    return partitions


def verify_partitions(
    records: Sequence[dict[str, Any]],
    partitions: Mapping[str, Sequence[dict[str, Any]]],
) -> None:
    """Assert group and utterance disjointness plus exact record coverage."""
    required = ("train", "validation", "test")
    group_sets = {
        name: {recording_group(row) for row in partitions[name]} for name in required
    }
    id_sets = {
        name: {str(row["utterance_id"]) for row in partitions[name]}
        for name in required
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = group_sets[left] & group_sets[right]
        if overlap:
            raise AssertionError(f"Recording leakage between {left} and {right}: {sorted(overlap)[:5]}")
        shared_ids = id_sets[left] & id_sets[right]
        if shared_ids:
            raise AssertionError(f"Utterance ID leakage between {left} and {right}: {sorted(shared_ids)[:5]}")

    source_ids = [str(row["utterance_id"]) for row in records]
    partition_ids = [item for name in required for item in id_sets[name]]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Eligible metadata contains duplicate utterance IDs")
    if set(partition_ids) != set(source_ids) or len(partition_ids) != len(source_ids):
        raise AssertionError("Partitions do not contain every eligible utterance exactly once")
    source_groups = {recording_group(row) for row in records}
    if set.union(*(group_sets[name] for name in required)) != source_groups:
        raise AssertionError("At least one eligible recording group was dropped")
