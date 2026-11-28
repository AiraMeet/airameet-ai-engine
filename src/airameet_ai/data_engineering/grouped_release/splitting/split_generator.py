"""GroupShuffleSplit implementation for verified v1.1 recording identities."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from sklearn.model_selection import GroupShuffleSplit

SEED_V11 = 42


class _DisjointSet:
    """Deterministic union-find for session and exact-audio constraints."""

    def __init__(self, values: Sequence[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        first, second = sorted((root_left, root_right))
        self.parent[second] = first


def assign_split_groups(records: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Join all utterances from one session and all exact duplicate audio."""
    keys = sorted({str(row["split_group_key"]) for row in records})
    sets = _DisjointSet(keys)
    by_hash: dict[str, list[str]] = defaultdict(list)
    for row in records:
        by_hash[row["source_audio_sha256"]].append(str(row["split_group_key"]))
    for group_keys in by_hash.values():
        distinct = sorted(set(group_keys))
        for key in distinct[1:]:
            sets.union(distinct[0], key)
    return {key: sets.find(key) for key in keys}


def split_records(
    records: Sequence[dict[str, Any]], *, seed: int = SEED_V11
) -> dict[str, list[dict[str, Any]]]:
    """Partition identity groups at 80/10/10 target group proportions."""
    if len(records) < 3:
        raise ValueError("At least three eligible utterances are required for splitting")
    group_map = assign_split_groups(records)
    for record in records:
        record["split_group_id"] = group_map[str(record["split_group_key"])]
    groups = sorted({record["split_group_id"] for record in records})
    if len(groups) < 3:
        raise ValueError(f"At least three recording groups are required; found {len(groups)}")

    group_indices = list(range(len(groups)))
    first = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    remaining, test = next(first.split(group_indices, groups=group_indices))
    second = GroupShuffleSplit(n_splits=1, test_size=1 / 9, random_state=seed)
    train_local, validation_local = next(second.split(remaining, groups=remaining))
    assigned = {
        **{groups[int(remaining[int(i)])]: "train" for i in train_local},
        **{groups[int(remaining[int(i)])]: "validation" for i in validation_local},
        **{groups[int(i)]: "test" for i in test},
    }
    partitions = {name: [] for name in ("train", "validation", "test")}
    for record in records:
        record["split"] = assigned[record["split_group_id"]]
        partitions[record["split"]].append(record)
    for rows in partitions.values():
        rows.sort(key=lambda row: row["utterance_id"])
    return partitions


def validate_partition_integrity(
    records: Sequence[dict[str, Any]],
    partitions: Mapping[str, Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    """Raise on record, session, identity-group, or exact-hash leakage."""
    names = ("train", "validation", "test")
    all_ids = [row["utterance_id"] for name in names for row in partitions[name]]
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("Duplicate utterance IDs occur in release partitions")
    source_ids = [row["utterance_id"] for row in records]
    if set(all_ids) != set(source_ids) or len(all_ids) != len(source_ids):
        raise AssertionError("Partition coverage differs from eligible records")

    splits_by_session: dict[str, set[str]] = defaultdict(set)
    splits_by_hash: dict[str, set[str]] = defaultdict(set)
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    speaker_sets: dict[str, set[tuple[str, str]]] = {name: set() for name in names}
    for split, rows in partitions.items():
        for row in rows:
            if row.get("session_id"):
                splits_by_session[row["session_id"]].add(split)
            splits_by_hash[row["source_audio_sha256"]].add(split)
            splits_by_group[row["split_group_id"]].add(split)
            if row.get("speaker_id") and row["speaker_id"].casefold() != "unknown":
                speaker_sets[split].add((row["source_dataset"], row["speaker_id"]))

    session_leaks = {key: sorted(value) for key, value in splits_by_session.items() if len(value) > 1}
    duplicate_leaks = {key: sorted(value) for key, value in splits_by_hash.items() if len(value) > 1}
    group_leaks = {key: sorted(value) for key, value in splits_by_group.items() if len(value) > 1}
    if session_leaks:
        raise AssertionError(f"Interview sessions leak across partitions: {list(session_leaks)[:5]}")
    if duplicate_leaks:
        raise AssertionError(f"Exact duplicate audio leaks across partitions: {list(duplicate_leaks)[:5]}")
    if group_leaks:
        raise AssertionError(f"Split identity groups leak across partitions: {list(group_leaks)[:5]}")

    group_counts = {name: len({row["split_group_id"] for row in rows}) for name, rows in partitions.items()}
    total_groups = sum(group_counts.values())
    row_counts = {name: len(rows) for name, rows in partitions.items()}
    total_rows = sum(row_counts.values())
    return {
        "status": "passed",
        "eligible_utterances": len(records),
        "partition_utterance_counts": row_counts,
        "partition_utterance_proportions": {
            name: count / total_rows for name, count in row_counts.items()
        },
        "partition_group_counts": group_counts,
        "partition_group_proportions": {
            name: count / total_groups for name, count in group_counts.items()
        },
        "verified_session_count": len(splits_by_session),
        "session_leaks": 0,
        "exact_duplicate_hash_groups": len(splits_by_hash),
        "exact_duplicate_leaks": 0,
        "identity_group_leaks": 0,
        "speaker_independent": False,
        "speaker_overlaps": {
            f"{left}/{right}": sorted(speaker_sets[left] & speaker_sets[right])
            for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
        },
    }
