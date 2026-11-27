"""Read-only regression checks for the immutable data-v1.1 artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from airameet_ai.core.config import PROJECT_ROOT
from airameet_ai.data_engineering.grouped_release.release.release_writer import (
    verify_checksums,
)

RELEASE_ROOT = PROJECT_ROOT / "data/releases/data-v1.1"
REFERENCE_CHECKSUM_INDEX_SHA256 = (
    "b9daf03330053e6f7841abce6b4a4a3b89f3d753bdf9beed851fb2975e90df99"
)


@pytest.mark.slow
def test_data_v11_release_matches_recorded_integrity_and_split_regression() -> None:
    if not RELEASE_ROOT.is_dir():
        pytest.skip("Local data-v1.1 release is not present")
    checksum_file = RELEASE_ROOT / "CHECKSUMS.sha256"
    checksum_hash = hashlib.sha256(checksum_file.read_bytes()).hexdigest()
    assert checksum_hash == REFERENCE_CHECKSUM_INDEX_SHA256, (
        "The v1.1 checksum index changed; confirm the release/source revision "
        "before updating expected regression values."
    )
    verification = verify_checksums(RELEASE_ROOT)
    assert verification == {"status": "passed", "files_verified": 18690}

    stats = json.loads((RELEASE_ROOT / "reports/dataset_stats.json").read_text(encoding="utf-8"))
    duplicates = json.loads((RELEASE_ROOT / "reports/duplicate_report.json").read_text(encoding="utf-8"))
    quality = json.loads((RELEASE_ROOT / "reports/audio_quality_report.json").read_text(encoding="utf-8"))
    integrity = json.loads((RELEASE_ROOT / "reports/split_integrity_report.json").read_text(encoding="utf-8"))

    assert stats["release"] == "data-v1.1"
    assert stats["total_utterances"] == 18_680
    assert stats["random_seed"] == 42
    assert duplicates["cross_dataset_duplicate_pairs"] == 1_422
    assert quality["summary"]["standardized_files"] == 18_680
    assert quality["input_format_distribution"] == {
        "16000Hz/1ch": 12_173,
        "44100Hz/2ch": 46,
        "48000Hz/2ch": 6_462,
    }
    assert sum(1 for _ in (RELEASE_ROOT / "audio").rglob("*.wav")) == 18_680

    expected_group = {"train": 0.799946, "validation": 0.100027, "test": 0.100027}
    expected_utterance = {"train": 0.795128, "validation": 0.079443, "test": 0.125428}
    for split, expected in expected_group.items():
        assert stats["splits"][split]["recording_group_proportion"] == pytest.approx(
            expected, abs=0.000001
        )
    for split, expected in expected_utterance.items():
        assert stats["splits"][split]["utterance_proportion"] == pytest.approx(
            expected, abs=0.000001
        )
    assert integrity["session_leaks"] == 0
    assert integrity["exact_duplicate_leaks"] == 0
    assert integrity["speaker_independent"] is False
    assert integrity["global_session_independence"] is False
    assert stats["arabic_english_unmatched_session_rows"] == 11_049
    assert stats["unknown_speaker_rows"] == 12_471

    all_ids: list[str] = []
    sessions_by_split: dict[str, set[str]] = {}
    hashes_by_split: dict[str, set[str]] = {}
    duplicate_group_splits: list[set[str]] = []
    for split, filename in (("train", "train.jsonl"), ("validation", "val.jsonl"), ("test", "test.jsonl")):
        sessions_by_split[split] = set()
        hashes_by_split[split] = set()
        manifest = RELEASE_ROOT / "manifests" / filename
        for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                pytest.fail(f"Malformed JSONL at {manifest}:{line_number}: {error}")
            assert {"utterance_id", "audio_path", "text", "recording_id", "source_dataset", "split"} <= set(record)
            assert record["split"] == split
            audio_path = Path(record["audio_path"])
            assert not audio_path.is_absolute()
            assert (RELEASE_ROOT / audio_path).is_file()
            all_ids.append(record["utterance_id"])
            if record.get("session_id"):
                sessions_by_split[split].add(record["session_id"])
            hashes_by_split[split].add(record["source_audio_sha256"])
    assert len(all_ids) == 18_680
    assert len(all_ids) == len(set(all_ids))
    assert all(
        not sessions_by_split[left] & sessions_by_split[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    assert all(
        not hashes_by_split[left] & hashes_by_split[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    for duplicate in duplicates["groups"]:
        duplicate_group_splits.append({row["split"] for row in duplicate["records"]})
    assert all(len(splits) == 1 for splits in duplicate_group_splits)

    audit = (RELEASE_ROOT / "reports/release_audit.md").read_text(encoding="utf-8")
    normalized_audit = " ".join(audit.split())
    assert "not certified session-independent" in normalized_audit
    assert "Speaker-independent splitting was not applied" in normalized_audit
