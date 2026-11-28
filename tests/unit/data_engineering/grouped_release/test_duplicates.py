"""Exact byte duplicate and unreadable-audio handling tests."""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from airameet_ai.data_engineering.grouped_release.grouping.duplicate_detector import (
    analyze_duplicates,
)
from airameet_ai.data_engineering.quality import audit_audio_files


def _record(path: Path, source: str, utterance_id: str, text: str) -> dict:
    return {
        "source_path": path,
        "source_audio_path": path.name,
        "source_dataset": source,
        "utterance_id": utterance_id,
        "recording_id": f"rec-{utterance_id}",
        "session_id": "C03-R01" if source == "Erzen" else None,
        "split_group_key": "session:C03-R01" if source == "Erzen" else f"recording:{utterance_id}",
        "corrected_row": {},
        "text": text,
    }


def test_exact_cross_dataset_audio_pairs_and_transcript_variants_are_retained(
    tmp_path: Path, make_wav_file
) -> None:
    erzen = make_wav_file(tmp_path / "erzen.wav")
    arabic = tmp_path / "arabic.wav"
    shutil.copyfile(erzen, arabic)
    records = [
        _record(erzen, "Erzen", "e-1", "Hello world"),
        _record(arabic, "Arabic-English", "a-1", "مرحبا world"),
    ]

    by_hash, report = analyze_duplicates(records)

    assert len(by_hash) == 1
    assert report["files_hashed"] == 2
    assert report["duplicate_hash_groups"] == 1
    assert report["cross_dataset_duplicate_pairs"] == 1
    assert report["transcript_mismatches_unadjudicated"] == 1
    assert {row["utterance_id"] for row in report["groups"][0]["records"]} == {"e-1", "a-1"}
    assert records[1]["session_id"] == "C03-R01"
    assert records[1]["identity_link_basis"] == "exact_sha256_match_to_erzen"


def test_different_audio_is_not_classified_as_duplicate(tmp_path: Path, make_wav_file) -> None:
    first = make_wav_file(tmp_path / "first.wav", freq=440)
    second = make_wav_file(tmp_path / "second.wav", freq=880)

    _, report = analyze_duplicates(
        [_record(first, "Arabic-English", "a-1", "same"), _record(second, "Erzen", "e-1", "same")]
    )

    assert report["unique_audio_hashes"] == 2
    assert report["duplicate_hash_groups"] == 0
    assert report["cross_dataset_duplicate_pairs"] == 0


def test_missing_audio_raises_file_error(tmp_path: Path) -> None:
    record = _record(tmp_path / "missing.wav", "Erzen", "e-1", "text")
    with pytest.raises(FileNotFoundError):
        analyze_duplicates([record])


def test_corrupt_audio_is_reported_unreadable_by_audio_auditor(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"not a wav file")
    report = audit_audio_files(
        pd.DataFrame({"audio_path": ["corrupt.wav"]}), tmp_path
    )
    assert not bool(report.loc[0, "readable"])
    assert report.loc[0, "snr_status"] == "failed"
    assert report.loc[0, "error"]
