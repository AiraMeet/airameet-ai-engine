"""Checksum generation, determinism, and corruption detection tests."""

from pathlib import Path

import pytest

from airameet_ai.data_engineering.grouped_release.release.release_writer import (
    verify_checksums,
    write_checksums,
)


def test_checksum_output_is_deterministic_and_verifies_files(tmp_path: Path) -> None:
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests/train.jsonl").write_bytes(b'{"id":"one"}\n')
    (tmp_path / "reports.json").write_text('{"ready":true}\n', encoding="utf-8")

    first = write_checksums(tmp_path)
    checksum_bytes = (tmp_path / "CHECKSUMS.sha256").read_bytes()
    second = write_checksums(tmp_path)

    assert first == second
    assert (tmp_path / "CHECKSUMS.sha256").read_bytes() == checksum_bytes
    assert verify_checksums(tmp_path) == {"status": "passed", "files_verified": 2}


def test_checksum_verification_detects_modified_file(tmp_path: Path) -> None:
    payload = tmp_path / "manifest.jsonl"
    payload.write_text("original\n", encoding="utf-8")
    write_checksums(tmp_path)
    payload.write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_checksums(tmp_path)


@pytest.mark.parametrize("change", ["delete", "add"])
def test_checksum_verification_detects_missing_or_unlisted_files(
    tmp_path: Path, change: str
) -> None:
    payload = tmp_path / "manifest.jsonl"
    payload.write_text("payload\n", encoding="utf-8")
    write_checksums(tmp_path)
    if change == "delete":
        payload.unlink()
    else:
        (tmp_path / "extra.txt").write_text("unexpected\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exact release file set"):
        verify_checksums(tmp_path)


def test_checksum_file_without_index_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        verify_checksums(tmp_path)
