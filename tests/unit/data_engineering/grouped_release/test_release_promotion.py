"""Atomic release staging and no-overwrite guarantees."""

from pathlib import Path

import pytest

from airameet_ai.data_engineering.grouped_release.release.release_writer import (
    create_staging_directory,
    discard_staging_directory,
    promote_release,
    verify_checksums,
    write_checksums,
)


def test_release_is_promoted_after_files_are_checksummed(tmp_path: Path) -> None:
    release = tmp_path / "data-vX"
    stage = create_staging_directory(release)
    (stage / "manifests").mkdir()
    (stage / "manifests/train.jsonl").write_text('{"utterance_id":"a"}\n', encoding="utf-8")
    write_checksums(stage)
    assert verify_checksums(stage)["status"] == "passed"

    promote_release(stage, release)

    assert release.is_dir()
    assert not stage.exists()
    assert (release / "CHECKSUMS.sha256").is_file()


def test_existing_release_is_never_overwritten(tmp_path: Path) -> None:
    release = tmp_path / "existing"
    release.mkdir()
    sentinel = release / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        create_staging_directory(release)
    with pytest.raises(FileExistsError):
        promote_release(tmp_path, release)

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_failed_stage_cleanup_removes_only_given_temporary_directory(tmp_path: Path) -> None:
    stage = create_staging_directory(tmp_path / "release")
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (stage / "partial.txt").write_text("partial", encoding="utf-8")

    discard_staging_directory(stage)

    assert not stage.exists()
    assert unrelated.is_dir()
