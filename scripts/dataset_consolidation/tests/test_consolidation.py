"""Synthetic safety tests for the standalone consolidation package."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.dataset_consolidation.audio_manager import copy_audio, destination_path
from scripts.dataset_consolidation.checksum_manager import (
    generate_checksums,
    verify_checksums,
)
from scripts.dataset_consolidation.consolidate_asr_dataset import consolidate, verify
from scripts.dataset_consolidation.manifest_manager import migrate_manifests
from scripts.dataset_consolidation.metadata_merger import merge_metadata
from scripts.dataset_consolidation.validator import validate_dataset


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    processed, releases, destination = root / "data/processed", root / "data/releases", root / "data/asr-dataset"
    processed.mkdir(parents=True)
    release = releases / "data-v1.1"
    rows = []
    for dataset, filename, utterance, recording, speaker in (
        ("Erzen", "Erzen_corrected_metadata.csv", "e1", "recording-e", "speaker-e"),
        ("Arabic-English", "Arabic_English_corrected_metadata.csv", "a1", "recording-a", "unknown"),
    ):
        rel = f"audio/{dataset}/audio/train/{utterance}.wav"
        audio = release / rel
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes((dataset + " audio").encode())
        rows.append({"utterance_id": utterance, "audio_path": rel, "text": "transcript " + utterance, "recording_id": recording, "session_id": "session-" + utterance, "speaker_id": speaker, "source_dataset": dataset, "split": "train", "source_audio_sha256": _sha(audio)})
        metadata = release / "metadata" / filename
        metadata.parent.mkdir(parents=True, exist_ok=True)
        fields = ["utterance_id", "audio_path", "text", "text_variant", "recording_id", "session_id", "speaker_id", "source_dataset", "source_only", "release_eligible"]
        with metadata.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerow({key: rows[-1].get(key, dataset if key == "source_only" else "alternate " + utterance if key == "text_variant" else "True" if key == "release_eligible" else "") for key in fields})
    manifests = release / "manifests"
    manifests.mkdir(parents=True)
    for filename, split in (("train.jsonl", "train"), ("val.jsonl", "validation"), ("test.jsonl", "test")):
        subset = rows if split == "train" else []
        (manifests / filename).write_text("".join(json.dumps(row) + "\n" for row in subset), encoding="utf-8")
    # Match the v1.1 two-space checksum format.
    records = []
    for path in sorted(release.rglob("*")):
        if path.is_file() and path.name != "CHECKSUMS.sha256":
            records.append(f"{_sha(path)}  {path.relative_to(release).as_posix()}\n")
    (release / "CHECKSUMS.sha256").write_text("".join(records), encoding="utf-8")
    return processed, releases, destination


def test_metadata_audio_and_manifest_migration(tmp_path: Path) -> None:
    processed, releases, destination = _fixture(tmp_path)
    release = releases / "data-v1.1"
    rows = [json.loads(line) for line in (release / "manifests/train.jsonl").read_text().splitlines()]
    stage = tmp_path / "stage"
    stage.mkdir()
    mapping = copy_audio(release, stage, rows)
    merged = merge_metadata(release, stage, mapping)
    migrated = migrate_manifests(release, stage, mapping)
    assert len(merged) == 2
    assert {"source_only", "recording_id", "session_id", "speaker_id", "text_variant"} <= set(merged.columns)
    assert merged.loc[0, "text_variant"] == "alternate e1"
    assert migrated[0]["audio_path"] == "audio/Erzen/train/e1.wav"
    assert (stage / migrated[0]["audio_path"]).read_bytes() == (release / rows[0]["audio_path"]).read_bytes()
    assert processed.is_dir() and destination.is_dir() is False


def test_checksum_generation_and_modified_file_detection(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "metadata").mkdir(parents=True)
    (root / "metadata/file.csv").write_text("x\n")
    generate_checksums(root)
    assert verify_checksums(root)["status"] == "passed"
    (root / "metadata/file.csv").write_text("changed\n")
    result = verify_checksums(root)
    assert result["status"] == "failed"
    assert result["hash_mismatches"] == ["metadata/file.csv"]


def test_checksum_verifier_detects_missing_and_unindexed_files(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "indexed.txt").write_text("verified")
    generate_checksums(root)
    (root / "indexed.txt").unlink()
    (root / "extra.txt").write_text("unindexed")
    result = verify_checksums(root)
    assert result["missing_files"] == ["indexed.txt"]
    assert result["unindexed_files"] == ["extra.txt"]


def test_checksum_verifier_reports_malformed_entries(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "checksums").mkdir(parents=True)
    (root / "checksums/SHA256SUMS").write_text("not-a-digest  ../outside\n")
    report = verify_checksums(root)
    assert report["status"] == "failed"
    assert report["invalid_checksum_entries"] == ["line:1"]


def test_checksum_verifier_rejects_symlinks_outside_dataset(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    (root / "link.txt").symlink_to(outside)
    (root / "checksums").mkdir()
    (root / "checksums/SHA256SUMS").write_text(f"{_sha(outside)}  link.txt\n")
    report = verify_checksums(root)
    assert report["invalid_checksum_entries"] == ["line:1"]


def test_verification_is_read_only_and_validates_manifest_coverage(tmp_path: Path) -> None:
    processed, releases, destination = _fixture(tmp_path)
    result = consolidate(processed=processed, releases=releases, destination=destination)
    assert result["verification"]["checksums"]["files_checked"] == 6
    report = destination / "checksums/checksum_verification.json"
    before = report.read_bytes(), report.stat().st_mtime_ns
    assert verify(destination)["status"] == "passed"
    assert before == (report.read_bytes(), report.stat().st_mtime_ns)
    structure = validate_dataset(destination)
    assert structure["manifest_rows"] == 2
    assert structure["status"] == "passed"


def test_dry_run_and_merge_preserve_sources(tmp_path: Path) -> None:
    processed, releases, destination = _fixture(tmp_path)
    snapshot = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in releases.rglob("*") if path.is_file()}
    preview = consolidate(processed=processed, releases=releases, destination=destination, dry_run=True)
    assert preview["status"] == "dry-run" and not destination.exists()
    assert snapshot == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in releases.rglob("*") if path.is_file()}
    result = consolidate(processed=processed, releases=releases, destination=destination)
    assert result["status"] == "passed"
    assert verify(destination)["status"] == "passed"
    assert processed.is_dir() and releases.is_dir()
    assert not list(destination.parent.glob(".asr-dataset.staging-*"))


def test_cleanup_requires_confirmation_and_failure_recovers_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    processed, releases, destination = _fixture(tmp_path)
    with pytest.raises(ValueError, match="confirm-delete"):
        consolidate(processed=processed, releases=releases, destination=destination, mode="merge-clean")
    import scripts.dataset_consolidation.consolidate_asr_dataset as module

    original = module.copy_audio

    def fail_after_copy(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(module, "copy_audio", fail_after_copy)
    with pytest.raises(RuntimeError, match="simulated"):
        consolidate(processed=processed, releases=releases, destination=destination, mode="merge-clean", confirm_delete=True)
    assert processed.is_dir() and releases.is_dir() and not destination.exists()
    assert not list(destination.parent.glob(".asr-dataset.staging-*"))


def test_existing_destination_is_protected(tmp_path: Path) -> None:
    processed, releases, destination = _fixture(tmp_path)
    destination.mkdir()
    (destination / "keep").write_text("safe")
    with pytest.raises(FileExistsError):
        consolidate(processed=processed, releases=releases, destination=destination)
    assert (destination / "keep").read_text() == "safe"


def test_confirmed_cleanup_removes_only_approved_synthetic_sources(tmp_path: Path) -> None:
    processed, releases, destination = _fixture(tmp_path)
    result = consolidate(processed=processed, releases=releases, destination=destination, mode="merge-clean", confirm_delete=True)
    assert result["status"] == "passed"
    assert result["cleanup"] is True
    assert not processed.exists() and not releases.exists()
    assert destination.is_dir()


def test_failed_final_verification_restores_overwritten_destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    processed, releases, destination = _fixture(tmp_path)
    destination.mkdir()
    (destination / "old.txt").write_text("previous destination")
    import scripts.dataset_consolidation.consolidate_asr_dataset as module

    original_verify = module.verify
    calls = 0

    def fail_final(path: Path, *, write_report: bool = False):
        nonlocal calls
        calls += 1
        if calls == 2:
            return {"status": "failed", "structure": {}, "checksums": {}}
        return original_verify(path, write_report=write_report)

    monkeypatch.setattr(module, "verify", fail_final)
    with pytest.raises(ValueError, match="Final verification"):
        consolidate(processed=processed, releases=releases, destination=destination, overwrite=True)
    assert (destination / "old.txt").read_text() == "previous destination"
    assert not list(destination.parent.glob(".asr-dataset.staging-*"))
    assert not list(destination.parent.glob(".asr-dataset.backup-*"))


def test_cleanup_failure_restores_sources_and_old_destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    processed, releases, destination = _fixture(tmp_path)
    destination.mkdir()
    (destination / "old.txt").write_text("prior")
    import scripts.dataset_consolidation.consolidate_asr_dataset as module

    original_rmtree = module.shutil.rmtree
    failed_once = False

    def fail_releases_once(path, *args, **kwargs):
        nonlocal failed_once
        if Path(path) == releases and not failed_once:
            failed_once = True
            raise OSError("simulated interrupted cleanup")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(module.shutil, "rmtree", fail_releases_once)
    with pytest.raises(OSError, match="interrupted cleanup"):
        consolidate(processed=processed, releases=releases, destination=destination, mode="merge-clean", confirm_delete=True, overwrite=True)
    assert processed.is_dir() and releases.is_dir()
    assert (destination / "old.txt").read_text() == "prior"


def test_direct_script_and_main_dispatch_with_synthetic_sources(tmp_path: Path) -> None:
    processed, releases, destination = _fixture(tmp_path)
    repo = Path(__file__).resolve().parents[3]
    standalone = subprocess.run(
        [sys.executable, str(repo / "scripts/dataset_consolidation/consolidate_asr_dataset.py"), "--mode", "merge", "--processed", str(processed), "--releases", str(releases), "--destination", str(destination)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert standalone.returncode == 0, standalone.stderr
    assert json.loads(standalone.stdout)["status"] == "passed"
    report = destination / "checksums/checksum_verification.json"
    previous_report = report.read_bytes()
    standalone_verify = subprocess.run(
        [sys.executable, str(repo / "scripts/dataset_consolidation/consolidate_asr_dataset.py"), "--mode", "verify", "--destination", str(destination)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert standalone_verify.returncode == 0, standalone_verify.stderr
    assert report.read_bytes() == previous_report

    processed2, releases2, destination2 = _fixture(tmp_path / "main-cli")
    main_cli = subprocess.run(
        [sys.executable, str(repo / "main.py"), "--task", "DATASET-CONSOLIDATION", "--mode", "merge", "--input", str(processed2), "--releases", str(releases2), "--output", str(destination2)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert main_cli.returncode == 0, main_cli.stderr
    assert destination2.is_dir()


def test_main_task_is_registered() -> None:
    repo = Path(__file__).resolve().parents[3]
    result = subprocess.run([sys.executable, str(repo / "main.py"), "--list-tasks"], cwd=repo, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "DATASET-CONSOLIDATION:" in result.stdout


def test_audio_paths_reject_traversal_and_keep_dataset_separation() -> None:
    assert destination_path("audio/Erzen/audio/train/a.wav", "Erzen") == "audio/Erzen/train/a.wav"
    assert destination_path("audio/Arabic-English/audio/train/a.wav", "Arabic-English") == "audio/Arabic-English-ASR/train/a.wav"
    with pytest.raises(ValueError, match="Unsafe"):
        destination_path("audio/Erzen/../../outside.wav", "Erzen")
