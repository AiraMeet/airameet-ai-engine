"""End-to-end grouped-release tests using tiny generated audio inputs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from airameet_ai.data_engineering.grouped_release import DatasetReleasePipeline
from airameet_ai.data_engineering.grouped_release import pipeline as pipeline_module
from airameet_ai.data_engineering.grouped_release.release.release_writer import (
    verify_checksums,
)


def _install_fast_audio_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    def audit(_self, records, _audio_roots):
        audits = {
            row["utterance_id"]: {
                "readable": True,
                "is_silent": False,
                "sample_rate": 16000,
                "channels": 1,
            }
            for row in records
        }
        summary = {
            "input_format_distribution": {"16000Hz/1ch": len(records)},
            "by_source": {},
            "files": [],
        }
        return audits, summary

    def convert(self, staging_root, records, _audits):
        converted = []
        details = []
        for row in records:
            relative_path = Path("audio") / row["source_dataset"] / Path(row["source_audio_path"]).name
            destination = staging_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(row["source_path"], destination)
            row["audio_path"] = relative_path.as_posix()
            row["corrected_row"]["audio_path"] = row["audio_path"]
            converted.append(row)
            details.append(
                {
                    "source_dataset": row["source_dataset"],
                    "utterance_id": row["utterance_id"],
                    "source_audio_path": row["source_audio_path"],
                    "output_audio_path": row["audio_path"],
                    "status": "success",
                }
            )
        summary = {
            "attempted": len(details),
            "standardized_files": len(details),
            "failed": 0,
            "failed_rows": [],
            "files": details,
        }
        return converted, details, summary

    monkeypatch.setattr(DatasetReleasePipeline, "_audit_audio", audit)
    monkeypatch.setattr(DatasetReleasePipeline, "_convert_audio", convert)


def test_synthetic_release_pipeline_writes_complete_integrity_checked_release(
    tmp_path: Path,
    synthetic_release_inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fast_audio_stages(monkeypatch)
    metadata_dir, data_root = synthetic_release_inputs
    release_dir = tmp_path / "releases" / "data-v1.1-test"
    report_root = tmp_path / "artifacts/data_engineering/reports"

    result = DatasetReleasePipeline(metadata_dir, data_root, release_dir, report_root).run()

    assert result["release"] == "data-v1.1"
    assert result["statistics"]["total_utterances"] == 13
    assert result["statistics"]["integrity"]["session_leaks"] == 0
    assert result["statistics"]["integrity"]["exact_duplicate_leaks"] == 0
    assert result["statistics"]["integrity"]["global_session_independence"] is False
    assert result["statistics"]["methodology"]["speaker_independent_split"] is False
    assert result["statistics"]["cross_dataset_duplicate_pairs"] == 1
    assert verify_checksums(release_dir)["status"] == "passed"

    split_ids: dict[str, set[str]] = {}
    all_ids: list[str] = []
    for name in ("train", "val", "test"):
        manifest_path = release_dir / "manifests" / f"{name}.jsonl"
        rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
        split_name = "validation" if name == "val" else name
        assert rows
        assert all(row["split"] == split_name for row in rows)
        assert all(not Path(row["audio_path"]).is_absolute() for row in rows)
        assert all((release_dir / row["audio_path"]).is_file() for row in rows)
        assert all("مرحبا" in row["text"] or "matching" in row["text"] for row in rows)
        split_ids[split_name] = {row["utterance_id"] for row in rows}
        all_ids.extend(row["utterance_id"] for row in rows)
    assert len(all_ids) == len(set(all_ids)) == 13

    linked_arabic = next(
        row for name in ("train", "val", "test")
        for row in [json.loads(line) for line in (release_dir / "manifests" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
        if row["source_dataset"] == "Arabic-English" and row.get("session_id") == "C03-R01"
    )
    erzen_split = next(
        split for split, ids in split_ids.items() if "I01-C03-R01_000000-000100" in ids
    )
    assert linked_arabic["split"] == erzen_split
    for category, filename in (
        ("statistics", "dataset_stats.json"),
        ("duplicates", "duplicate_report.json"),
        ("audio_quality", "audio_quality_report.json"),
        ("splitting", "split_integrity_report.json"),
        ("release", "release_audit.md"),
    ):
        assert (report_root / category / release_dir.name / filename).is_file()


def test_failed_integrity_validation_prevents_promotion_and_cleans_stage(
    tmp_path: Path,
    synthetic_release_inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fast_audio_stages(monkeypatch)
    metadata_dir, data_root = synthetic_release_inputs
    release_dir = tmp_path / "releases" / "failed-release"

    def fail_validation(*_args, **_kwargs):
        raise AssertionError("injected integrity failure")

    monkeypatch.setattr(pipeline_module, "validate_partition_integrity", fail_validation)
    with pytest.raises(AssertionError, match="injected integrity failure"):
        DatasetReleasePipeline(metadata_dir, data_root, release_dir).run()

    assert not release_dir.exists()
    assert not list(release_dir.parent.glob(f".{release_dir.name}.staging-*"))


def test_preexisting_partial_release_is_preserved(
    tmp_path: Path,
    synthetic_release_inputs: tuple[Path, Path],
) -> None:
    metadata_dir, data_root = synthetic_release_inputs
    release_dir = tmp_path / "releases" / "partial-release"
    release_dir.mkdir(parents=True)
    sentinel = release_dir / "partial.txt"
    sentinel.write_text("do not replace", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        DatasetReleasePipeline(metadata_dir, data_root, release_dir).run()

    assert sentinel.read_text(encoding="utf-8") == "do not replace"
