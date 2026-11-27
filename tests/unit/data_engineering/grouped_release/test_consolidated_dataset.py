"""AI-DS-03 consumes the existing consolidated manifests and audio."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from airameet_ai.data_engineering.grouped_release.consolidated_dataset import (
    ConsolidatedDatasetPipeline,
)


def _dataset(root: Path) -> Path:
    for split, dataset, recording in (
        ("train", "Erzen", "recording-1"),
        ("validation", "Arabic-English", "recording-2"),
        ("test", "Erzen", "recording-3"),
    ):
        audio = Path("audio") / ("Arabic-English-ASR" if dataset == "Arabic-English" else "Erzen") / f"{split}.wav"
        audio_path = root / audio
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"synthetic audio")
        manifest = root / "manifests" / f"{split}.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "utterance_id": f"utt-{split}",
            "audio_path": audio.as_posix(),
            "text": f"transcript {split}",
            "recording_id": recording,
            "session_id": recording,
            "speaker_id": "speaker-1",
            "source_dataset": dataset,
            "duration": 1.25,
            "split": split,
        }
        manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return root


def test_ai_ds_03_reads_all_existing_splits_and_relative_audio(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "asr-dataset")
    before = {
        path.relative_to(dataset).as_posix(): path.read_bytes()
        for path in dataset.rglob("*")
        if path.is_file()
    }
    result = ConsolidatedDatasetPipeline(dataset).run()
    after = {
        path.relative_to(dataset).as_posix(): path.read_bytes()
        for path in dataset.rglob("*")
        if path.is_file()
    }
    assert result["manifest_rows"] == 3
    assert result["audio_files_validated"] == 3
    assert result["statistics"]["splits"]["train"]["utterances"] == 1
    assert result["statistics"]["splits"]["validation"]["utterances"] == 1
    assert result["statistics"]["splits"]["test"]["utterances"] == 1
    assert before == after


def test_ai_ds_03_writes_only_report_outside_dataset(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "asr-dataset")
    reports = tmp_path / "reports"
    result = ConsolidatedDatasetPipeline(dataset, reports).run()
    report = Path(result["report"])
    assert report == reports / "ai_ds_03_statistics.json"
    assert json.loads(report.read_text(encoding="utf-8"))["manifest_rows"] == 3
    assert not (dataset / "reports").exists()


def test_ai_ds_03_rejects_missing_audio_and_wrong_split(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "asr-dataset")
    train = dataset / "manifests/train.jsonl"
    record = json.loads(train.read_text(encoding="utf-8"))
    record["audio_path"] = "audio/Erzen/missing.wav"
    train.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Manifest audio is missing"):
        ConsolidatedDatasetPipeline(dataset).run()

    record["audio_path"] = "audio/Erzen/train.wav"
    record["split"] = "test"
    train.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Split assignment mismatch"):
        ConsolidatedDatasetPipeline(dataset).run()


def test_ai_ds_03_rejects_paths_that_escape_dataset_root(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "asr-dataset")
    train = dataset / "manifests/train.jsonl"
    record = json.loads(train.read_text(encoding="utf-8"))
    record["audio_path"] = "../../outside.wav"
    train.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsafe relative audio_path"):
        ConsolidatedDatasetPipeline(dataset).run()
