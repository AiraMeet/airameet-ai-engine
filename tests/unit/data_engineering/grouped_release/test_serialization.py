"""Deterministic JSONL serialization and portable manifest path tests."""

import json
from pathlib import Path

import pytest

from airameet_ai.data_engineering.grouped_release.release.release_writer import (
    render_manifest,
)


def _row(utterance_id: str, audio_path: str, text: str) -> dict:
    return {
        "utterance_id": utterance_id,
        "audio_path": audio_path,
        "text": text,
        "recording_id": "C03-R01",
        "session_id": "C03-R01",
        "source_recording_id": "C03-R01-I01",
        "speaker_id": "I01",
        "start": 0.0,
        "end": 1.0,
        "duration": 1.0,
        "source_dataset": "Erzen",
        "split": "train",
        "source_audio_sha256": "a" * 64,
        "identity_link_basis": None,
        "private_field": "not-in-manifest",
    }


def test_manifest_is_stable_jsonl_with_all_expected_fields_and_unicode(
    tmp_path: Path, make_wav_file
) -> None:
    relative_path = "audio/arabic.wav"
    audio_path = make_wav_file(tmp_path / relative_path)
    records = [
        _row("utt-2", relative_path, "مرحبا hello this is English"),
        _row("utt-1", relative_path, "السلام عليكم"),
    ]

    first = render_manifest(records)
    second = render_manifest(list(reversed(records)))
    lines = first.decode("utf-8").splitlines()
    parsed = [json.loads(line) for line in lines]

    assert first == second
    assert [row["utterance_id"] for row in parsed] == ["utt-1", "utt-2"]
    assert all(set(row) == {
        "utterance_id", "audio_path", "text", "recording_id", "session_id",
        "source_recording_id", "speaker_id", "start", "end", "duration",
        "source_dataset", "split", "source_audio_sha256",
    } for row in parsed)
    assert parsed[0]["text"] == "السلام عليكم"
    assert parsed[1]["text"] == "مرحبا hello this is English"
    assert all((tmp_path / row["audio_path"]).is_file() for row in parsed)
    assert all("private_field" not in row for row in parsed)
    assert audio_path.is_file()


def test_null_optional_fields_are_omitted_and_empty_input_is_empty_bytes() -> None:
    row = _row("utt", "audio.wav", "text")
    row["session_id"] = None
    row["source_recording_id"] = None

    parsed = json.loads(render_manifest([row]).decode("utf-8"))

    assert "session_id" not in parsed
    assert "source_recording_id" not in parsed
    assert render_manifest([]) == b""


def test_malformed_jsonl_is_rejected_by_standard_decoder() -> None:
    with pytest.raises(json.JSONDecodeError):
        json.loads('{"utterance_id": "missing quote}')


def test_manifest_serialization_preserves_audio_path_without_rewriting(tmp_path: Path) -> None:
    row = _row("utt", str(tmp_path / "audio.wav"), "text")
    decoded = json.loads(render_manifest([row]).decode("utf-8"))
    assert Path(decoded["audio_path"]).is_absolute()
