"""Source identity correction and validation tests."""

from pathlib import Path

import pandas as pd

from airameet_ai.data_engineering.grouped_release.metadata.metadata_corrector import (
    prepare_source_records,
)


def _source(name: str, rows: list[dict], path: Path) -> dict:
    return {name: (path, pd.DataFrame(rows))}


def test_erzen_rows_use_verified_session_and_keep_track_id(
    tmp_path: Path, make_wav_file
) -> None:
    audio_root = tmp_path / "erzen"
    relative = "audio/train/track.wav"
    make_wav_file(audio_root / relative)
    row = {
        "utt_id": "I01-C03-R01_000000-000100",
        "recording_id": "C03-R01-I01",
        "speaker_id": "I01",
        "audio_path": relative,
        "text": "Hello",
        "duration": 0.1,
        "status": "success",
    }

    records, corrected, _ = prepare_source_records(
        _source("Erzen", [row], tmp_path / "Erzen_cleaned_metadata.csv"),
        {"Erzen": audio_root},
    )

    assert records[0]["session_id"] == "C03-R01"
    assert records[0]["recording_id"] == "C03-R01"
    assert records[0]["source_recording_id"] == "C03-R01-I01"
    assert records[0]["split_group_key"] == "session:C03-R01"
    assert corrected[0]["session_id"] == "C03-R01"


def test_arabic_english_recording_ids_are_per_wav_and_session_unknown(
    tmp_path: Path, make_wav_file
) -> None:
    audio_root = tmp_path / "arabic"
    rows = []
    for stem in ("000001", "000002"):
        relative = f"audio/train/{stem}.wav"
        make_wav_file(audio_root / relative, freq=440 if stem == "000001" else 660)
        rows.append(
            {
                "utterance_id": f"utt-{stem}",
                "recording_id": "legacy-fake-group",
                "audio_path": relative,
                "text": "السلام hello",
                "duration": 0.1,
                "speaker_id": "unknown",
            }
        )

    records, corrected, _ = prepare_source_records(
        _source("Arabic-English", rows, tmp_path / "Arabic_English_cleaned_metadata.csv"),
        {"Arabic-English": audio_root},
    )

    assert [row["recording_id"] for row in records] == ["areng:000001", "areng:000002"]
    assert len({row["recording_id"] for row in records}) == 2
    assert all(row["session_id"] is None for row in records)
    assert all(row["legacy_recording_id"] == "legacy-fake-group" for row in corrected)
    assert all("legacy-fake-group" not in row["split_group_key"] for row in records)


def test_invalid_erzen_identity_is_retained_as_ineligible(
    tmp_path: Path, make_wav_file
) -> None:
    audio_root = tmp_path / "erzen"
    relative = "audio/track.wav"
    make_wav_file(audio_root / relative)
    row = {
        "utt_id": "unknown-C03-R01_000000-000100",
        "recording_id": "invented-session-track",
        "speaker_id": "unknown",
        "audio_path": relative,
        "text": "transcript",
        "duration": 0.1,
        "status": "success",
    }

    records, corrected, summary = prepare_source_records(
        _source("Erzen", [row], tmp_path / "metadata.csv"), {"Erzen": audio_root}
    )

    assert records == []
    assert corrected[0]["release_eligible"] is False
    assert corrected[0]["session_id"] is None
    assert "invalid_erzen_recording_id" in corrected[0]["release_exclusion_reasons"]
    assert summary["Erzen"]["initially_eligible_rows"] == 0


def test_missing_audio_path_is_rejected_without_fabricating_identity(tmp_path: Path) -> None:
    row = {
        "utterance_id": "utt-missing",
        "recording_id": "legacy",
        "audio_path": None,
        "text": "hello",
        "duration": 1.0,
    }
    records, corrected, _ = prepare_source_records(
        _source(
            "Arabic-English",
            [row],
            tmp_path / "metadata.csv",
        ),
        {"Arabic-English": tmp_path},
    )

    assert records == []
    assert corrected[0]["release_eligible"] is False
    assert corrected[0]["recording_id"] == ""
    assert corrected[0]["session_id"] is None
    assert "missing_audio_path" in corrected[0]["release_exclusion_reasons"]


def test_duplicate_eligible_utterance_ids_across_sources_raise(
    tmp_path: Path, make_wav_file
) -> None:
    erzen_root, arabic_root = tmp_path / "erzen", tmp_path / "arabic"
    erzen_rel, arabic_rel = "track.wav", "000001.wav"
    make_wav_file(erzen_root / erzen_rel)
    make_wav_file(arabic_root / arabic_rel, freq=660)
    erzen = {
        "utt_id": "I01-C03-R01_000000-000100",
        "recording_id": "C03-R01-I01",
        "speaker_id": "I01",
        "audio_path": erzen_rel,
        "text": "one",
        "duration": 0.1,
        "status": "success",
    }
    arabic = {
        "utterance_id": "I01-C03-R01_000000-000100",
        "audio_path": arabic_rel,
        "text": "two",
        "duration": 0.1,
    }

    import pytest

    with pytest.raises(ValueError, match="Duplicate utterance IDs"):
        prepare_source_records(
            {
                **_source("Erzen", [erzen], tmp_path / "erzen.csv"),
                **_source("Arabic-English", [arabic], tmp_path / "arabic.csv"),
            },
            {"Erzen": erzen_root, "Arabic-English": arabic_root},
        )
