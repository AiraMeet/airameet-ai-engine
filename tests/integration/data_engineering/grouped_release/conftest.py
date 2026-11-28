"""Small, isolated dataset fixture for release workflow integration tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def synthetic_release_inputs(tmp_path: Path, make_wav_file) -> tuple[Path, Path]:
    metadata_dir = tmp_path / "processed"
    data_root = tmp_path / "raw"
    metadata_dir.mkdir()
    arabic_root = data_root / "Arabic-English-ASR"
    erzen_root = data_root / "ArZen_Prepared"
    arabic_rows = []
    for index in range(1, 13):
        name = f"{index:06d}.wav"
        path = arabic_root / "audio/train" / name
        make_wav_file(path, freq=400 + index * 20)
        arabic_rows.append(
            {
                "utterance_id": f"AR-{index:03d}",
                "recording_id": "legacy-row-group",
                "speaker_id": "unknown",
                "audio_path": f"audio/train/{name}",
                "text": "مرحبا hello",
                "start": 0.0,
                "end": 0.1,
                "duration": 0.1,
            }
        )

    erzen_name = "I01-C03-R01_000000-000100.wav"
    erzen_path = erzen_root / "audio/train" / erzen_name
    erzen_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(arabic_root / "audio/train/000001.wav", erzen_path)
    erzen_rows = [
        {
            "utt_id": "I01-C03-R01_000000-000100",
            "recording_id": "C03-R01-I01",
            "speaker_id": "I01",
            "audio_path": f"audio/train/{erzen_name}",
            "text": "matching transcript",
            "start": 0.0,
            "end": 0.1,
            "duration": 0.1,
            "status": "success",
        }
    ]
    pd.DataFrame(arabic_rows).to_csv(
        metadata_dir / "Arabic_English_cleaned_metadata.csv", index=False
    )
    pd.DataFrame(erzen_rows).to_csv(
        metadata_dir / "Erzen_cleaned_metadata.csv", index=False
    )
    return metadata_dir, data_root
