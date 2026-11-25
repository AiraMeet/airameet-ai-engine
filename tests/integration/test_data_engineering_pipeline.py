"""End-to-end tests for the modular data engineering pipeline."""

from pathlib import Path

import pandas as pd
import soundfile as sf

from airameet_ai.core.config import Settings
from airameet_ai.data_engineering import pipeline as pipeline_module
from airameet_ai.data_engineering.pipeline import DataEngineeringPipeline


def test_pipeline_runs_in_notebook_order_and_keeps_raw_audio(
    tmp_path: Path,
    make_wav_file,
    monkeypatch,
) -> None:
    raw_root = tmp_path / "data"
    processed_root = tmp_path / "data" / "processed"
    artifacts_root = tmp_path / "artifacts"
    arzen_root = raw_root / "ArZen_Prepared"
    arabic_root = raw_root / "Arabic-English-ASR"

    arzen_audio = make_wav_file(arzen_root / "audio" / "arzen.wav", duration=1.0)
    arabic_audio = make_wav_file(
        arabic_root / "audio" / "arabic.wav",
        sample_rate=8000,
        duration=1.0,
        channels=2,
    )
    metadata = pd.DataFrame(
        {
            "meeting": ["M1"],
            "speaker": ["S1"],
            "audio_path": ["audio/arzen.wav"],
            "text": ["hello"],
            "start": [0.0],
            "end": [1.0],
            "duration": [1.0],
        }
    )
    arabic_metadata = metadata.assign(audio_path="audio/arabic.wav")
    metadata_sources = {"ArZen": metadata, "Arabic-English-ASR": arabic_metadata}
    monkeypatch.setattr(
        pipeline_module,
        "load_metadata",
        lambda path: metadata_sources[
            "ArZen" if "ArZen" in str(path) else "Arabic-English-ASR"
        ],
    )

    pipeline = DataEngineeringPipeline(
        Settings(
            raw_data_dir=raw_root,
            processed_data_dir=processed_root,
            artifacts_dir=artifacts_root,
        )
    )
    stage_order = []
    for stage_name in (
        "load_datasets",
        "inspect_metadata",
        "audit_arzen",
        "convert_nonstandard_audio",
        "audit_arabic_english",
        "write_reports",
        "create_handoff",
    ):
        original_stage = getattr(pipeline, stage_name)

        def record_stage(method=original_stage, name=stage_name):
            stage_order.append(name)
            return method()

        monkeypatch.setattr(pipeline, stage_name, record_stage)

    results = pipeline.run()

    assert stage_order == [
        "load_datasets",
        "inspect_metadata",
        "audit_arzen",
        "convert_nonstandard_audio",
        "audit_arabic_english",
        "write_reports",
        "create_handoff",
    ]
    converted = processed_root / "Arabic-English-ASR" / "audio" / "arabic.wav"
    converted_info = sf.info(converted)
    assert converted_info.samplerate == 16000
    assert converted_info.channels == 1
    assert sf.info(arabic_audio).samplerate == 8000
    assert sf.info(arzen_audio).samplerate == 16000
    assert results.audio_audits["ArZen"].loc[0, "final_audit_status"]
    assert results.report_paths["ArZen_report"].is_file()
    assert results.report_paths["Arabic-English-ASR_ledger"].is_file()
    assert results.handoff_dir is not None
    assert (results.handoff_dir / "audio" / "arabic.wav").is_file()
    assert len(results.conversions) == 1
