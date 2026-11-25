"""Unit tests for DataEngineeringPipeline helpers and stage guards."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from airameet_ai.core.config import Settings
from airameet_ai.data_engineering.pipeline import (
    DataEngineeringPipeline,
    PipelineResults,
)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        artifacts_dir=tmp_path / "artifacts",
    )


@pytest.fixture
def pipeline(settings: Settings) -> DataEngineeringPipeline:
    return DataEngineeringPipeline(settings)


class TestPipelineResults:
    def test_defaults_are_empty(self) -> None:
        results = PipelineResults()
        assert results.metadata_inspections == {}
        assert results.audio_audits == {}
        assert results.conversions == []
        assert results.handoff_dir is None


class TestClippingLevel:
    def test_unknown_for_nan(self) -> None:
        assert DataEngineeringPipeline._clipping_level(float("nan")) == "Unknown"

    def test_no_clipping(self) -> None:
        assert DataEngineeringPipeline._clipping_level(0) == "No Clipping"

    def test_very_low(self) -> None:
        assert DataEngineeringPipeline._clipping_level(0.05) == "Very Low"

    def test_moderate(self) -> None:
        assert DataEngineeringPipeline._clipping_level(0.5) == "Moderate"

    def test_high(self) -> None:
        assert DataEngineeringPipeline._clipping_level(2.0) == "High"


class TestStageGuards:
    def test_require_datasets_raises_when_empty(
        self, pipeline: DataEngineeringPipeline
    ) -> None:
        with pytest.raises(RuntimeError, match="Load datasets before"):
            pipeline.inspect_metadata()

    def test_audit_arabic_english_requires_resolved_paths(
        self, pipeline: DataEngineeringPipeline
    ) -> None:
        pipeline.datasets = {
            "Arabic-English-ASR": (
                Path("/tmp"),
                pd.DataFrame({"audio_path": ["a.wav"]}),
            )
        }
        with pytest.raises(RuntimeError, match="Standardize and resolve"):
            pipeline.audit_arabic_english()

    def test_write_reports_requires_audits(
        self, pipeline: DataEngineeringPipeline
    ) -> None:
        with pytest.raises(RuntimeError, match="Audio audits must run"):
            pipeline.write_reports()

    def test_create_handoff_requires_resolved_paths(
        self, pipeline: DataEngineeringPipeline
    ) -> None:
        with pytest.raises(RuntimeError, match="Audio conversion and path"):
            pipeline.create_handoff()


class TestLoadDatasetsPathResolution:
    """Verify dataset roots resolve from AIRAMEET_RAW_DATA_DIR layouts."""

    def test_resolves_flat_arzen_prepared_under_data_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Matches the real layout: data/ArZen_Prepared/metadata.csv."""
        raw_root = tmp_path / "data"
        arzen_root = raw_root / "ArZen_Prepared"
        arabic_root = raw_root / "Arabic-English-ASR"
        arzen_root.mkdir(parents=True)
        arabic_root.mkdir(parents=True)
        (arzen_root / "metadata.csv").write_text(
            "meeting,speaker,audio_path,text,start,end,duration\n"
            "M1,S1,audio/a.wav,hi,0,1,1\n",
            encoding="utf-8-sig",
        )
        (arabic_root / "metadata.xlsx").write_text("placeholder", encoding="utf-8")

        from airameet_ai.data_engineering import pipeline as pipeline_module

        def fake_load_metadata(path: Path) -> pd.DataFrame:
            path = Path(path)
            if path.name == "metadata.csv":
                return pd.read_csv(path, encoding="utf-8-sig")
            if "Arabic-English-ASR" not in str(path):
                raise FileNotFoundError(path)
            return pd.DataFrame(
                {
                    "meeting": ["M1"],
                    "speaker": ["S1"],
                    "audio_path": ["audio/b.wav"],
                    "text": ["hi"],
                    "start": [0.0],
                    "end": [1.0],
                    "duration": [1.0],
                }
            )

        monkeypatch.setattr(pipeline_module, "load_metadata", fake_load_metadata)

        pipeline = DataEngineeringPipeline(
            Settings(
                raw_data_dir=raw_root,
                processed_data_dir=tmp_path / "processed",
                artifacts_dir=tmp_path / "artifacts",
            )
        )
        loaded = pipeline.load_datasets()

        assert loaded["ArZen"][0] == arzen_root
        assert (loaded["ArZen"][0] / "metadata.csv").is_file()
        assert loaded["Arabic-English-ASR"][0] == arabic_root

    def test_missing_metadata_raises_file_not_found(self, tmp_path: Path) -> None:
        pipeline = DataEngineeringPipeline(
            Settings(
                raw_data_dir=tmp_path / "data",
                processed_data_dir=tmp_path / "processed",
                artifacts_dir=tmp_path / "artifacts",
            )
        )
        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            pipeline.load_datasets()

    def test_get_settings_default_resolves_project_data_dir(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from airameet_ai.core.config import PROJECT_ROOT, get_settings

        monkeypatch.delenv("AIRAMEET_RAW_DATA_DIR", raising=False)
        settings = get_settings()
        assert settings.raw_data_dir == PROJECT_ROOT / "data"
        expected_metadata = settings.raw_data_dir / "ArZen_Prepared" / "metadata.csv"
        assert expected_metadata == PROJECT_ROOT / "data" / "ArZen_Prepared" / "metadata.csv"
