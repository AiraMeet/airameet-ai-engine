"""Unit tests for the application entry point."""

from __future__ import annotations

from pathlib import Path

import pytest

import main as app_main
from airameet_ai.core.config import Settings
from airameet_ai.data_engineering.pipeline import PipelineResults


def test_main_returns_zero_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        raw_data_dir=Path("/tmp/raw"),
        processed_data_dir=Path("/tmp/processed"),
        artifacts_dir=Path("/tmp/artifacts"),
    )
    results = PipelineResults(
        report_paths={"demo": Path("/tmp/report.json")},
        handoff_dir=Path("/tmp/handoff"),
    )

    monkeypatch.setattr(app_main, "get_settings", lambda: settings)
    monkeypatch.setattr(app_main, "configure_logging", lambda _level: None)

    class FakePipeline:
        def __init__(self, _settings: Settings) -> None:
            pass

        def run(self) -> PipelineResults:
            return results

    monkeypatch.setattr(app_main, "DataEngineeringPipeline", FakePipeline)

    assert app_main.main() == 0


def test_main_returns_one_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings()
    monkeypatch.setattr(app_main, "get_settings", lambda: settings)
    monkeypatch.setattr(app_main, "configure_logging", lambda _level: None)

    class FailingPipeline:
        def __init__(self, _settings: Settings) -> None:
            pass

        def run(self) -> PipelineResults:
            raise RuntimeError("boom")

    monkeypatch.setattr(app_main, "DataEngineeringPipeline", FailingPipeline)

    assert app_main.main() == 1
