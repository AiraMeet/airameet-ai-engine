"""CLI registration, argument validation, and failure propagation tests."""

from pathlib import Path

import pytest

import main as app_main
from airameet_ai.core.config import Settings


def test_ai_ds_03_cli_reads_consolidated_dataset_with_overridden_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        artifacts_dir=tmp_path / "artifacts",
    )
    monkeypatch.setattr(app_main, "get_settings", lambda: settings)
    monkeypatch.setattr(app_main, "configure_logging", lambda _level: None)
    captured = {}

    class FakeConsolidatedPipeline:
        def __init__(self, dataset_root, reports_root):
            captured.update(
                dataset_root=dataset_root,
                reports_root=reports_root,
            )

        def run(self):
            return {"dataset": "asr-dataset"}

    monkeypatch.setattr(app_main, "ConsolidatedDatasetPipeline", FakeConsolidatedPipeline)
    dataset = tmp_path / "custom-dataset"
    reports = tmp_path / "custom-reports"

    assert app_main.main(["--task", "AI-DS-03", "--input", str(dataset), "--output", str(reports)]) == 0
    assert captured == {
        "dataset_root": dataset,
        "reports_root": reports,
    }


def test_ai_ds_03_cli_uses_project_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = Settings(
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        artifacts_dir=tmp_path / "artifacts",
    )
    monkeypatch.setattr(app_main, "get_settings", lambda: settings)
    monkeypatch.setattr(app_main, "configure_logging", lambda _level: None)
    captured = {}

    class FakeConsolidatedPipeline:
        def __init__(self, dataset_root, reports_root):
            captured["dataset_root"] = dataset_root
            captured["reports_root"] = reports_root

        def run(self):
            return {"dataset": "asr-dataset"}

    monkeypatch.setattr(app_main, "ConsolidatedDatasetPipeline", FakeConsolidatedPipeline)
    assert app_main.main(["--task", "AI-DS-03"]) == 0
    assert captured == {
        "dataset_root": app_main.PROJECT_ROOT / "data" / "asr-dataset",
        "reports_root": settings.artifacts_dir / "data_engineering" / "reports",
    }


def test_ai_ds_03_never_invokes_v11_release_builder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        artifacts_dir=tmp_path / "artifacts",
    )
    monkeypatch.setattr(app_main, "get_settings", lambda: settings)
    monkeypatch.setattr(app_main, "configure_logging", lambda _level: None)

    class FakeConsolidatedPipeline:
        def __init__(self, *_args):
            pass

        def run(self):
            return {"dataset": "asr-dataset"}

    class ForbiddenReleaseBuilder:
        def __init__(self, *_args):
            raise AssertionError("AI-DS-03 must not invoke the v1.1 release builder")

    monkeypatch.setattr(app_main, "ConsolidatedDatasetPipeline", FakeConsolidatedPipeline)
    monkeypatch.setattr(app_main, "DatasetReleasePipeline", ForbiddenReleaseBuilder, raising=False)
    assert app_main.main(["--task", "AI-DS-03"]) == 0


def test_invalid_cli_task_is_rejected_by_parser() -> None:
    with pytest.raises(SystemExit) as result:
        app_main.main(["--task", "AI-DS-99"])
    assert result.value.code == 2


def test_consolidated_dataset_pipeline_failure_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_main, "get_settings", Settings)
    monkeypatch.setattr(app_main, "configure_logging", lambda _level: None)

    class FailingConsolidatedPipeline:
        def __init__(self, *_args):
            pass

        def run(self):
            raise RuntimeError("synthetic pipeline failure")

    monkeypatch.setattr(app_main, "ConsolidatedDatasetPipeline", FailingConsolidatedPipeline)
    assert app_main.main(["--task", "AI-DS-03"]) == 1
