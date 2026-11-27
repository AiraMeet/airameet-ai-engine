"""CLI registration, argument validation, and failure propagation tests."""

from pathlib import Path

import pytest

import main as app_main
from airameet_ai.core.config import Settings


def test_ai_ds_03_cli_invokes_pipeline_with_configured_and_overridden_paths(
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

    class FakeReleasePipeline:
        def __init__(self, metadata_dir, data_root, release_dir, reports_root):
            captured.update(
                metadata_dir=metadata_dir,
                data_root=data_root,
                release_dir=release_dir,
                reports_root=reports_root,
            )

        def run(self):
            return {"release": "data-v1.1"}

    monkeypatch.setattr(app_main, "DatasetReleasePipeline", FakeReleasePipeline)
    metadata = tmp_path / "custom-metadata"
    release = tmp_path / "custom-release"

    assert app_main.main(["--task", "AI-DS-03", "--input", str(metadata), "--output", str(release)]) == 0
    assert captured == {
        "metadata_dir": metadata,
        "data_root": settings.raw_data_dir,
        "release_dir": release,
        "reports_root": settings.artifacts_dir / "data_engineering" / "reports",
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

    class FakeReleasePipeline:
        def __init__(self, *args):
            captured["args"] = args

        def run(self):
            return {"release": "data-v1.1"}

    monkeypatch.setattr(app_main, "DatasetReleasePipeline", FakeReleasePipeline)
    assert app_main.main(["--task", "AI-DS-03"]) == 0
    assert captured["args"][0] == settings.processed_data_dir / "Arabic-English-ASR"
    assert captured["args"][1] == settings.raw_data_dir
    assert captured["args"][2].name == "data-v1.1"


def test_invalid_cli_task_is_rejected_by_parser() -> None:
    with pytest.raises(SystemExit) as result:
        app_main.main(["--task", "AI-DS-99"])
    assert result.value.code == 2


def test_release_pipeline_failure_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_main, "get_settings", Settings)
    monkeypatch.setattr(app_main, "configure_logging", lambda _level: None)

    class FailingReleasePipeline:
        def __init__(self, *_args):
            pass

        def run(self):
            raise RuntimeError("synthetic pipeline failure")

    monkeypatch.setattr(app_main, "DatasetReleasePipeline", FailingReleasePipeline)
    assert app_main.main(["--task", "AI-DS-03"]) == 1
