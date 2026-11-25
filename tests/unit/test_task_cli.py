from pathlib import Path

import pandas as pd
import pytest

import main


@pytest.fixture
def cli_inputs(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame(
        {
            "utt_id": ["one"],
            "audio_path": ["a.wav"],
            "text": ["أَحْمَد"],
            "split": ["train"],
            "speaker_id": ["s"],
            "recording_id": ["r"],
            "start": [0.0],
            "end": [1.0],
            "duration": [1.0],
            "status": ["success"],
        }
    ).to_csv(source / "Erzen_metadata.csv", index=False)
    pd.DataFrame(
        {
            "meeting": ["m"],
            "speaker": ["s"],
            "audio_path": ["b.wav"],
            "text": ["Hello"],
            "start": [0.0],
            "end": [1.0],
            "duration": [1.0],
        }
    ).to_excel(source / "arabic-english-metadata.xlsx", index=False)
    return source


def test_list_tasks(capsys: pytest.CaptureFixture[str]) -> None:
    assert main.main(["--list-tasks"]) == 0
    output = capsys.readouterr().out
    assert "AI-DS-01" in output
    assert "AI-DS-02" in output


def test_ai_ds_02_runs_only_with_custom_paths(
    cli_inputs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main, "configure_logging", lambda _level: None)
    output = tmp_path / "custom-output"
    assert (
        main.main(
            [
                "--task",
                "AI-DS-02",
                "--input",
                str(cli_inputs),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert (output / "Erzen_cleaned_metadata.csv").is_file()
    assert (output / "Arabic_English_cleaned_metadata.csv").is_file()


def test_unknown_task_has_argument_error() -> None:
    with pytest.raises(SystemExit) as exc:
        main.main(["--task", "NOPE"])
    assert exc.value.code == 2


def test_task_failure_returns_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main, "configure_logging", lambda _level: None)
    assert main.main(["--task", "AI-DS-02", "--input", str(tmp_path)]) == 1
