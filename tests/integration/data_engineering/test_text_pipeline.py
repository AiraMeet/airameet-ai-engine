from pathlib import Path

import pandas as pd
import pytest

from airameet_ai.data_engineering.text_pipeline import TextNormalizationPipeline


@pytest.fixture
def notebook_inputs(tmp_path: Path) -> Path:
    source = tmp_path / "inputs"
    source.mkdir()
    erzen = pd.DataFrame(
        {
            "utt_id": ["e1", "e2"],
            "audio_path": ["a.wav", "b.wav"],
            "text": ["أَحْمَد", None],
            "split": ["train", "test"],
            "speaker_id": ["s1", "s2"],
            "recording_id": ["r1", "r2"],
            "start": [0.0, 1.0],
            "end": [1.0, 2.0],
            "duration": [1.0, 1.0],
            "status": ["success", "failed"],
            "notes": ["keep", "keep too"],
        }
    )
    arabic = pd.DataFrame(
        {
            "meeting": ["m1"],
            "speaker": ["s1"],
            "audio_path": ["c.wav"],
            "text": ['"Hello يا أحمد"'],
            "start": [0.0],
            "end": [2.0],
            "duration": [2.0],
            "extra": ["preserve"],
        }
    )
    erzen.to_csv(source / "Erzen_metadata.csv", index=False, encoding="utf-8-sig")
    arabic.to_excel(source / "arabic-english-metadata.xlsx", index=False)
    return source


def test_pipeline_exports_utf8_and_preserves_input_columns(
    notebook_inputs: Path, tmp_path: Path
) -> None:
    output = tmp_path / "processed" / "Arabic-English-ASR"
    source_bytes = (notebook_inputs / "Erzen_metadata.csv").read_bytes()
    result = TextNormalizationPipeline(notebook_inputs, output).run()

    assert set(result["outputs"]) == {"Erzen", "Arabic-English"}
    assert all(Path(path).is_file() for path in result["outputs"].values())
    erzen = pd.read_csv(result["outputs"]["Erzen"], encoding="utf-8-sig")
    arabic = pd.read_csv(result["outputs"]["Arabic-English"], encoding="utf-8-sig")
    assert erzen.columns.tolist() == [
        "utt_id",
        "audio_path",
        "text",
        "split",
        "speaker_id",
        "recording_id",
        "start",
        "end",
        "duration",
        "status",
        "notes",
    ]
    assert erzen["text"].iloc[0] == "احمد"
    assert pd.isna(erzen["text"].iloc[1])
    assert erzen["notes"].tolist() == ["keep", "keep too"]
    assert arabic["text"].iloc[0] == "Hello يا احمد"
    assert arabic["extra"].iloc[0] == "preserve"
    assert (notebook_inputs / "Erzen_metadata.csv").read_bytes() == source_bytes
    assert result["stats"]["Erzen"]["total_records"] == 2


def test_pipeline_handles_empty_datasets(notebook_inputs: Path, tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "utt_id": pd.Series(dtype=str),
            "audio_path": pd.Series(dtype=str),
            "text": pd.Series(dtype=str),
            "split": pd.Series(dtype=str),
            "speaker_id": pd.Series(dtype=str),
            "recording_id": pd.Series(dtype=str),
            "start": pd.Series(dtype=float),
            "end": pd.Series(dtype=float),
            "duration": pd.Series(dtype=float),
            "status": pd.Series(dtype=str),
        }
    ).to_csv(notebook_inputs / "Erzen_metadata.csv", index=False)
    pd.DataFrame(
        {
            "meeting": pd.Series(dtype=str),
            "speaker": pd.Series(dtype=str),
            "audio_path": pd.Series(dtype=str),
            "text": pd.Series(dtype=str),
            "start": pd.Series(dtype=float),
            "end": pd.Series(dtype=float),
            "duration": pd.Series(dtype=float),
        }
    ).to_excel(notebook_inputs / "arabic-english-metadata.xlsx", index=False)
    result = TextNormalizationPipeline(notebook_inputs, tmp_path / "out").run()
    assert result["stats"]["Erzen"]["total_records"] == 0


def test_pipeline_reports_missing_input_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Could not find metadata"):
        TextNormalizationPipeline(tmp_path, tmp_path / "out").run()


def test_pipeline_rejects_missing_text_column(
    notebook_inputs: Path, tmp_path: Path
) -> None:
    pd.DataFrame({"other": ["value"]}).to_csv(
        notebook_inputs / "Erzen_metadata.csv", index=False
    )
    with pytest.raises(ValueError, match="column 'text'"):
        TextNormalizationPipeline(notebook_inputs, tmp_path / "out").run()


def test_pipeline_rejects_malformed_csv(notebook_inputs: Path, tmp_path: Path) -> None:
    (notebook_inputs / "Erzen_metadata.csv").write_text(
        'utt_id,text\n"unterminated,entry\n', encoding="utf-8"
    )
    with pytest.raises(pd.errors.ParserError):
        TextNormalizationPipeline(notebook_inputs, tmp_path / "out").run()
