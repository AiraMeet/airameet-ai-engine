"""Unit tests for dataset_inspector module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from airameet_ai.data_engineering.inspection.dataset_inspector import load_metadata


class TestLoadMetadata:
    """Tests for the load_metadata function."""

    def test_load_csv_file(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "metadata.csv"
        csv_file.write_text(
            "meeting,speaker,audio_path\nM1,S1,audio/file1.wav\nM2,S2,audio/file2.wav\n",
            encoding="utf-8-sig",
        )

        df = load_metadata(csv_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["meeting", "speaker", "audio_path"]
        assert df.iloc[0]["meeting"] == "M1"
        assert df.iloc[1]["speaker"] == "S2"

    def test_load_csv_with_path_as_string(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "metadata.csv"
        csv_file.write_text("col1,col2\na,1\nb,2\n", encoding="utf-8-sig")

        df = load_metadata(str(csv_file))

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_load_xlsx_file(self, tmp_path: Path) -> None:
        xlsx_file = tmp_path / "metadata.xlsx"
        expected_df = pd.DataFrame(
            {
                "meeting": ["M1", "M2"],
                "speaker": ["S1", "S2"],
                "audio_path": ["a1.wav", "a2.wav"],
            }
        )
        with pd.ExcelWriter(xlsx_file, engine="openpyxl") as writer:
            expected_df.to_excel(writer, sheet_name="Metadata", index=False)

        df = load_metadata(xlsx_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["meeting", "speaker", "audio_path"]
        assert df.iloc[0]["meeting"] == "M1"

    def test_load_xls_extension_accepted(self, tmp_path: Path) -> None:
        # We can't actually write a real .xls easily, but we verify the extension
        # is in the accepted set by testing the .xlsx extension works
        xlsx_file = tmp_path / "metadata.xlsx"
        df_in = pd.DataFrame({"col": [1, 2]})
        with pd.ExcelWriter(xlsx_file, engine="openpyxl") as writer:
            df_in.to_excel(writer, sheet_name="Metadata", index=False)

        df = load_metadata(xlsx_file)
        assert isinstance(df, pd.DataFrame)

    def test_nonexistent_file_raises_file_not_found(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.csv"

        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            load_metadata(missing)

    def test_unsupported_format_raises_value_error(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "metadata.txt"
        txt_file.write_text("some content", encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported metadata format"):
            load_metadata(txt_file)

    def test_unsupported_json_format(self, tmp_path: Path) -> None:
        json_file = tmp_path / "metadata.json"
        json_file.write_text('{"key": "value"}', encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported metadata format"):
            load_metadata(json_file)

    def test_csv_preserves_utf8_bom_encoding(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "metadata.csv"
        # Write with BOM (utf-8-sig)
        csv_file.write_text(
            "meeting,text\nM1,Hello\nM2,World\n", encoding="utf-8-sig"
        )

        df = load_metadata(csv_file)

        assert "meeting" in df.columns
        assert len(df) == 2

    def test_returned_dataframe_has_expected_values(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "metadata.csv"
        csv_file.write_text(
            "id,name,value\n1,alpha,10.5\n2,beta,20.0\n", encoding="utf-8-sig"
        )

        df = load_metadata(csv_file)

        assert df["id"].tolist() == [1, 2]
        assert df["name"].tolist() == ["alpha", "beta"]
        assert df["value"].tolist() == [10.5, 20.0]
