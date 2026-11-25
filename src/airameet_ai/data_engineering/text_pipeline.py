"""Orchestration for the AI-DS-02 text normalization workflow."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from airameet_ai.data_engineering.conversion import normalize_text
from airameet_ai.data_engineering.exporting import export_metadata
from airameet_ai.data_engineering.inspection import load_metadata, profile_text
from airameet_ai.data_engineering.validation.text_validator import (
    validate_cleaned_dataset,
    validate_unified_metadata,
)

LOGGER = logging.getLogger(__name__)
COMMON_COLUMNS = [
    "utterance_id",
    "audio_path",
    "text",
    "split",
    "speaker_id",
    "recording_id",
    "start",
    "end",
    "duration",
    "source_dataset",
]


def process_dataset(
    frame: pd.DataFrame, dataset_name: str
) -> tuple[pd.DataFrame, dict]:
    """Normalize string values in text while retaining all other values."""
    if "text" not in frame.columns:
        raise ValueError(f"{dataset_name}: required column 'text' was not found")
    cleaned = frame.copy(deep=True)
    original = cleaned["text"].copy()
    cleaned["text"] = cleaned["text"].map(
        lambda value: normalize_text(value) if isinstance(value, str) else value
    )
    changed = original.ne(cleaned["text"]) & ~(original.isna() & cleaned["text"].isna())
    return cleaned, {
        "dataset": dataset_name,
        "total_records": len(cleaned),
        "changed_texts": int(changed.sum()),
        "unchanged_texts": len(cleaned) - int(changed.sum()),
        "missing_texts": int(cleaned["text"].isna().sum()),
    }


def build_unified_metadata(
    erzen: pd.DataFrame, arabic_english: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform the notebook's final source-specific schema standardization."""
    required_erzen = [
        "status",
        "utt_id",
        *COMMON_COLUMNS[1:3],
        "split",
        "speaker_id",
        "recording_id",
        "start",
        "end",
        "duration",
    ]
    missing = [column for column in required_erzen if column not in erzen.columns]
    if missing:
        raise ValueError(f"Erzen metadata is missing columns: {missing}")
    erzen_rows = erzen.loc[
        erzen["status"]
        .astype("string")
        .str.strip()
        .str.lower()
        .eq("success")
        .fillna(False)
    ].copy()
    erzen_rows = erzen_rows.rename(columns={"utt_id": "utterance_id"})
    erzen_rows["source_dataset"] = "Erzen"
    erzen_rows["utterance_id"] = "erzen_" + erzen_rows["utterance_id"].astype(str)

    required_arabic = [
        "meeting",
        "speaker",
        "audio_path",
        "text",
        "start",
        "end",
        "duration",
    ]
    missing = [
        column for column in required_arabic if column not in arabic_english.columns
    ]
    if missing:
        raise ValueError(f"Arabic-English metadata is missing columns: {missing}")
    arabic_rows = arabic_english.copy()
    arabic_rows = arabic_rows.rename(
        columns={"meeting": "recording_id", "speaker": "speaker_id"}
    )
    arabic_rows["utterance_id"] = "arabic_english_" + arabic_rows.index.astype(
        str
    ).str.zfill(6)
    arabic_rows["split"] = "unknown"
    arabic_rows["source_dataset"] = "Arabic-English"
    erzen_final, arabic_final = (
        erzen_rows[COMMON_COLUMNS].copy(),
        arabic_rows[COMMON_COLUMNS].copy(),
    )
    for final in (erzen_final, arabic_final):
        validate_unified_metadata(final)
    if not erzen_final["source_dataset"].eq("Erzen").all():
        raise ValueError("Erzen source labels are inconsistent")
    if not arabic_final["source_dataset"].eq("Arabic-English").all():
        raise ValueError("Arabic-English source labels are inconsistent")
    return erzen_final, arabic_final


class TextNormalizationPipeline:
    """Load, normalize, validate, and export AI-DS-02 metadata sources."""

    def __init__(self, input_dir: str | Path, output_dir: str | Path) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

    def _find_input(self, candidates: tuple[str, ...]) -> Path:
        for candidate in candidates:
            path = self.input_dir / candidate
            if path.is_file():
                return path
        options = ", ".join(candidates)
        raise FileNotFoundError(
            f"Could not find metadata under {self.input_dir}; expected one of: {options}"
        )

    def run(self) -> dict[str, object]:
        """Run notebook workflow and return output paths and processing metrics."""
        erzen_path = self._find_input(
            (
                "Erzen_metadata.csv",
                "ArZen_Prepared/metadata.csv",
                "ArzEn_Prepared/metadata.csv",
                "ArZen/ArZen_Prepared/metadata.csv",
            )
        )
        arabic_path = self._find_input(
            (
                "arabic-english-metadata.xlsx",
                "Arabic-English-ASR/metadata.xlsx",
                "Arabic-English-ASR/Arabic-English-ASR/metadata.xlsx",
            )
        )
        originals = {
            "Erzen": load_metadata(erzen_path),
            "Arabic-English": load_metadata(arabic_path, sheet_name=0),
        }
        cleaned: dict[str, pd.DataFrame] = {}
        stats: dict[str, dict] = {}
        profiles: dict[str, dict] = {}
        for name, frame in originals.items():
            LOGGER.info("Inspecting and normalizing %s: %s", name, frame.shape)
            profiles[name] = profile_text(frame)
            cleaned[name], stats[name] = process_dataset(frame, name)
            validate_cleaned_dataset(frame, cleaned[name])
        # This is the notebook's in-memory unified-schema/readiness validation.
        build_unified_metadata(cleaned["Erzen"], cleaned["Arabic-English"])
        outputs = {
            "Erzen": export_metadata(
                cleaned["Erzen"], self.output_dir / "Erzen_cleaned_metadata.csv"
            ),
            "Arabic-English": export_metadata(
                cleaned["Arabic-English"],
                self.output_dir / "Arabic_English_cleaned_metadata.csv",
            ),
        }
        for name, path in outputs.items():
            reloaded = load_metadata(path)
            if list(reloaded.columns) != list(cleaned[name].columns) or len(
                reloaded
            ) != len(cleaned[name]):
                raise ValueError(f"Export validation failed for {path}")
            try:
                pd.testing.assert_frame_equal(
                    cleaned[name].reset_index(drop=True),
                    reloaded.reset_index(drop=True),
                    check_dtype=False,
                    check_categorical=False,
                )
            except AssertionError as error:
                raise ValueError(
                    f"Export contents failed validation for {path}"
                ) from error
        return {"outputs": outputs, "stats": stats, "profiles": profiles}
