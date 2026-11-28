"""Coordinated, non-destructive implementation of the AI-DS-01 workflow."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from airameet_ai.core.config import Settings
from airameet_ai.data_engineering.conversion import convert_audio_batch
from airameet_ai.data_engineering.inspection import load_metadata
from airameet_ai.data_engineering.quality import audit_audio_files
from airameet_ai.data_engineering.validation import (
    inspect_metadata as inspect_metadata_frame,
)
from airameet_ai.data_engineering.validation import (
    metadata_quality_summary,
    validate_audio_paths,
    validate_metadata_schema,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineResults:
    """Useful artifacts and summaries produced by one pipeline run."""

    metadata_inspections: dict[str, dict] = field(default_factory=dict)
    metadata_quality: dict[str, dict] = field(default_factory=dict)
    schema_reports: dict[str, dict] = field(default_factory=dict)
    path_reports: dict[str, pd.DataFrame] = field(default_factory=dict)
    audio_audits: dict[str, pd.DataFrame] = field(default_factory=dict)
    conversions: list[dict] = field(default_factory=list)
    report_paths: dict[str, Path] = field(default_factory=dict)
    handoff_dir: Path | None = None


class DataEngineeringPipeline:
    """Run the inspection, conversion, audit, reporting, and handoff stages."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.results = PipelineResults()
        self.datasets: dict[str, tuple[Path, pd.DataFrame]] = {}
        self._arabic_audio_paths: pd.Series | None = None

    def load_datasets(self) -> dict[str, tuple[Path, pd.DataFrame]]:
        """Load the two metadata sources referenced by AI-DS-01."""
        raw_root = self.settings.raw_data_dir
        # Candidates are relative to AIRAMEET_RAW_DATA_DIR (default: data/).
        # Prefer the flat on-disk layout used in this repo, then nested variants.
        dataset_specs = {
            "ArZen": (
                (
                    raw_root / "ArZen_Prepared",
                    raw_root / "ArzEn_Prepared",
                    raw_root / "ArZen" / "ArZen_Prepared",
                    raw_root / "ArZen",
                    raw_root / "ArzEn",
                ),
                "metadata.csv",
            ),
            "Arabic-English-ASR": (
                (
                    raw_root / "Arabic-English-ASR",
                    raw_root / "Arabic-English-ASR" / "Arabic-English-ASR",
                ),
                "metadata.xlsx",
            ),
        }
        loaded = {}
        for name, (dataset_roots, metadata_name) in dataset_specs.items():
            dataset_root = next(
                (
                    candidate
                    for candidate in dataset_roots
                    if (candidate / metadata_name).is_file()
                ),
                dataset_roots[0],
            )
            metadata = load_metadata(dataset_root / metadata_name)
            if "audio_path" not in metadata.columns:
                raise ValueError(f"{name} metadata is missing the audio_path column")
            loaded[name] = (dataset_root, metadata)
        self.datasets = loaded
        return loaded

    def inspect_metadata(self) -> PipelineResults:
        """Record the notebook's metadata summaries and path validations."""
        self._require_datasets()
        for name, (dataset_root, metadata) in self.datasets.items():
            LOGGER.info("Inspecting %s metadata", name)
            self.results.metadata_inspections[name] = inspect_metadata_frame(
                metadata, name
            )
            self.results.metadata_quality[name] = metadata_quality_summary(
                metadata, name
            )
            if name == "Arabic-English-ASR":
                self.results.schema_reports[name] = validate_metadata_schema(
                    metadata, name
                )
            self.results.path_reports[name] = validate_audio_paths(
                metadata, dataset_root
            )
        return self.results

    def convert_nonstandard_audio(self) -> list[dict]:
        """Standardize Arabic-English audio into a separate processed tree."""
        self._require_datasets()
        name = "Arabic-English-ASR"
        dataset_root, metadata = self.datasets[name]
        LOGGER.info("Inspecting source audio formats for %s", name)
        format_rows: list[dict[str, object]] = []
        for raw_path in metadata["audio_path"]:
            path = Path(str(raw_path))
            source = (dataset_root / path).resolve()
            row: dict[str, object] = {
                "audio_path": raw_path,
                "needs_conversion": False,
                "status": "unreadable",
            }
            try:
                if path.is_absolute() or not source.is_relative_to(
                    dataset_root.resolve()
                ):
                    raise ValueError("Audio path points outside the dataset root")
                info = sf.info(source)
                row.update(
                    sample_rate=info.samplerate,
                    channels=info.channels,
                    format=info.format,
                    subtype=info.subtype,
                    status="readable",
                    needs_conversion=(
                        info.samplerate != 16000
                        or info.channels != 1
                        or info.format != "WAV"
                        or info.subtype != "PCM_16"
                    ),
                )
            except (OSError, RuntimeError, ValueError) as error:
                row["error"] = str(error)
            format_rows.append(row)

        format_report = pd.DataFrame(format_rows)
        to_convert = format_report.loc[
            format_report["needs_conversion"] & (format_report["status"] == "readable")
        ]
        processed_root = self.settings.processed_data_dir / name
        self.results.conversions = convert_audio_batch(
            to_convert,
            source_root=dataset_root,
            output_root=processed_root,
            path_column="audio_path",
            overwrite=False,
        )

        paths: list[str | None] = []
        for relative_path in metadata["audio_path"]:
            relative = Path(str(relative_path))
            processed_path = (processed_root / relative).resolve()
            raw_path = (dataset_root / relative).resolve()
            if (
                not relative.is_absolute()
                and processed_path.is_relative_to(processed_root.resolve())
                and processed_path.is_file()
            ):
                paths.append(str(processed_path.resolve()))
            elif (
                not relative.is_absolute()
                and raw_path.is_relative_to(dataset_root.resolve())
                and raw_path.is_file()
            ):
                paths.append(str(raw_path.resolve()))
            else:
                paths.append(None)
        self._arabic_audio_paths = pd.Series(paths, index=metadata.index)
        return self.results.conversions

    def audit_audio(self) -> dict[str, pd.DataFrame]:
        """Run full audio audits and apply the notebook's review rules."""
        self._require_datasets()
        for name in self.datasets:
            self._audit_dataset(name)
        return self.results.audio_audits

    def audit_arzen(self) -> pd.DataFrame:
        """Run the ArZen audio quality audit stage."""
        self._require_datasets()
        return self._audit_dataset("ArZen")

    def audit_arabic_english(self) -> pd.DataFrame:
        """Run the final Arabic-English audio quality audit stage."""
        self._require_datasets()
        if self._arabic_audio_paths is None:
            raise RuntimeError("Standardize and resolve audio before auditing")
        return self._audit_dataset("Arabic-English-ASR")

    def _audit_dataset(self, name: str) -> pd.DataFrame:
        dataset_root, metadata = self.datasets[name]
        LOGGER.info("Auditing audio quality for %s", name)
        resolved_paths = (
            self._arabic_audio_paths if name == "Arabic-English-ASR" else None
        )
        # The configured data root contains both raw and processed inputs.
        audit_root = dataset_root
        audio_report = audit_audio_files(
            metadata,
            audit_root,
            audio_paths=resolved_paths,
            allowed_roots=(self.settings.processed_data_dir / "Arabic-English-ASR",)
            if resolved_paths is not None
            else (),
        )
        audio_report.insert(0, "dataset", name)
        for column in reversed(metadata.columns.tolist()):
            if column != "audio_path":
                report_column = (
                    f"metadata_{column}" if column in audio_report.columns else column
                )
                audio_report.insert(1, report_column, metadata[column].to_numpy())
        audio_report["duration_valid"] = audio_report["duration"].between(
            0.5, 30.0, inclusive="both"
        )
        audio_report["snr_at_boundary"] = audio_report["snr_db"].isin([-20, 100])
        audio_report["clipping_level"] = audio_report["clipping_percentage_gt_099"].map(
            self._clipping_level
        )

        if name == "ArZen":
            audio_report["final_audit_status"] = np.select(
                [
                    ~audio_report["readable"],
                    audio_report["duration"] < 0.5,
                    audio_report["snr_status"] != "success",
                    audio_report["snr_at_boundary"],
                    audio_report["clipping_level"].isin(["Moderate", "High"]),
                ],
                [
                    "quarantine_unreadable",
                    "quarantine_short_audio",
                    "quarantine_snr_review",
                    "quarantine_snr_review",
                    "quarantine_clipping_review",
                ],
                default="valid",
            )
        else:
            audio_report["invalid_duration"] = ~audio_report["duration_valid"].fillna(
                False
            )
            audio_report["invalid_audio"] = ~audio_report["readable"] | audio_report[
                "has_nan_inf"
            ].fillna(False)
            audio_report["final_audit_status"] = np.select(
                [
                    audio_report["invalid_duration"],
                    audio_report["invalid_audio"],
                    audio_report["snr_status"] != "success",
                    audio_report["peak_amplitude"] > 0.99,
                    audio_report["snr_at_boundary"],
                ],
                [
                    "quarantine_short_audio",
                    "quarantine_invalid_audio",
                    "review_snr_failed",
                    "review_clipping",
                    "review_snr_boundary",
                ],
                default="valid",
            )
            reason_conditions = [
                (audio_report["invalid_duration"], "short_audio"),
                (audio_report["invalid_audio"], "invalid_audio"),
                (
                    audio_report["snr_status"] != "success",
                    "snr_estimation_failed",
                ),
                (
                    audio_report["peak_amplitude"] > 0.99,
                    "clipping_review",
                ),
                (audio_report["snr_at_boundary"], "snr_boundary_review"),
            ]
            audio_report["audit_reasons"] = [
                "; ".join(
                    reason
                    for condition, reason in reason_conditions
                    if condition.iloc[index]
                )
                or "none"
                for index in range(len(audio_report))
            ]

        self.results.audio_audits[name] = audio_report
        return audio_report

    def write_reports(self) -> dict[str, Path]:
        """Write audit summaries and quarantine/review ledgers to artifacts."""
        if not self.results.audio_audits:
            raise RuntimeError("Audio audits must run before writing reports")
        report_root = self.settings.artifacts_dir / "data_engineering" / "reports"
        written: dict[str, Path] = {}
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        for name, audit in self.results.audio_audits.items():
            stem = name.replace("-", "_").lower()
            report_dir = report_root / "audio_quality" / stem / run_id
            report_dir.mkdir(parents=True, exist_ok=False)
            ledger_path = report_dir / "quarantine_ledger.csv"
            report_path = report_dir / "audit_report.json"
            quarantine = audit.loc[audit["final_audit_status"] != "valid"]
            quarantine.to_csv(ledger_path, index=False, encoding="utf-8-sig")
            report = {
                "task": "AI-DS-01",
                "dataset": name,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "total_records": int(len(audit)),
                "valid_records": int((audit["final_audit_status"] == "valid").sum()),
                "review_and_quarantine_records": int(len(quarantine)),
                "status_counts": {
                    str(key): int(value)
                    for key, value in audit["final_audit_status"]
                    .value_counts(dropna=False)
                    .items()
                },
                "original_audio_modified": False,
            }
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            written[f"{name}_ledger"] = ledger_path
            written[f"{name}_report"] = report_path
        self.results.report_paths = written
        return written

    def create_handoff(self) -> Path:
        """Copy resolved Arabic-English audio into a complete handoff tree."""
        if self._arabic_audio_paths is None:
            raise RuntimeError("Audio conversion and path resolution must run first")
        name = "Arabic-English-ASR"
        _, metadata = self.datasets[name]
        handoff_dir = self.settings.processed_data_dir / name / "complete_audio"
        copied = 0
        missing: list[str] = []
        for relative, raw_source in zip(
            metadata["audio_path"], self._arabic_audio_paths, strict=True
        ):
            if pd.isna(raw_source):
                missing.append(str(relative))
                continue
            source = Path(str(raw_source))
            relative_path = Path(str(relative))
            destination = (handoff_dir / relative_path).resolve()
            if relative_path.is_absolute() or not destination.is_relative_to(
                handoff_dir.resolve()
            ):
                raise ValueError(f"Invalid handoff audio path: {relative}")
            if not source.is_file():
                missing.append(str(relative))
                continue
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            copied += 1
        if missing:
            raise FileNotFoundError(
                f"Cannot complete audio handoff; {len(missing)} audio files are missing"
            )
        expected = len(metadata)
        actual = sum(1 for path in handoff_dir.rglob("*.wav") if path.is_file())
        if actual != expected:
            raise RuntimeError(
                f"Handoff file count mismatch: expected {expected}, found {actual}"
            )
        LOGGER.info("Handoff ready: %s files in %s", copied, handoff_dir)
        self.results.handoff_dir = handoff_dir
        return handoff_dir

    def run(self) -> PipelineResults:
        """Execute each concrete stage in notebook order exactly once."""
        stages = (
            ("load metadata", self.load_datasets),
            ("metadata inspection and validation", self.inspect_metadata),
            ("ArZen audio quality audit", self.audit_arzen),
            ("audio standardization", self.convert_nonstandard_audio),
            ("Arabic-English audio quality audit", self.audit_arabic_english),
            ("audit report generation", self.write_reports),
            ("complete audio handoff", self.create_handoff),
        )
        for name, stage in stages:
            LOGGER.info("Starting stage: %s", name)
            stage()
            LOGGER.info("Completed stage: %s", name)
        return self.results

    def _require_datasets(self) -> None:
        if not self.datasets:
            raise RuntimeError("Load datasets before running this pipeline stage")

    @staticmethod
    def _clipping_level(value: object) -> str:
        if pd.isna(value):
            return "Unknown"
        percentage = float(str(value))
        if percentage == 0:
            return "No Clipping"
        if percentage <= 0.1:
            return "Very Low"
        if percentage <= 1.0:
            return "Moderate"
        return "High"
