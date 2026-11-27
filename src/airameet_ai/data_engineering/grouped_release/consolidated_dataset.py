"""Read and summarize the existing consolidated dataset without rebuilding it."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from .statistics.statistics_generator import build_statistics

SPLITS = ("train", "validation", "test")
REQUIRED_FIELDS = (
    "utterance_id",
    "audio_path",
    "text",
    "recording_id",
    "source_dataset",
    "duration",
    "split",
)


class ConsolidatedDatasetPipeline:
    """Validate consolidated manifests and compute their existing statistics."""

    def __init__(
        self,
        dataset_root: str | Path,
        reports_root: str | Path | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.reports_root = Path(reports_root) if reports_root else None

    def _load_partitions(self) -> dict[str, list[dict[str, Any]]]:
        partitions: dict[str, list[dict[str, Any]]] = {}
        seen_ids: set[str] = set()
        for split in SPLITS:
            manifest = self.dataset_root / "manifests" / f"{split}.jsonl"
            if not manifest.is_file():
                raise FileNotFoundError(f"Missing consolidated {split} manifest: {manifest}")
            records = []
            for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {manifest}:{line_number}") from error
                missing = [field for field in REQUIRED_FIELDS if field not in record or record[field] is None]
                if missing:
                    raise ValueError(f"Missing required fields {missing} at {manifest}:{line_number}")
                if record["split"] != split:
                    raise ValueError(
                        f"Split assignment mismatch at {manifest}:{line_number}: {record['split']}"
                    )
                utterance_id = str(record["utterance_id"])
                if not utterance_id or utterance_id in seen_ids:
                    raise ValueError(f"Missing or duplicate utterance_id: {utterance_id}")
                seen_ids.add(utterance_id)
                audio_path = Path(str(record["audio_path"]))
                if audio_path.is_absolute() or ".." in audio_path.parts:
                    raise ValueError(f"Unsafe relative audio_path at {manifest}:{line_number}")
                resolved_audio = (self.dataset_root / audio_path).resolve()
                if not resolved_audio.is_relative_to(self.dataset_root):
                    raise ValueError(f"audio_path escapes consolidated dataset: {audio_path}")
                if not resolved_audio.is_file():
                    raise FileNotFoundError(f"Manifest audio is missing: {audio_path}")
                try:
                    duration = float(record["duration"])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"Invalid duration at {manifest}:{line_number}") from error
                if not math.isfinite(duration) or duration < 0:
                    raise ValueError(f"Invalid duration at {manifest}:{line_number}")
                record["duration"] = duration
                records.append(record)
            partitions[split] = records
        return partitions

    def run(self) -> dict[str, Any]:
        """Summarize the existing manifests; never split, copy, or rewrite data."""
        partitions = self._load_partitions()
        statistics = build_statistics(
            partitions,
            rejected_by_source={},
            excluded_audio_by_source={},
        )
        result: dict[str, Any] = {
            "dataset": "asr-dataset",
            "path": str(self.dataset_root),
            "manifest_rows": sum(len(records) for records in partitions.values()),
            "audio_files_validated": sum(len(records) for records in partitions.values()),
            "statistics": statistics,
        }
        if self.reports_root is not None:
            self.reports_root.mkdir(parents=True, exist_ok=True)
            report_path = self.reports_root / "ai_ds_03_statistics.json"
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".ai_ds_03_statistics-", suffix=".tmp", dir=self.reports_root
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
                    stream.write("\n")
                os.replace(temporary_name, report_path)
            except Exception:
                Path(temporary_name).unlink(missing_ok=True)
                raise
            result["report"] = str(report_path)
        return result
