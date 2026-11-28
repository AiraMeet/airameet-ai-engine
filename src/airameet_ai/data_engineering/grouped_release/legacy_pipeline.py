"""Orchestration for the AI-DS-03 data-v1.0 release checkpoint."""

from __future__ import annotations

import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .metadata.recording_metadata import is_missing, load_cleaned_sources
from .release.legacy_release_writer import render_release, write_release
from .splitting.group_splitter import SEED, TARGET_RATIOS, split_grouped_records
from .statistics.statistics_generator import build_statistics

LOGGER = logging.getLogger(__name__)


class GroupedReleasePipeline:
    """Build deterministic recording-disjoint manifests and statistics."""

    def __init__(
        self,
        metadata_dir: str | Path,
        release_root: str | Path,
        audio_root: str | Path | None = None,
    ) -> None:
        self.metadata_dir = Path(metadata_dir)
        self.release_root = Path(release_root)
        self.audio_root = Path(audio_root) if audio_root else self.release_root / "data"

    def _resolve_audio(self, raw_value: Any, source: str, metadata_path: Path) -> Path | None:
        """Resolve supported relative paths while requiring an actual file."""
        if is_missing(raw_value):
            return None
        supplied = Path(str(raw_value).strip())
        if supplied.is_absolute():
            candidates = [supplied]
        else:
            candidates = [metadata_path.parent / supplied, self.audio_root / supplied]
            folder = "ArZen_Prepared" if source == "Erzen" else "Arabic-English-ASR"
            candidates.append(self.audio_root / folder / supplied)
        for candidate in candidates:
            try:
                resolved = candidate.resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            if resolved.is_file():
                return resolved
        return None

    @staticmethod
    def _plain(value: Any) -> Any:
        """Convert pandas/numpy scalar values to JSON-safe Python values."""
        if is_missing(value):
            return None
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    def _collect(self, sources: dict[str, tuple[Path, pd.DataFrame]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]], dict[str, int]]:
        eligible: list[dict[str, Any]] = []
        rejected: dict[str, Counter[str]] = {source: Counter() for source in sources}
        unresolved: Counter[str] = Counter()
        for source, (metadata_path, frame) in sources.items():
            for row in frame.to_dict(orient="records"):
                utterance_id = row.get("utterance_id", row.get("utt_id"))
                recording_id = row.get("recording_id")
                # `meeting` is a source split label in this dataset, not identity.
                checks = {
                    "utterance_id": utterance_id,
                    "recording_id": recording_id,
                    "text": row.get("text"),
                    "duration": row.get("duration"),
                }
                reasons = [f"missing_or_invalid:{key}" for key, value in checks.items() if is_missing(value)]
                if not is_missing(recording_id) and not str(recording_id).strip():
                    reasons.append("invalid:recording_id")
                try:
                    duration = float(row.get("duration"))
                    if not math.isfinite(duration) or duration < 0:
                        reasons.append("invalid:duration")
                except (TypeError, ValueError):
                    reasons.append("invalid:duration")
                    duration = 0.0
                if reasons:
                    rejected[source].update(reasons)
                    continue
                audio_path = self._resolve_audio(row.get("audio_path"), source, metadata_path)
                if audio_path is None:
                    unresolved[source] += 1
                    continue
                try:
                    portable_audio = audio_path.relative_to(self.release_root.resolve()).as_posix()
                except ValueError:
                    # Absolute inputs outside the release root are not portable.
                    rejected[source]["invalid:non_portable_audio_path"] += 1
                    continue
                speaker = row.get("speaker_id", row.get("speaker"))
                record = {
                    "utterance_id": str(utterance_id).strip(),
                    "audio_path": portable_audio,
                    "text": str(row["text"]),
                    "recording_id": str(recording_id).strip(),
                    "source_dataset": source,
                    "duration": duration,
                }
                for name in ("speaker_id", "start", "end", "sample_rate", "status"):
                    value = speaker if name == "speaker_id" else row.get(name)
                    if value is not None or name in row or name == "speaker_id":
                        record[name] = self._plain(value)
                eligible.append(record)
        return eligible, {source: dict(counts) for source, counts in rejected.items()}, dict(unresolved)

    def run(self) -> dict[str, object]:
        """Build, verify, and safely write all data-v1.0 release artifacts."""
        sources = load_cleaned_sources(self.metadata_dir)
        records, rejected, unresolved = self._collect(sources)
        if not records:
            raise ValueError("No eligible records have valid IDs, durations, and resolvable audio")
        partitions = split_grouped_records(records)
        stats = build_statistics(
            partitions,
            rejected_by_source=rejected,
            excluded_audio_by_source=unresolved,
        )
        stats.update(
            {
                "seed": SEED,
                "target_proportions": TARGET_RATIOS,
                "group_key": ["source_dataset", "recording_id"],
                "split_method": (
                    "GroupShuffleSplit over unique composite groups: test_size=0.1, "
                    "then validation test_size=1/9 of remaining groups; random_state=42. "
                    "The unavailable notebook helper's exact recipe could not be recovered."
                ),
                "limitations": [
                    "Ratios target recording-group counts and may differ after integer group rounding.",
                    "Utterance proportions may differ because recordings contain unequal utterance counts.",
                    "Speaker-level disjointness is not enforced.",
                ],
            }
        )
        outputs = render_release(partitions, stats)
        written = write_release(self.release_root, outputs)
        LOGGER.info(
            "Release data-v1.0 ready: %d utterances, %d recording groups, artifacts=%s",
            stats["total_utterances"],
            stats["total_recording_groups"],
            [str(path) for path in written],
        )
        return {"release": "data-v1.0", "artifacts": [str(p) for p in written], "statistics": stats}
