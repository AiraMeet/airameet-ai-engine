"""Source-aware identity normalization for the data-v1.1 release."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from airameet_ai.data_engineering.validation import validate_audio_paths

ERZEN_SESSION_PATTERN = re.compile(r"^(C\d+-R\d+)-([A-Za-z]\d+|S\d+)$")
ERZEN_UTTERANCE_PATTERN = re.compile(r"^([A-Za-z]\d+|S\d+)-(C\d+-R\d+)_")


def plain_value(value: Any) -> Any:
    """Convert pandas/numpy scalar values into JSON/CSV-safe Python values."""
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _required_text(value: Any) -> str | None:
    plain = plain_value(value)
    if plain is None:
        return None
    text = str(plain).strip()
    return text or None


def prepare_source_records(
    sources: dict[str, tuple[Path, pd.DataFrame]],
    audio_roots: dict[str, Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Normalize available rows and retain ineligible rows with reasons.

    Audio references are resolved through the project's existing path
    validator. Erzen session IDs are derived from source recording IDs and
    cross-checked against utterance and speaker IDs. Arabic-English recording
    IDs identify one WAV asset each; legacy group labels are retained only for
    audit and never used for partitioning.
    """
    records: list[dict[str, Any]] = []
    corrected_rows: list[dict[str, Any]] = []
    input_summary: dict[str, Any] = {}

    for source in sorted(sources):
        metadata_path, frame = sources[source]
        audio_root = audio_roots[source]
        validated = validate_audio_paths(frame, audio_root)
        source_records = 0
        source_exclusions: dict[str, int] = {}

        for position, original in enumerate(frame.to_dict(orient="records")):
            row = {key: plain_value(value) for key, value in original.items()}
            raw_audio = _required_text(row.get("audio_path"))
            raw_id = _required_text(row.get("utterance_id") or row.get("utt_id"))
            raw_recording_id = _required_text(row.get("recording_id"))
            text_value = _required_text(row.get("text"))
            speaker = _required_text(row.get("speaker_id") or row.get("speaker"))
            reasons: list[str] = []
            if not raw_id:
                reasons.append("missing_utterance_id")
            if not raw_audio:
                reasons.append("missing_audio_path")
            if not text_value:
                reasons.append("missing_text")
            status = _required_text(row.get("status"))
            if source == "Erzen" and status and status.casefold() != "success":
                reasons.append(f"source_status_{status.casefold()}")

            duration_value = row.get("duration")
            try:
                duration = float(duration_value)
                if not math.isfinite(duration) or duration <= 0:
                    reasons.append("invalid_duration")
            except (TypeError, ValueError):
                duration = None
                reasons.append("missing_or_invalid_duration")

            resolution = validated.iloc[position]
            if resolution["path_status"] != "found":
                reasons.append(f"audio_path_{resolution['path_status']}")
            resolved_path = (
                Path(str(resolution["resolved_audio_path"]))
                if resolution["audio_exists"]
                else None
            )

            old_recording_id = raw_recording_id
            if source == "Erzen":
                match = ERZEN_SESSION_PATTERN.fullmatch(raw_recording_id or "")
                utterance_match = ERZEN_UTTERANCE_PATTERN.match(raw_id or "")
                if match is None:
                    session_id = None
                    reasons.append("invalid_erzen_recording_id")
                else:
                    session_id = match.group(1)
                    track_speaker = match.group(2)
                    if utterance_match is None or utterance_match.group(2) != session_id:
                        reasons.append("erzen_utterance_session_mismatch")
                    if speaker and track_speaker != speaker:
                        reasons.append("erzen_track_speaker_mismatch")
                recording_id = session_id or ""
                source_recording_id = old_recording_id
                legacy_recording_id = old_recording_id
            else:
                session_id = None
                if not raw_audio:
                    recording_id = ""
                    reasons.append("invalid_arabic_recording_id")
                else:
                    stem = Path(raw_audio).stem
                    if not stem.isdigit():
                        reasons.append("invalid_arabic_audio_stem")
                    recording_id = f"areng:{stem}"
                source_recording_id = None
                legacy_recording_id = old_recording_id

            if resolved_path is None and "audio_path_file_not_found" not in reasons:
                reasons.append("audio_path_unresolved")
            deduped_reasons = sorted(set(reasons))
            relative_source_audio = raw_audio
            corrected = {
                **row,
                "source_dataset": source,
                "utterance_id": raw_id or "",
                "source_audio_path": relative_source_audio,
                "audio_path": None,
                "legacy_recording_id": legacy_recording_id,
                "recording_id": recording_id,
                "source_recording_id": source_recording_id,
                "session_id": session_id,
                "speaker_id": speaker,
                "text": text_value,
                "duration": duration,
                "release_eligible": not bool(deduped_reasons),
                "release_exclusion_reasons": ";".join(deduped_reasons),
            }
            corrected_rows.append(corrected)
            if deduped_reasons:
                for reason in deduped_reasons:
                    source_exclusions[reason] = source_exclusions.get(reason, 0) + 1
                continue

            split_group_key = (
                f"session:{session_id}"
                if session_id
                else f"recording:{source}:{recording_id}"
            )
            record = {
                "source_dataset": source,
                "utterance_id": raw_id,
                "source_audio_path": relative_source_audio,
                "source_path": resolved_path,
                "audio_path": None,
                "recording_id": recording_id,
                "session_id": session_id,
                "source_recording_id": source_recording_id,
                "legacy_recording_id": legacy_recording_id,
                "speaker_id": speaker,
                "text": text_value,
                "duration": duration,
                "start": plain_value(row.get("start")),
                "end": plain_value(row.get("end")),
                "sample_rate_metadata": plain_value(row.get("sample_rate")),
                "status": status,
                "metadata_position": position,
                "split_group_key": split_group_key,
                "corrected_row": corrected,
            }
            records.append(record)
            source_records += 1

        input_summary[source] = {
            "input_rows": len(frame),
            "initially_eligible_rows": source_records,
            "exclusions_by_reason": dict(sorted(source_exclusions.items())),
            "metadata_file": metadata_path.name,
        }

    id_counts = Counter(record["utterance_id"] for record in records)
    duplicates = sorted(value for value, count in id_counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate utterance IDs across sources: {duplicates[:10]}")
    arabic_recording_ids = [
        row["recording_id"] for row in records if row["source_dataset"] == "Arabic-English"
    ]
    duplicate_recording_ids = sorted(
        value
        for value, count in Counter(arabic_recording_ids).items()
        if count > 1
    )
    if duplicate_recording_ids:
        raise ValueError(
            f"Arabic-English WAV recording IDs are not unique: {duplicate_recording_ids[:10]}"
        )
    return records, corrected_rows, input_summary
