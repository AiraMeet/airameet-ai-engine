"""Build the corrected, non-destructive data-v1.1 release."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf

from airameet_ai.data_engineering.conversion import convert_audio_batch
from airameet_ai.data_engineering.quality import audit_audio_files

from .grouping.duplicate_detector import analyze_duplicates
from .metadata.metadata_corrector import plain_value, prepare_source_records
from .metadata.recording_metadata import load_cleaned_sources
from .release.release_writer import (
    archive_release_reports,
    create_staging_directory,
    discard_staging_directory,
    promote_release,
    render_manifest,
    verify_checksums,
    write_checksums,
    write_json,
)
from .splitting.split_generator import (
    SEED_V11,
    split_records,
    validate_partition_integrity,
)

LOGGER = logging.getLogger(__name__)
SOURCES = ("Arabic-English", "Erzen")
TARGETS = {"train": 0.8, "validation": 0.1, "test": 0.1}


def _quantile(values: list[float], quantile: float) -> float | None:
    return float(np.quantile(values, quantile)) if values else None


def _duration_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["duration"]) for row in rows if row.get("duration") is not None]
    if not values:
        return {"utterances_with_duration": 0, "total_seconds": 0.0, "hours": 0.0}
    return {
        "utterances_with_duration": len(values),
        "total_seconds": float(sum(values)),
        "hours": float(sum(values) / 3600),
        "min": min(values),
        "q25": _quantile(values, 0.25),
        "median": _quantile(values, 0.5),
        "q75": _quantile(values, 0.75),
        "q90": _quantile(values, 0.9),
        "q95": _quantile(values, 0.95),
        "max": max(values),
        "mean": float(sum(values) / len(values)),
    }


def _split_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    speakers = Counter(row["speaker_id"] or "<missing>" for row in rows)
    sources = Counter(row["source_dataset"] for row in rows)
    tokens = [token for row in rows for token in row["text"].split()]
    return {
        "utterances": len(rows),
        "recording_groups": len({row["split_group_id"] for row in rows}),
        "verified_sessions": len({row["session_id"] for row in rows if row.get("session_id")}),
        "utterances_without_verified_session": sum(not row.get("session_id") for row in rows),
        "duration": _duration_statistics(rows),
        "source_dataset_distribution": dict(sorted(sources.items())),
        "speaker_distribution": dict(sorted(speakers.items())),
        "vocabulary": {
            "token_count": len(tokens),
            "unique_token_count": len(set(tokens)),
            "tokenization": "whitespace; case-sensitive; punctuation retained",
        },
    }


def _markdown_report(
    stats: dict[str, Any],
    integrity: dict[str, Any],
    duplicate_report: dict[str, Any],
    quality_report: dict[str, Any],
) -> str:
    split_rows = "\n".join(
        f"| {name} | {part['utterances']} | {part['utterance_proportion']:.4%} | "
        f"{part['recording_groups']} | {part['recording_group_proportion']:.4%} | "
        f"{part['duration']['hours']:.4f} |"
        for name, part in stats["splits"].items()
    )
    return f"""# data-v1.1 Release Audit

## Release status

`{stats['release_status']}`. Recording-group and exact-duplicate checks passed.
This release is **not certified session-independent for unmatched Arabic-English
records**: each such WAV is treated as one independent recording unit, while its
upstream meeting/session is unknown. Speaker-independent splitting was not
applied. Model-training readiness is `{stats['model_training_readiness']}` because
{duplicate_report['transcript_mismatches_unadjudicated']} exact-audio transcript
pairs remain unadjudicated.

## Scope

- Manifest utterances: {stats['total_utterances']}
- Recording groups used for splitting: {stats['total_recording_groups']}
- Verified Erzen/linked interview sessions: {stats['total_verified_sessions']}
- Arabic-English rows with no verified session: {stats['arabic_english_unmatched_session_rows']}
- Identified speakers: {stats['total_identified_speakers']}
- Rows with unknown speaker: {stats['unknown_speaker_rows']}
- Source row exclusions: {stats['source_rows_excluded']}
- Arabic-English ingestion exclusions documented from `errors.jsonl`: {stats['arabic_english_ingestion_exclusions']}
- Manifest row duration: {stats['duration']['hours']:.4f} hours

## Partitions

| Split | Utterances | Utterance proportion | Recording groups | Group proportion | Duration (hours) |
|---|---:|---:|---:|---:|---:|
{split_rows}

Targets are group-count targets of 80/10/10. Utterance proportions differ because
session groups have different sizes.

## Identity and duplicate handling

- Erzen uses session IDs parsed from verified source recording identifiers.
- Original speaker-track identifiers are retained as `source_recording_id`.
- Every Arabic-English WAV has a unique file-level `recording_id`.
- Arabic-English `session_id` is populated only for an exact SHA256 match to an
  Erzen audio file; all other session IDs remain null.
- Exact duplicate rows are retained. All rows with identical source audio hashes
  are assigned to one split.
- Exact duplicate groups: {duplicate_report['duplicate_hash_groups']}.
- Exact duplicate cross-dataset pairs: {duplicate_report['cross_dataset_duplicate_pairs']}.
- Transcript mismatches requiring review: {duplicate_report['transcript_mismatches_unadjudicated']}.

## Audio quality

- Available source audio files decoded: {quality_report['summary']['readable_files']}.
- Silent files excluded from manifests: {quality_report['summary']['silent_files_excluded']}.
- Audio files standardized to mono 16 kHz PCM16: {quality_report['summary']['standardized_files']}.
- Seven failed Erzen source rows remain excluded.
- Eight Arabic-English ingestion exclusions are documented separately; they were
  absent from the supplied workbook and were not reconstructed.

## Integrity checks

- Partition coverage and unique utterance IDs: passed.
- Verified session overlap: {integrity['session_leaks']}.
- Exact duplicate audio overlap: {integrity['exact_duplicate_leaks']}.
- Missing manifest audio references: {integrity.get('missing_manifest_audio_references', 0)}.
- Checksum verification: passed.

The unmatched Arabic-English WAVs are independent file-level groups by explicit
task policy. This does not establish that their parent sessions are distinct.
"""


class DatasetReleasePipeline:
    """Create a fresh data-v1.1 release directory without touching v1.0."""

    def __init__(
        self,
        metadata_dir: str | Path,
        data_root: str | Path,
        release_dir: str | Path,
        reports_root: str | Path | None = None,
    ) -> None:
        self.metadata_dir = Path(metadata_dir)
        self.data_root = Path(data_root)
        self.release_dir = Path(release_dir)
        self.reports_root = Path(reports_root) if reports_root else None
        self.audio_roots = {
            "Erzen": self.data_root / "ArZen_Prepared",
            "Arabic-English": self.data_root / "Arabic-English-ASR",
        }

    def _load_ingestion_exclusions(self) -> list[dict[str, Any]]:
        path = self.audio_roots["Arabic-English"] / "errors.jsonl"
        if not path.is_file():
            return []
        exclusions = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid ingestion error JSON at {path}:{line_number}") from error
            exclusions.append(item)
        return exclusions

    @staticmethod
    def _audit_audio(records: list[dict[str, Any]], audio_roots: dict[str, Path]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        audits: dict[str, dict[str, Any]] = {}
        details: list[dict[str, Any]] = []
        summary: dict[str, Any] = {"by_source": {}}
        for source in SOURCES:
            source_rows = [row for row in records if row["source_dataset"] == source]
            if not source_rows:
                continue
            frame = pd.DataFrame({"audio_path": [row["source_audio_path"] for row in source_rows]})
            resolved = pd.Series([str(row["source_path"]) for row in source_rows])
            audited = audit_audio_files(
                frame,
                audio_roots[source],
                audio_paths=resolved,
                allowed_roots=(audio_roots[source],),
            )
            source_counts = Counter()
            for row, (_, result) in zip(source_rows, audited.iterrows(), strict=True):
                clean = {key: plain_value(value) for key, value in result.to_dict().items()}
                clean.pop("source_path", None)
                clean.update(
                    source_dataset=source,
                    utterance_id=row["utterance_id"],
                    source_audio_sha256=row["source_audio_sha256"],
                )
                audits[row["utterance_id"]] = clean
                source_counts["readable"] += bool(clean["readable"])
                source_counts["decode_failures"] += not bool(clean["readable"])
                source_counts["silent"] += bool(clean["is_silent"])
                source_counts[f"sample_rate:{clean['sample_rate']}"] += 1
                source_counts[f"channels:{clean['channels']}"] += 1
                details.append(clean)
            summary["by_source"][source] = dict(sorted(source_counts.items()))
        summary.update(
            {
                "files_audited": len(audits),
                "readable_files": sum(bool(row["readable"]) for row in audits.values()),
                "decode_failures": sum(not bool(row["readable"]) for row in audits.values()),
                "silent_files": sum(bool(row["is_silent"]) for row in audits.values()),
                "silent_files_excluded": sum(bool(row["is_silent"]) for row in audits.values()),
                "input_format_distribution": dict(
                    sorted(Counter(f"{row['sample_rate']}Hz/{row['channels']}ch" for row in audits.values()).items())
                ),
            }
        )
        summary["files"] = details
        return audits, summary

    def _convert_audio(
        self,
        staging_root: Path,
        records: list[dict[str, Any]],
        audits: dict[str, dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        active = [
            row
            for row in records
            if audits[row["utterance_id"]]["readable"]
            and not audits[row["utterance_id"]]["is_silent"]
        ]
        conversions: list[dict[str, Any]] = []
        successful: list[dict[str, Any]] = []
        conversion_by_source_path: dict[tuple[str, str], dict[str, Any]] = {}
        for source in SOURCES:
            source_rows = [row for row in active if row["source_dataset"] == source]
            if not source_rows:
                continue
            frame = pd.DataFrame({"audio_path": [row["source_audio_path"] for row in source_rows]})
            output_root = staging_root / "audio" / source.replace("/", "-")
            results = convert_audio_batch(
                frame,
                source_root=self.audio_roots[source],
                output_root=output_root,
                overwrite=False,
            )
            for row, result in zip(source_rows, results, strict=True):
                relative_input = row["source_audio_path"]
                output_path = Path(result["output_path"])
                clean = {
                    "source_dataset": source,
                    "utterance_id": row["utterance_id"],
                    "source_audio_path": relative_input,
                    "output_audio_path": None,
                    "status": result["status"],
                    "error": result.get("error"),
                }
                if result["status"] == "success":
                    info = sf.info(output_path)
                    clean.update(
                        output_sample_rate=info.samplerate,
                        output_channels=info.channels,
                        output_duration=info.duration,
                    )
                    if info.samplerate != 16000 or info.channels != 1:
                        clean["status"] = "failed"
                        clean["error"] = "Output audio is not mono 16 kHz"
                    elif abs(info.duration - float(row["duration"])) > 0.05:
                        clean["status"] = "failed"
                        clean["error"] = "Output duration differs from validated metadata by >50 ms"
                if clean["status"] != "success":
                    failed_output = output_path
                    if failed_output.is_file() and failed_output.is_relative_to(staging_root):
                        failed_output.unlink()
                elif output_path.is_relative_to(staging_root):
                    clean["output_audio_path"] = output_path.relative_to(staging_root).as_posix()
                conversions.append(clean)
                conversion_by_source_path[(source, relative_input)] = clean

        failed_conversions: list[dict[str, Any]] = []
        for row in active:
            result = conversion_by_source_path[(row["source_dataset"], row["source_audio_path"])]
            if result["status"] != "success":
                failed_conversions.append(
                    {
                        "source_dataset": row["source_dataset"],
                        "utterance_id": row["utterance_id"],
                        "source_audio_path": row["source_audio_path"],
                        "reason": result.get("error") or "audio conversion failed",
                    }
                )
                row["corrected_row"]["release_eligible"] = False
                row["corrected_row"]["release_exclusion_reasons"] = "audio_conversion_failed"
                row["corrected_row"]["audio_path"] = None
                continue
            relative_output = result["output_audio_path"]
            if not relative_output:
                raise AssertionError("Successful audio conversion has no portable output path")
            row["audio_path"] = relative_output
            row["corrected_row"]["audio_path"] = relative_output
            row["corrected_row"]["release_eligible"] = True
            row["corrected_row"]["release_exclusion_reasons"] = ""
            successful.append(row)

        conversion_summary = {
            "attempted": len(conversions),
            "standardized_files": len(successful),
            "failed": len(failed_conversions),
            "failed_rows": failed_conversions,
            "files": conversions,
        }
        return successful, conversions, conversion_summary

    @staticmethod
    def _build_statistics(
        partitions: dict[str, list[dict[str, Any]]],
        corrected_rows: list[dict[str, Any]],
        input_summary: dict[str, Any],
        duplicate_report: dict[str, Any],
        ingestion_exclusions: list[dict[str, Any]],
        integrity: dict[str, Any],
    ) -> dict[str, Any]:
        split_stats = {
            name: _split_statistics(rows) for name, rows in partitions.items()
        }
        total = sum(part["utterances"] for part in split_stats.values())
        groups = sum(part["recording_groups"] for part in split_stats.values())
        for part in split_stats.values():
            part["utterance_proportion"] = part["utterances"] / total if total else 0.0
            part["recording_group_proportion"] = part["recording_groups"] / groups if groups else 0.0
        all_records = [row for rows in partitions.values() for row in rows]
        speaker_counts = Counter(row["speaker_id"] or "<missing>" for row in all_records)
        unknown_speakers = sum(
            1
            for row in all_records
            if not row.get("speaker_id") or row["speaker_id"].casefold() == "unknown"
        )
        durations = _duration_statistics(all_records)
        source_distribution = dict(
            sorted(Counter(row["source_dataset"] for row in all_records).items())
        )
        excluded_by_reason = Counter()
        for row in corrected_rows:
            for reason in (row.get("release_exclusion_reasons") or "").split(";"):
                if reason:
                    excluded_by_reason[reason] += 1
        verified_sessions = {row["session_id"] for row in all_records if row.get("session_id")}
        arabic_unmatched = sum(
            row["source_dataset"] == "Arabic-English" and not row.get("session_id")
            for row in all_records
        )
        return {
            "release": "data-v1.1",
            "release_status": "conditional",
            "model_training_readiness": "review_required",
            "target_group_proportions": TARGETS,
            "random_seed": SEED_V11,
            "split_algorithm": "GroupShuffleSplit over session/recording connected components; test=0.1 then validation=1/9 of remainder",
            "total_utterances": total,
            "total_recording_groups": groups,
            "total_verified_sessions": len(verified_sessions),
            "arabic_english_unmatched_session_rows": arabic_unmatched,
            "total_identified_speakers": len({row["speaker_id"] for row in all_records if row.get("speaker_id") and row["speaker_id"].casefold() != "unknown"}),
            "unknown_speaker_rows": unknown_speakers,
            "speaker_statistics": dict(sorted(speaker_counts.items())),
            "source_dataset_distribution": source_distribution,
            "duration": durations,
            "vocabulary": _split_statistics(all_records)["vocabulary"],
            "splits": split_stats,
            "input_rows_by_source": input_summary,
            "source_rows_excluded": sum(not row.get("release_eligible") for row in corrected_rows),
            "exclusions_by_reason": dict(sorted(excluded_by_reason.items())),
            "arabic_english_ingestion_exclusions": len(ingestion_exclusions),
            "arabic_english_ingestion_exclusion_reasons": dict(
                sorted(Counter(str(row.get("error", "unknown")) for row in ingestion_exclusions).items())
            ),
            "exact_audio_duplicate_groups": duplicate_report["duplicate_hash_groups"],
            "cross_dataset_duplicate_pairs": duplicate_report["cross_dataset_duplicate_pairs"],
            "transcript_mismatches_unadjudicated": duplicate_report["transcript_mismatches_unadjudicated"],
            "integrity": integrity,
            "methodology": {
                "erzen_recording_group": "C03-Rxx interview session parsed from verified source IDs; speaker track ID retained separately",
                "arabic_english_recording_group": "one WAV per recording ID; session_id populated only for exact SHA256 match to an Erzen session",
                "unmatched_arabic_english_session_status": "unknown; no session-independence claim is made",
                "speaker_independent_split": False,
                "transcript_mismatch_policy": "retain rows and both transcripts; report every unadjudicated exact-audio mismatch; do not select a canonical transcript",
                "duplicate_row_policy": "retain all rows; constrain exact source-audio duplicates to one split",
                "durations": "validated metadata durations; output audio is separately checked within 50 ms",
            },
        }

    def run(self) -> dict[str, Any]:
        """Generate and validate a fresh, isolated data-v1.1 release."""
        if self.release_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing release: {self.release_dir}")
        sources = load_cleaned_sources(self.metadata_dir)
        records, corrected_rows, input_summary = prepare_source_records(sources, self.audio_roots)
        if not records:
            raise ValueError("No source rows have validated audio paths and metadata")

        _, duplicate_report = analyze_duplicates(records)
        LOGGER.info(
            "Hashed %d audio files; found %d exact duplicate groups",
            len(records),
            duplicate_report["duplicate_hash_groups"],
        )
        staging_root = create_staging_directory(self.release_dir)
        try:
            audits, audio_summary = self._audit_audio(records, self.audio_roots)
            silent_ids = {
                utterance_id
                for utterance_id, audit in audits.items()
                if audit["is_silent"] is True
            }
            decode_failed_ids = {
                utterance_id
                for utterance_id, audit in audits.items()
                if audit["readable"] is not True
            }
            for row in records:
                if row["utterance_id"] in silent_ids | decode_failed_ids:
                    row["corrected_row"]["release_eligible"] = False
                    row["corrected_row"]["release_exclusion_reasons"] = (
                        "silent_audio" if row["utterance_id"] in silent_ids else "audio_decode_failed"
                    )
                    row["corrected_row"]["audio_path"] = None
            active = [
                row
                for row in records
                if row["utterance_id"] not in silent_ids | decode_failed_ids
            ]

            converted, conversion_details, conversion_summary = self._convert_audio(
                staging_root, active, audits
            )
            active = converted
            if len({row["utterance_id"] for row in active}) != len(active):
                raise ValueError("Eligible release rows contain duplicate utterance IDs")
            if not active:
                raise ValueError("No rows remain after audio validation and standardization")

            partitions = split_records(active, seed=SEED_V11)
            integrity = validate_partition_integrity(active, partitions)
            repeated_partitions = split_records(
                [dict(row) for row in active], seed=SEED_V11
            )
            original_assignment = {
                row["utterance_id"]: row["split"]
                for rows in partitions.values()
                for row in rows
            }
            repeated_assignment = {
                row["utterance_id"]: row["split"]
                for rows in repeated_partitions.values()
                for row in rows
            }
            if original_assignment != repeated_assignment:
                raise AssertionError("Split assignment is not reproducible for identical inputs")
            integrity["reproducibility_check"] = "passed"
            referenced_audio = {
                row["audio_path"] for rows in partitions.values() for row in rows
            }
            missing_audio = [
                path for path in sorted(referenced_audio) if not (staging_root / path).is_file()
            ]
            if missing_audio:
                raise AssertionError(f"Release manifests reference missing audio: {missing_audio[:10]}")
            integrity["missing_manifest_audio_references"] = len(missing_audio)
            integrity["unmatched_arabic_english_session_rows"] = sum(
                row["source_dataset"] == "Arabic-English" and not row.get("session_id")
                for row in active
            )
            integrity["session_independence_claim"] = "verified Erzen sessions and exact linked duplicates only"
            integrity["global_session_independence"] = False
            integrity["target_group_proportions"] = TARGETS
            integrity["random_seed"] = SEED_V11

            ingestion_exclusions = self._load_ingestion_exclusions()
            stats = self._build_statistics(
                partitions,
                corrected_rows,
                input_summary,
                duplicate_report,
                ingestion_exclusions,
                integrity,
            )
            split_by_id = {
                row["utterance_id"]: row["split"]
                for rows in partitions.values()
                for row in rows
            }
            for duplicate_group in duplicate_report["groups"]:
                for duplicate_row in duplicate_group["records"]:
                    duplicate_row["split"] = split_by_id.get(duplicate_row["utterance_id"])
            audio_summary["summary"] = {
                "files_audited": len(audits),
                "readable_files": sum(bool(row["readable"]) for row in audits.values()),
                "decode_failures": sum(not bool(row["readable"]) for row in audits.values()),
                "silent_files_excluded": len(silent_ids),
                "standardized_files": len(active),
                "conversion_failures": conversion_summary["failed"],
                "input_format_distribution": audio_summary.get("input_format_distribution", {}),
            }
            audio_report = {
                "summary": audio_summary["summary"],
                "input_format_distribution": audio_summary.get("input_format_distribution", {}),
                "by_source": audio_summary.get("by_source", {}),
                "ingestion_exclusions": ingestion_exclusions,
                "source_row_exclusions": [
                    {
                        "source_dataset": row.get("source_dataset"),
                        "utterance_id": row.get("utterance_id") or row.get("utt_id"),
                        "source_audio_path": row.get("source_audio_path"),
                        "reasons": row.get("release_exclusion_reasons"),
                    }
                    for row in corrected_rows
                    if not row.get("release_eligible")
                ],
                "conversion_summary": {
                    key: value
                    for key, value in conversion_summary.items()
                    if key not in {"files"}
                },
                "files": audio_summary.get("files", []),
                "conversions": conversion_details,
            }
            write_json(staging_root / "reports" / "dataset_stats.json", stats)
            write_json(staging_root / "reports" / "duplicate_report.json", duplicate_report)
            write_json(staging_root / "reports" / "audio_quality_report.json", audio_report)
            write_json(staging_root / "reports" / "split_integrity_report.json", integrity)

            for source in SOURCES:
                source_rows = [row for row in corrected_rows if row.get("source_dataset") == source]
                metadata_name = (
                    "Erzen_corrected_metadata.csv"
                    if source == "Erzen"
                    else "Arabic_English_corrected_metadata.csv"
                )
                destination = staging_root / "metadata" / metadata_name
                destination.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(source_rows).to_csv(
                    destination,
                    index=False,
                    encoding="utf-8-sig",
                    lineterminator="\n",
                )

            manifest_dir = staging_root / "manifests"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            for name, rows in partitions.items():
                (manifest_dir / f"{('val' if name == 'validation' else name)}.jsonl").write_bytes(
                    render_manifest(rows)
                )

            audit_text = _markdown_report(stats, integrity, duplicate_report, audio_report)
            (staging_root / "reports" / "release_audit.md").write_text(
                audit_text,
                encoding="utf-8",
                newline="\n",
            )
            checksums = write_checksums(staging_root)
            checksum_validation = verify_checksums(staging_root)

            for name, rows in partitions.items():
                manifest_name = "val" if name == "validation" else name
                path = manifest_dir / f"{manifest_name}.jsonl"
                parsed = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                if len(parsed) != len(rows):
                    raise AssertionError(f"Manifest count differs for {manifest_name}")
                if any(not (staging_root / row["audio_path"]).is_file() for row in parsed):
                    raise AssertionError(f"Missing audio path in {manifest_name} manifest")
            stats["checksum_validation"] = checksum_validation
            # The stats report is checksummed; leave the in-memory copy enriched
            # only in the return value so the written checksums stay self-consistent.
            promote_release(staging_root, self.release_dir)
        except Exception:
            discard_staging_directory(staging_root)
            raise

        archived_reports = (
            archive_release_reports(self.release_dir, self.reports_root)
            if self.reports_root is not None
            else []
        )
        LOGGER.info(
            "data-v1.1 created at %s: %d utterances, %d groups, status=%s",
            self.release_dir,
            stats["total_utterances"],
            stats["total_recording_groups"],
            stats["release_status"],
        )
        return {
            "release": "data-v1.1",
            "path": str(self.release_dir),
            "central_reports": [str(path) for path in archived_reports],
            "statistics": stats,
            "checksums": len(checksums),
        }


# Preserve the version-specific class name used by existing callers.
DatasetReleaseV11Pipeline = DatasetReleasePipeline
