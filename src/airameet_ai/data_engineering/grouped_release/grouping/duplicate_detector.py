"""Global SHA256 duplicate detection and verified cross-source linking."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    """Hash a file incrementally to keep memory use bounded."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_transcript(text: str) -> str:
    """Normalize only Unicode, case, and whitespace for comparison."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def analyze_duplicates(
    records: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Hash all resolved source audio and link only exact byte duplicates.

    Matching Arabic-English clips inherit an Erzen session only when every
    matching Erzen file identifies the same verified session. Ambiguous links
    remain unknown, while their exact-hash duplicate constraint is preserved.
    """
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        record["source_audio_sha256"] = sha256_file(record["source_path"])
        by_hash[record["source_audio_sha256"]].append(record)

    report_groups: list[dict[str, Any]] = []
    cross_dataset_pairs = 0
    transcript_matches = 0
    transcript_mismatches = 0
    linked_arabic_rows = 0
    ambiguous_arabic_rows = 0

    for digest, group in sorted(by_hash.items()):
        if len(group) < 2:
            continue
        erzen_sessions = {
            row["session_id"]
            for row in group
            if row["source_dataset"] == "Erzen" and row["session_id"]
        }
        arabic_rows = [row for row in group if row["source_dataset"] == "Arabic-English"]
        if len(erzen_sessions) == 1:
            verified_session = next(iter(erzen_sessions))
            for row in arabic_rows:
                row["session_id"] = verified_session
                row["identity_link_basis"] = "exact_sha256_match_to_erzen"
                row["split_group_key"] = f"session:{verified_session}"
                row["corrected_row"]["session_id"] = verified_session
                row["corrected_row"]["identity_link_basis"] = "exact_sha256_match_to_erzen"
                linked_arabic_rows += 1
        elif arabic_rows and erzen_sessions:
            ambiguous_arabic_rows += len(arabic_rows)

        sources = sorted({row["source_dataset"] for row in group})
        is_cross_dataset = len(sources) > 1
        if is_cross_dataset:
            cross_dataset_pairs += sum(
                1
                for left in range(len(group))
                for right in range(left + 1, len(group))
                if group[left]["source_dataset"] != group[right]["source_dataset"]
            )

        mismatches: list[dict[str, Any]] = []
        pairwise: list[dict[str, Any]] = []
        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                a, b = group[left], group[right]
                same_text = normalize_transcript(a["text"]) == normalize_transcript(b["text"])
                pair = {
                    "left_utterance_id": a["utterance_id"],
                    "left_source_dataset": a["source_dataset"],
                    "left_transcript": a["text"],
                    "right_utterance_id": b["utterance_id"],
                    "right_source_dataset": b["source_dataset"],
                    "right_transcript": b["text"],
                    "normalized_text_match": same_text,
                    "review_status": "match_under_normalization" if same_text else "unadjudicated_mismatch",
                }
                pairwise.append(pair)
                if same_text:
                    transcript_matches += 1
                else:
                    transcript_mismatches += 1
                    mismatches.append(pair)

        report_groups.append(
            {
                "sha256": digest,
                "record_count": len(group),
                "sources": sources,
                "cross_dataset_duplicate": is_cross_dataset,
                "verified_erzen_sessions": sorted(erzen_sessions),
                "arabic_rows_linked_to_session": sum(
                    row.get("identity_link_basis") == "exact_sha256_match_to_erzen"
                    for row in arabic_rows
                ),
                "records": [
                    {
                        "source_dataset": row["source_dataset"],
                        "utterance_id": row["utterance_id"],
                        "source_audio_path": row["source_audio_path"],
                        "recording_id": row["recording_id"],
                        "session_id": row["session_id"],
                        "transcript": row["text"],
                        "split": None,
                    }
                    for row in group
                ],
                "transcript_comparisons": pairwise,
                "transcript_mismatches": mismatches,
            }
        )

    report = {
        "algorithm": "SHA256 of original referenced WAV bytes",
        "files_hashed": len(records),
        "unique_audio_hashes": len(by_hash),
        "duplicate_hash_groups": len(report_groups),
        "duplicate_rows_beyond_unique_content": sum(group["record_count"] - 1 for group in report_groups),
        "cross_dataset_duplicate_groups": sum(group["cross_dataset_duplicate"] for group in report_groups),
        "cross_dataset_duplicate_pairs": cross_dataset_pairs,
        "arabic_rows_linked_to_verified_erzen_session": linked_arabic_rows,
        "arabic_rows_with_ambiguous_erzen_link": ambiguous_arabic_rows,
        "transcript_comparison_pairs": transcript_matches + transcript_mismatches,
        "transcript_matches_after_unicode_case_whitespace_normalization": transcript_matches,
        "transcript_mismatches_unadjudicated": transcript_mismatches,
        "groups": report_groups,
    }
    return dict(by_hash), report
