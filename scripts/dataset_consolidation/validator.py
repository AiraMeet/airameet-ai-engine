"""Validate manifest integrity and all consolidated audio references."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def validate_dataset(root: Path) -> dict:
    errors, seen, references, count = [], set(), set(), 0
    for filename in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        manifest = root / "manifests" / filename
        if not manifest.is_file():
            errors.append(f"missing manifest: {filename}")
            continue
        for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            try:
                row = json.loads(line)
                audio = Path(row["audio_path"])
                if audio.is_absolute() or ".." in audio.parts:
                    raise ValueError("audio_path must be a safe relative path")
                resolved_audio = (root / audio).resolve()
                if not resolved_audio.is_relative_to(root.resolve()):
                    raise ValueError("audio_path resolves outside dataset root")
                if not resolved_audio.is_file():
                    errors.append(f"missing audio: {audio}")
                references.add(audio.as_posix())
                expected_split = "validation" if filename == "validation.jsonl" else filename.removesuffix(".jsonl")
                if row.get("split") != expected_split:
                    errors.append(f"wrong split assignment at {filename}:{number}")
                uid = row.get("utterance_id")
                if not uid or uid in seen:
                    errors.append(f"missing or duplicate utterance_id at {filename}:{number}")
                seen.add(uid)
                count += 1
            except (KeyError, ValueError, json.JSONDecodeError) as error:
                errors.append(f"{filename}:{number}: {error}")
    metadata = root / "metadata" / "merged_metadata.csv"
    if not metadata.is_file():
        errors.append("missing metadata/merged_metadata.csv")
    else:
        with metadata.open(encoding="utf-8-sig", newline="") as stream:
            metadata_rows = list(csv.DictReader(stream))
        manifest_ids = seen
        eligible_ids = {row.get("utterance_id", "") for row in metadata_rows if row.get("release_eligible", "").lower() == "true"}
        if eligible_ids != manifest_ids:
            errors.append(f"eligible metadata/manifest ID mismatch: {len(eligible_ids ^ manifest_ids)}")
        for row in metadata_rows:
            if row.get("release_eligible", "").lower() != "true" or not row.get("audio_path"):
                continue
            audio_path = Path(row["audio_path"])
            resolved_audio = (root / audio_path).resolve()
            if audio_path.is_absolute() or ".." in audio_path.parts or not resolved_audio.is_relative_to(root.resolve()) or not resolved_audio.is_file():
                errors.append(f"metadata audio path does not resolve: {row['audio_path']}")
    present = {path.relative_to(root).as_posix() for path in (root / "audio").rglob("*") if path.is_file()}
    extra = sorted(present - references)
    if extra:
        errors.append(f"unreferenced audio files: {len(extra)}")
    return {"status": "passed" if not errors else "failed", "manifest_rows": count, "errors": errors}
