"""Migrate release manifests without changing split assignments or fields."""

from __future__ import annotations

import json
from pathlib import Path

SPLITS = {"train": "train.jsonl", "validation": "val.jsonl", "test": "test.jsonl"}


def read_release_manifests(release: Path) -> list[dict]:
    rows = []
    for filename in SPLITS.values():
        path = release / "manifests" / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing release manifest: {path}")
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return rows


def migrate_manifests(release: Path, stage: Path, audio_map: dict[str, str]) -> list[dict]:
    out = stage / "manifests"
    out.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    all_rows = []
    for split, filename in SPLITS.items():
        rows = [json.loads(line) for line in (release / "manifests" / filename).read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            uid = row.get("utterance_id")
            if not uid or uid in seen:
                raise ValueError(f"Missing or duplicate manifest ID: {uid}")
            seen.add(uid)
            old = row["audio_path"]
            if old not in audio_map:
                raise ValueError(f"Manifest audio is not in canonical inventory: {old}")
            row["audio_path"] = audio_map[old]
            if row.get("split") != split:
                raise ValueError(f"Split mismatch for {uid}")
        name = "validation.jsonl" if split == "validation" else filename
        (out / name).write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8", newline="\n")
        all_rows.extend(rows)
    return all_rows
