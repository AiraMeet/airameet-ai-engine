"""Copy and verify canonical release audio."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def destination_path(path: str, dataset: str) -> str:
    source_path = Path(path)
    if source_path.is_absolute() or ".." in source_path.parts:
        raise ValueError(f"Unsafe release audio path: {path}")
    prefix = Path("audio") / dataset
    tail = source_path.relative_to(prefix)
    if tail.parts and tail.parts[0] == "audio":
        tail = tail.relative_to("audio")
    name = "Arabic-English-ASR" if dataset == "Arabic-English" else "Erzen"
    return (Path("audio") / name / tail).as_posix()


def copy_audio(release: Path, stage: Path, rows: list[dict]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    reverse: dict[str, str] = {}
    for row in rows:
        old = row["audio_path"]
        new = destination_path(old, row["source_dataset"])
        if old in mapping and mapping[old] != new:
            raise ValueError(f"Audio source has incompatible destinations: {old}")
        if new in reverse and reverse[new] != old:
            raise ValueError(f"Audio destination collision: {new}")
        mapping[old] = new
        reverse[new] = old
    for old, new in sorted(mapping.items()):
        source, target = release / old, stage / new
        if not source.resolve().is_relative_to(release.resolve()) or not source.is_file():
            raise FileNotFoundError(f"Missing normalized release audio: {source}")
        if not target.resolve().is_relative_to(stage.resolve()):
            raise ValueError(f"Audio destination escapes staging directory: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256(source) != sha256(target):
            raise OSError(f"Audio copy verification failed: {old}")
    return mapping
