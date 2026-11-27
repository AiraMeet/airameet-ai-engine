"""Safe v1.1 release serialization, checksums, and atomic promotion."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def json_bytes(value: Any, *, indent: int | None = None) -> bytes:
    """Return deterministic UTF-8 JSON bytes."""
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
    )
    return (text + "\n").encode("utf-8")


def write_json(path: Path, value: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value, indent=indent))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def render_manifest(rows: list[dict[str, Any]]) -> bytes:
    """Serialize stable JSONL, excluding implementation-only keys."""
    fields = (
        "utterance_id",
        "audio_path",
        "text",
        "recording_id",
        "session_id",
        "source_recording_id",
        "speaker_id",
        "start",
        "end",
        "duration",
        "source_dataset",
        "split",
        "source_audio_sha256",
        "identity_link_basis",
    )
    ordered = sorted(rows, key=lambda row: row["utterance_id"])
    lines = [
        json.dumps(
            {field: row[field] for field in fields if row.get(field) is not None},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in ordered
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def write_checksums(root: Path) -> dict[str, str]:
    """Checksum every release file except the checksum file itself."""
    checksum_path = root / "CHECKSUMS.sha256"
    digests: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item != checksum_path):
        relative = path.relative_to(root).as_posix()
        digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
        digests[relative] = digest
    checksum_path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(digests.items())),
        encoding="utf-8",
        newline="\n",
    )
    return digests


def verify_checksums(root: Path) -> dict[str, Any]:
    """Verify every named release file and reject unlisted release files."""
    checksum_path = root / "CHECKSUMS.sha256"
    declared: dict[str, str] = {}
    for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64 or relative in declared:
            raise ValueError(f"Invalid checksum line {line_number}")
        declared[relative] = digest
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != checksum_path
    }
    if actual_files != set(declared):
        raise ValueError("Release checksum index does not cover the exact release file set")
    mismatches = []
    for relative, expected in declared.items():
        actual = hashlib.file_digest((root / relative).open("rb"), "sha256").hexdigest()
        if actual != expected:
            mismatches.append(relative)
    if mismatches:
        raise ValueError(f"Checksum mismatch for release files: {mismatches[:10]}")
    return {"status": "passed", "files_verified": len(declared)}


def archive_release_reports(release_root: Path, reports_root: Path) -> list[Path]:
    """Copy release reports to categorized central storage without replacing history."""
    report_categories = {
        "dataset_stats.json": "statistics",
        "duplicate_report.json": "duplicates",
        "audio_quality_report.json": "audio_quality",
        "split_integrity_report.json": "splitting",
        "release_audit.md": "release",
    }
    version = release_root.name
    written: list[Path] = []
    for name, category in report_categories.items():
        source = release_root / "reports" / name
        if not source.is_file():
            raise FileNotFoundError(f"Expected release report is missing: {source}")
        destination = reports_root / category / version / name
        content = source.read_bytes()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_bytes() != content:
                raise FileExistsError(f"Refusing to replace historical report: {destination}")
        else:
            with destination.open("xb") as stream:
                stream.write(content)
        written.append(destination)
    return written


def create_staging_directory(final_root: Path) -> Path:
    """Create a same-filesystem staging directory without touching releases."""
    if final_root.exists():
        raise FileExistsError(f"Release destination already exists: {final_root}")
    final_root.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{final_root.name}.staging-", dir=final_root.parent))


def promote_release(staging_root: Path, final_root: Path) -> None:
    """Atomically publish a complete release to a previously absent path."""
    if final_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing release: {final_root}")
    os.rename(staging_root, final_root)


def discard_staging_directory(staging_root: Path) -> None:
    """Remove only the temporary directory created for the failed run."""
    if staging_root.exists():
        shutil.rmtree(staging_root)
