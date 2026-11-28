"""Deterministic checksum index creation and complete verification."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .audio_manager import sha256


def generate_checksums(root: Path, release_index: dict[str, str] | None = None, audio_map: dict[str, str] | None = None) -> None:
    index = root / "checksums" / "SHA256SUMS"
    index.parent.mkdir(parents=True, exist_ok=True)
    reusable = {audio_map[k]: v for k, v in (release_index or {}).items() if audio_map and k in audio_map}
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item != index and item.name != "checksum_verification.json"):
        name = path.relative_to(root).as_posix()
        actual = sha256(path)
        if name in reusable and reusable[name] != actual:
            raise ValueError(f"Release checksum does not match copied audio: {name}")
        lines.append(f"{reusable.get(name, actual)}  {name}\n")
    index.write_text("".join(lines), encoding="utf-8", newline="\n")


def verify_checksums(root: Path, *, write_report: bool = False) -> dict:
    index = root / "checksums" / "SHA256SUMS"
    declared, invalid, missing, mismatches = {}, [], [], []
    if index.is_file():
        for number, line in enumerate(index.read_text(encoding="utf-8").splitlines(), 1):
            digest, separator, name = line.partition("  ")
            path = Path(name)
            resolved = (root / path).resolve()
            if (
                not separator
                or not name
                or len(digest) != 64
                or any(ch not in "0123456789abcdef" for ch in digest)
                or path.is_absolute()
                or ".." in path.parts
                or not resolved.is_relative_to(root.resolve())
                or name in declared
            ):
                invalid.append(f"line:{number}")
            else:
                declared[name] = digest
    else:
        invalid.append("missing checksums/SHA256SUMS")
    passed = 0
    for name, digest in declared.items():
        path = root / name
        if not path.is_file():
            missing.append(name)
        elif sha256(path) != digest:
            mismatches.append(name)
        else:
            passed += 1
    indexed_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != index and path.name != "checksum_verification.json"
    }
    unindexed = sorted(indexed_files - set(declared))
    report = {"files_checked": len(declared), "files_passed": passed, "missing_files": missing, "hash_mismatches": mismatches, "invalid_checksum_entries": invalid, "unindexed_files": unindexed, "verification_timestamp": datetime.now(UTC).isoformat(), "status": "passed" if not (missing or mismatches or invalid or unindexed) else "failed"}
    report_path = root / "checksums" / "checksum_verification.json"
    if write_report:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
