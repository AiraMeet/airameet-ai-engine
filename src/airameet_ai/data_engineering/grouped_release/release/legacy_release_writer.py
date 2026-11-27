"""Deterministic manifest serialization, checksums, and safe release writing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def render_release(
    partitions: dict[str, list[dict[str, Any]]], statistics: dict[str, Any]
) -> dict[Path, bytes]:
    """Render all release files in memory using stable UTF-8 serialization."""
    manifest_dir = Path("data/manifests")
    outputs: dict[Path, bytes] = {}
    for split, filename in (("train", "train.jsonl"), ("validation", "val.jsonl"), ("test", "test.jsonl")):
        ordered = sorted(partitions[split], key=lambda row: row["utterance_id"])
        lines = [
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for row in ordered
        ]
        outputs[manifest_dir / filename] = (("\n".join(lines) + "\n") if lines else "").encode("utf-8")
    checksum_text = "".join(
        f"{hashlib.sha256(outputs[manifest_dir / name]).hexdigest()}  {name}\n"
        for name in ("train.jsonl", "val.jsonl", "test.jsonl")
    )
    outputs[manifest_dir / "CHECKSUMS.sha256"] = checksum_text.encode("utf-8")
    statistics["release"] = "data-v1.0"
    outputs[Path("artifacts/data_engineering/reports/statistics/data-v1.0/dataset_stats.json")] = (
        json.dumps(statistics, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    return outputs


def write_release(root: str | Path, outputs: dict[Path, bytes]) -> list[Path]:
    """Create a release only when all targets are absent or match byte-for-byte."""
    destination = Path(root)
    targets = {destination / relative: content for relative, content in outputs.items()}
    existing = [path for path in targets if path.exists()]
    if existing:
        if len(existing) != len(targets):
            raise FileExistsError(
                "Partial release artifacts already exist; refusing to overwrite or complete them"
            )
        differing = [
            path for path, expected in targets.items() if path.read_bytes() != expected
        ]
        if differing:
            names = ", ".join(str(path.relative_to(destination)) for path in differing)
            raise FileExistsError(f"Existing release differs from deterministic output: {names}")
    else:
        for path in targets:
            path.parent.mkdir(parents=True, exist_ok=True)
        for path, content in targets.items():
            with path.open("xb") as stream:
                stream.write(content)

    for path in targets:
        if path.name.endswith(".jsonl"):
            continue
        if path.name == "CHECKSUMS.sha256":
            for line in path.read_text(encoding="utf-8").splitlines():
                digest, filename = line.split(maxsplit=1)
                manifest = path.parent / filename.strip()
                if hashlib.sha256(manifest.read_bytes()).hexdigest() != digest:
                    raise OSError(f"Checksum verification failed for {manifest}")
    return list(targets)
