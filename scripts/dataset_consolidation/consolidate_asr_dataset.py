"""CLI and orchestration for safe AiraMeet ASR dataset consolidation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.dataset_consolidation.audio_manager import copy_audio, sha256
    from scripts.dataset_consolidation.checksum_manager import (
        generate_checksums,
        verify_checksums,
    )
    from scripts.dataset_consolidation.manifest_manager import (
        migrate_manifests,
        read_release_manifests,
    )
    from scripts.dataset_consolidation.metadata_merger import FILES, merge_metadata
    from scripts.dataset_consolidation.validator import validate_dataset
else:
    from .audio_manager import copy_audio, sha256
    from .checksum_manager import generate_checksums, verify_checksums
    from .manifest_manager import migrate_manifests, read_release_manifests
    from .metadata_merger import FILES, merge_metadata
    from .validator import validate_dataset


def verify(destination: Path, *, write_report: bool = False) -> dict:
    structure = validate_dataset(destination)
    checksums = verify_checksums(destination, write_report=write_report)
    return {"status": "passed" if structure["status"] == checksums["status"] == "passed" else "failed", "structure": structure, "checksums": checksums}


def _checksum_index(release: Path) -> dict[str, str]:
    entries = {}
    for line in (release / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
        digest, sep, relative = line.partition("  ")
        if sep and len(digest) == 64:
            entries[relative] = digest
    return entries


def _source_inventory(processed: Path, releases: Path, release: Path, rows: list[dict]) -> dict:
    return {
        "processed": {"path": str(processed), "audio_files": sum(1 for p in processed.rglob("*.wav") if p.is_file()) if processed.exists() else 0},
        "releases": {"path": str(releases)},
        "release": str(release),
        "normalized_audio_files": sum(p.is_file() for p in (release / "audio").rglob("*")),
        "manifests": {name: sum(row.get("split") == name for row in rows) for name in ("train", "validation", "test")},
        "manifest_rows": len(rows),
        "metadata_rows": {name: sum(1 for _ in (release / "metadata" / filename).open(encoding="utf-8-sig")) - 1 for name, filename in FILES.items()},
    }


def consolidate(*, processed: Path, releases: Path, destination: Path, mode: str = "merge", dry_run: bool = False, overwrite: bool = False, confirm_delete: bool = False) -> dict:
    if mode not in {"merge", "merge-clean"}:
        raise ValueError(f"Unsupported merge mode: {mode}")
    if mode == "merge-clean" and not confirm_delete:
        raise ValueError("merge-clean requires --confirm-delete")
    release = releases / "data-v1.1"
    if not processed.is_dir() or not releases.is_dir() or not release.is_dir():
        raise FileNotFoundError("Expected data/processed and data/releases/data-v1.1 source directories")
    rows = read_release_manifests(release)
    inventory = _source_inventory(processed, releases, release, rows)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination exists; pass --overwrite: {destination}")
    if dry_run:
        return {"status": "dry-run", "inventory": inventory, "destination": str(destination), "cleanup": False}

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    backup: Path | None = None
    promoted = False
    try:
        audio_map = copy_audio(release, stage, rows)
        merge_metadata(release, stage, audio_map)
        migrated = migrate_manifests(release, stage, audio_map)
        generate_checksums(stage, _checksum_index(release), audio_map)
        staged_check = verify(stage, write_report=True)
        if staged_check["status"] != "passed" or len(migrated) != len(rows):
            raise ValueError(f"Staging verification failed: {staged_check}")
        if destination.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{destination.name}.backup-", dir=destination.parent))
            backup.rmdir()
            os.replace(destination, backup)
        os.replace(stage, destination)
        promoted = True
        final_check = verify(destination, write_report=True)
        if final_check["status"] != "passed":
            raise ValueError(f"Final verification failed: {final_check}")
        if mode == "merge-clean":
            data_root = destination.parent.resolve()
            if processed.resolve() != data_root / "processed" or releases.resolve() != data_root / "releases":
                raise ValueError("Refusing cleanup: source paths are outside approved data directories")
            original_hashes = _checksum_index(release)
            for old, new in audio_map.items():
                expected = original_hashes.get(old)
                if not expected or sha256(release / old) != expected or sha256(destination / new) != expected:
                    raise ValueError(f"Refusing cleanup: source/destination audio hash mismatch: {old}")
            cleanup_backup = Path(tempfile.mkdtemp(prefix=".asr-cleanup-backup-", dir=data_root))
            cleanup_started = False
            try:
                shutil.copytree(processed, cleanup_backup / "processed")
                shutil.copytree(releases, cleanup_backup / "releases")
                cleanup_started = True
                shutil.rmtree(processed)
                shutil.rmtree(releases)
            except Exception:
                if cleanup_started:
                    shutil.rmtree(processed, ignore_errors=True)
                    shutil.rmtree(releases, ignore_errors=True)
                if cleanup_started and (cleanup_backup / "processed").exists():
                    os.replace(cleanup_backup / "processed", processed)
                if cleanup_started and (cleanup_backup / "releases").exists():
                    os.replace(cleanup_backup / "releases", releases)
                raise
            finally:
                if cleanup_backup.exists():
                    shutil.rmtree(cleanup_backup, ignore_errors=True)
        if backup and backup.exists():
            # A failure to discard the old copy must not trigger rollback from
            # a partly removed backup. Keep any leftovers for manual cleanup.
            shutil.rmtree(backup, ignore_errors=True)
        return {"status": "passed", "inventory": inventory, "destination": str(destination), "verification": final_check, "cleanup": mode == "merge-clean"}
    except Exception:
        if promoted and destination.exists():
            shutil.rmtree(destination)
        if backup and backup.exists():
            os.replace(backup, destination)
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Consolidate the AiraMeet ASR dataset")
    parser.add_argument("--mode", choices=("merge", "merge-clean", "verify"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--confirm-delete", action="store_true")
    parser.add_argument("--processed", type=Path)
    parser.add_argument("--releases", type=Path)
    parser.add_argument("--destination", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    processed = args.processed or repo / "data" / "processed"
    releases = args.releases or repo / "data" / "releases"
    destination = args.destination or repo / "data" / "asr-dataset"
    try:
        result = verify(destination) if args.mode == "verify" else consolidate(processed=processed, releases=releases, destination=destination, mode=args.mode, dry_run=args.dry_run, overwrite=args.overwrite, confirm_delete=args.confirm_delete)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] in {"passed", "dry-run"} else 1
    except Exception as error:
        print(f"Dataset consolidation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
