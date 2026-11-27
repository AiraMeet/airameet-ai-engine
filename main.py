"""Command-line entry point for independently runnable data tasks."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from airameet_ai.core.config import PROJECT_ROOT, get_settings
from airameet_ai.core.logger import configure_logging
from airameet_ai.data_engineering.grouped_release.consolidated_dataset import (
    ConsolidatedDatasetPipeline,
)
from airameet_ai.data_engineering.pipeline import DataEngineeringPipeline
from airameet_ai.data_engineering.text_pipeline import TextNormalizationPipeline
from scripts.dataset_consolidation.consolidate_asr_dataset import consolidate, verify

TASKS = {
    "AI-DS-01": "Dataset inspection, audio audit, standardization, and handoff",
    "AI-DS-02": "Arabic and English code-switching text normalization",
    "AI-DS-03": "Leak-free grouped manifests and dataset release statistics",
    "DATASET-CONSOLIDATION": "Safely consolidate and verify the normalized ASR dataset",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AiraMeet AI Engine")
    parser.add_argument(
        "--list-tasks", action="store_true", help="List registered tasks"
    )
    parser.add_argument(
        "--task", choices=tuple(TASKS), help="Run one task independently"
    )
    parser.add_argument("--input", type=Path, help="Override the task input directory")
    parser.add_argument(
        "--output", type=Path, help="Override the task output directory"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--mode", choices=("merge", "merge-clean", "verify"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--confirm-delete", action="store_true")
    parser.add_argument("--releases", type=Path, help="Override the release source directory")
    return parser


def main(argv: Sequence[str] = ()) -> int:
    """Run a selected task; preserve AI-DS-01 as the default command."""
    args = _parser().parse_args(argv)
    if args.list_tasks:
        for task_id, description in TASKS.items():
            print(f"{task_id}: {description}")
        return 0

    settings = get_settings()
    configure_logging("DEBUG" if args.verbose else settings.log_level)
    logger = logging.getLogger(__name__)
    task = args.task or "AI-DS-01"
    try:
        if task == "AI-DS-02":
            input_dir = args.input or settings.raw_data_dir
            output_dir = (
                args.output or settings.processed_data_dir / "Arabic-English-ASR"
            )
            result = TextNormalizationPipeline(input_dir, output_dir).run()
            logger.info("AI-DS-02 completed. Outputs: %s", result["outputs"])
        elif task == "DATASET-CONSOLIDATION":
            mode = args.mode or "merge"
            destination = args.output or PROJECT_ROOT / "data" / "asr-dataset"
            if mode == "verify":
                result = verify(destination)
            else:
                result = consolidate(
                    processed=args.input or PROJECT_ROOT / "data" / "processed",
                    releases=args.releases or PROJECT_ROOT / "data" / "releases",
                    destination=destination,
                    mode=mode,
                    dry_run=args.dry_run,
                    overwrite=args.overwrite,
                    confirm_delete=args.confirm_delete,
                )
            logger.info("Dataset consolidation %s: %s", mode, result)
            if result["status"] not in {"passed", "dry-run"}:
                return 1
        elif task == "AI-DS-03":
            result = ConsolidatedDatasetPipeline(
                args.input or PROJECT_ROOT / "data" / "asr-dataset",
                args.output
                or settings.artifacts_dir / "data_engineering" / "reports",
            ).run()
            logger.info("AI-DS-03 consolidated dataset analysis completed: %s", result)
        else:
            task_settings = settings
            if args.input:
                task_settings = replace(task_settings, raw_data_dir=args.input)
            if args.output:
                task_settings = replace(task_settings, processed_data_dir=args.output)
            result = DataEngineeringPipeline(task_settings).run()
            logger.info(
                "AI-DS-01 completed. Reports: %s; handoff: %s",
                result.report_paths,
                result.handoff_dir,
            )
    except Exception:
        logger.exception("%s failed", task)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
