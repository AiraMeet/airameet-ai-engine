"""AI-DS-01: Arabic-English ASR dataset preprocessing.

Interactive VS Code / Jupyter cells (`# %%`) that run the production
`DataEngineeringPipeline` stages in notebook order.

Fixes vs the original notebook:
- Uses shared code under ``src/airameet_ai`` (no duplicated helpers).
- Resolves project root via ``pyproject.toml`` (works from any cwd).
- Reads settings from ``.env`` / ``AIRAMEET_*`` (default ``data``).
- Writes reports under ``artifacts/data_engineering/reports``.
- Avoids Jupyter-only APIs (``display``, ``%pip``).
"""

# %% 1. Bootstrap project path and imports

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve()
while PROJECT_ROOT != PROJECT_ROOT.parent and not (
    PROJECT_ROOT / "pyproject.toml"
).is_file():
    PROJECT_ROOT = PROJECT_ROOT.parent

if not (PROJECT_ROOT / "pyproject.toml").is_file():
    raise FileNotFoundError(
        "Open this file from within the AiraMeet project (pyproject.toml not found)."
    )

src_path = str(PROJECT_ROOT / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from airameet_ai.core.config import get_settings  # noqa: E402
from airameet_ai.core.logger import configure_logging  # noqa: E402
from airameet_ai.data_engineering.pipeline import DataEngineeringPipeline  # noqa: E402

print("Project root:", PROJECT_ROOT)
print("Python:", sys.executable)

# %% 2. Initialize settings, logging, and pipeline

settings = get_settings()
configure_logging(settings.log_level)
pipeline = DataEngineeringPipeline(settings)

print("Environment:", settings.environment)
print("Raw data:", settings.raw_data_dir)
print("Processed data:", settings.processed_data_dir)
print("Artifacts:", settings.artifacts_dir)

# %% 3. Load original metadata (read-only)

datasets = pipeline.load_datasets()
for name, (root, metadata) in datasets.items():
    print(f"{name}: {len(metadata)} rows @ {root}")

# %% 4. Inspect metadata and validate audio paths

metadata_results = pipeline.inspect_metadata()

for name, quality in metadata_results.metadata_quality.items():
    print(f"\n=== {name} quality ===")
    for key, value in quality.items():
        print(f"  {key}: {value}")

for name, path_report in metadata_results.path_reports.items():
    print(f"\n=== {name} path status ===")
    print(path_report["path_status"].value_counts(dropna=False).to_string())

# %% 5. Audit ArZen audio quality

arzen_audio_audit = pipeline.audit_arzen()
print("ArZen final_audit_status:")
print(arzen_audio_audit["final_audit_status"].value_counts(dropna=False).to_string())

# %% 6. Standardize nonstandard Arabic-English audio into processed/

conversion_results = pipeline.convert_nonstandard_audio()
print(f"Conversion results: {len(conversion_results)}")
if conversion_results:
    statuses = {}
    for row in conversion_results:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    print("Status counts:", statuses)

# %% 7. Audit resolved Arabic-English audio

arabic_audio_audit = pipeline.audit_arabic_english()
print("Arabic-English-ASR final_audit_status:")
print(
    arabic_audio_audit["final_audit_status"].value_counts(dropna=False).to_string()
)

# %% 8. Write audit reports and quarantine / review ledgers

report_paths = pipeline.write_reports()
for label, path in report_paths.items():
    print(f"{label}: {path}")

# %% 9. Create complete audio handoff tree

handoff_dir = pipeline.create_handoff()
wav_count = sum(1 for path in handoff_dir.rglob("*.wav") if path.is_file())
print("Handoff directory:", handoff_dir)
print("WAV files copied:", wav_count)
print("Pipeline complete.")
