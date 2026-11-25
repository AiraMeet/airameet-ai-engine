<p align="center">
  <strong>A I R A M E E T</strong><br>
  <sub>AI ENGINE</sub>
</p>

<h1 align="center">AiraMeet AI Engine</h1>

<p align="center">
  The AI processing and intelligence engine behind the AiraMeet platform.
</p>

<p align="center">
  Designed to process meeting content through Automatic Speech Recognition (ASR), including Egyptian Arabic and English code-switching, and speaker diarization with speaker-attributed transcription. The planned intelligence workflows extract decisions, tasks, deadlines, and unresolved issues from meetings.
</p>

<p align="center">
  <strong>Project status:</strong> Dataset inspection, audio quality auditing, standardization, reporting, and handoff are implemented. AI services are planned.
</p>

<p align="center">
  <a href="https://github.com/AiraMeet/airameet-ai-engine"><img src="https://img.shields.io/badge/Repository-AiraMeet%2Fairameet--ai--engine-181717?logo=github" alt="AiraMeet AI Engine repository"></a>
</p>

---

## Project Objectives

The AI Engine aims to:

- Convert meeting audio into accurate text transcripts.
- Identify and distinguish individual speakers.
- Support Egyptian Arabic and English technical terminology, including code-switching.
- Generate structured meeting summaries.
- Extract decisions, action items, deadlines, and unresolved issues.
- Provide intelligent knowledge retrieval across meetings.

## Architecture Overview

Production code lives under `src/airameet_ai`. The AI-DS-01 workflow is implemented as reusable stages coordinated by `DataEngineeringPipeline`.

```text
src/airameet_ai/
├── core/                      # Settings, logging, base exceptions
└── data_engineering/
    ├── conversion/            # WAV mono 16 kHz PCM standardization
    ├── validation/            # Metadata inspection and path checks
    ├── inspection/            # Metadata loaders (CSV / Excel)
    ├── quality/               # Audio audit + WADA-SNR
    └── pipeline.py            # End-to-end orchestration
notebooks/
└── AI-DS-01.py                # Interactive # %% cells for the same pipeline
```

## Project Structure

### 1. `src/airameet_ai/core/`

Application configuration, logging, and shared exceptions.

### 2. `src/airameet_ai/data_engineering/`

| Directory | Responsibility |
|---|---|
| `inspection/` | Load CSV/Excel metadata without modifying sources. |
| `validation/` | Schema, quality, and audio-path validation. |
| `conversion/` | Convert audio to WAV / mono / 16 kHz / PCM_16. |
| `quality/` | Full-file audio audit and WADA-SNR. |
| `pipeline.py` | Ordered stages: load → inspect → audit → convert → report → handoff. |

Source datasets remain read-only. Converted audio and handoff trees are written under `data/processed/`. Reports go under `artifacts/data_engineering/reports/`.

### 3. Planned modules (not implemented yet)

ASR inference, API layer, and training pipelines will be added when Phase 2 starts. Empty placeholder packages were removed so `src/` only contains working code.

### ASR Dataset Setup

Create a `data/` directory in the repository root:

```text
data/
├── ArZen_Prepared/               # metadata.csv (+ audio/)
├── Arabic-English-ASR/           # metadata.xlsx (+ audio/)
└── processed/
```

`AIRAMEET_RAW_DATA_DIR` defaults to `data` (relative to the project root). Nested layouts such as `data/ArZen/ArZen_Prepared/` are still accepted. Configure alternate locations with `AIRAMEET_RAW_DATA_DIR`, `AIRAMEET_PROCESSED_DATA_DIR`, and `AIRAMEET_ARTIFACTS_DIR`.

## Technology Stack

- **Python** `>= 3.10`
- **NumPy, Pandas, OpenPyXL, SoundFile, SciPy**
- **Pytest** for tests

## Running the Application

```bash
python -m pip install -e ".[dev]"
python main.py
```

Copy `.env.example` to `.env` for local configuration.

To run stages interactively, open `notebooks/AI-DS-01.py` in VS Code and execute the `# %%` cells in order.

## Development Roadmap

### Phase 1 – Initial Setup

- [x] Establish the project package layout.
- [x] Add dependency management via `pyproject.toml`.
- [x] Implement configuration and logging.
- [x] Implement the dataset engineering workflow.

### Phase 2 – Automatic Speech Recognition

- [ ] Audio preprocessing.
- [ ] Initial ASR model integration.
- [ ] Transcription inference and postprocessing.
- [ ] Quality evaluation.

### Future Development

- Speaker Diarization.
- Meeting Summarization.
- Meeting Intelligence.
- Action Item and Decision Extraction.
- Knowledge Retrieval.

## Testing and Quality Assurance

See [docs/testing.md](docs/testing.md).

```bash
pytest
pytest -m 'unit and not integration'
pytest -m integration
pytest --cov=airameet_ai --cov-branch --cov-report=term-missing --cov-report=xml --cov-report=html
ruff check src tests main.py notebooks/AI-DS-01.py .github/scripts/ci_summary.py
ruff format --check src tests main.py notebooks/AI-DS-01.py .github/scripts/ci_summary.py
mypy src tests main.py
bandit -r src
pip-audit
```

See [docs/testing.md](docs/testing.md) for category markers, JUnit output, detailed report commands, and CI artifact guidance.

## Integration with AiraMeet

The AI Engine processes audio and produces structured outputs for the AiraMeet Core backend. Core remains responsible for business logic, users, meetings, and persistence.

## Development Guidelines

- Keep AI modules independent and modular.
- Separate model training from production inference.
- Keep large datasets and model weights outside Git.
- Store configuration examples without secrets.
- Add tests when implementing new functionality.

---

**Repository:** [AiraMeet/airameet-ai-engine](https://github.com/AiraMeet/airameet-ai-engine)

**Project:** AiraMeet – AI-Powered Meeting and Knowledge Management Platform.
