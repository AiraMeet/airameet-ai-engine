# Testing and quality guide

## Testing architecture

Tests exercise the implemented dataset engineering workflow with small synthetic fixtures. They do not load the production dataset or contact an external AI service. The current suite has unit and integration tests; `tests/e2e/` is reserved and contains no end-to-end tests yet.

| Category | Scope | Marker |
|---|---|---|
| Unit | Individual functions, modules, validation rules, and error behavior | `unit` |
| Integration | Interactions across metadata loading, validation, audio processing, reporting, and handoff | `integration` |
| Data validation | Metadata schema, values, timing, duplicates, and audio paths | `data_validation` |
| Audio processing | Synthetic WAV loading, conversion, audio auditing, and SNR | `audio_processing` |
| Error handling | Invalid input and error paths, selected by test name | `error_handling` |

Markers are assigned during collection from the directory layout and test names in `tests/conftest.py`. Unit and integration are assigned from their directories. Data validation and audio processing overlap unit tests by design. The error handling marker is a convenience category; review its selected tests before relying on it as a complete error inventory.

```text
tests/
├── conftest.py                 # Synthetic fixtures and category markers
├── unit/
│   ├── core/
│   └── data_engineering/       # Conversion, inspection, quality, validation, pipeline
├── integration/                # Workflow level checks
└── e2e/                        # Reserved; no end-to-end tests currently
```

## Install development dependencies

From the repository root, use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The same tools can be installed from `requirements-dev.txt`. The editable install uses the dependency declarations in `pyproject.toml`.

## Run tests

```bash
# All tests
pytest

# Unit and integration categories
pytest -m 'unit and not integration'
pytest -m integration

# Cross-cutting categories
pytest -m data_validation
pytest -m audio_processing
pytest -m error_handling

# Individual test file, verbose output, and slowest test timings
pytest -v tests/unit/data_engineering/conversion/test_audio_converter.py
pytest --durations=0
```

## Coverage reports

Coverage is configured for branch measurement. The current `fail_under = 0` setting reports coverage without introducing a threshold unsupported by a stable baseline.

```bash
# Terminal summary and missing lines
pytest --cov=airameet_ai --cov-branch --cov-report=term-missing

# Terminal, XML, and browsable HTML reports
pytest --cov=airameet_ai --cov-branch \
  --cov-report=term-missing \
  --cov-report=xml:coverage.xml \
  --cov-report=html:htmlcov

# JUnit test results alongside coverage
pytest --cov=airameet_ai --cov-branch \
  --cov-report=term-missing --cov-report=xml:coverage.xml \
  --cov-report=html:htmlcov --junitxml=junit.xml
```

The terminal report lists statements, branches, missing lines, and percentages by module. Open `htmlcov/index.html` for line-level detail. `coverage.xml` is intended for CI integrations. Total coverage alone can hide untested modules, so review each module and its missing lines.

## Code quality

```bash
ruff check src tests main.py notebooks/AI-DS-01.py .github/scripts/ci_summary.py
ruff format --check src tests main.py notebooks/AI-DS-01.py .github/scripts/ci_summary.py
mypy --show-error-codes src tests main.py
```

Ruff reports lint rule codes and file locations; format check reports files that need formatting. MyPy reports file, line, column, and error code where available. Apply formatting to tests and source files intentionally with `ruff format <paths>`; a format check does not edit files.

## Security audits

```bash
bandit -r src
bandit -r src -f json -o bandit.json
pip-audit
pip-audit -f json -o pip-audit.json
```

Bandit examines repository source and reports code locations, rule IDs, severity, and confidence. `pip-audit` checks installed/project dependencies against its vulnerability data source and reports affected versions, vulnerability IDs, and fixed versions when known. The audit needs network access to its vulnerability service. Findings remain visible and fail the relevant CI job; do not suppress them just to make the job pass.

## GitHub Actions

`.github/workflows/ci.yml` runs separate Code Quality, Security Audit, Unit Tests, Integration Tests, Python Compatibility, Coverage Report, and CI Summary jobs. The full suite is run on Python 3.10, 3.11, and 3.12. The coverage job runs on Python 3.11 and produces terminal, XML, HTML, and JUnit output.

The run's **Summary** tab contains overall job status, aggregate JUnit test counts, coverage percentages, quality/security finding counts, compatibility status, and links to the run's artifact area. Download **coverage-and-test-reports** for HTML coverage, XML coverage, JUnit XML, and logs. Other artifacts contain code quality, security, unit, integration, and per-version compatibility results. Artifacts are retained for 14 days. Reports are uploaded even when their test or audit command fails, when the files were generated.

JUnit XML supplies test counts and durations. Compatibility status is reported as the matrix job result; open each Python version job and its uploaded JUnit/log artifact to identify a version-specific failure. Ruff JSON and MyPy logs include locations and diagnostics. Bandit JSON and pip-audit JSON retain detailed findings and package versions.

## Troubleshooting

- **No tests collected for a marker:** run `pytest --collect-only -m <marker>` and inspect the category assignment in `tests/conftest.py`.
- **Audio test failures:** tests use `tmp_path` and generated WAVs; confirm SoundFile and SciPy installed from the development dependencies.
- **Coverage report missing:** ensure pytest-cov is installed and the test process completed far enough to write reports.
- **pip-audit cannot connect:** rerun in an environment with access to its vulnerability service. A network failure is not evidence of a clean audit.
- **Python matrix install/test failure:** inspect that version's job log for dependency resolution, wheel availability, and the failing test or import.
- **Formatting and lint differ:** run the Ruff commands independently; formatting and lint rules are separate checks.
