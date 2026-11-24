# Testing and Quality Assurance Guide

This document details the automated testing environment, static code analysis, security auditing, and continuous integration (CI) workflows for the **AiraMeet AI Engine**.

---

## 1. Testing Environment

### Supported Python Versions
- **Primary:** Python `>= 3.10` (tested on 3.10, 3.11, 3.12, 3.15)

### Development Dependencies
The testing and quality tooling is managed via `pyproject.toml` and `requirements-dev.txt`:

| Package | Purpose |
|---|---|
| `pytest` | Test execution framework |
| `pytest-cov` | Code coverage reporting (terminal & XML) |
| `pytest-mock` | Fixtures and monkeypatching helpers |
| `ruff` | Fast Python linter and code formatter |
| `mypy` | Static type analysis |
| `bandit` | Common security vulnerabilities scanner |
| `pip-audit` | Vulnerability scanner for installed dependencies |

### Environment Setup
Create an isolated virtual environment and install dependencies:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade packaging tools
pip install --upgrade pip

# Install runtime and development dependencies
pip install -r requirements-dev.txt

# Or install in editable mode with dev extras
pip install -e ".[dev]"
```

---

## 2. Test Suite Structure

The test suite is structured logically under `tests/`:

```text
tests/
├── conftest.py                       # Shared test fixtures (WAV generators, sample DataFrames)
├── unit/
│   └── dataset/
│       ├── test_audio_converter.py   # Unit tests for format conversion and resampling
│       ├── test_metadata_validator.py# Unit tests for metadata quality and schema validation
│       └── test_package_exports.py   # Public interface and __all__ exports validation
└── integration/
    └── test_dataset_pipeline.py      # Multi-stage validation and conversion pipeline integration
```

### Safety and Synthetic Data
- **No Real Datasets:** The test suite does not require or load real datasets (e.g., AMI or ArzEn).
- **Synthetic Audio Generation:** The `make_wav_file` fixture in `conftest.py` generates deterministic, lightweight 16-bit PCM WAV audio waveforms in milliseconds using pure standard library tools.
- **Mocking & Isolation:** Tests utilize `tmp_path` and `monkeypatch` to ensure isolation and zero filesystem residue.

---

## 3. Running Tests

### Run all tests
```bash
pytest
```

### Run tests with verbose output
```bash
pytest -v
```

### Run a specific test module
```bash
# Unit tests
pytest tests/unit/dataset/test_audio_converter.py
pytest tests/unit/dataset/test_metadata_validator.py
pytest tests/unit/dataset/test_package_exports.py

# Integration tests
pytest tests/integration/test_dataset_pipeline.py
```

### Run tests with coverage reporting
```bash
# Terminal summary with missing line numbers
pytest --cov=airameet_ai --cov-report=term-missing

# Generate XML coverage for CI/SonarQube
pytest --cov=airameet_ai --cov-report=xml:coverage.xml
```

---

## 4. Code Quality & Security Tools

### Ruff (Linting & Formatting)
Ruff enforces code quality, style, and import sorting:

```bash
# Check for lint and import sorting issues
ruff check src tests

# Automatically fix fixable issues
ruff check --fix src tests

# Check code formatting
ruff format --check src tests

# Apply code formatting
ruff format src tests
```

### MyPy (Static Type Checking)
MyPy verifies type annotations across production and test code:

```bash
mypy src tests
```

### Bandit (Security Scanning)
Bandit scans Python source code for common security vulnerabilities:

```bash
bandit -r src
```

### pip-audit (Dependency Vulnerability Audit)
pip-audit verifies that installed packages do not contain known CVE vulnerabilities:

```bash
pip-audit
```

---

## 5. Continuous Integration (GitHub Actions)

The repository automated workflow is defined in `.github/workflows/ci.yml`.

### Triggers
- Pull requests targeting `main`.
- Pushes to `main`.

### CI Jobs
1. **Code Quality & Security:**
   - Runs `ruff check` for style and import ordering.
   - Runs `mypy` for static typing.
   - Runs `bandit` to identify security risks.
   - Runs `pip-audit` to detect vulnerable dependencies.
2. **Test Suite Execution (Matrix):**
   - Executes across Python `3.10`, `3.11`, and `3.12`.
   - Runs all unit and integration tests.
   - Generates coverage reports and uploads `coverage.xml` as an artifact.

### Resolving CI Failures
- **Linter failure:** Run `ruff check --fix src tests` locally.
- **Type errors:** Run `mypy src tests` and verify type hints.
- **Test failure:** Run `pytest -v tests/path/to/test.py` to isolate the failing assertion.
- **Audit failure:** Inspect vulnerability IDs from `pip-audit` and update dependencies in `requirements.txt`.
