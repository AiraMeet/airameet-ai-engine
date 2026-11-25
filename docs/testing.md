# Testing and Quality Assurance Guide

This document details the automated testing environment, static analysis, security auditing, and CI for the **AiraMeet AI Engine**.

---

## 1. Testing Environment

### Supported Python Versions
- **Primary:** Python `>= 3.10` (CI matrix: 3.10, 3.11, 3.12)

### Development Dependencies

| Package | Purpose |
|---|---|
| `pytest` | Test execution |
| `pytest-cov` | Coverage reporting |
| `pytest-mock` | Monkeypatching helpers |
| `ruff` | Lint and format |
| `mypy` | Static typing |
| `bandit` | Security scan |
| `pip-audit` | Dependency CVE scan |

### Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

---

## 2. Test Suite Structure

```text
tests/
├── conftest.py
├── unit/
│   ├── test_main.py
│   ├── core/
│   │   ├── test_config.py
│   │   ├── test_exceptions.py
│   │   └── test_logger.py
│   └── data_engineering/
│       ├── test_package_exports.py
│       ├── test_pipeline.py
│       ├── conversion/
│       │   └── test_audio_converter.py
│       ├── validation/
│       │   └── test_metadata.py
│       ├── inspection/
│       │   └── test_load_metadata.py
│       └── quality/
│           ├── test_audio_auditor.py
│           └── test_snr.py
└── integration/
    ├── test_data_engineering_pipeline.py
    └── test_dataset_pipeline.py
```

### Safety and Synthetic Data
- Tests never load real datasets.
- `make_wav_file` in `conftest.py` builds tiny deterministic WAV fixtures.
- Isolation uses `tmp_path` and `monkeypatch`.

---

## 3. Running Tests

```bash
pytest
pytest -v
pytest tests/unit/data_engineering/conversion/test_audio_converter.py
pytest --cov=airameet_ai --cov-report=term-missing
```

---

## 4. Code Quality & Security

```bash
ruff check src tests main.py notebooks/AI-DS-01.py
ruff check --fix src tests main.py notebooks/AI-DS-01.py
ruff format src tests main.py notebooks/AI-DS-01.py
mypy src tests main.py
bandit -r src
pip-audit
```

---

## 5. Continuous Integration

Workflow: `.github/workflows/ci.yml`

- Pull requests and pushes to `main`.
- Lint / mypy / bandit / pip-audit on Python 3.11.
- Pytest + coverage across Python 3.10–3.12.
