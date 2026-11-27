# Testing the AiraMeet Data Engineering Workflows

## Install test dependencies

From the repository root, install the development extra into the active virtual
environment:

```bash
python -m pip install -e '.[dev]'
```

The project declares pytest, pytest-cov, pytest-mock, Ruff, and scikit-learn in
its existing project configuration. `requirements.txt` remains the runtime
dependency list.

## Commands

Run the full test suite:

```bash
python -m pytest
```

Run grouped-release tests only:

```bash
python -m pytest tests/unit/data_engineering/grouped_release tests/integration/data_engineering/test_grouped_release_pipeline.py
```

Run integration tests:

```bash
python -m pytest tests/integration
```

Run with coverage and create an HTML report:

```bash
python -m pytest --cov=airameet_ai.data_engineering --cov-report=term-missing --cov-report=html
```

The `slow` marker covers read-only regression checks against the local
data-v1.1 release, including verification of its full checksum index. Exclude
those release-backed tests when the artifact is not mounted:

```bash
python -m pytest -m 'not slow'
```

## Grouped-release test modules

Tests follow the existing `tests/unit` and `tests/integration` layout.

| Module | Coverage |
|---|---|
| `test_identity.py` | Verified Erzen session parsing, speaker-track retention, per-WAV Arabic-English IDs, unknown sessions, and invalid/missing identity metadata. |
| `test_duplicates.py` | Exact file hashes, cross-dataset matching, transcript variants, different audio, missing files, and corrupt audio audit handling. |
| `test_grouped_split.py` | Seed 42 reproducibility, recording/session/hash disjointness, complete row assignment, variable group sizes, separate row/group proportions, and no speaker-independence claim. |
| `test_statistics.py` | Counts, durations, source/speaker distributions, duplicate counters, exclusions, unknown speakers, and empty data. |
| `test_serialization.py` | Deterministic JSONL, required fields, Unicode text, optional identities, relative paths in integration output, and malformed JSON rejection. |
| `test_checksums.py` | Deterministic SHA256 index, modified/missing/unlisted files, and missing index errors. |
| `test_release_promotion.py` | Checksummed staging, atomic promotion, existing release protection, and safe temporary cleanup. |
| `test_cli.py` | Registered AI-DS-03 arguments, configured paths, parser errors, and nonzero failures. |
| `tests/integration/data_engineering/grouped_release/test_pipeline.py` | Tiny isolated release generation, manifests/audio/report completeness, duplicate linkage, report copies, and failure cleanup; expensive audio stages are mocked. |
| `test_release_regression.py` | Read-only reference checks against the existing v1.1 checksum index, audio count, duplicate count, split ratios, manifests, and integrity claims. |

Fixtures create tiny synthetic WAV datasets under pytest temporary directories.
Tests do not regenerate or write into `data/releases/data-v1.1`.

## Release limitations captured by tests

Speaker-independent evaluation is not implemented or claimed. Erzen interview
sessions are verified from source identifiers. Arabic-English session identity
remains unknown unless exact audio identity links a file to a verified Erzen
session. The reference-release regression tests preserve these limitations.
Group proportions and utterance proportions are measured separately; uneven
recording/session sizes mean they need not match.
