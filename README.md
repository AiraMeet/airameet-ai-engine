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
  <strong>Project status:</strong> Initial setup. Architecture and folder structure are established; implementation is planned.
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

The project follows a modular architecture that separates API communication, AI processing modules, training workflows, integrations, and shared infrastructure.

```text
src/
└── airameet_ai/
    ├── api/
    ├── core/
    ├── modules/
    ├── pipelines/
    ├── integrations/
    ├── schemas/
    └── utils/
```

## Project Structure

### 1. `src/airameet_ai/`

Contains the main application source code.

| Directory | Responsibility |
|---|---|
| `api/` | API endpoints and request handling. |
| `api/routes/` | API route definitions. |
| `api/middleware/` | Request processing middleware. |
| `core/` | Application configuration, logging, and shared infrastructure. |
| `modules/` | Independent AI processing modules. |
| `pipelines/` | Orchestration of multi-stage AI workflows. |
| `integrations/` | Integration with external services and AiraMeet Core. |
| `schemas/` | Shared request, response, and data schemas. |
| `utils/` | Reusable helper utilities. |

### 2. `modules/asr/` – Automatic Speech Recognition

Responsible for converting meeting audio into text.

| Directory | Responsibility |
|---|---|
| `models/` | ASR model definitions and loading components. |
| `preprocessing/` | Audio preparation and preprocessing. |
| `inference/` | Model inference and transcription execution. |
| `postprocessing/` | Text normalization, timestamp handling, and transcript formatting. |

**Initial ASR Target:**

- English speech recognition.
- Egyptian Arabic speech recognition.
- Arabic-English code-switching.
- Timestamp-aligned transcription.
- Structured transcription output.

### 3. `training/asr/`

Dedicated to ASR model development and experimentation.

| Directory | Responsibility |
|---|---|
| `datasets/` | Dataset preparation and organization. |
| `preprocessing/` | Training data preprocessing workflows. |
| `experiments/` | Model training experiments and configurations. |
| `evaluation/` | Model evaluation and performance metrics. |

### 4. Supporting Directories

| Directory | Responsibility |
|---|---|
| `configs/` | Application and AI model configurations. |
| `tests/` | Unit, integration, and end-to-end tests. |
| `scripts/` | Development and automation scripts. |
| `docs/` | Technical documentation and architecture references. |
| `.github/workflows/` | GitHub Actions automation workflows. |

---

## Technology Stack

The planned technology stack includes:

- **Python** – Primary development language.
- **FastAPI** – API framework.
- **PyTorch** – Deep Learning framework.
- **Hugging Face** – Model and dataset ecosystem.
- **Pydantic** – Data validation and schemas.
- **Pytest** – Testing framework.
- **Docker** – Containerization and deployment.

Technology selection and implementation details may evolve during development.

## Development Roadmap

### Phase 1 – Initial Setup

- [x] Establish the initial project directory structure.
- [ ] Configure FastAPI.
- [ ] Add project dependency management using `pyproject.toml`.
- [ ] Implement application configuration and logging.
- [ ] Add health check endpoint.
- [ ] Create Dockerfile.

### Phase 2 – Automatic Speech Recognition

- [ ] Implement audio preprocessing.
- [ ] Integrate the initial ASR model.
- [ ] Implement transcription inference.
- [ ] Support timestamp-aligned transcription.
- [ ] Implement text postprocessing.
- [ ] Evaluate transcription quality.

### Future Development

- Speaker Diarization.
- Meeting Summarization.
- Meeting Intelligence.
- Action Item and Decision Extraction.
- Knowledge Retrieval.

## Integration with AiraMeet

The AI Engine is designed to work alongside the AiraMeet Core backend.

Its responsibilities include processing audio, generating structured AI outputs, and exposing these capabilities through APIs for consumption by the main backend.

The Core backend remains responsible for application business logic, user management, meeting management, and persistent application data.

## Development Guidelines

- Keep AI modules independent and modular.
- Separate model training from production inference.
- Keep large datasets and model weights outside the Git repository.
- Store configuration examples without exposing secrets.
- Maintain consistent data schemas between services.
- Add tests when implementing new functionality.
- Document important architectural and technical decisions.

---

**Repository:** [AiraMeet/airameet-ai-engine](https://github.com/AiraMeet/airameet-ai-engine)

**Project:** AiraMeet – AI-Powered Meeting and Knowledge Management Platform.
