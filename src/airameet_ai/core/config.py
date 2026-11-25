"""Application settings loaded from the environment and an optional .env file."""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _configured_path(name: str, default: str) -> Path:
    """Resolve a configured path relative to the project when not absolute."""
    path = Path(os.getenv(name, default)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    """Small set of settings needed to initialize the application."""

    app_name: str = "AiraMeet AI Engine"
    environment: str = "development"
    log_level: str = "INFO"
    raw_data_dir: Path = PROJECT_ROOT / "data"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"


def get_settings() -> Settings:
    """Load settings, using development-friendly defaults."""
    env_file = PROJECT_ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            entry = line.strip()
            if not entry or entry.startswith("#") or "=" not in entry:
                continue
            key, value = entry.split("=", maxsplit=1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))

    return Settings(
        environment=os.getenv("AIRAMEET_ENV", "development"),
        log_level=os.getenv("AIRAMEET_LOG_LEVEL", "INFO").upper(),
        raw_data_dir=_configured_path("AIRAMEET_RAW_DATA_DIR", "data"),
        processed_data_dir=_configured_path(
            "AIRAMEET_PROCESSED_DATA_DIR", "data/processed"
        ),
        artifacts_dir=_configured_path("AIRAMEET_ARTIFACTS_DIR", "artifacts"),
    )
