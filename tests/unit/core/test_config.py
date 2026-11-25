"""Unit tests for core config module."""

from __future__ import annotations

from pathlib import Path

import pytest

from airameet_ai.core.config import Settings, get_settings


class TestSettings:
    """Tests for the Settings dataclass."""

    def test_default_app_name(self) -> None:
        settings = Settings()
        assert settings.app_name == "AiraMeet AI Engine"

    def test_default_environment(self) -> None:
        settings = Settings()
        assert settings.environment == "development"

    def test_default_log_level(self) -> None:
        settings = Settings()
        assert settings.log_level == "INFO"

    def test_default_raw_data_dir_is_path(self) -> None:
        settings = Settings()
        assert isinstance(settings.raw_data_dir, Path)

    def test_default_raw_data_dir_points_to_data(self) -> None:
        settings = Settings()
        assert settings.raw_data_dir.name == "data"

    def test_default_processed_data_dir_is_path(self) -> None:
        settings = Settings()
        assert isinstance(settings.processed_data_dir, Path)

    def test_default_artifacts_dir_is_path(self) -> None:
        settings = Settings()
        assert isinstance(settings.artifacts_dir, Path)

    def test_settings_is_frozen(self) -> None:
        settings = Settings()
        with pytest.raises((AttributeError, TypeError)):
            settings.app_name = "changed"  # type: ignore[misc]

    def test_custom_environment(self) -> None:
        settings = Settings(environment="production")
        assert settings.environment == "production"

    def test_custom_log_level(self) -> None:
        settings = Settings(log_level="DEBUG")
        assert settings.log_level == "DEBUG"

    def test_custom_raw_data_dir(self, tmp_path: Path) -> None:
        settings = Settings(raw_data_dir=tmp_path / "custom_raw")
        assert settings.raw_data_dir == tmp_path / "custom_raw"

    def test_custom_processed_data_dir(self, tmp_path: Path) -> None:
        settings = Settings(processed_data_dir=tmp_path / "custom_processed")
        assert settings.processed_data_dir == tmp_path / "custom_processed"

    def test_custom_artifacts_dir(self, tmp_path: Path) -> None:
        settings = Settings(artifacts_dir=tmp_path / "custom_artifacts")
        assert settings.artifacts_dir == tmp_path / "custom_artifacts"


class TestGetSettings:
    """Tests for the get_settings factory function."""

    def test_returns_settings_instance(self) -> None:
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_default_environment_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AIRAMEET_ENV", "staging")
        settings = get_settings()
        assert settings.environment == "staging"

    def test_default_log_level_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AIRAMEET_LOG_LEVEL", "debug")
        settings = get_settings()
        assert settings.log_level == "DEBUG"

    def test_log_level_is_uppercased(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AIRAMEET_LOG_LEVEL", "warning")
        settings = get_settings()
        assert settings.log_level == "WARNING"

    def test_raw_data_dir_from_env_relative(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AIRAMEET_RAW_DATA_DIR", "custom/raw")
        settings = get_settings()
        assert settings.raw_data_dir.is_absolute()
        assert settings.raw_data_dir.name == "raw"
        assert settings.raw_data_dir.parent.name == "custom"

    def test_raw_data_dir_from_env_absolute(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        abs_dir = str(tmp_path / "abs_raw")
        monkeypatch.setenv("AIRAMEET_RAW_DATA_DIR", abs_dir)
        settings = get_settings()
        assert settings.raw_data_dir == Path(abs_dir)

    def test_processed_data_dir_from_env_relative(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AIRAMEET_PROCESSED_DATA_DIR", "custom/processed")
        settings = get_settings()
        assert settings.processed_data_dir.is_absolute()
        assert settings.processed_data_dir.name == "processed"

    def test_artifacts_dir_from_env_relative(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AIRAMEET_ARTIFACTS_DIR", "custom/artifacts")
        settings = get_settings()
        assert settings.artifacts_dir.is_absolute()
        assert settings.artifacts_dir.name == "artifacts"

    def test_env_file_loading(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("AIRAMEET_ENV=test_from_file\n", encoding="utf-8")

        # Patch PROJECT_ROOT to point to tmp_path so get_settings finds the .env
        import airameet_ai.core.config as config_module

        original_root = config_module.PROJECT_ROOT
        monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
        # Clear any existing env override
        monkeypatch.delenv("AIRAMEET_ENV", raising=False)
        try:
            settings = get_settings()
            assert settings.environment == "test_from_file"
        finally:
            monkeypatch.setattr(config_module, "PROJECT_ROOT", original_root)

    def test_env_file_ignores_comments(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# This is a comment\nAIRAMEET_ENV=from_file\n", encoding="utf-8"
        )

        import airameet_ai.core.config as config_module

        original_root = config_module.PROJECT_ROOT
        monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
        monkeypatch.delenv("AIRAMEET_ENV", raising=False)
        try:
            settings = get_settings()
            assert settings.environment == "from_file"
        finally:
            monkeypatch.setattr(config_module, "PROJECT_ROOT", original_root)

    def test_env_file_quoted_values(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text('AIRAMEET_ENV="quoted_value"\n', encoding="utf-8")

        import airameet_ai.core.config as config_module

        original_root = config_module.PROJECT_ROOT
        monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
        monkeypatch.delenv("AIRAMEET_ENV", raising=False)
        try:
            settings = get_settings()
            assert settings.environment == "quoted_value"
        finally:
            monkeypatch.setattr(config_module, "PROJECT_ROOT", original_root)

    def test_explicit_env_vars_take_priority_over_env_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("AIRAMEET_ENV=from_file\n", encoding="utf-8")

        import airameet_ai.core.config as config_module

        original_root = config_module.PROJECT_ROOT
        monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
        monkeypatch.setenv("AIRAMEET_ENV", "from_env_var")
        try:
            settings = get_settings()
            # os.environ.setdefault means existing env vars take priority
            assert settings.environment == "from_env_var"
        finally:
            monkeypatch.setattr(config_module, "PROJECT_ROOT", original_root)
