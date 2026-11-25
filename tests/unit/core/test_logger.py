"""Unit tests for core logger module."""

from __future__ import annotations

import logging

import pytest

from airameet_ai.core.logger import configure_logging


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configures_info_level(self) -> None:
        configure_logging("INFO")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    def test_configures_debug_level(self) -> None:
        configure_logging("DEBUG")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_configures_warning_level(self) -> None:
        configure_logging("WARNING")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING

    def test_configures_error_level(self) -> None:
        configure_logging("ERROR")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.ERROR

    def test_configures_critical_level(self) -> None:
        configure_logging("CRITICAL")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.CRITICAL

    def test_case_insensitive_level(self) -> None:
        configure_logging("info")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    def test_mixed_case_level(self) -> None:
        configure_logging("Warning")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING

    def test_invalid_level_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid log level"):
            configure_logging("INVALID_LEVEL")

    def test_returns_none(self) -> None:
        assert configure_logging("INFO") is None  # type: ignore[func-returns-value]

    def test_default_level_is_info(self) -> None:
        configure_logging()
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    def test_configures_handler(self) -> None:
        configure_logging("INFO")
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 0

    def test_format_contains_expected_fields(self) -> None:
        configure_logging("INFO")
        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]
        formatter = handler.formatter
        assert formatter is not None
        fmt = formatter._fmt or ""
        assert "%(asctime)s" in fmt
        assert "%(levelname)s" in fmt
        assert "%(name)s" in fmt
        assert "%(message)s" in fmt
