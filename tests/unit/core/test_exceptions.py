"""Unit tests for core exceptions module."""

from __future__ import annotations

import pytest

from airameet_ai.core.exceptions import AiraMeetError


class TestAiraMeetError:
    """Tests for AiraMeetError base exception."""

    def test_is_subclass_of_exception(self) -> None:
        assert issubclass(AiraMeetError, Exception)

    def test_can_raise_and_catch(self) -> None:
        with pytest.raises(AiraMeetError):
            raise AiraMeetError("something went wrong")

    def test_message_is_preserved(self) -> None:
        msg = "detailed error message"
        error = AiraMeetError(msg)
        assert str(error) == msg

    def test_can_raise_without_message(self) -> None:
        with pytest.raises(AiraMeetError):
            raise AiraMeetError

    def test_can_be_caught_as_exception(self) -> None:
        caught = False
        try:
            raise AiraMeetError("oops")
        except Exception:
            caught = True
        assert caught

    def test_can_be_subclassed(self) -> None:
        class SpecificError(AiraMeetError):
            """Specific domain error."""

        with pytest.raises(AiraMeetError):
            raise SpecificError("specific problem")

    def test_subclass_is_also_airameet_error(self) -> None:
        class SpecificError(AiraMeetError):
            pass

        assert issubclass(SpecificError, AiraMeetError)
        assert issubclass(SpecificError, Exception)
