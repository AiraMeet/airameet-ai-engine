"""Fixtures for deterministic grouped-release unit tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest


@pytest.fixture
def grouped_records_factory() -> Callable[..., list[dict[str, Any]]]:
    """Build varied recording groups with stable hashes and known identities."""

    def make_records(
        group_sizes: tuple[int, ...] = tuple(range(1, 21)),
        *,
        speaker: str | None = "speaker-1",
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for group_index, size in enumerate(group_sizes):
            session = f"C03-R{group_index + 1:02d}" if group_index % 4 == 0 else None
            key = f"session:{session}" if session else f"recording:source-{group_index}"
            for row_index in range(size):
                records.append(
                    {
                        "utterance_id": f"utt-{group_index:03d}-{row_index:03d}",
                        "source_dataset": "Erzen" if session else "Arabic-English",
                        "recording_id": session or f"areng:{group_index:03d}-{row_index:03d}",
                        "source_recording_id": f"{session}-I01" if session else None,
                        "session_id": session,
                        "split_group_key": key,
                        "split_group_id": key,
                        "source_audio_sha256": f"hash-{group_index:03d}-{row_index:03d}",
                        "speaker_id": speaker,
                        "duration": float(row_index + 1),
                        "text": "مرحبا hello",
                    }
                )
        return records

    return make_records
