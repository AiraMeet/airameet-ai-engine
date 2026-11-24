"""Integration tests for dataset pipeline workflow."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import soundfile as sf

from airameet_ai.dataset import (
    convert_audio_batch,
    validate_audio_paths,
    validate_metadata_schema,
    wada_snr,
)


class TestDatasetPipelineIntegration:
    """Integration test suite chaining metadata validation and audio conversion."""

    def test_end_to_end_validation_and_conversion_pipeline(
        self,
        tmp_path: Path,
        make_wav_file: Callable[..., Path],
    ) -> None:
        source_root = tmp_path / "raw_dataset"
        output_root = tmp_path / "processed_dataset"
        audio_dir = source_root / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Create 2 valid files (one 8kHz stereo, one 16kHz mono) and 1 non-existent
        make_wav_file(
            audio_dir / "valid_stereo_8k.wav",
            sample_rate=8000,
            channels=2,
            duration=0.1,
            freq=440.0,
        )
        make_wav_file(
            audio_dir / "valid_mono_16k.wav",
            sample_rate=16000,
            channels=1,
            duration=0.1,
            freq=880.0,
        )

        raw_metadata = pd.DataFrame(
            {
                "meeting": ["M01", "M01", "M02"],
                "speaker": ["S01", "S02", "S01"],
                "audio_path": [
                    "audio/valid_stereo_8k.wav",
                    "audio/valid_mono_16k.wav",
                    "audio/missing.wav",
                ],
                "text": [
                    "Discussion on architecture",
                    "Agreed on plan",
                    "Missing file test",
                ],
                "start": [0.0, 5.0, 10.0],
                "end": [5.0, 10.0, 15.0],
                "duration": [5.0, 5.0, 5.0],
            }
        )

        # Step 1: Validate Schema
        schema_report = validate_metadata_schema(raw_metadata, "raw_dataset")
        assert schema_report["schema_valid"] is True
        assert schema_report["total_records"] == 3

        # Step 2: Validate Audio Paths
        path_validation = validate_audio_paths(raw_metadata, source_root)
        assert len(path_validation) == 3
        found_records = path_validation[path_validation["path_status"] == "found"]
        missing_records = path_validation[
            path_validation["path_status"] == "file_not_found"
        ]

        assert len(found_records) == 2
        assert len(missing_records) == 1

        # Step 3: Batch Conversion on found records only
        conversion_results = convert_audio_batch(
            found_records,
            source_root=source_root,
            output_root=output_root,
            path_column="audio_path",
            overwrite=False,
        )

        assert len(conversion_results) == 2
        assert all(r["status"] == "success" for r in conversion_results)

        # Step 4: Verify converted files and compute SNR
        for res in conversion_results:
            out_file = Path(res["output_path"])
            assert out_file.exists()

            info = sf.info(str(out_file))
            assert info.samplerate == 16000
            assert info.channels == 1
            assert info.subtype == "PCM_16"
            assert info.format == "WAV"

            audio_data, _ = sf.read(str(out_file))
            snr = wada_snr(audio_data)
            assert isinstance(snr, float)
            assert -20.0 <= snr <= 100.0
