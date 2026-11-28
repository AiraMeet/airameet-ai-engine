"""Statistics and unknown identity accounting tests."""

from airameet_ai.data_engineering.grouped_release.pipeline import DatasetReleasePipeline


def test_statistics_match_rows_groups_duration_sources_and_unknowns(grouped_records_factory) -> None:
    records = grouped_records_factory(group_sizes=(2, 1, 3), speaker=None)
    records[0]["source_dataset"] = "Arabic-English"
    records[1]["source_dataset"] = "Arabic-English"
    records[0]["speaker_id"] = None
    partitions = {
        "train": records[:2],
        "validation": records[2:3],
        "test": records[3:],
    }
    duplicate_report = {
        "duplicate_hash_groups": 1,
        "cross_dataset_duplicate_pairs": 2,
        "transcript_mismatches_unadjudicated": 1,
    }

    stats = DatasetReleasePipeline._build_statistics(
        partitions,
        corrected_rows=[{"release_eligible": False, "release_exclusion_reasons": "missing_text"}],
        input_summary={"Arabic-English": {"input_rows": 2}},
        duplicate_report=duplicate_report,
        ingestion_exclusions=[],
        integrity={"speaker_independent": False},
    )

    assert stats["total_utterances"] == 6
    assert stats["splits"]["train"]["utterances"] == 2
    assert stats["splits"]["train"]["duration"]["total_seconds"] == 3.0
    assert stats["duration"]["total_seconds"] == 10.0
    assert stats["source_dataset_distribution"]["Arabic-English"] == 6
    assert stats["unknown_speaker_rows"] == 6
    assert stats["exact_audio_duplicate_groups"] == 1
    assert stats["cross_dataset_duplicate_pairs"] == 2
    assert stats["transcript_mismatches_unadjudicated"] == 1
    assert stats["exclusions_by_reason"] == {"missing_text": 1}


def test_empty_partitions_report_zero_duration_without_inventing_values() -> None:
    stats = DatasetReleasePipeline._build_statistics(
        {"train": [], "validation": [], "test": []},
        corrected_rows=[],
        input_summary={},
        duplicate_report={
            "duplicate_hash_groups": 0,
            "cross_dataset_duplicate_pairs": 0,
            "transcript_mismatches_unadjudicated": 0,
        },
        ingestion_exclusions=[],
        integrity={},
    )

    assert stats["total_utterances"] == 0
    assert stats["duration"]["hours"] == 0.0
    assert "min" not in stats["duration"]
    assert stats["speaker_statistics"] == {}
