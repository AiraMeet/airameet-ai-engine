"""Combine corrected release metadata without dropping columns or rows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FILES = {
    "Erzen": "Erzen_corrected_metadata.csv",
    "Arabic-English": "Arabic_English_corrected_metadata.csv",
}


def merge_metadata(release: Path, stage: Path, audio_map: dict[str, str]) -> pd.DataFrame:
    frames = []
    for dataset, filename in FILES.items():
        path = release / "metadata" / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing metadata: {path}")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "source_dataset" not in frame:
            frame["source_dataset"] = dataset
        frame["source_dataset"] = frame["source_dataset"].replace("", dataset)
        if "audio_path" in frame:
            eligible = frame.get("release_eligible", pd.Series("True", index=frame.index)).str.lower().eq("true")
            missing = eligible & ~frame["audio_path"].isin(audio_map)
            if missing.any():
                raise ValueError(f"Eligible metadata references audio absent from manifests: {frame.loc[missing, 'audio_path'].iloc[0]}")
            frame["audio_path"] = frame["audio_path"].map(lambda value: audio_map.get(value, "") if value else "")
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True, sort=False).fillna("")
    leading = [key for key in ("utterance_id", "recording_id", "session_id", "speaker_id", "source_dataset", "audio_path", "text", "start", "end", "duration", "split") if key in merged]
    merged = merged[leading + [column for column in merged if column not in leading]]
    output = stage / "metadata" / "merged_metadata.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False, lineterminator="\n")
    return merged
