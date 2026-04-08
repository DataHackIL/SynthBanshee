"""Manifest CSV generation for the AVDP dataset (milestone 1.4).

Scans a data directory for clip JSON metadata files, parses them as
ClipMetadata, and writes a flat manifest CSV with one row per clip.
The manifest is the primary delivery artefact for AI teams.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from synthbanshee.labels.schema import ClipMetadata

_MANIFEST_COLUMNS = [
    "clip_id",
    "project",
    "violence_typology",
    "tier",
    "duration_seconds",
    "speaker_ids",
    "has_violence",
    "max_intensity",
    "quality_flags",
    "split",
    "wav_path",
]


@dataclass
class ManifestRow:
    """One row in the manifest CSV."""

    clip_id: str
    project: str
    violence_typology: str
    tier: str
    duration_seconds: float
    speaker_ids: str  # pipe-separated: "AGG_M_30-45_001|VIC_F_25-40_002"
    has_violence: bool
    max_intensity: int
    quality_flags: str  # comma-separated, empty string if none
    split: str  # "train" | "val" | "test" | "" (empty if unassigned)
    wav_path: str


def generate_manifest(
    data_dir: Path,
    output_path: Path,
    *,
    splits: dict[str, str] | None = None,
    clip_ids: set[str] | None = None,
) -> list[ManifestRow]:
    """Scan data_dir recursively for clip JSON files and write a manifest CSV.

    Files whose stems contain "_dirty" are skipped (they are pre-processing
    originals, not valid clips). JSON files that fail to parse as ClipMetadata
    are silently skipped (they may be non-clip JSON).

    Args:
        data_dir: Root directory to scan (e.g. ``data/he``).
        output_path: Destination path for the manifest CSV.
        splits: Optional mapping of clip_id → split name ("train"/"val"/"test").
            Clips not present in this dict receive an empty split column.
        clip_ids: Optional allow-list of clip IDs. When provided, only clips
            whose clip_id is in this set are included in the manifest. Use this
            to restrict the manifest to a specific generation run when
            ``data_dir`` may contain clips from previous runs.

    Returns:
        List of ManifestRow objects that were written to output_path.
        Rows are ordered by clip_id (lexicographic).
    """
    rows: list[ManifestRow] = []

    for json_path in sorted(data_dir.rglob("*.json")):
        if "_dirty" in json_path.stem:
            continue
        try:
            metadata = ClipMetadata.model_validate_json(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if clip_ids is not None and metadata.clip_id not in clip_ids:
            continue

        rows.append(
            ManifestRow(
                clip_id=metadata.clip_id,
                project=metadata.project,
                violence_typology=metadata.violence_typology,
                tier=metadata.tier,
                duration_seconds=metadata.duration_seconds,
                speaker_ids="|".join(s.speaker_id for s in metadata.speakers),
                has_violence=metadata.weak_label.has_violence,
                max_intensity=metadata.weak_label.max_intensity,
                quality_flags=",".join(metadata.quality_flags),
                split=(splits or {}).get(metadata.clip_id, ""),
                wav_path=str(json_path.with_suffix(".wav")),
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "clip_id": row.clip_id,
                    "project": row.project,
                    "violence_typology": row.violence_typology,
                    "tier": row.tier,
                    "duration_seconds": row.duration_seconds,
                    "speaker_ids": row.speaker_ids,
                    "has_violence": row.has_violence,
                    "max_intensity": row.max_intensity,
                    "quality_flags": row.quality_flags,
                    "split": row.split,
                    "wav_path": row.wav_path,
                }
            )

    return rows
