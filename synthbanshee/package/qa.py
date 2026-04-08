"""Automated QA suite for the AVDP dataset (milestone 1.5).

Runs validate_clip() on every WAV in a data directory and accumulates
aggregate statistics. Produces a QAReport that AI teams can inspect before
loading the dataset for training.

Checks performed per clip:
  1. All three required files present (.wav, .txt, .json)
  2. WAV: 16 kHz, mono, 16-bit PCM, −1.0 dBFS peak, ≥ 0.5 s silence padding
  3. Metadata JSON parses as ClipMetadata with is_synthetic=True
  4. Filename stem is ASCII-only lowercase

Dataset-level checks (via report.passed):
  - Failure rate ≤ max_failure_rate (default 2 %)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pydantic

from synthbanshee.labels.schema import ClipMetadata
from synthbanshee.package.validator import validate_clip


@dataclass
class DatasetStats:
    """Aggregate statistics computed over a dataset directory."""

    total_clips: int = 0
    total_duration_seconds: float = 0.0
    clips_by_typology: dict[str, int] = field(default_factory=dict)
    clips_by_split: dict[str, int] = field(default_factory=dict)
    clips_by_tier: dict[str, int] = field(default_factory=dict)
    intensity_distribution: dict[int, int] = field(default_factory=dict)
    speaker_count: int = 0
    failed_clips: int = 0
    quality_flagged_clips: int = 0


@dataclass
class QAReport:
    """Results of an automated QA pass on a dataset directory."""

    data_dir: str
    stats: DatasetStats = field(default_factory=DatasetStats)
    failed_clip_ids: list[str] = field(default_factory=list)
    quality_flagged: dict[str, list[str]] = field(default_factory=dict)
    passed: bool = False
    failure_rate: float = 0.0


def run_qa(
    data_dir: Path,
    *,
    splits: dict[str, str] | None = None,
    max_failure_rate: float = 0.02,
) -> QAReport:
    """Run automated QA checks on all clips in data_dir.

    Args:
        data_dir: Root directory to scan. Expected layout:
            ``data_dir/{speaker_id}/{clip_id}.{wav,txt,json}``
        splits: Optional mapping of clip_id → split name for per-split stats.
            Clips not in this dict appear under the "unassigned" key in
            ``stats.clips_by_split``.
        max_failure_rate: Fraction of clips that may fail before
            ``report.passed`` is set to False (default: 0.02 = 2 %).

    Returns:
        QAReport with per-clip failure list and aggregate statistics.
        An empty data_dir passes automatically with zero clips.
    """
    report = QAReport(data_dir=str(data_dir))
    stats = DatasetStats()

    _by_typology: dict[str, int] = defaultdict(int)
    _by_split: dict[str, int] = defaultdict(int)
    _by_tier: dict[str, int] = defaultdict(int)
    _intensity: dict[int, int] = defaultdict(int)
    _speakers: set[str] = set()

    wav_paths = [p for p in sorted(data_dir.rglob("*.wav")) if "_dirty" not in p.stem]

    if not wav_paths:
        report.stats = stats
        report.passed = True
        return report

    for wav_path in wav_paths:
        validation = validate_clip(wav_path)
        if not validation.is_valid:
            stats.failed_clips += 1
            report.failed_clip_ids.append(wav_path.stem)
            continue

        json_path = wav_path.with_suffix(".json")
        try:
            metadata = ClipMetadata.model_validate_json(json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, pydantic.ValidationError):
            stats.failed_clips += 1
            report.failed_clip_ids.append(wav_path.stem)
            continue

        stats.total_clips += 1
        stats.total_duration_seconds += metadata.duration_seconds
        _by_typology[metadata.violence_typology] += 1
        _by_tier[metadata.tier] += 1
        _by_split[(splits or {}).get(metadata.clip_id, "unassigned")] += 1
        _intensity[metadata.weak_label.max_intensity] = (
            _intensity.get(metadata.weak_label.max_intensity, 0) + 1
        )
        for spk in metadata.speakers:
            _speakers.add(spk.speaker_id)

        if metadata.quality_flags:
            stats.quality_flagged_clips += 1
            report.quality_flagged[metadata.clip_id] = list(metadata.quality_flags)

    stats.clips_by_typology = dict(_by_typology)
    stats.clips_by_split = dict(_by_split)
    stats.clips_by_tier = dict(_by_tier)
    stats.intensity_distribution = dict(_intensity)
    stats.speaker_count = len(_speakers)

    total_attempted = stats.total_clips + stats.failed_clips
    report.failure_rate = stats.failed_clips / total_attempted if total_attempted else 0.0
    report.passed = report.failure_rate <= max_failure_rate
    report.stats = stats
    return report
