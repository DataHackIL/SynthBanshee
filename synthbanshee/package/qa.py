"""Automated QA suite for the AVDP dataset (milestone 1.5 + M10a + M10b).

Runs validate_clip() on every WAV in a data directory and accumulates
aggregate statistics. Produces a QAReport that AI teams can inspect before
loading the dataset for training.

Checks performed per clip (Stage 5 — Validation):
  1. All three required files present (.wav, .txt, .json)
  2. WAV: 16 kHz, mono, 16-bit PCM, −1.0 dBFS peak, ≥ 0.5 s silence padding
  3. Metadata JSON parses as ClipMetadata with is_synthetic=True
  4. Filename stem is ASCII-only lowercase
  5. Strong labels JSONL present alongside the clip (warning — not a hard error)

M10a per-clip acoustic checks (when JSONL is present):
  - vic_f0_high: any VIC turn at I4–I5 with F0 > 250 Hz
  - agg_no_escalation: AGG RMS range (I5 − I1) < 6 dB

M10b per-clip checks (when JSONL is present, always active):
  - warn_no_overlap: no temporal overlap between any two events in a clip
    that contains I4+ turns
  - warn_emotion_downgrade: count of neutral emotional_state events exceeds
    count of low-intensity (I1–I2) events

M10b run-level aggregation (RunSummary, enabled via run_summary=True):
  - F0/RMS/LUFS distributions by role and intensity
  - Voice and backend diversity counts
  - Overlap and emotion-downgrade ratios
  - Outlier clip detection (> 2σ from run mean)
  - Run-level warning flags

Note: The design doc also specifies distributions by typology/project and
mix_mode distribution.  mix_mode is not persisted in EventLabel/ClipMetadata
today; typology/project breakdowns are deferred to a follow-up.

Dataset-level checks (via report.passed):
  - Failure rate ≤ max_failure_rate (default 2 %)
  - clips_missing_strong_labels is reported for observability but does not
    affect report.passed (missing JSONL is a warning, not a failure)
"""

from __future__ import annotations

import logging
import warnings as _warnings_mod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydantic
import soundfile as sf

from synthbanshee.labels.prosody_metrics import (
    RoleIntensityStats,
    TurnMetrics,
    aggregate_metrics,
    measure_events,
)
from synthbanshee.labels.schema import ClipMetadata, EventLabel
from synthbanshee.package.validator import validate_clip

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# M10a constants
# ---------------------------------------------------------------------------

# QA warning thresholds — intentionally looser than the hard pass/fail
# thresholds in prosody_metrics.py (AGG_ESCALATION_MIN_DB = 8 dB).  The
# 6 dB threshold here is a *warning* that flags clips for manual review;
# the 8 dB threshold is a hard gate for merging prosody-config PRs.
_VIC_F0_HIGH_HZ: float = 250.0
_AGG_ESCALATION_MIN_DB: float = 6.0

# ---------------------------------------------------------------------------
# M10b constants
# ---------------------------------------------------------------------------

_OUTLIER_SIGMA: float = 2.0
_HIGH_WARNING_RATE: float = 0.10
_HIGH_EMOTION_DOWNGRADE_RATE: float = 0.05
_MIN_VOICE_DIVERSITY: int = 3


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


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
    clips_missing_strong_labels: int = 0  # Stage 4b JSONL absent (warning only)
    # M10a acoustic metric counters
    acoustic_warnings: dict[str, int] = field(default_factory=dict)
    clips_with_acoustic_warnings: int = 0
    # M10b structural warning counters (separate from acoustic)
    structural_warnings: dict[str, int] = field(default_factory=dict)
    clips_with_structural_warnings: int = 0


@dataclass
class RunSummary:
    """M10b run-level aggregation metrics.

    Computed across all valid clips in a QA run to detect systematic
    biases that per-clip checks cannot catch.
    """

    # Acoustic distributions by (role, intensity)
    role_intensity_stats: list[RoleIntensityStats] = field(default_factory=list)

    # Voice and backend diversity
    voices_by_gender: dict[str, int] = field(default_factory=dict)
    backend_count: int = 0
    clips_by_tts_engine: dict[str, int] = field(default_factory=dict)

    # Overlap and emotion-downgrade ratios
    overlap_ratio: float = 0.0
    clips_with_i4_plus: int = 0
    emotion_downgrade_ratio: float = 0.0

    # Outlier clips (F0 or RMS > 2σ from run mean for their role/intensity)
    outlier_clip_ids: list[str] = field(default_factory=list)

    # Run-level warning strings
    run_warnings: list[str] = field(default_factory=list)


@dataclass
class QAReport:
    """Results of an automated QA pass on a dataset directory."""

    data_dir: str
    stats: DatasetStats = field(default_factory=DatasetStats)
    failed_clip_ids: list[str] = field(default_factory=list)
    quality_flagged: dict[str, list[str]] = field(default_factory=dict)
    acoustic_warnings: dict[str, list[str]] = field(default_factory=dict)
    structural_warnings: dict[str, list[str]] = field(default_factory=dict)
    passed: bool = False
    failure_rate: float = 0.0
    run_summary: RunSummary | None = None


# ---------------------------------------------------------------------------
# M10a per-clip acoustic checks
# ---------------------------------------------------------------------------


def _check_acoustic_warnings(
    turns: list[TurnMetrics],
) -> list[str]:
    """Evaluate M10a acoustic thresholds for one clip.

    Returns a list of triggered warning flag strings.
    """
    warnings: list[str] = []

    # vic_f0_high: any VIC turn at I4–I5 with F0 > 250 Hz
    vic_high_intensity = [
        t
        for t in turns
        if t.speaker_role == "VIC" and t.intensity in (4, 5) and t.f0_median_hz is not None
    ]
    if any(
        t.f0_median_hz > _VIC_F0_HIGH_HZ for t in vic_high_intensity if t.f0_median_hz is not None
    ):
        warnings.append("vic_f0_high")

    # agg_no_escalation: AGG RMS range (I5 − I1) < 6 dB
    agg_i1 = [t.rms_db for t in turns if t.speaker_role == "AGG" and t.intensity == 1]
    agg_i5 = [t.rms_db for t in turns if t.speaker_role == "AGG" and t.intensity == 5]
    if agg_i1 and agg_i5:
        delta = float(np.mean(agg_i5)) - float(np.mean(agg_i1))
        if delta < _AGG_ESCALATION_MIN_DB:
            warnings.append("agg_no_escalation")

    return warnings


# ---------------------------------------------------------------------------
# M10b per-clip checks
# ---------------------------------------------------------------------------


def _has_overlap(events: list[EventLabel]) -> bool:
    """Return True if any two events in the list overlap temporally.

    Uses sort-and-scan (O(n log n)) instead of pairwise comparison.
    Two events overlap when ``onset_i < offset_j`` and ``onset_j < offset_i``.
    """
    if len(events) < 2:
        return False
    sorted_events = sorted(events, key=lambda e: e.onset)
    max_offset = sorted_events[0].offset
    for e in sorted_events[1:]:
        if e.onset < max_offset:
            return True
        max_offset = max(max_offset, e.offset)
    return False


def _check_emotion_downgrade(events: list[EventLabel]) -> bool:
    """Return True if neutral emotional states outnumber low-intensity turns.

    A clip is flagged when the count of events with ``emotional_state ==
    "neutral"`` exceeds the count of events with intensity ≤ 2.  This
    indicates the LLM is defaulting to neutral for turns that should
    carry emotional weight.

    Expects ``role_events`` (events with a non-None ``speaker_role``);
    ambient events (``emotional_state=None``) are safe to include but
    will not count as neutral.
    """
    neutral_count = sum(1 for e in events if e.emotional_state == "neutral")
    low_intensity_count = sum(1 for e in events if e.intensity <= 2)
    return neutral_count > low_intensity_count


def _parse_jsonl_events(jsonl_path: Path) -> list[EventLabel]:
    """Parse a JSONL file into a list of EventLabel objects.

    Malformed lines are skipped with a warning.
    """
    events: list[EventLabel] = []
    for raw_line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            events.append(EventLabel.model_validate_json(line))
        except Exception as exc:
            _warnings_mod.warn(
                f"Skipping malformed label line in {jsonl_path.name}: {exc}",
                stacklevel=2,
            )
    return events


def _detect_outliers(
    all_turns: list[TurnMetrics],
) -> list[str]:
    """Return clip IDs whose F0 or RMS is > 2σ from run mean.

    Uses mean (not median) as center for both F0 and RMS so that the
    center and spread come from the same distribution.  Only buckets
    with ≥ 2 samples are checked (σ requires at least 2 data points).
    A clip ID appears at most once in the result.
    """
    # Build per-(role, intensity) lists
    buckets: dict[tuple[str, int], list[TurnMetrics]] = defaultdict(list)
    for t in all_turns:
        buckets[(t.speaker_role, t.intensity)].append(t)

    # Pre-compute mean and σ for F0 and RMS per bucket
    f0_stats: dict[tuple[str, int], tuple[float, float]] = {}  # (mean, sigma)
    rms_stats: dict[tuple[str, int], tuple[float, float]] = {}
    for key, turns in buckets.items():
        f0_vals = [t.f0_median_hz for t in turns if t.f0_median_hz is not None]
        if len(f0_vals) >= 2:
            f0_stats[key] = (float(np.mean(f0_vals)), float(np.std(f0_vals)))
        rms_vals = [t.rms_db for t in turns]
        if len(rms_vals) >= 2:
            rms_stats[key] = (float(np.mean(rms_vals)), float(np.std(rms_vals)))

    outlier_ids: set[str] = set()
    for t in all_turns:
        key = (t.speaker_role, t.intensity)

        # F0 outlier
        if t.f0_median_hz is not None and key in f0_stats:
            mean, sigma = f0_stats[key]
            if sigma > 0 and abs(t.f0_median_hz - mean) > _OUTLIER_SIGMA * sigma:
                outlier_ids.add(t.clip_id)

        # RMS outlier
        if key in rms_stats:
            mean, sigma = rms_stats[key]
            if sigma > 0 and abs(t.rms_db - mean) > _OUTLIER_SIGMA * sigma:
                outlier_ids.add(t.clip_id)

    return sorted(outlier_ids)


def _compute_run_warnings(
    summary: RunSummary,
    total_clips: int,
    clips_with_acoustic_warnings: int,
) -> list[str]:
    """Evaluate run-level warning conditions and return triggered flags."""
    warnings: list[str] = []

    # Voice diversity (only meaningful when clips exist)
    if total_clips > 0:
        for gender in ("male", "female"):
            count = summary.voices_by_gender.get(gender, 0)
            if count < _MIN_VOICE_DIVERSITY:
                warnings.append(f"low_voice_diversity_{gender}")

    # Backend diversity
    if summary.backend_count <= 1 and total_clips > 0:
        warnings.append("single_backend")

    # Zero overlap — only warn when there are I4+ clips that *should* have overlap
    if summary.clips_with_i4_plus > 0 and summary.overlap_ratio == 0.0:
        warnings.append("zero_overlap")

    # High emotion-downgrade rate
    if summary.emotion_downgrade_ratio > _HIGH_EMOTION_DOWNGRADE_RATE:
        warnings.append("high_emotion_downgrade")

    # High acoustic-warning rate (acoustic only, not structural)
    if total_clips > 0:
        warning_rate = clips_with_acoustic_warnings / total_clips
        if warning_rate > _HIGH_WARNING_RATE:
            warnings.append("high_warning_rate")

    return warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_qa(
    data_dir: Path,
    *,
    splits: dict[str, str] | None = None,
    max_failure_rate: float = 0.02,
    run_summary: bool = False,
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
        run_summary: If True, compute M10b run-level aggregation metrics
            and attach a :class:`RunSummary` to the report.

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
    _acoustic_warnings: dict[str, int] = defaultdict(int)
    _structural_warnings: dict[str, int] = defaultdict(int)

    # M10b accumulators
    _all_turns: list[TurnMetrics] = []
    _voices_by_gender: dict[str, set[str]] = defaultdict(set)
    _tts_engines: set[str] = set()
    _clips_by_engine: dict[str, int] = defaultdict(int)
    _clips_with_overlap: int = 0
    _clips_with_emotion_downgrade: int = 0
    _clips_with_i4_plus: int = 0  # denominator for overlap ratio

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

        # Count Stage 4b JSONL warnings surfaced by validate_clip()
        if any("Strong labels JSONL missing" in w for w in validation.warnings):
            stats.clips_missing_strong_labels += 1

        # M10b: track voice and backend diversity
        if run_summary:
            _tts_engines.add(metadata.tts_engine)
            _clips_by_engine[metadata.tts_engine] += 1
            for spk in metadata.speakers:
                _voices_by_gender[spk.gender].add(spk.tts_voice_id)

        # --- JSONL-based checks (M10a acoustic + M10b structural) ---
        # Parse JSONL once, share events between acoustic measurement and
        # structural checks to avoid double-parsing.
        jsonl_path = wav_path.with_suffix(".jsonl")
        if jsonl_path.exists():
            acoustic_clip_warnings: list[str] = []
            structural_clip_warnings: list[str] = []

            try:
                events = _parse_jsonl_events(jsonl_path)
                role_events = [e for e in events if e.speaker_role is not None]
            except Exception:
                logger.warning("JSONL parse failed for %s", wav_path.stem, exc_info=True)
                events = []
                role_events = []

            # M10a: acoustic measurements using pre-parsed events
            if role_events:
                try:
                    samples, sr = sf.read(str(wav_path), dtype="int16")
                    if samples.ndim > 1:
                        samples = samples[:, 0]
                    turns = measure_events(samples, sr, role_events)
                    if turns:
                        acoustic_clip_warnings.extend(_check_acoustic_warnings(turns))
                        if run_summary:
                            _all_turns.extend(turns)
                except Exception:
                    logger.warning(
                        "Acoustic measurement failed for %s", wav_path.stem, exc_info=True
                    )

            # M10b: structural checks (always active, not gated on run_summary)
            if role_events:
                try:
                    has_i4_plus = any(e.intensity >= 4 for e in role_events)
                    has_overlap = _has_overlap(role_events) if has_i4_plus else False
                    has_emotion_dg = _check_emotion_downgrade(role_events)

                    # Per-clip structural warnings (always fire)
                    if has_i4_plus and not has_overlap:
                        structural_clip_warnings.append("warn_no_overlap")
                    if has_emotion_dg:
                        structural_clip_warnings.append("warn_emotion_downgrade")

                    # Run-level counters (only needed for RunSummary)
                    if run_summary:
                        if has_i4_plus:
                            _clips_with_i4_plus += 1
                            if has_overlap:
                                _clips_with_overlap += 1
                        if has_emotion_dg:
                            _clips_with_emotion_downgrade += 1
                except Exception:
                    logger.warning(
                        "M10b event analysis failed for %s", wav_path.stem, exc_info=True
                    )

            # Record acoustic warnings
            if acoustic_clip_warnings:
                stats.clips_with_acoustic_warnings += 1
                report.acoustic_warnings[metadata.clip_id] = acoustic_clip_warnings
                for w in acoustic_clip_warnings:
                    _acoustic_warnings[w] += 1

            # Record structural warnings (separate from acoustic)
            if structural_clip_warnings:
                stats.clips_with_structural_warnings += 1
                report.structural_warnings[metadata.clip_id] = structural_clip_warnings
                for w in structural_clip_warnings:
                    _structural_warnings[w] += 1

    stats.clips_by_typology = dict(_by_typology)
    stats.clips_by_split = dict(_by_split)
    stats.clips_by_tier = dict(_by_tier)
    stats.intensity_distribution = dict(_intensity)
    stats.speaker_count = len(_speakers)
    stats.acoustic_warnings = dict(_acoustic_warnings)
    stats.structural_warnings = dict(_structural_warnings)

    total_attempted = stats.total_clips + stats.failed_clips
    report.failure_rate = stats.failed_clips / total_attempted if total_attempted else 0.0
    report.passed = report.failure_rate <= max_failure_rate
    report.stats = stats

    # --- M10b: build run summary ---
    if run_summary and stats.total_clips > 0:
        ri_stats = aggregate_metrics(_all_turns) if _all_turns else []

        summary = RunSummary(
            role_intensity_stats=ri_stats,
            voices_by_gender={g: len(v) for g, v in _voices_by_gender.items()},
            backend_count=len(_tts_engines),
            clips_by_tts_engine=dict(_clips_by_engine),
            overlap_ratio=(
                _clips_with_overlap / _clips_with_i4_plus if _clips_with_i4_plus > 0 else 0.0
            ),
            clips_with_i4_plus=_clips_with_i4_plus,
            emotion_downgrade_ratio=_clips_with_emotion_downgrade / stats.total_clips,
            outlier_clip_ids=_detect_outliers(_all_turns),
        )
        summary.run_warnings = _compute_run_warnings(
            summary, stats.total_clips, stats.clips_with_acoustic_warnings
        )
        report.run_summary = summary

    return report
