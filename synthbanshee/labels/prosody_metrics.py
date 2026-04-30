"""Per-turn F0 and RMS measurement for M2a prosody validation.

Reads generated clips (WAV + strong-label JSONL) and returns per-turn
prosody metrics grouped by speaker role and intensity level.  Used by
the ``measure-prosody`` CLI command to validate that the §4.2a speaker
config targets are met before merging prosody-config PRs.

Thresholds (from docs/audio_generation_v3_design.md §4.2a):
  - VIC I1 median F0 ≤ 200 Hz  (adult female baseline; avoids child-voice range)
  - VIC I4 median F0 < 250 Hz  (pitch cap — distress via rate/timing, not pitch)
  - VIC I5 median F0 < 250 Hz  (hard cap)
  - AGG I5 RMS − AGG I1 RMS ≥ 8 dB  (loudness escalation check)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from synthbanshee.labels.schema import EventLabel

import numpy as np
import soundfile as sf


class SegmentMeasurement(NamedTuple):
    """Acoustic measurements returned by :func:`_measure_segment`."""

    f0_median_hz: float | None
    f0_std_hz: float | None
    rms_db: float
    lufs_db: float | None


# ---------------------------------------------------------------------------
# Threshold constants
# ---------------------------------------------------------------------------

VIC_I1_F0_MAX_HZ: float = 200.0
VIC_I4_F0_MAX_HZ: float = 250.0
VIC_I5_F0_MAX_HZ: float = 250.0
AGG_ESCALATION_MIN_DB: float = 8.0


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class TurnMetrics:
    """Prosody measurements for one EventLabel segment.

    Attributes:
        clip_id: Source clip identifier.
        speaker_role: Role of the speaker (e.g. ``"AGG"``, ``"VIC"``).
        intensity: Turn intensity level 1–5.
        f0_median_hz: Median voiced F0 in Hz, or ``None`` if the segment is
            too short, fully unvoiced, or librosa is unavailable.
        f0_std_hz: Standard deviation of voiced F0 frames, or ``None``.
        rms_db: RMS level in dBFS (always present).
        lufs_db: Integrated LUFS (ITU-R BS.1770-4) or ``None`` if the segment
            is too short or pyloudnorm is unavailable.
    """

    clip_id: str
    speaker_role: str
    intensity: int
    f0_median_hz: float | None
    f0_std_hz: float | None
    rms_db: float
    lufs_db: float | None = None


@dataclass
class RoleIntensityStats:
    """Aggregated statistics for one (role, intensity) bucket.

    Attributes:
        role: Speaker role string.
        intensity: Intensity level 1–5.
        n_turns: Number of turns in this bucket.
        f0_median_hz: Median of per-turn F0 medians, or ``None`` if no
            voiced turns were measured.
        f0_std_hz_mean: Mean of per-turn F0 standard deviations, or ``None``.
            Note: this is the average of per-turn stddevs, not a true pooled
            standard deviation (which would require frame-level counts).
        rms_db_mean: Mean RMS across turns in dBFS.
    """

    role: str
    intensity: int
    n_turns: int
    f0_median_hz: float | None
    f0_std_hz_mean: float | None
    rms_db_mean: float
    lufs_db_mean: float | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_lufs_meters: dict[int, Any] = {}


def _get_lufs_meter(sr: int) -> Any:
    """Return a cached ``pyloudnorm.Meter`` for the given sample rate."""
    import pyloudnorm

    if sr not in _lufs_meters:
        _lufs_meters[sr] = pyloudnorm.Meter(sr)
    return _lufs_meters[sr]


def _measure_segment(
    samples: np.ndarray,
    sr: int,
    onset_s: float,
    offset_s: float,
) -> SegmentMeasurement:
    """Return acoustic measurements for one audio segment.

    Args:
        samples: Mono int16 or float32 audio array.
        sr: Sample rate in Hz.
        onset_s: Segment start time in seconds.
        offset_s: Segment end time in seconds.

    Returns:
        :class:`SegmentMeasurement` with F0, RMS, and LUFS fields.
    """
    start = max(0, int(onset_s * sr))
    end = min(len(samples), int(offset_s * sr))
    seg = samples[start:end].astype(np.float64)

    if len(seg) < int(sr * 0.05):  # < 50 ms — too short to analyse
        return SegmentMeasurement(None, None, -96.0, None)

    # RMS in dBFS
    rms = float(np.sqrt(np.mean(seg**2)))
    # Normalize to [-1, 1] range for dBFS if samples are int16
    if samples.dtype == np.int16:
        rms /= 32768.0
    rms_db = 20.0 * np.log10(max(rms, 1e-9))

    # LUFS via pyloudnorm — graceful fallback if unavailable
    lufs_db: float | None = None
    try:
        seg_float = seg.astype(np.float64) / 32768.0 if samples.dtype == np.int16 else seg
        meter = _get_lufs_meter(sr)
        lufs_db = float(meter.integrated_loudness(seg_float))
        if not np.isfinite(lufs_db):
            lufs_db = None
    except (ImportError, ValueError):
        pass

    # F0 via librosa pyin — graceful fallback if unavailable
    try:
        import librosa

        f0, voiced_flag, _ = librosa.pyin(
            seg.astype(np.float32) / 32768.0
            if samples.dtype == np.int16
            else seg.astype(np.float32),
            fmin=float(librosa.note_to_hz("C2")),
            fmax=float(librosa.note_to_hz("C7")),
            sr=sr,
        )
        voiced_f0 = f0[voiced_flag & np.isfinite(f0)]
        if len(voiced_f0) == 0:
            return SegmentMeasurement(None, None, rms_db, lufs_db)
        return SegmentMeasurement(
            float(np.median(voiced_f0)), float(np.std(voiced_f0)), rms_db, lufs_db
        )
    except ImportError:
        return SegmentMeasurement(None, None, rms_db, lufs_db)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def measure_events(
    samples: np.ndarray,
    sr: int,
    role_events: list[EventLabel],
) -> list[TurnMetrics]:
    """Return per-turn prosody metrics for pre-parsed events.

    This is the lower-level entry point used when the caller has already
    parsed the JSONL and loaded the WAV (e.g. ``run_qa`` in ``qa.py``
    which needs the parsed events for M10b overlap/emotion checks).

    Args:
        samples: Mono audio array (int16 or float32).
        sr: Sample rate in Hz.
        role_events: List of ``EventLabel`` objects that have a non-None
            ``speaker_role``.

    Returns:
        List of :class:`TurnMetrics`, one per event.
    """
    results: list[TurnMetrics] = []
    for ev in role_events:
        m = _measure_segment(samples, sr, ev.onset, ev.offset)
        assert ev.speaker_role is not None
        results.append(
            TurnMetrics(
                clip_id=ev.clip_id,
                speaker_role=ev.speaker_role,
                intensity=ev.intensity,
                f0_median_hz=m.f0_median_hz,
                f0_std_hz=m.f0_std_hz,
                rms_db=m.rms_db,
                lufs_db=m.lufs_db,
            )
        )
    return results


def parse_jsonl_events(jsonl_path: Path) -> list[EventLabel]:
    """Parse a JSONL strong-label file into a list of EventLabel objects.

    Malformed lines are skipped with a warning.  Blank lines are ignored.

    Args:
        jsonl_path: Path to the ``.jsonl`` file.

    Returns:
        List of :class:`EventLabel` objects (may be empty).
    """
    from synthbanshee.labels.schema import EventLabel

    events: list[EventLabel] = []
    for raw_line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            events.append(EventLabel.model_validate_json(line))
        except Exception as exc:  # pydantic.ValidationError or JSON parse error
            warnings.warn(
                f"Skipping malformed label line in {jsonl_path.name}: {exc}",
                stacklevel=2,
            )
    return events


def measure_clip(clip_path: Path) -> list[TurnMetrics]:
    """Return per-turn prosody metrics for one generated clip.

    Reads ``{clip_path.stem}.jsonl`` for event boundaries, speaker roles,
    and intensity levels.  Events without a ``speaker_role`` (e.g. ambient
    noise events) are skipped.

    Args:
        clip_path: Path to the ``.wav`` file.  The matching ``.jsonl``
            strong-label file must exist alongside it.

    Returns:
        List of :class:`TurnMetrics`, one per qualifying event.  Empty if
        the JSONL is missing or contains no speaker-role events.
    """
    jsonl_path = clip_path.with_suffix(".jsonl")
    if not jsonl_path.exists():
        return []

    events = parse_jsonl_events(jsonl_path)
    role_events = [e for e in events if e.speaker_role is not None]
    if not role_events:
        return []

    samples, sr = sf.read(str(clip_path), dtype="int16")
    if samples.ndim > 1:
        samples = samples[:, 0]

    return measure_events(samples, sr, role_events)


def aggregate_metrics(turns: list[TurnMetrics]) -> list[RoleIntensityStats]:
    """Aggregate per-turn metrics into (role, intensity) buckets.

    Args:
        turns: Flat list of :class:`TurnMetrics` from one or more clips.

    Returns:
        List of :class:`RoleIntensityStats` sorted by role then intensity.
    """
    from collections import defaultdict

    buckets: dict[tuple[str, int], list[TurnMetrics]] = defaultdict(list)
    for t in turns:
        buckets[(t.speaker_role, t.intensity)].append(t)

    stats: list[RoleIntensityStats] = []
    for (role, intensity), bucket in sorted(buckets.items()):
        voiced = [t.f0_median_hz for t in bucket if t.f0_median_hz is not None]
        stds = [t.f0_std_hz for t in bucket if t.f0_std_hz is not None]
        rms_values = [t.rms_db for t in bucket]
        lufs_values = [t.lufs_db for t in bucket if t.lufs_db is not None]
        stats.append(
            RoleIntensityStats(
                role=role,
                intensity=intensity,
                n_turns=len(bucket),
                f0_median_hz=float(np.median(voiced)) if voiced else None,
                f0_std_hz_mean=float(np.mean(stds)) if stds else None,
                rms_db_mean=float(np.mean(rms_values)),
                lufs_db_mean=float(np.mean(lufs_values)) if lufs_values else None,
            )
        )
    return stats


def run_threshold_checks(
    stats: list[RoleIntensityStats],
    include_roles: set[str] | None = None,
) -> list[tuple[str, bool, str]]:
    """Evaluate §4.2a pass/fail thresholds against aggregated stats.

    Args:
        stats: Aggregated per-role/intensity stats from :func:`aggregate_metrics`.
        include_roles: If provided, only emit checks for roles in this set.
            Pass ``None`` (default) to run all checks.

    Returns a list of ``(label, passed, detail)`` tuples — one per check.
    Checks that cannot be evaluated (missing data) are reported as
    inconclusive (``passed=False``, detail explains why).
    """
    by_key: dict[tuple[str, int], RoleIntensityStats] = {(s.role, s.intensity): s for s in stats}

    checks: list[tuple[str, bool, str]] = []

    def _f0_check(
        role: str, intensity: int, max_hz: float, label: str, *, strict: bool = False
    ) -> None:
        s = by_key.get((role, intensity))
        if s is None or s.f0_median_hz is None:
            checks.append((label, False, "no data"))
            return
        passed = s.f0_median_hz < max_hz if strict else s.f0_median_hz <= max_hz
        checks.append((label, passed, f"{s.f0_median_hz:.1f} Hz"))

    if include_roles is None or "VIC" in include_roles:
        _f0_check("VIC", 1, VIC_I1_F0_MAX_HZ, f"VIC I1 median F0 ≤ {VIC_I1_F0_MAX_HZ:.0f} Hz")
        _f0_check(
            "VIC",
            4,
            VIC_I4_F0_MAX_HZ,
            f"VIC I4 median F0 < {VIC_I4_F0_MAX_HZ:.0f} Hz",
            strict=True,
        )
        _f0_check(
            "VIC",
            5,
            VIC_I5_F0_MAX_HZ,
            f"VIC I5 median F0 < {VIC_I5_F0_MAX_HZ:.0f} Hz",
            strict=True,
        )

    if include_roles is None or "AGG" in include_roles:
        agg_i1 = by_key.get(("AGG", 1))
        agg_i5 = by_key.get(("AGG", 5))
        label = f"AGG I5 − I1 RMS ≥ {AGG_ESCALATION_MIN_DB:.0f} dB"
        if agg_i1 is None or agg_i5 is None:
            checks.append((label, False, "no data"))
        else:
            delta = agg_i5.rms_db_mean - agg_i1.rms_db_mean
            checks.append((label, delta >= AGG_ESCALATION_MIN_DB, f"{delta:+.1f} dB"))

    return checks
