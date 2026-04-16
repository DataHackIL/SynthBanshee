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

import numpy as np
import soundfile as sf

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
    """

    clip_id: str
    speaker_role: str
    intensity: int
    f0_median_hz: float | None
    f0_std_hz: float | None
    rms_db: float


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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _measure_segment(
    samples: np.ndarray,
    sr: int,
    onset_s: float,
    offset_s: float,
) -> tuple[float | None, float | None, float]:
    """Return ``(f0_median_hz, f0_std_hz, rms_db)`` for one audio segment.

    Args:
        samples: Mono int16 or float32 audio array.
        sr: Sample rate in Hz.
        onset_s: Segment start time in seconds.
        offset_s: Segment end time in seconds.

    Returns:
        Tuple of F0 median (Hz or None), F0 std (Hz or None), RMS (dBFS).
    """
    start = max(0, int(onset_s * sr))
    end = min(len(samples), int(offset_s * sr))
    seg = samples[start:end].astype(np.float64)

    if len(seg) < int(sr * 0.05):  # < 50 ms — too short to analyse
        return None, None, -96.0

    # RMS in dBFS
    rms = float(np.sqrt(np.mean(seg**2)))
    # Normalize to [-1, 1] range for dBFS if samples are int16
    if samples.dtype == np.int16:
        rms /= 32768.0
    rms_db = 20.0 * np.log10(max(rms, 1e-9))

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
            return None, None, rms_db
        return float(np.median(voiced_f0)), float(np.std(voiced_f0)), rms_db
    except ImportError:
        return None, None, rms_db


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
    from synthbanshee.labels.schema import EventLabel

    jsonl_path = clip_path.with_suffix(".jsonl")
    if not jsonl_path.exists():
        return []

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

    role_events = [e for e in events if e.speaker_role is not None]
    if not role_events:
        return []

    samples, sr = sf.read(str(clip_path), dtype="int16")
    if samples.ndim > 1:
        samples = samples[:, 0]

    results: list[TurnMetrics] = []
    for ev in role_events:
        f0_med, f0_std, rms_db = _measure_segment(samples, sr, ev.onset, ev.offset)
        assert ev.speaker_role is not None  # narrowed above
        results.append(
            TurnMetrics(
                clip_id=ev.clip_id,
                speaker_role=ev.speaker_role,
                intensity=ev.intensity,
                f0_median_hz=f0_med,
                f0_std_hz=f0_std,
                rms_db=rms_db,
            )
        )
    return results


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
        stats.append(
            RoleIntensityStats(
                role=role,
                intensity=intensity,
                n_turns=len(bucket),
                f0_median_hz=float(np.median(voiced)) if voiced else None,
                f0_std_hz_mean=float(np.mean(stds)) if stds else None,
                rms_db_mean=float(np.mean(rms_values)),
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
