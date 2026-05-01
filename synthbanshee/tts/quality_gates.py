"""M15 — Turn-level quality gates for post-TTS render validation.

Quality gates reject or flag individual turn renders that exhibit unrealistic
acoustic properties.  Gates run after TTS synthesis but before mixing, so
bad turns can be re-rendered or excluded early.

Research reference: wiki/topics/research-synthesis.md §Quality Gates (lines 159-171).

Three gates implemented:
1. Sustained-vowel detection — reject turns with >2.8 s single sustained voiced segment.
2. F0 guardrails — reject if median F0 outside gender-specific range.
3. Click detection — flag DC-offset jumps between SSML blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Thresholds (from research-synthesis.md §Quality Gates)
# ---------------------------------------------------------------------------

# Maximum sustained voiced segment duration (seconds).
MAX_SUSTAINED_VOICED_S: float = 2.8

# F0 guardrail ranges (Hz) — reject if median F0 outside.
MALE_F0_MIN_HZ: float = 80.0
MALE_F0_MAX_HZ: float = 180.0
FEMALE_F0_MIN_HZ: float = 150.0
FEMALE_F0_MAX_HZ: float = 290.0

# Click detection: minimum absolute sample-to-sample jump (normalized to [-1,1])
# that qualifies as a DC-offset click.  Empirically: 0.05 of full-scale.
CLICK_THRESHOLD: float = 0.05

# Minimum number of click events to flag a turn.
CLICK_COUNT_THRESHOLD: int = 3


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    """Result of a quality gate check on a single turn render.

    Attributes:
        passed: True if the turn passes all gates.
        gate_name: Name of the first gate that failed (None if passed).
        detail: Human-readable detail about the failure.
    """

    passed: bool
    gate_name: str | None = None
    detail: str | None = None


# ---------------------------------------------------------------------------
# Gate implementations
# ---------------------------------------------------------------------------


def _wav_bytes_to_samples(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Parse raw WAV bytes into float32 samples and sample rate.

    Handles 16-bit PCM WAV (the only format in this pipeline).
    Returns samples normalized to [-1.0, 1.0].
    """
    import io

    import soundfile as sf

    with io.BytesIO(wav_bytes) as buf:
        samples, sr = sf.read(buf, dtype="float32")
    # Ensure mono
    if samples.ndim > 1:
        samples = samples[:, 0]
    return samples, int(sr)


def check_sustained_vowel(samples: np.ndarray, sr: int) -> GateResult:
    """Detect sustained voiced segments longer than MAX_SUSTAINED_VOICED_S.

    Uses energy-based voice activity detection: a frame is "voiced" if its
    RMS exceeds -40 dBFS.  Consecutive voiced frames form a segment.

    Args:
        samples: Float32 audio samples, mono, normalized to [-1, 1].
        sr: Sample rate (Hz).

    Returns:
        GateResult indicating pass/fail.
    """
    frame_len = int(0.020 * sr)  # 20ms frames
    hop = int(0.010 * sr)  # 10ms hop
    threshold_linear = 10 ** (-40 / 20)  # -40 dBFS

    n_frames = max(0, (len(samples) - frame_len) // hop + 1)
    if n_frames == 0:
        return GateResult(passed=True)

    max_voiced_frames = 0
    current_run = 0

    for i in range(n_frames):
        start = i * hop
        frame = samples[start : start + frame_len]
        rms = np.sqrt(np.mean(frame**2))
        if rms > threshold_linear:
            current_run += 1
        else:
            max_voiced_frames = max(max_voiced_frames, current_run)
            current_run = 0
    max_voiced_frames = max(max_voiced_frames, current_run)

    max_duration_s = max_voiced_frames * (hop / sr)
    if max_duration_s > MAX_SUSTAINED_VOICED_S:
        return GateResult(
            passed=False,
            gate_name="sustained_vowel",
            detail=f"Sustained voiced segment {max_duration_s:.2f}s > {MAX_SUSTAINED_VOICED_S}s",
        )
    return GateResult(passed=True)


def check_f0_guardrails(
    samples: np.ndarray,
    sr: int,
    gender: Literal["male", "female"],
) -> GateResult:
    """Check that median F0 falls within gender-appropriate range.

    Uses autocorrelation-based pitch estimation on voiced frames.

    Args:
        samples: Float32 audio samples, mono.
        sr: Sample rate (Hz).
        gender: Speaker gender for range selection.

    Returns:
        GateResult indicating pass/fail.
    """
    f0_min = MALE_F0_MIN_HZ if gender == "male" else FEMALE_F0_MIN_HZ
    f0_max = MALE_F0_MAX_HZ if gender == "male" else FEMALE_F0_MAX_HZ

    # Estimate F0 using autocorrelation on 30ms frames
    frame_len = int(0.030 * sr)
    hop = int(0.010 * sr)
    n_frames = max(0, (len(samples) - frame_len) // hop + 1)

    if n_frames == 0:
        return GateResult(passed=True)

    f0_estimates: list[float] = []
    # Search range in lag samples
    min_lag = int(sr / 500)  # 500 Hz max
    max_lag = int(sr / 50)  # 50 Hz min

    for i in range(n_frames):
        start = i * hop
        frame = samples[start : start + frame_len]
        # Skip silent frames
        if np.sqrt(np.mean(frame**2)) < 10 ** (-40 / 20):
            continue
        # Autocorrelation
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2 :]  # positive lags only
        if max_lag >= len(corr):
            continue
        # Find peak in valid range
        search = corr[min_lag : max_lag + 1]
        if len(search) == 0:
            continue
        peak_idx = int(np.argmax(search)) + min_lag
        # Voicing confidence: ratio of peak to zero-lag
        if corr[0] > 0 and corr[peak_idx] / corr[0] > 0.3:
            f0_estimates.append(sr / peak_idx)

    if len(f0_estimates) < 5:
        # Too few voiced frames to make a judgment — pass by default
        return GateResult(passed=True)

    median_f0 = float(np.median(f0_estimates))

    if median_f0 < f0_min or median_f0 > f0_max:
        return GateResult(
            passed=False,
            gate_name="f0_guardrails",
            detail=(f"Median F0 {median_f0:.1f} Hz outside {gender} range [{f0_min}, {f0_max}] Hz"),
        )
    return GateResult(passed=True)


def check_clicks(samples: np.ndarray, sr: int) -> GateResult:
    """Detect DC-offset jump clicks in the audio.

    Clicks manifest as sudden large sample-to-sample amplitude changes,
    typically at SSML block boundaries.

    Args:
        samples: Float32 audio samples, mono, normalized to [-1, 1].
        sr: Sample rate (Hz).

    Returns:
        GateResult indicating pass/fail.
    """
    if len(samples) < 2:
        return GateResult(passed=True)

    # Compute absolute sample-to-sample differences
    diffs = np.abs(np.diff(samples))
    click_count = int(np.sum(diffs > CLICK_THRESHOLD))

    if click_count >= CLICK_COUNT_THRESHOLD:
        return GateResult(
            passed=False,
            gate_name="click_detection",
            detail=f"Detected {click_count} click events (threshold: {CLICK_COUNT_THRESHOLD})",
        )
    return GateResult(passed=True)


# ---------------------------------------------------------------------------
# Composite gate runner
# ---------------------------------------------------------------------------


def run_quality_gates(
    wav_bytes: bytes,
    gender: Literal["male", "female"],
) -> GateResult:
    """Run all turn-level quality gates on rendered WAV bytes.

    Gates are evaluated in order; first failure is returned.
    If all pass, returns a passing GateResult.

    Args:
        wav_bytes: Raw WAV file bytes (16 kHz, mono, PCM16 or float32).
        gender: Speaker gender for F0 range selection.

    Returns:
        GateResult from the first failing gate, or a pass result.
    """
    samples, sr = _wav_bytes_to_samples(wav_bytes)

    # Gate 1: sustained vowel
    result = check_sustained_vowel(samples, sr)
    if not result.passed:
        return result

    # Gate 2: F0 guardrails
    result = check_f0_guardrails(samples, sr, gender)
    if not result.passed:
        return result

    # Gate 3: click detection
    result = check_clicks(samples, sr)
    if not result.passed:
        return result

    return GateResult(passed=True)
