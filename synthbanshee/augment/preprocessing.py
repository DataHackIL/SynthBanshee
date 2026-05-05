"""Preprocessing pipeline: resample → downmix → high-pass → denoise → normalize → pad → validate.

Spec reference: docs/spec.md §3.1 (M14 supersedes §3.1 filter and denoise defaults;
see wiki/topics/research-synthesis.md for rationale)

Order of operations is fixed. The "dirty" pre-pipeline file is retained in assets/
and must never be overwritten by this module.

Uses scipy + soundfile exclusively (no torchaudio dependency at import time)
to avoid torch version incompatibilities.

M14 changes (2026-05-01):
- Removed 7500 Hz lowpass filter (was destroying sibilants and breathiness cues
  at 16 kHz sample rate — Nyquist is 8 kHz, so the LPF removed real signal content).
- Added 80 Hz highpass filter for DC/rumble removal.
- Changed wiener_denoise default to False (Wiener denoising on clean TTS output
  causes muffled sound; only enable for clips with real added noise).
See wiki/topics/research-synthesis.md for full rationale and citations.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt, wiener

from synthbanshee.config.preprocessing_config import PreprocessingConfig

_TARGET_SR = 16_000
_TARGET_CHANNELS = 1
_PEAK_DBFS = -1.0  # −1.0 dBFS — safety ceiling enforced by validate_audio()
_HP_CUTOFF_HZ = 80  # high-pass cutoff for DC/rumble removal
_HP_ORDER = 2  # Butterworth order (gentle slope)
_SILENCE_PAD_S = 0.5  # minimum silence padding (seconds)
_MIN_DURATION_S = 3.0  # clips below this are invalid (spec §3)
_MAX_DURATION_S = 300.0  # clips above this must be segmented (spec §3)

# Re-export so callers can do: from synthbanshee.augment.preprocessing import PreprocessingConfig
__all__ = [
    "PreprocessingConfig",
    "PreprocessingResult",
    "peak_normalize_to_target",
    "preprocess",
    "validate_audio",
]


@dataclass
class PreprocessingResult:
    output_path: Path
    dirty_path: Path | None
    sample_rate: int
    channels: int
    duration_seconds: float
    peak_dbfs: float
    """Measured peak in the written PCM_16 file."""

    target_peak_dbfs: float | None = None
    """Target peak the loudness step aimed for (#78), in dBFS.  Pair this
    with ``peak_dbfs`` to compute deviation; QA tooling consumes this as
    structured data instead of regexing a step-name string."""

    silence_pad_applied_s: float = _SILENCE_PAD_S
    steps_applied: list[str] = field(default_factory=list)
    """Ordered list of step *names* (literal tokens like ``peak_normalize``
    and ``peak_limit``).  Numeric parameters are *not* embedded in the
    step name — they live in dedicated fields like ``target_peak_dbfs``
    and ``peak_dbfs`` so QA grep contracts don't break under config
    overrides."""

    warnings: list[str] = field(default_factory=list)


def _to_dbfs(samples: np.ndarray) -> float:
    """Return peak dBFS of a float32 array normalised to ±1.0."""
    peak = float(np.max(np.abs(samples)))
    if peak == 0.0:
        return -math.inf
    return 20.0 * math.log10(peak)


def _peak_limit(samples: np.ndarray, ceiling_dbfs: float = _PEAK_DBFS) -> np.ndarray:
    """Attenuate samples so peak ≤ ceiling_dbfs; never scale up.

    Unlike peak normalization, this only applies gain when the signal
    exceeds the ceiling.  Quieter signals are returned unchanged.  After
    #78 this is a safety net behind ``peak_normalize_to_target``; on its
    own it would let M3a-shaped clips sit ~6 dB below useful ASR / MOS
    operating range.
    """
    peak = float(np.max(np.abs(samples)))
    if peak == 0.0:
        return samples
    ceiling_linear = 10.0 ** (ceiling_dbfs / 20.0)
    if peak <= ceiling_linear:
        return samples  # already within limit — do not scale up
    return (samples * (ceiling_linear / peak)).astype(np.float32)


def peak_normalize_to_target(samples: np.ndarray, target_dbfs: float) -> np.ndarray:
    """Scale *samples* by a single global gain so their peak equals *target_dbfs*.

    Used in two places that must stay consistent:

    - ``preprocess()`` (Stage 3a) — Tier A clip-level loudness target.
    - ``cli.py`` (Stage 3b post-augment) — Tier B/C re-normalization after
      room IR + noise mixing reshapes the peak.  Both call sites read
      ``target_dbfs`` from the same ``PreprocessingConfig.target_peak_dbfs``
      so all tiers exit at the same absolute peak.

    Why a single global gain (#78): per-turn RMS targeting (M3a) creates
    the within-clip loudness trajectory.  We must not collapse it back
    into a flat shape, which is what the M3b commit (PR #27) deliberately
    avoided.  A single multiplicative factor preserves all per-turn RMS
    *ratios* exactly while pinning the absolute peak to a configured value.

    Silent or near-silent input (**peak** below ~−80 dBFS, the 1e-4
    threshold below) is returned unchanged.  This is a *peak* guard —
    distinct from ``_apply_rms_gain``'s identical numeric threshold,
    which is an *RMS* guard.  Same number, different physical meaning.

    Note: this helper does not clip.  Callers must constrain
    ``target_dbfs`` so the output stays in valid PCM range; the
    Pydantic ``[-12, -1.5]`` constraint on ``target_peak_dbfs``
    is the canonical guarantee.
    """
    peak = float(np.max(np.abs(samples)))
    if peak < 1e-4:  # peak < ~−80 dBFS — treat as silence, do not amplify
        return samples
    target_linear = 10.0 ** (target_dbfs / 20.0)
    return (samples * (target_linear / peak)).astype(np.float32)


def _resample(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample using polyphase filtering (scipy.signal.resample_poly)."""
    from math import gcd

    g = gcd(dst_sr, src_sr)
    up, down = dst_sr // g, src_sr // g
    resampled = resample_poly(samples.astype(np.float64), up, down)
    return resampled.astype(np.float32)


def _butterworth_highpass(samples: np.ndarray, sr: int, cutoff: int, order: int) -> np.ndarray:
    """Apply a Butterworth high-pass filter (sos form for numerical stability).

    Removes DC offset and sub-bass rumble that small phone microphones
    cannot capture.  At 80 Hz / order 2 this has negligible effect on
    speech content (fundamental frequency for adult males starts ~100 Hz).
    """
    nyq = sr / 2.0
    sos = butter(order, cutoff / nyq, btype="high", output="sos")
    return sosfilt(sos, samples).astype(np.float32)


def _wiener_denoise(samples: np.ndarray) -> np.ndarray:
    """Apply Wiener filter for noise reduction.

    Uses scipy.signal.wiener with a window that approximates spectral
    smoothing. For Phase 0 this removes high-frequency electrical hum.
    """
    denoised = wiener(samples.astype(np.float64), mysize=29)
    return denoised.astype(np.float32)


def _ensure_silence_pad(samples: np.ndarray, sr: int, pad_s: float) -> np.ndarray:
    """Prepend/append silence padding. The spec requires ≥ 0.5 s of ambient baseline."""
    pad_n = int(pad_s * sr)
    silence = np.zeros(pad_n, dtype=np.float32)
    return np.concatenate([silence, samples, silence])


def preprocess(
    input_path: Path | str,
    output_path: Path | str,
    dirty_dir: Path | str | None = None,
    config: PreprocessingConfig | None = None,
) -> PreprocessingResult:
    """Run the full preprocessing pipeline on a WAV file.

    Args:
        input_path: Source audio file (any sample rate / channel count).
        output_path: Destination for the processed 16 kHz mono WAV.
        dirty_dir: If given, copy the raw input here before processing
                   (to retain the "dirty" original as required by spec §3.1).
        config: Optional per-scene preprocessing flags.  ``None`` uses
                ``PreprocessingConfig()`` defaults (all steps enabled).
                Currently controls ``wiener_denoise`` — set to ``False``
                to skip the Wiener filter for clean TTS Tier A scenes.

    Returns:
        PreprocessingResult with metadata about the processing applied.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    cfg = config or PreprocessingConfig()
    steps: list[str] = []
    warnings: list[str] = []

    # --- Retain dirty original -----------------------------------------------
    # Use output_path.stem as the dirty filename so that two different inputs
    # that share the same basename (e.g. "raw.wav") never collide.
    dirty_path: Path | None = None
    if dirty_dir is not None:
        dirty_dir = Path(dirty_dir)
        dirty_dir.mkdir(parents=True, exist_ok=True)
        dirty_path = dirty_dir / f"{output_path.stem}_dirty{input_path.suffix}"
        if not dirty_path.exists():
            shutil.copy2(input_path, dirty_path)

    # --- Load audio ----------------------------------------------------------
    data, src_sr = sf.read(str(input_path), dtype="float32", always_2d=True)
    # data: (n_samples, n_channels), float32

    # --- 1. Resample ---------------------------------------------------------
    if src_sr != _TARGET_SR:
        resampled_channels = [
            _resample(data[:, c], src_sr, _TARGET_SR) for c in range(data.shape[1])
        ]
        data = np.stack(resampled_channels, axis=1)
        steps.append(f"resample_{src_sr}_to_{_TARGET_SR}")
    sr = _TARGET_SR

    # --- 2. Downmix to mono --------------------------------------------------
    if data.shape[1] > 1:
        data = data.mean(axis=1)
        steps.append("downmix_to_mono")
    else:
        data = data[:, 0]
    samples: np.ndarray = data  # (N,) float32

    # --- 3. High-pass filter at 80 Hz (DC/rumble removal) --------------------
    samples = _butterworth_highpass(samples, sr, _HP_CUTOFF_HZ, _HP_ORDER)
    steps.append(f"highpass_{_HP_CUTOFF_HZ}Hz_order{_HP_ORDER}")

    # --- 4. Wiener denoising (configurable — skip for clean TTS Tier A) ------
    if cfg.wiener_denoise:
        samples = _wiener_denoise(samples)
        steps.append("wiener_denoise")

    # --- 5a. Peak-normalize to target (#78) ----------------------------------
    # Single global gain — preserves per-turn RMS contrast (M3a) while pinning
    # absolute peak to a configured value.  Without this step, M3a-shaped
    # clips peak wherever per-turn RMS happens to land (~−6 dBFS for Tier A
    # baseline targets), which made it impossible to define a clip-level
    # loudness contract or detect downstream regressions from metadata alone.
    #
    # Note: empirically (lever probe, 2026-05-05) Whisper WER does not respond
    # to the absolute peak in the spec range — its log-mel feature extractor
    # internally normalizes — so the rationale for this step is contract
    # clarity + metadata-trail, not ASR recovery.  The actual ASR regression
    # tracked at the M17 spike has a different cause (likely the M15/#70
    # prosody changes); see the prosody-bisect follow-up issue.
    samples = peak_normalize_to_target(samples, cfg.target_peak_dbfs)
    steps.append("peak_normalize")

    # --- 5b. Safety limiter at −1.0 dBFS (never scale up) --------------------
    # Defence-in-depth.  Pydantic constrains target_peak_dbfs ≤ −1.5, so this
    # is a guaranteed no-op in normal flow — it only fires if upstream code
    # somehow bypasses step 5a, in which case clamping at the spec ceiling is
    # still strictly better than shipping a clipped clip.
    samples = _peak_limit(samples, _PEAK_DBFS)
    steps.append("peak_limit")

    # --- 6. Silence pad (≥ 0.5 s at head and tail) ---------------------------
    samples = _ensure_silence_pad(samples, sr, _SILENCE_PAD_S)
    steps.append("silence_pad")

    # --- Clip duration checks ------------------------------------------------
    duration = len(samples) / sr
    if duration < _MIN_DURATION_S:
        warnings.append(f"duration {duration:.2f} s is below minimum {_MIN_DURATION_S} s")
    if duration > _MAX_DURATION_S:
        warnings.append(f"duration {duration:.2f} s exceeds maximum {_MAX_DURATION_S} s")

    # --- Write output --------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(output_path),
        samples,
        sr,
        subtype="PCM_16",
    )

    return PreprocessingResult(
        output_path=output_path,
        dirty_path=dirty_path,
        sample_rate=sr,
        channels=_TARGET_CHANNELS,
        duration_seconds=duration,
        peak_dbfs=_to_dbfs(samples),
        target_peak_dbfs=cfg.target_peak_dbfs,
        steps_applied=steps,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_audio(path: Path | str) -> tuple[bool, list[str]]:
    """Validate that an audio file meets AVDP spec requirements.

    Returns:
        (is_valid, list_of_error_messages)
    """
    path = Path(path)
    errors: list[str] = []

    if not path.exists():
        return False, [f"File not found: {path}"]

    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        return False, [f"Cannot load audio: {exc}"]

    # Sample rate
    if sr != _TARGET_SR:
        errors.append(f"sample_rate {sr} != {_TARGET_SR}")

    # Channels
    n_channels = data.shape[1]
    if n_channels != _TARGET_CHANNELS:
        errors.append(f"channels {n_channels} != {_TARGET_CHANNELS}")

    samples = data[:, 0] if data.ndim == 2 else data

    # NaN / Inf
    if not np.isfinite(samples).all():
        errors.append("audio contains NaN or Inf samples")

    # Duration — use the file's actual sr so the value is accurate even when
    # sr != _TARGET_SR (the sample_rate error above will still be reported).
    duration = len(samples) / sr
    if duration < _MIN_DURATION_S:
        errors.append(f"duration {duration:.2f} s < minimum {_MIN_DURATION_S} s")

    # Peak limit — clip must not exceed ceiling (allow 0.5 dB tolerance for
    # PCM_16 quantisation rounding); clips quieter than the ceiling are fine.
    peak_dbfs = _to_dbfs(samples)
    if not math.isinf(peak_dbfs) and peak_dbfs > _PEAK_DBFS + 0.5:
        errors.append(f"peak {peak_dbfs:.2f} dBFS exceeds ceiling {_PEAK_DBFS} dBFS")

    # Silence padding: check first and last 0.5 s are below -40 dBFS
    pad_n = int(_SILENCE_PAD_S * sr)
    if len(samples) > 2 * pad_n:
        head_peak = _to_dbfs(samples[:pad_n])
        tail_peak = _to_dbfs(samples[-pad_n:])
        if not math.isinf(head_peak) and head_peak > -40.0:
            errors.append(
                f"head silence peak {head_peak:.1f} dBFS > -40 dBFS "
                "(insufficient silence padding at start)"
            )
        if not math.isinf(tail_peak) and tail_peak > -40.0:
            errors.append(
                f"tail silence peak {tail_peak:.1f} dBFS > -40 dBFS "
                "(insufficient silence padding at end)"
            )

    return len(errors) == 0, errors
