"""Pydantic model for per-scene preprocessing options."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PreprocessingConfig(BaseModel):
    """Per-scene preprocessing options.

    Passed to ``preprocess()`` to control which pipeline steps are applied.

    M14 (2026-05-01): ``wiener_denoise`` default changed from ``True`` to
    ``False`` — Wiener denoising on clean TTS output over-smooths
    high-frequency transients.  Enable explicitly for Tier B/C clips
    with real added noise.

    #78 (2026-05-05): ``target_peak_dbfs`` added.  Per-turn RMS targeting
    (M3a) plus the M3b limiter-instead-of-normalizer change left Tier A
    clips with peaks well below the −1.0 dBFS ceiling (~−6 dBFS), which
    confused downstream Whisper/UTMOS scoring.  ``preprocess()`` now
    peak-normalizes the mixed scene to this target before the safety
    limiter, while the per-turn RMS contrast within the clip is preserved
    by applying a single global gain.
    """

    wiener_denoise: bool = False
    """Apply Wiener noise reduction (step 4).  Defaults to False because
    Wiener denoising on clean TTS output causes muffled sound by
    over-smoothing high-frequency transients (confirmed by three
    independent research reports).  Enable only for clips with real
    added noise (Tier B/C after acoustic augmentation)."""

    target_peak_dbfs: float = Field(default=-2.0, ge=-12.0, le=-1.0)
    """Target peak level for clip-level loudness normalization (#78).

    The mixed scene is scaled by a single global gain so its absolute peak
    lands at this dBFS value, preserving per-turn RMS contrast (M3a) while
    ensuring the clip reaches a useful absolute loudness for downstream
    ASR / MOS evaluation.  The −1.0 dBFS safety limiter still runs after
    this step; values closer to the ceiling leave less headroom for
    downstream room-IR / device augmentation.  Range is bounded
    [−12.0, −1.0] dBFS — louder than −1.0 would clip past the spec
    ceiling, quieter than −12.0 reproduces the very deficiency this
    setting exists to fix."""
