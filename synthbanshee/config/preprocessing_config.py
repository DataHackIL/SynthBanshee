"""Pydantic model for per-scene preprocessing options."""

from __future__ import annotations

from pydantic import BaseModel


class PreprocessingConfig(BaseModel):
    """Per-scene preprocessing options.

    Passed to ``preprocess()`` to control which pipeline steps are applied.

    M14 (2026-05-01): ``wiener_denoise`` default changed from ``True`` to
    ``False`` — Wiener denoising on clean TTS output over-smooths
    high-frequency transients.  Enable explicitly for Tier B/C clips
    with real added noise.
    """

    wiener_denoise: bool = False
    """Apply Wiener noise reduction (step 4).  Defaults to False because
    Wiener denoising on clean TTS output causes muffled sound by
    over-smoothing high-frequency transients (confirmed by three
    independent research reports).  Enable only for clips with real
    added noise (Tier B/C after acoustic augmentation)."""
