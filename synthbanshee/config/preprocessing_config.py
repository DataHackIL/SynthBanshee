"""Pydantic model for per-scene preprocessing options."""

from __future__ import annotations

from pydantic import BaseModel


class PreprocessingConfig(BaseModel):
    """Per-scene preprocessing options.

    Passed to ``preprocess()`` to control which pipeline steps are applied.
    All flags default to the spec-mandated behaviour so existing call sites
    that omit the config are unaffected.
    """

    wiener_denoise: bool = False
    """Apply Wiener noise reduction (step 4).  Defaults to False because
    Wiener denoising on clean TTS output causes muffled sound by
    over-smoothing high-frequency transients (confirmed by three
    independent research reports).  Enable only for clips with real
    added noise (Tier B/C after acoustic augmentation)."""
