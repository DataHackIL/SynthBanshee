"""Pydantic model for per-scene preprocessing options."""

from __future__ import annotations

from pydantic import BaseModel


class PreprocessingConfig(BaseModel):
    """Per-scene preprocessing options.

    Passed to ``preprocess()`` to control which pipeline steps are applied.
    All flags default to the spec-mandated behaviour so existing call sites
    that omit the config are unaffected.
    """

    wiener_denoise: bool = True
    """Apply Wiener noise reduction (step 4).  Set to False for Tier A scenes
    where the clean TTS signal should be preserved without spectral smoothing."""
