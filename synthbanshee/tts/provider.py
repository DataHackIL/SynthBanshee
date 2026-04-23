"""TTS provider abstraction: ProviderCapabilities + TTSProvider Protocol.

Each backend declares its capabilities at instantiation so the render
pipeline can adapt the SSML payload (or skip unsupported tags) rather than
silently losing expressiveness.

Design reference: docs/audio_generation_v3_design.md §4.10 / M9a
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ProviderCapabilities:
    """Declares what a TTS backend can and cannot do.

    Attributes:
        supports_ssml: Backend accepts SSML input (vs. plain text only).
        supports_style_tags: Backend supports ``<mstts:express-as>`` style
            elements (Azure-specific; must be omitted for all other backends).
        supports_phoneme_tags: Backend honours ``<phoneme alphabet="ipa">``
            elements for explicit pronunciation control.
        supports_api_emotion_sliders: Backend exposes per-request API
            parameters for emotion/stability (e.g. ElevenLabs
            ``stability`` / ``style_exaggeration``).
        max_volume_delta_db: Hard ceiling on volume offset accepted by the
            backend, in dB.  ``None`` means unlimited (backend clips
            internally or has no documented limit).
    """

    supports_ssml: bool
    supports_style_tags: bool
    supports_phoneme_tags: bool
    supports_api_emotion_sliders: bool
    max_volume_delta_db: float | None


@runtime_checkable
class TTSProvider(Protocol):
    """Structural protocol satisfied by AzureProvider and GoogleProvider.

    Providers are duck-typed via this protocol; no explicit base class
    is required.  The renderer uses ``isinstance(p, TTSProvider)`` only
    for documentation/type-narrowing; it does not enforce it at runtime.
    """

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return the capability matrix for this backend."""
        ...

    def synthesize(self, ssml: str) -> bytes:
        """Synthesize *ssml* and return raw WAV bytes.

        The returned bytes must be a valid WAV file at 24 kHz mono (before
        the preprocessing stage downsamples to 16 kHz).

        Raises:
            RuntimeError: If synthesis fails or credentials are absent.
        """
        ...

    def is_configured(self) -> bool:
        """Return True if the provider has the credentials it needs."""
        ...
