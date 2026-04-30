"""Google Cloud TTS provider — Chirp 3 HD he-IL voices.

Uses the ``google-cloud-texttospeech`` client library (optional extra
``[google-tts]``).  Credentials are discovered via Google Application
Default Credentials (ADC):

    - ``GOOGLE_APPLICATION_CREDENTIALS`` pointing to a service-account JSON
    - ``gcloud auth application-default login`` (local development)
    - GCE metadata server / Workload Identity (CI / cloud environments)

See https://cloud.google.com/docs/authentication/application-default-credentials

Voice IDs to use in speaker YAMLs (verify against the current catalog):

    python scripts/check_google_voices.py

Known Chirp 3 HD he-IL voices (as of 2026-04):
    he-IL-Chirp3-HD-Achird   — male
    he-IL-Chirp3-HD-Achernar — female

Set ``tts_provider: google`` and ``tts_voice_id: he-IL-Chirp3-HD-Achird``
(or Achernar) in a speaker YAML to route synthesis through this provider.
"""

from __future__ import annotations

import io
import wave

from synthbanshee.tts.provider import ProviderCapabilities

_GOOGLE_CAPABILITIES = ProviderCapabilities(
    supports_ssml=True,
    supports_style_tags=False,  # <mstts:express-as> is Azure-only
    supports_phoneme_tags=True,  # <phoneme alphabet="ipa"> is W3C SSML
    supports_api_emotion_sliders=False,
    max_volume_delta_db=None,
)

# Google TTS outputs LINEAR16 at this sample rate.  The preprocessing stage
# will downsample to 16 kHz — same pipeline as Azure.
_OUTPUT_SAMPLE_RATE_HZ = 24_000


def _pcm_to_wav(pcm: bytes, sample_rate: int = _OUTPUT_SAMPLE_RATE_HZ) -> bytes:
    """Wrap raw LINEAR16 (16-bit signed LE mono) PCM in a RIFF/WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


class GoogleProvider:
    """Synthesize SSML to WAV bytes using Google Cloud TTS Chirp 3 HD.

    In unit tests, inject a mock via the ``client_factory`` parameter to
    avoid real network calls (identical pattern to AzureProvider).

    Args:
        client_factory: Optional callable() -> Google TTS client.  When
            provided, the real ``google.cloud.texttospeech`` import is
            bypassed entirely.
    """

    def __init__(self, *, client_factory=None) -> None:
        self._client_factory = client_factory

    # ------------------------------------------------------------------
    # ProviderCapabilities
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return Google TTS capability matrix."""
        return _GOOGLE_CAPABILITIES

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client_factory is not None:
            return self._client_factory()

        try:
            from google.cloud import texttospeech  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "google-cloud-texttospeech is required for GoogleProvider. "
                "Install it with: pip install 'synthbanshee[google-tts]'"
            ) from exc

        return texttospeech.TextToSpeechClient()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(self, ssml: str) -> bytes:
        """Synthesize *ssml* and return WAV bytes (24 kHz, 16-bit mono).

        Google Cloud TTS returns raw LINEAR16 PCM for
        ``AudioEncoding.LINEAR16``.  This method wraps the PCM in a
        RIFF/WAV container so downstream code (``SceneMixer``,
        ``soundfile.read()``) receives a valid WAV file — the same
        contract as ``AzureProvider.synthesize()``.

        The voice name is extracted from the ``<voice name="...">``
        element of the SSML.  If no voice name is found, a ``RuntimeError``
        is raised immediately (rather than forwarding malformed params to
        the API).

        Args:
            ssml: A complete SSML document.  Must not contain
                ``<mstts:express-as>`` elements (Google ignores/rejects them).
                The SSMLBuilder strips these automatically when
                ``supports_style_tags=False``.

        Returns:
            WAV file content as bytes (24 kHz, 16-bit PCM mono).

        Raises:
            RuntimeError: If synthesis fails, credentials cannot be
                discovered, or the SSML is missing a ``<voice>`` element.
        """
        voice_name = _extract_voice_name(ssml)
        if not voice_name:
            raise RuntimeError(
                'SSML document is missing a <voice name="..."> element. '
                "GoogleProvider requires the voice ID to be embedded in the SSML."
            )

        try:
            from google.cloud import texttospeech  # type: ignore[import-untyped]
        except ImportError:
            texttospeech = None  # type: ignore[assignment]

        client = self._get_client()

        synthesis_input = _build_input(ssml, texttospeech)
        voice_params = _build_voice_params(voice_name, texttospeech)
        audio_config = _build_audio_config(_OUTPUT_SAMPLE_RATE_HZ, texttospeech)

        try:
            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=audio_config,
            )
        except Exception as exc:
            raise RuntimeError(f"Google TTS synthesis failed: {exc}") from exc

        audio = response.audio_content
        if not isinstance(audio, bytes | bytearray):
            raise RuntimeError(f"Unexpected audio_content type: {type(audio)}")
        return _pcm_to_wav(bytes(audio))

    def is_configured(self) -> bool:
        """Return True if ADC credentials can likely be discovered.

        Checks for the most common ADC signal
        (``GOOGLE_APPLICATION_CREDENTIALS``).  This is a fast heuristic;
        credentials may also be available via ``gcloud auth`` or the
        metadata server even when this returns False.
        """
        import os

        return bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))


# ---------------------------------------------------------------------------
# Private helpers — accept the texttospeech module as a param so they work
# in both real and mock contexts.
# ---------------------------------------------------------------------------


def _build_input(ssml: str, tts_module):
    """Build a SynthesisInput from an SSML string."""
    if tts_module is not None:
        return tts_module.SynthesisInput(ssml=ssml)
    return {"ssml": ssml}


def _extract_voice_name(ssml: str) -> str:
    """Extract the voice name from the first <voice name="..."> element."""
    import re

    match = re.search(r'<voice\s[^>]*name=["\']([^"\']+)["\']', ssml)
    return match.group(1) if match else ""


def _build_voice_params(voice_name: str, tts_module):
    """Build VoiceSelectionParams from a voice name."""
    lang_code = "-".join(voice_name.split("-")[:2])
    if tts_module is not None:
        return tts_module.VoiceSelectionParams(
            language_code=lang_code,
            name=voice_name,
        )
    return {"language_code": lang_code, "name": voice_name}


def _build_audio_config(sample_rate_hz: int, tts_module):
    """Build AudioConfig for LINEAR16 output."""
    if tts_module is not None:
        return tts_module.AudioConfig(
            audio_encoding=tts_module.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate_hz,
        )
    return {"audio_encoding": "LINEAR16", "sample_rate_hertz": sample_rate_hz}
