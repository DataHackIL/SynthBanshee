"""Azure Cognitive Services TTS provider for he-IL voices.

Wraps the azure-cognitiveservices-speech SDK to synthesize SSML documents
into WAV audio. Credentials are read from environment variables:

    AZURE_TTS_KEY    — Azure subscription key
    AZURE_TTS_REGION — Azure region (e.g. "eastus")
"""

from __future__ import annotations

import os
from typing import Protocol


class SpeechSDKProtocol(Protocol):
    """Protocol for dependency injection / testing without real Azure credentials."""

    def synthesize_ssml_async(self, ssml: str): ...


class AzureProvider:
    """Synthesize SSML to WAV bytes using Azure Cognitive Services TTS.

    In unit tests, inject a mock via the `sdk_factory` parameter to avoid
    real network calls.
    """

    def __init__(
        self,
        subscription_key: str | None = None,
        region: str | None = None,
        *,
        sdk_factory=None,
    ) -> None:
        self._key = subscription_key or os.environ.get("AZURE_TTS_KEY", "")
        self._region = region or os.environ.get("AZURE_TTS_REGION", "eastus")
        self._sdk_factory = sdk_factory  # callable(key, region) -> SDK synthesizer

    def _get_synthesizer(self):
        """Return an Azure SpeechSynthesizer, lazily constructed."""
        if self._sdk_factory is not None:
            return self._sdk_factory(self._key, self._region)

        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError as exc:
            raise ImportError(
                "azure-cognitiveservices-speech is required for AzureProvider. "
                "Install it with: pip install azure-cognitiveservices-speech"
            ) from exc

        speech_config = speechsdk.SpeechConfig(
            subscription=self._key, region=self._region
        )
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
        )
        # Stream output into memory
        audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=False)
        return speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=audio_config
        )

    def synthesize(self, ssml: str) -> bytes:
        """Synthesize SSML and return raw WAV bytes (Riff24Khz16BitMonoPcm).

        Args:
            ssml: A complete SSML document string.

        Returns:
            WAV file content as bytes (24 kHz, 16-bit PCM mono).
            The preprocessing pipeline will downsample to 16 kHz.

        Raises:
            RuntimeError: If synthesis fails or credentials are missing.
        """
        synthesizer = self._get_synthesizer()
        result = synthesizer.speak_ssml_async(ssml).get()

        # Handle both real SDK result and mock result
        if hasattr(result, "reason"):
            try:
                import azure.cognitiveservices.speech as speechsdk

                if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
                    cancellation = result.cancellation_details
                    raise RuntimeError(
                        f"Azure TTS synthesis failed: {result.reason}. "
                        f"Details: {cancellation.reason} / {cancellation.error_details}"
                    )
            except ImportError:
                pass  # Mock mode — assume success

        audio_data = result.audio_data if hasattr(result, "audio_data") else result
        if not isinstance(audio_data, (bytes, bytearray)):
            raise RuntimeError(f"Unexpected audio_data type: {type(audio_data)}")
        return bytes(audio_data)

    def is_configured(self) -> bool:
        """Return True if credentials are present in the environment."""
        return bool(self._key and self._region)
