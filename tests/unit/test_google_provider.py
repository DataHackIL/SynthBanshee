"""Unit tests for GoogleProvider and multi-provider TTSRenderer dispatch.

The Google TTS client is mocked — no real API credentials are required.
"""

from __future__ import annotations

import io
import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from synthbanshee.config.speaker_config import SpeakerConfig
from synthbanshee.tts.google_provider import GoogleProvider, _extract_voice_name
from synthbanshee.tts.provider import TTSProvider
from synthbanshee.tts.renderer import TTSRenderer
from synthbanshee.tts.ssml_builder import SSMLBuilder, UtteranceSpec

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "configs" / "examples"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav_bytes(sample_rate: int = 24000, duration_s: float = 1.0) -> bytes:
    n_samples = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        data = struct.pack(f"<{n_samples}h", *([1000] * n_samples))
        w.writeframes(data)
    return buf.getvalue()


def _mock_google_client_factory():
    """Return a factory that produces a mock Google TTS client."""

    def factory():
        client = MagicMock()
        response = MagicMock()
        response.audio_content = _make_wav_bytes()
        client.synthesize_speech.return_value = response
        return client

    return factory


# ---------------------------------------------------------------------------
# ProviderCapabilities
# ---------------------------------------------------------------------------


class TestProviderCapabilities:
    def test_azure_capabilities(self):
        from synthbanshee.tts.azure_provider import AzureProvider

        p = AzureProvider(sdk_factory=lambda k, r: MagicMock())
        caps = p.capabilities
        assert caps.supports_ssml is True
        assert caps.supports_style_tags is True
        assert caps.supports_phoneme_tags is True
        assert caps.supports_api_emotion_sliders is False

    def test_google_capabilities(self):
        p = GoogleProvider(client_factory=_mock_google_client_factory())
        caps = p.capabilities
        assert caps.supports_ssml is True
        assert caps.supports_style_tags is False  # no <mstts:express-as>
        assert caps.supports_phoneme_tags is True
        assert caps.supports_api_emotion_sliders is False

    def test_capabilities_is_frozen(self):
        from synthbanshee.tts.azure_provider import AzureProvider

        p = AzureProvider(sdk_factory=lambda k, r: MagicMock())
        with pytest.raises((AttributeError, TypeError)):
            p.capabilities.supports_ssml = False  # type: ignore[misc]

    def test_provider_protocol_satisfied(self):
        from synthbanshee.tts.azure_provider import AzureProvider

        azure = AzureProvider(sdk_factory=lambda k, r: MagicMock())
        google = GoogleProvider(client_factory=_mock_google_client_factory())
        assert isinstance(azure, TTSProvider)
        assert isinstance(google, TTSProvider)


# ---------------------------------------------------------------------------
# GoogleProvider.synthesize (mocked)
# ---------------------------------------------------------------------------


class TestGoogleProvider:
    def _make_provider(self) -> GoogleProvider:
        return GoogleProvider(client_factory=_mock_google_client_factory())

    def test_synthesize_returns_bytes(self):
        provider = self._make_provider()
        builder = SSMLBuilder()
        ssml = builder.build_from_speaker_config(
            text="שלום",
            voice_id="he-IL-Chirp3-HD-Alef",
            style="General",
            supports_style_tags=False,
        )
        result = provider.synthesize(ssml)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_synthesize_calls_client_once(self):
        mock_client = MagicMock()
        mock_client.synthesize_speech.return_value.audio_content = _make_wav_bytes()
        provider = GoogleProvider(client_factory=lambda: mock_client)

        builder = SSMLBuilder()
        ssml = builder.build_from_speaker_config(
            text="שלום",
            voice_id="he-IL-Chirp3-HD-Bet",
            style="General",
            supports_style_tags=False,
        )
        provider.synthesize(ssml)
        mock_client.synthesize_speech.assert_called_once()

    def test_is_configured_with_env(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/fake/path.json")
        provider = GoogleProvider()
        assert provider.is_configured()

    def test_not_configured_without_env(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        provider = GoogleProvider()
        assert not provider.is_configured()

    def test_synthesize_raises_on_client_error(self):
        def bad_client():
            c = MagicMock()
            c.synthesize_speech.side_effect = Exception("quota exceeded")
            return c

        provider = GoogleProvider(client_factory=bad_client)
        builder = SSMLBuilder()
        ssml = builder.build_from_speaker_config(
            text="test",
            voice_id="he-IL-Chirp3-HD-Alef",
            style="General",
            supports_style_tags=False,
        )
        with pytest.raises(RuntimeError, match="quota exceeded"):
            provider.synthesize(ssml)

    def test_missing_google_package_raises_import_error(self, monkeypatch):
        """GoogleProvider raises ImportError when the SDK is absent and no factory."""
        import sys

        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/fake.json")
        # Simulate missing package by removing it from sys.modules and blocking import
        monkeypatch.setitem(sys.modules, "google", None)  # type: ignore[arg-type]
        monkeypatch.setitem(sys.modules, "google.cloud", None)  # type: ignore[arg-type]
        monkeypatch.setitem(sys.modules, "google.cloud.texttospeech", None)  # type: ignore[arg-type]

        provider = GoogleProvider()  # no factory
        with pytest.raises((ImportError, RuntimeError)):
            provider.synthesize("<speak/>")


# ---------------------------------------------------------------------------
# SSMLBuilder — supports_style_tags=False
# ---------------------------------------------------------------------------


class TestSSMLBuilderGoogleMode:
    def setup_method(self):
        self.builder = SSMLBuilder()

    def test_no_mstts_namespace_for_google(self):
        utt = UtteranceSpec(
            text="hello",
            voice_id="he-IL-Chirp3-HD-Alef",
            style="angry",
        )
        ssml = self.builder.build_single(utt, supports_style_tags=False)
        assert "mstts" not in ssml
        assert "express-as" not in ssml

    def test_no_express_as_even_with_non_general_style(self):
        utt = UtteranceSpec(text="hi", voice_id="he-IL-Chirp3-HD-Bet", style="angry")
        ssml = self.builder.build_single(utt, supports_style_tags=False)
        assert "express-as" not in ssml

    def test_azure_mode_still_includes_express_as(self):
        utt = UtteranceSpec(text="hi", voice_id="he-IL-AvriNeural", style="angry")
        ssml = self.builder.build_single(utt, supports_style_tags=True)
        assert "express-as" in ssml
        assert "mstts" in ssml

    def test_prosody_still_present_in_google_mode(self):
        utt = UtteranceSpec(
            text="hi",
            voice_id="he-IL-Chirp3-HD-Alef",
            style="General",
            rate_multiplier=1.2,
            pitch_delta_st=2.0,
        )
        ssml = self.builder.build_single(utt, supports_style_tags=False)
        assert "prosody" in ssml

    def test_build_from_speaker_config_google(self):
        ssml = self.builder.build_from_speaker_config(
            text="test",
            voice_id="he-IL-Chirp3-HD-Alef",
            style="angry",
            rate_multiplier=1.1,
            supports_style_tags=False,
        )
        assert "mstts" not in ssml
        assert "he-IL-Chirp3-HD-Alef" in ssml


# ---------------------------------------------------------------------------
# TTSRenderer — multi-provider dispatch
# ---------------------------------------------------------------------------


def _mock_azure_factory(key, region):
    synth = MagicMock()
    mock_result = MagicMock()
    mock_result.audio_data = _make_wav_bytes()
    del mock_result.reason
    synth.speak_ssml_async.return_value.get.return_value = mock_result
    return synth


class TestTTSRendererMultiProvider:
    def _make_renderer(self, tmp_path) -> TTSRenderer:
        from synthbanshee.tts.azure_provider import AzureProvider

        azure = AzureProvider(sdk_factory=_mock_azure_factory)
        google = GoogleProvider(client_factory=_mock_google_client_factory())
        return TTSRenderer(
            providers={"azure": azure, "google": google},
            cache_dir=tmp_path / "cache",
        )

    def test_azure_speaker_uses_azure_provider(self, tmp_path):
        renderer = self._make_renderer(tmp_path)
        speaker = SpeakerConfig.from_yaml(EXAMPLES_DIR / "speaker_AGG_M_30-45_001.yaml")
        assert speaker.tts_provider == "azure"
        wav, _, hit = renderer.render_utterance("שלום", speaker, intensity=1)
        assert isinstance(wav, bytes)
        assert not hit

    def test_google_speaker_uses_google_provider(self, tmp_path):
        renderer = self._make_renderer(tmp_path)
        speaker = SpeakerConfig.from_yaml(EXAMPLES_DIR / "speaker_AGG_M_30-45_002.yaml")
        assert speaker.tts_provider == "google"
        wav, _, hit = renderer.render_utterance("שלום", speaker, intensity=1)
        assert isinstance(wav, bytes)
        assert not hit

    def test_google_ssml_has_no_mstts(self, tmp_path):
        """SSML built for a Google speaker must not contain mstts tags."""
        from synthbanshee.tts.azure_provider import AzureProvider

        captured_ssml: list[str] = []

        class CapturingGoogle(GoogleProvider):
            def synthesize(self, ssml: str) -> bytes:
                captured_ssml.append(ssml)
                return _make_wav_bytes()

        azure = AzureProvider(sdk_factory=_mock_azure_factory)
        renderer = TTSRenderer(
            providers={"azure": azure, "google": CapturingGoogle()},
            cache_dir=tmp_path / "cache",
        )
        speaker = SpeakerConfig.from_yaml(EXAMPLES_DIR / "speaker_AGG_M_30-45_002.yaml")
        renderer.render_utterance("שלום", speaker, intensity=3)
        assert captured_ssml, "Google synthesize was not called"
        assert "mstts" not in captured_ssml[0]
        assert "express-as" not in captured_ssml[0]

    def test_azure_ssml_has_mstts_at_intensity_3(self, tmp_path):
        """Azure speaker at intensity 3 (angry style) must include express-as."""
        from synthbanshee.tts.azure_provider import AzureProvider

        captured_ssml: list[str] = []

        class CapturingAzure(AzureProvider):
            def synthesize(self, ssml: str) -> bytes:
                captured_ssml.append(ssml)
                return _make_wav_bytes()

        renderer = TTSRenderer(
            providers={"azure": CapturingAzure(sdk_factory=_mock_azure_factory)},
            cache_dir=tmp_path / "cache",
        )
        speaker = SpeakerConfig.from_yaml(EXAMPLES_DIR / "speaker_AGG_M_30-45_001.yaml")
        renderer.render_utterance("שלום", speaker, intensity=3)
        assert captured_ssml
        assert "express-as" in captured_ssml[0]

    def test_provider_and_providers_mutually_exclusive(self, tmp_path):
        from synthbanshee.tts.azure_provider import AzureProvider

        azure = AzureProvider(sdk_factory=_mock_azure_factory)
        with pytest.raises(ValueError, match="not both"):
            TTSRenderer(
                provider=azure,
                providers={"azure": azure},
                cache_dir=tmp_path / "cache",
            )

    def test_unknown_provider_raises_key_error(self, tmp_path):
        from synthbanshee.config.speaker_config import SpeakerConfig
        from synthbanshee.tts.azure_provider import AzureProvider

        azure = AzureProvider(sdk_factory=_mock_azure_factory)
        renderer = TTSRenderer(
            providers={"azure": azure},
            cache_dir=tmp_path / "cache",
        )
        # Build a speaker whose tts_provider is "google" but no google provider is registered.
        speaker = SpeakerConfig.from_yaml(EXAMPLES_DIR / "speaker_AGG_M_30-45_002.yaml")
        assert speaker.tts_provider == "google"
        with pytest.raises(KeyError, match="google"):
            renderer.render_utterance("שלום", speaker, intensity=1)

    def test_legacy_single_provider_mode(self, tmp_path):
        """Single-provider (legacy) mode: provider kwarg is still accepted."""
        from synthbanshee.tts.azure_provider import AzureProvider

        azure = AzureProvider(sdk_factory=_mock_azure_factory)
        renderer = TTSRenderer(provider=azure, cache_dir=tmp_path / "cache")
        speaker = SpeakerConfig.from_yaml(EXAMPLES_DIR / "speaker_AGG_M_30-45_001.yaml")
        wav, _, _ = renderer.render_utterance("שלום", speaker, intensity=1)
        assert isinstance(wav, bytes)


# ---------------------------------------------------------------------------
# _extract_voice_name helper
# ---------------------------------------------------------------------------


class TestExtractVoiceName:
    def test_extracts_double_quoted(self):
        ssml = '<speak><voice name="he-IL-Chirp3-HD-Alef">text</voice></speak>'
        assert _extract_voice_name(ssml) == "he-IL-Chirp3-HD-Alef"

    def test_extracts_single_quoted(self):
        ssml = "<speak><voice name='he-IL-Chirp3-HD-Bet'>text</voice></speak>"
        assert _extract_voice_name(ssml) == "he-IL-Chirp3-HD-Bet"

    def test_returns_empty_when_absent(self):
        assert _extract_voice_name("<speak>no voice</speak>") == ""
