"""Unit tests for GoogleProvider and multi-provider TTSRenderer dispatch.

The Google TTS client is mocked — no real API credentials are required.
COPILOT-4/5: mocks return raw PCM (not WAV) to match real API behaviour;
tests assert that GoogleProvider.synthesize() returns a valid WAV container.
"""

from __future__ import annotations

import io
import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from synthbanshee.config.speaker_config import SpeakerConfig
from synthbanshee.tts.google_provider import (
    GoogleProvider,
    _extract_voice_name,
    _pcm_to_wav,
)
from synthbanshee.tts.provider import TTSProvider
from synthbanshee.tts.renderer import TTSRenderer
from synthbanshee.tts.ssml_builder import SSMLBuilder, UtteranceSpec

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "configs" / "examples"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_pcm(sample_rate: int = 24000, duration_s: float = 1.0) -> bytes:
    """Generate raw LINEAR16 PCM bytes (no RIFF header) — matches real API."""
    n_samples = int(sample_rate * duration_s)
    return struct.pack(f"<{n_samples}h", *([1000] * n_samples))


def _make_wav_bytes(sample_rate: int = 24000, duration_s: float = 1.0) -> bytes:
    """Generate a valid WAV container (for Azure mock and assertion helpers)."""
    buf = io.BytesIO()
    with wave.open(buf, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(_make_raw_pcm(sample_rate, duration_s))
    return buf.getvalue()


def _mock_google_client_factory():
    """Return a factory that produces a mock Google TTS client.

    The mock returns raw PCM (no RIFF header) to match real API behaviour.
    """

    def factory():
        client = MagicMock()
        response = MagicMock()
        response.audio_content = _make_raw_pcm()
        client.synthesize_speech.return_value = response
        return client

    return factory


def _assert_valid_wav(data: bytes, expected_rate: int = 24000) -> None:
    """Assert that *data* is a valid WAV container with the expected properties."""
    assert data[:4] == b"RIFF", "Output must start with RIFF header"
    with wave.open(io.BytesIO(data), "r") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == expected_rate


# ---------------------------------------------------------------------------
# ProviderCapabilities
# ---------------------------------------------------------------------------


class TestProviderCapabilities:
    def test_azure_capabilities(self):
        from synthbanshee.tts.azure_provider import AzureProvider

        p = AzureProvider(sdk_factory=lambda k, r: MagicMock())
        caps = p.capabilities
        assert caps.supports_ssml is True
        # #97: M14's False default was speculative; flipped pending the spike's
        # listening-test verdict.  Production speakers all use style="General"
        # so this is a no-op for default renders.
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
# _pcm_to_wav
# ---------------------------------------------------------------------------


class TestPcmToWav:
    def test_wraps_pcm_in_valid_wav(self):
        pcm = _make_raw_pcm(duration_s=0.1)
        wav = _pcm_to_wav(pcm)
        _assert_valid_wav(wav)

    def test_preserves_sample_count(self):
        n_samples = 480
        pcm = struct.pack(f"<{n_samples}h", *([500] * n_samples))
        wav = _pcm_to_wav(pcm)
        with wave.open(io.BytesIO(wav), "r") as w:
            assert w.getnframes() == n_samples


# ---------------------------------------------------------------------------
# GoogleProvider.synthesize (mocked)
# ---------------------------------------------------------------------------


class TestGoogleProvider:
    def _make_provider(self) -> GoogleProvider:
        return GoogleProvider(client_factory=_mock_google_client_factory())

    def test_synthesize_returns_valid_wav(self):
        """Synthesize returns a WAV container, not raw PCM."""
        provider = self._make_provider()
        builder = SSMLBuilder()
        ssml = builder.build_from_speaker_config(
            text="שלום",
            voice_id="he-IL-Chirp3-HD-Achird",
            style="General",
            supports_style_tags=False,
        )
        result = provider.synthesize(ssml)
        _assert_valid_wav(result)

    def test_synthesize_calls_client_once(self):
        mock_client = MagicMock()
        mock_client.synthesize_speech.return_value.audio_content = _make_raw_pcm()
        provider = GoogleProvider(client_factory=lambda: mock_client)

        builder = SSMLBuilder()
        ssml = builder.build_from_speaker_config(
            text="שלום",
            voice_id="he-IL-Chirp3-HD-Achernar",
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
            voice_id="he-IL-Chirp3-HD-Achird",
            style="General",
            supports_style_tags=False,
        )
        with pytest.raises(RuntimeError, match="quota exceeded"):
            provider.synthesize(ssml)

    def test_get_client_without_factory_returns_real_client(self, monkeypatch):
        """When no client_factory, _get_client() calls TextToSpeechClient().

        Covers google_provider.py line 96.  Works in CI where
        google-cloud-texttospeech is not installed by injecting a mock
        module into sys.modules.
        """
        import sys
        import types

        mock_tts = types.ModuleType("google.cloud.texttospeech")
        mock_client_instance = MagicMock()
        mock_tts.TextToSpeechClient = MagicMock(return_value=mock_client_instance)  # type: ignore[attr-defined]

        # Ensure the parent packages exist so `from google.cloud import texttospeech` works.
        google_pkg = types.ModuleType("google")
        google_pkg.__path__ = []  # type: ignore[attr-defined]
        google_cloud_pkg = types.ModuleType("google.cloud")
        google_cloud_pkg.__path__ = []  # type: ignore[attr-defined]

        monkeypatch.setitem(sys.modules, "google", google_pkg)
        monkeypatch.setitem(sys.modules, "google.cloud", google_cloud_pkg)
        monkeypatch.setitem(sys.modules, "google.cloud.texttospeech", mock_tts)

        provider = GoogleProvider()  # no factory
        client = provider._get_client()
        assert client is mock_client_instance
        mock_tts.TextToSpeechClient.assert_called_once()

    def test_client_is_cached_across_calls(self):
        """_get_client() returns the same instance on repeated calls."""
        call_count = 0

        def counting_factory():
            nonlocal call_count
            call_count += 1
            return MagicMock()

        provider = GoogleProvider(client_factory=counting_factory)
        c1 = provider._get_client()
        c2 = provider._get_client()
        assert c1 is c2
        assert call_count == 1

    def test_missing_google_package_raises_import_error(self, monkeypatch):
        """GoogleProvider raises ImportError when the SDK is absent and no factory."""
        import sys

        monkeypatch.setitem(sys.modules, "google", None)  # type: ignore[arg-type]
        monkeypatch.setitem(sys.modules, "google.cloud", None)  # type: ignore[arg-type]
        monkeypatch.setitem(sys.modules, "google.cloud.texttospeech", None)  # type: ignore[arg-type]

        provider = GoogleProvider()  # no factory
        with pytest.raises((ImportError, RuntimeError)):
            provider.synthesize('<speak><voice name="he-IL-Chirp3-HD-Achird">x</voice></speak>')

    def test_synthesize_raises_on_missing_voice_name(self):
        """COPILOT-2: SSML without <voice name=...> raises RuntimeError."""
        provider = self._make_provider()
        with pytest.raises(RuntimeError, match="missing a <voice"):
            provider.synthesize("<speak>no voice element</speak>")

    def test_synthesize_raises_on_unexpected_audio_type(self):
        """Covers the audio_content type-check branch."""

        def bad_type_client():
            c = MagicMock()
            resp = MagicMock()
            resp.audio_content = 12345  # not bytes
            c.synthesize_speech.return_value = resp
            return c

        provider = GoogleProvider(client_factory=bad_type_client)
        ssml = '<speak><voice name="he-IL-Chirp3-HD-Achird">x</voice></speak>'
        with pytest.raises(RuntimeError, match="Unexpected audio_content type"):
            provider.synthesize(ssml)


# ---------------------------------------------------------------------------
# SSMLBuilder — supports_style_tags=False
# ---------------------------------------------------------------------------


class TestSSMLBuilderGoogleMode:
    def setup_method(self):
        self.builder = SSMLBuilder()

    def test_no_mstts_namespace_for_google(self):
        utt = UtteranceSpec(
            text="hello",
            voice_id="he-IL-Chirp3-HD-Achird",
            style="angry",
        )
        ssml = self.builder.build_single(utt, supports_style_tags=False)
        assert "mstts" not in ssml
        assert "express-as" not in ssml

    def test_no_express_as_even_with_non_general_style(self):
        utt = UtteranceSpec(text="hi", voice_id="he-IL-Chirp3-HD-Achernar", style="angry")
        ssml = self.builder.build_single(utt, supports_style_tags=False)
        assert "express-as" not in ssml

    def test_style_tags_true_flag_includes_express_as(self):
        """SSMLBuilder with supports_style_tags=True emits express-as (flag-level, not provider)."""
        utt = UtteranceSpec(text="hi", voice_id="he-IL-AvriNeural", style="angry")
        ssml = self.builder.build_single(utt, supports_style_tags=True)
        assert "express-as" in ssml
        assert "mstts" in ssml

    def test_prosody_still_present_in_google_mode(self):
        utt = UtteranceSpec(
            text="hi",
            voice_id="he-IL-Chirp3-HD-Achird",
            style="General",
            rate_multiplier=1.2,
            pitch_delta_st=2.0,
        )
        ssml = self.builder.build_single(utt, supports_style_tags=False)
        assert "prosody" in ssml

    def test_build_from_speaker_config_google(self):
        ssml = self.builder.build_from_speaker_config(
            text="test",
            voice_id="he-IL-Chirp3-HD-Achird",
            style="angry",
            rate_multiplier=1.1,
            supports_style_tags=False,
        )
        assert "mstts" not in ssml
        assert "he-IL-Chirp3-HD-Achird" in ssml


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
        _assert_valid_wav(wav)
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

    def test_azure_ssml_no_express_as_at_intensity_3(self, tmp_path):
        """Azure speaker at intensity 3 must NOT include express-as (M14: disabled for he-IL)."""
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
        assert "express-as" not in captured_ssml[0]

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
        from synthbanshee.tts.azure_provider import AzureProvider

        azure = AzureProvider(sdk_factory=_mock_azure_factory)
        renderer = TTSRenderer(
            providers={"azure": azure},
            cache_dir=tmp_path / "cache",
        )
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

    def test_default_renderer_creates_azure_provider(self, tmp_path):
        """TTSRenderer() with no args lazy-imports and creates AzureProvider.

        Covers renderer.py lines 67, 69.
        """
        from unittest.mock import patch

        mock_azure = MagicMock()
        mock_azure.capabilities = MagicMock()
        mock_azure.capabilities.supports_style_tags = False

        with patch(
            "synthbanshee.tts.azure_provider.AzureProvider", return_value=mock_azure
        ) as mock_cls:
            renderer = TTSRenderer(cache_dir=tmp_path / "cache")
            mock_cls.assert_called_once()
        assert "azure" in renderer._providers

    def test_single_provider_rejects_mismatched_speaker(self, tmp_path):
        """Single-provider mode raises KeyError for unregistered tts_provider.

        When using ``provider=`` (registered as "azure"), a Google speaker
        must fail — no silent fallback.
        """
        from synthbanshee.tts.azure_provider import AzureProvider

        azure = AzureProvider(sdk_factory=_mock_azure_factory)
        renderer = TTSRenderer(provider=azure, cache_dir=tmp_path / "cache")
        speaker = SpeakerConfig.from_yaml(EXAMPLES_DIR / "speaker_AGG_M_30-45_002.yaml")
        assert speaker.tts_provider == "google"
        with pytest.raises(KeyError, match="google"):
            renderer.render_utterance("שלום", speaker, intensity=1)


# ---------------------------------------------------------------------------
# _extract_voice_name helper
# ---------------------------------------------------------------------------


class TestExtractVoiceName:
    def test_extracts_double_quoted(self):
        ssml = '<speak><voice name="he-IL-Chirp3-HD-Achird">text</voice></speak>'
        assert _extract_voice_name(ssml) == "he-IL-Chirp3-HD-Achird"

    def test_extracts_single_quoted(self):
        ssml = "<speak><voice name='he-IL-Chirp3-HD-Achernar'>text</voice></speak>"
        assert _extract_voice_name(ssml) == "he-IL-Chirp3-HD-Achernar"

    def test_returns_empty_when_absent(self):
        assert _extract_voice_name("<speak>no voice</speak>") == ""
