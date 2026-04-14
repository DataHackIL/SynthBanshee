"""Unit tests for TTS renderer and SSML builder (Phase 0.3).

Azure TTS is mocked; no real API credentials are required.
"""

from __future__ import annotations

import io
import struct
import sys
import wave
from pathlib import Path
from unittest.mock import MagicMock

from synthbanshee.config.speaker_config import SpeakerConfig
from synthbanshee.tts.azure_provider import AzureProvider
from synthbanshee.tts.renderer import TTSRenderer
from synthbanshee.tts.ssml_builder import (
    SSMLBuilder,
    UtteranceSpec,
    _rate_to_string,
    _semitones_to_percent,
)

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "configs" / "examples"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav_bytes(
    sample_rate: int = 24000, duration_s: float = 1.0, amplitude: float = 0.3
) -> bytes:
    """Generate minimal valid WAV bytes (silence) for mock TTS responses."""
    n_samples = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        # Write near-silence (avoid all-zero to test normalization)
        data = struct.pack(f"<{n_samples}h", *([int(amplitude * 32767)] * n_samples))
        w.writeframes(data)
    return buf.getvalue()


def _mock_azure_factory(key, region):
    """Return a mock synthesizer that returns fake WAV bytes."""
    synth = MagicMock()
    mock_result = MagicMock()
    mock_result.audio_data = _make_wav_bytes()
    # Set reason to avoid import of azure SDK in mock mode
    del mock_result.reason  # remove reason so hasattr check fails
    synth.speak_ssml_async.return_value.get.return_value = mock_result
    return synth


# ---------------------------------------------------------------------------
# SSMLBuilder tests
# ---------------------------------------------------------------------------


class TestSSMLBuilder:
    def setup_method(self):
        self.builder = SSMLBuilder()

    def test_single_utterance_structure(self):
        utt = UtteranceSpec(
            text="hello",
            voice_id="he-IL-AvriNeural",
            style="angry",
            rate_multiplier=1.2,
            pitch_delta_st=3.0,
            volume_delta_db=5.0,
        )
        ssml = self.builder.build_single(utt)
        assert '<?xml version="1.0"' in ssml
        assert "he-IL-AvriNeural" in ssml
        assert 'style="angry"' in ssml
        assert "prosody" in ssml

    def test_general_style_no_express_as(self):
        utt = UtteranceSpec(
            text="hello",
            voice_id="he-IL-AvriNeural",
            style="General",
        )
        ssml = self.builder.build_single(utt)
        assert "express-as" not in ssml

    def test_no_prosody_when_defaults(self):
        utt = UtteranceSpec(
            text="hello",
            voice_id="he-IL-AvriNeural",
            rate_multiplier=1.0,
            pitch_delta_st=0.0,
            volume_delta_db=0.0,
        )
        ssml = self.builder.build_single(utt)
        assert "prosody" not in ssml

    def test_multi_utterance(self):
        utts = [
            UtteranceSpec(text="first", voice_id="he-IL-AvriNeural"),
            UtteranceSpec(text="second", voice_id="he-IL-HilaNeural"),
        ]
        ssml = self.builder.build_multi(utts)
        assert "he-IL-AvriNeural" in ssml
        assert "he-IL-HilaNeural" in ssml

    def test_rate_to_string(self):
        assert _rate_to_string(1.0) == "+0%"
        assert _rate_to_string(1.2) == "+20%"
        assert _rate_to_string(0.8) == "-20%"

    def test_semitones_to_percent(self):
        assert "+" in _semitones_to_percent(2.0)
        assert "-" in _semitones_to_percent(-2.0)

    def test_xml_is_well_formed(self):
        import xml.etree.ElementTree as ET

        utt = UtteranceSpec(
            text="test content",
            voice_id="he-IL-AvriNeural",
            style="angry",
            rate_multiplier=1.1,
        )
        ssml = self.builder.build_single(utt)
        # Strip XML declaration for ElementTree
        ssml_body = ssml.split("\n", 1)[1] if ssml.startswith("<?xml") else ssml
        ET.fromstring(ssml_body)  # Should not raise


# ---------------------------------------------------------------------------
# AzureProvider tests (mocked)
# ---------------------------------------------------------------------------


class TestAzureProvider:
    def test_synthesize_returns_bytes(self):
        provider = AzureProvider(sdk_factory=_mock_azure_factory)
        builder = SSMLBuilder()
        ssml = builder.build_from_speaker_config(
            text="hello",
            voice_id="he-IL-AvriNeural",
            style="General",
        )
        result = provider.synthesize(ssml)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_is_configured_with_env(self, monkeypatch):
        monkeypatch.setenv("AZURE_TTS_KEY", "test_key")
        monkeypatch.setenv("AZURE_TTS_REGION", "eastus")
        provider = AzureProvider()
        assert provider.is_configured()

    def test_not_configured_without_key(self, monkeypatch):
        monkeypatch.delenv("AZURE_TTS_KEY", raising=False)
        provider = AzureProvider(subscription_key="", region="eastus")
        assert not provider.is_configured()

    def test_get_synthesizer_passes_audio_config_none(self, monkeypatch):
        """Regression: _get_synthesizer must use audio_config=None for headless synthesis.

        Using AudioOutputConfig(use_default_speaker=False) triggers
        "default speaker needs to be explicitly activated" on macOS.
        """
        import types

        mock_synthesizer_cls = MagicMock()
        mock_speech_config = MagicMock()

        # Build a proper module object so the import machinery accepts it.
        fake_sdk = types.ModuleType("azure.cognitiveservices.speech")
        fake_sdk.SpeechConfig = MagicMock(return_value=mock_speech_config)
        fake_sdk.SpeechSynthesizer = mock_synthesizer_cls
        fake_sdk.SpeechSynthesisOutputFormat = MagicMock()
        fake_sdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm = "mock_format"

        # Patch the full dotted import path — parent packages must also be present
        # or the import statement raises ImportError before reaching the leaf module.
        monkeypatch.setitem(sys.modules, "azure", types.ModuleType("azure"))
        monkeypatch.setitem(
            sys.modules, "azure.cognitiveservices", types.ModuleType("azure.cognitiveservices")
        )
        monkeypatch.setitem(sys.modules, "azure.cognitiveservices.speech", fake_sdk)

        provider = AzureProvider(subscription_key="key", region="eastus")
        provider._get_synthesizer()

        mock_synthesizer_cls.assert_called_once_with(
            speech_config=mock_speech_config,
            audio_config=None,
        )


# ---------------------------------------------------------------------------
# TTSRenderer tests (mocked)
# ---------------------------------------------------------------------------


class TestTTSRenderer:
    def setup_method(self):
        provider = AzureProvider(sdk_factory=_mock_azure_factory)
        self.renderer = None
        self._provider = provider

    def _make_renderer(self, tmp_path):
        return TTSRenderer(provider=self._provider, cache_dir=tmp_path / "cache")

    def test_render_utterance_returns_bytes(self, tmp_path):
        renderer = self._make_renderer(tmp_path)
        speaker = SpeakerConfig.from_yaml(EXAMPLES_DIR / "speaker_AGG_M_30-45_001.yaml")
        wav_bytes, key, hit = renderer.render_utterance("hello", speaker, intensity=3)
        assert isinstance(wav_bytes, bytes)
        assert len(key) == 64  # SHA-256 hex digest
        assert hit is False  # first call — not a cache hit

    def test_cache_hit_on_second_call(self, tmp_path):
        renderer = self._make_renderer(tmp_path)
        speaker = SpeakerConfig.from_yaml(EXAMPLES_DIR / "speaker_AGG_M_30-45_001.yaml")

        wav1, key1, hit1 = renderer.render_utterance("hello", speaker, intensity=1)
        wav2, key2, hit2 = renderer.render_utterance("hello", speaker, intensity=1)
        assert key1 == key2
        assert wav1 == wav2
        assert hit1 is False  # first call — miss
        assert hit2 is True  # second call — cache hit

    def test_different_text_different_key(self, tmp_path):
        renderer = self._make_renderer(tmp_path)
        speaker = SpeakerConfig.from_yaml(EXAMPLES_DIR / "speaker_AGG_M_30-45_001.yaml")

        _, key1, _ = renderer.render_utterance("hello", speaker, intensity=1)
        _, key2, _ = renderer.render_utterance("world", speaker, intensity=1)
        assert key1 != key2

    def test_render_to_file(self, tmp_path):
        renderer = self._make_renderer(tmp_path)
        speaker = SpeakerConfig.from_yaml(EXAMPLES_DIR / "speaker_AGG_M_30-45_001.yaml")
        out = tmp_path / "output.wav"
        result_path = renderer.render_utterance_to_file("hello", speaker, out, intensity=1)
        assert result_path == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_render_scene_verbose_log(self, tmp_path):
        """verbose_log callable receives per-turn status messages during render_scene."""
        from synthbanshee.script.types import DialogueTurn

        renderer = self._make_renderer(tmp_path)
        speaker = SpeakerConfig.from_yaml(EXAMPLES_DIR / "speaker_AGG_M_30-45_001.yaml")
        turns = [DialogueTurn(speaker_id="AGG_M_30-45_001", text="שלום", intensity=1)]
        speakers = {"AGG_M_30-45_001": speaker}

        log_messages: list[str] = []
        renderer.render_scene(turns, speakers, verbose_log=log_messages.append)

        assert len(log_messages) >= 1
        assert any("turn" in m for m in log_messages)
