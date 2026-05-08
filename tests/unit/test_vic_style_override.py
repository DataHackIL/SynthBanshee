"""Tests for the VIC high-intensity express-as override in ``TTSRenderer``.

Covers the four-condition gate (env-set + role + intensity + provider-default)
and the per-call ``supports_style_tags=True`` forcing.  Originally added for the
#97 spike (see ``tests/fixtures/issue_97_fearful_ssml.json``); kept generic so
follow-up style spikes (terrified, whispering, …) need no test changes.
"""

from __future__ import annotations

import io
import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from synthbanshee.config.speaker_config import SpeakerConfig
from synthbanshee.tts.azure_provider import AzureProvider
from synthbanshee.tts.renderer import TTSRenderer

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "configs" / "examples"
SPEAKERS_DIR = Path(__file__).parent.parent.parent / "configs" / "speakers"
ENV_VAR = "SYNTHBANSHEE_VIC_STYLE_OVERRIDE"


def _make_wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(struct.pack("<24000h", *([1000] * 24000)))
    return buf.getvalue()


def _capturing_azure_factory(captured: list[str]):
    """Return an Azure SDK factory that appends every SSML it sees to *captured*."""

    def factory(key, region):
        def speak_ssml_async(ssml: str):
            captured.append(ssml)
            mock_result = MagicMock()
            mock_result.audio_data = _make_wav_bytes()
            del mock_result.reason
            future = MagicMock()
            future.get.return_value = mock_result
            return future

        synth = MagicMock()
        synth.speak_ssml_async.side_effect = speak_ssml_async
        return synth

    return factory


def _renderer(tmp_path: Path, captured: list[str]) -> tuple[TTSRenderer, AzureProvider]:
    azure = AzureProvider(sdk_factory=_capturing_azure_factory(captured))
    return TTSRenderer(providers={"azure": azure}, cache_dir=tmp_path / "cache"), azure


class TestVICStyleOverride:
    """The override fires only on (env-set + VIC + I≥4); provider default stays False."""

    def test_env_unset_emits_no_express_as_at_vic_i4(self, tmp_path, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        captured: list[str] = []
        renderer, _ = _renderer(tmp_path, captured)
        speaker = SpeakerConfig.from_yaml(SPEAKERS_DIR / "speaker_VIC_F_25-40_004.yaml")
        renderer.render_utterance("שלום", speaker, intensity=4)
        assert captured, "Azure must have been called"
        assert "express-as" not in captured[0]
        assert "xmlns:mstts" not in captured[0]

    def test_env_set_emits_express_as_at_vic_i4(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "fearful")
        captured: list[str] = []
        renderer, _ = _renderer(tmp_path, captured)
        speaker = SpeakerConfig.from_yaml(SPEAKERS_DIR / "speaker_VIC_F_25-40_004.yaml")
        renderer.render_utterance("שלום", speaker, intensity=4)
        assert captured
        assert 'mstts:express-as style="fearful"' in captured[0]
        assert "xmlns:mstts" in captured[0]

    def test_env_set_emits_express_as_at_vic_i5(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "terrified")
        captured: list[str] = []
        renderer, _ = _renderer(tmp_path, captured)
        speaker = SpeakerConfig.from_yaml(SPEAKERS_DIR / "speaker_VIC_F_25-40_004.yaml")
        renderer.render_utterance("שלום", speaker, intensity=5)
        assert 'mstts:express-as style="terrified"' in captured[0]

    def test_env_set_does_not_fire_below_min_intensity(self, tmp_path, monkeypatch):
        """I=3 is below the gate; baseline path is preserved."""
        monkeypatch.setenv(ENV_VAR, "fearful")
        captured: list[str] = []
        renderer, _ = _renderer(tmp_path, captured)
        speaker = SpeakerConfig.from_yaml(SPEAKERS_DIR / "speaker_VIC_F_25-40_004.yaml")
        renderer.render_utterance("שלום", speaker, intensity=3)
        assert "express-as" not in captured[0]
        assert "xmlns:mstts" not in captured[0]

    def test_env_set_does_not_fire_for_non_vic_role(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "fearful")
        captured: list[str] = []
        renderer, _ = _renderer(tmp_path, captured)
        speaker = SpeakerConfig.from_yaml(SPEAKERS_DIR / "speaker_AGG_M_30-45_003.yaml")
        renderer.render_utterance("שלום", speaker, intensity=5)
        assert "express-as" not in captured[0]
        assert "xmlns:mstts" not in captured[0]

    def test_provider_global_capability_unchanged(self):
        """The override must not flip the provider's default — only force per-call."""
        provider = AzureProvider(sdk_factory=lambda k, r: MagicMock())
        assert provider.capabilities.supports_style_tags is False

    @pytest.mark.parametrize("style", ["fearful", "terrified", "whispering", "sad"])
    def test_arbitrary_styles_route_through(self, tmp_path, monkeypatch, style):
        """Future style spikes need no code changes — the env var routes any value."""
        monkeypatch.setenv(ENV_VAR, style)
        captured: list[str] = []
        renderer, _ = _renderer(tmp_path, captured)
        speaker = SpeakerConfig.from_yaml(SPEAKERS_DIR / "speaker_VIC_F_25-40_004.yaml")
        renderer.render_utterance("שלום", speaker, intensity=4)
        assert f'mstts:express-as style="{style}"' in captured[0]
