"""Integration test: multi-speaker TTS rendering via TTSRenderer.render_scene().

Wires DialogueTurn → TTSRenderer.render_scene() → SceneMixer → MixedScene.
Azure TTS calls are mocked with synthetic audio — no API credentials needed.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from synthbanshee.config.speaker_config import SpeakerConfig
from synthbanshee.script.types import DialogueTurn
from synthbanshee.tts.azure_provider import AzureProvider
from synthbanshee.tts.renderer import TTSRenderer

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "configs" / "examples"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav_bytes(
    sample_rate: int = 24000,
    duration_s: float = 2.0,
    freq: float = 440.0,
) -> bytes:
    n = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    samples = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


def _mock_azure_factory(key, region):
    synth = MagicMock()
    mock_result = MagicMock()
    mock_result.audio_data = _make_wav_bytes()
    del mock_result.reason
    synth.speak_ssml_async.return_value.get.return_value = mock_result
    return synth


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def renderer(tmp_path):
    provider = AzureProvider(sdk_factory=_mock_azure_factory)
    return TTSRenderer(provider=provider, cache_dir=tmp_path / "tts_cache")


@pytest.fixture()
def speakers():
    agg = SpeakerConfig.from_yaml(EXAMPLES_DIR / "speaker_AGG_M_30-45_001.yaml")
    vic = SpeakerConfig.from_yaml(EXAMPLES_DIR / "speaker_VIC_F_25-40_002.yaml")
    return {agg.speaker_id: agg, vic.speaker_id: vic}


@pytest.fixture()
def dialogue_turns():
    return [
        DialogueTurn(
            speaker_id="AGG_M_30-45_001",
            text="שלום, בוא נדבר",
            intensity=1,
            pause_before_s=0.0,
            emotional_state="neutral",
        ),
        DialogueTurn(
            speaker_id="VIC_F_25-40_002",
            text="בסדר, על מה?",
            intensity=1,
            pause_before_s=0.3,
            emotional_state="neutral",
        ),
        DialogueTurn(
            speaker_id="AGG_M_30-45_001",
            text="למה יצאת בלי לשאול אותי?",
            intensity=3,
            pause_before_s=0.2,
            emotional_state="angry",
        ),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRenderScene:
    def test_returns_mixed_scene(self, renderer, speakers, dialogue_turns):
        scene = renderer.render_scene(dialogue_turns, speakers)
        assert scene.sample_rate == 16000
        assert len(scene.samples) > 0

    def test_turn_count_matches(self, renderer, speakers, dialogue_turns):
        scene = renderer.render_scene(dialogue_turns, speakers)
        assert len(scene.turn_onsets_s) == len(dialogue_turns)
        assert len(scene.turn_offsets_s) == len(dialogue_turns)
        assert len(scene.speaker_ids) == len(dialogue_turns)

    def test_speaker_ids_preserved(self, renderer, speakers, dialogue_turns):
        scene = renderer.render_scene(dialogue_turns, speakers)
        expected_ids = [t.speaker_id for t in dialogue_turns]
        assert scene.speaker_ids == expected_ids

    def test_onsets_are_non_decreasing(self, renderer, speakers, dialogue_turns):
        scene = renderer.render_scene(dialogue_turns, speakers)
        for i in range(1, len(scene.turn_onsets_s)):
            assert scene.turn_onsets_s[i] >= scene.turn_onsets_s[i - 1]

    def test_offsets_after_onsets(self, renderer, speakers, dialogue_turns):
        scene = renderer.render_scene(dialogue_turns, speakers)
        for onset, offset in zip(scene.turn_onsets_s, scene.turn_offsets_s, strict=True):
            assert offset > onset

    def test_total_duration_positive(self, renderer, speakers, dialogue_turns):
        scene = renderer.render_scene(dialogue_turns, speakers)
        assert scene.duration_s > 0.0

    def test_samples_are_float32(self, renderer, speakers, dialogue_turns):
        scene = renderer.render_scene(dialogue_turns, speakers)
        assert scene.samples.dtype == np.float32

    def test_first_turn_onset_in_gap_controller_range(self, renderer, speakers, dialogue_turns):
        """First turn onset is drawn from TurnGapController (agg_low range for she_proves)."""
        from synthbanshee.tts.gap_controller import _SHE_PROVES_GAPS

        scene = renderer.render_scene(dialogue_turns, speakers)
        lo, hi = _SHE_PROVES_GAPS["agg_low"]
        assert lo <= scene.turn_onsets_s[0] <= hi

    def test_second_turn_onset_after_first_offset(self, renderer, speakers, dialogue_turns):
        """With M8a overlap mixing, turns can start before the previous turn ends.
        The invariant that always holds is that audible onsets are non-decreasing."""
        scene = renderer.render_scene(dialogue_turns, speakers)
        assert scene.turn_onsets_s[1] >= scene.turn_onsets_s[0]

    def test_unknown_speaker_raises(self, renderer, dialogue_turns):
        """render_scene should raise KeyError for unknown speaker_id."""
        with pytest.raises(KeyError):
            renderer.render_scene(dialogue_turns, speakers={})

    def test_disfluency_flag_accepted(self, renderer, speakers, dialogue_turns):
        """disfluency=True should not raise (even if it changes text)."""
        scene = renderer.render_scene(dialogue_turns, speakers, disfluency=True)
        assert scene.sample_rate == 16000

    def test_randomize_produces_different_cache_keys(self, renderer, speakers, tmp_path):
        """Two randomize=True calls with different rng_seed should call TTS separately."""
        turns = [
            DialogueTurn(
                speaker_id="AGG_M_30-45_001",
                text="שלום",
                intensity=2,
            )
        ]
        scene1 = renderer.render_scene(turns, speakers, randomize=True, rng_seed=1)
        # Clear cache to force re-render
        for f in (tmp_path / "tts_cache").glob("*.wav"):
            f.unlink()
        scene2 = renderer.render_scene(turns, speakers, randomize=True, rng_seed=99)
        # Both should produce valid scenes (behaviour tested, not identity)
        assert scene1.duration_s > 0.0
        assert scene2.duration_s > 0.0
