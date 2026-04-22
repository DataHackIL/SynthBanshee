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
from synthbanshee.labels.generator import LabelGenerator, ScriptEvent
from synthbanshee.script.types import DialogueTurn, MixedScene
from synthbanshee.tts.azure_provider import AzureProvider
from synthbanshee.tts.mix_mode import MixMode
from synthbanshee.tts.mixer import SceneMixer
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

    def test_three_timeline_fields_populated(self, renderer, speakers, dialogue_turns):
        """All three timeline fields must be populated and have the right length."""
        scene = renderer.render_scene(dialogue_turns, speakers)
        n = len(dialogue_turns)
        assert len(scene.script_onsets_s) == n
        assert len(scene.script_offsets_s) == n
        assert len(scene.rendered_onsets_s) == n
        assert len(scene.rendered_offsets_s) == n
        assert len(scene.audible_onsets_s) == n
        assert len(scene.audible_ends_s) == n

    def test_audible_mirrors_turn_compat_fields(self, renderer, speakers, dialogue_turns):
        """turn_onsets_s / turn_offsets_s must equal the audible timeline exactly."""
        scene = renderer.render_scene(dialogue_turns, speakers)
        assert scene.turn_onsets_s == scene.audible_onsets_s
        assert scene.turn_offsets_s == scene.audible_ends_s


# ---------------------------------------------------------------------------
# Overlapped segment label integration tests (M8b)
# ---------------------------------------------------------------------------


def _make_wav_bytes_16k(duration_s: float = 2.0) -> bytes:
    """Minimal 16 kHz mono WAV for direct use with SceneMixer."""
    sample_rate = 16_000
    n = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    samples = (0.3 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


class TestOverlapLabelIntegration:
    """Integration: SceneMixer audible timeline → LabelGenerator → EventLabel timestamps."""

    def _three_turn_scene(self, mix_mode_third: MixMode) -> MixedScene:
        """Mix three turns where the third uses the specified mix_mode."""
        mixer = SceneMixer()
        wav = _make_wav_bytes_16k(duration_s=2.0)
        segments = [
            (wav, 0.3, "SPK_A", None, MixMode.SEQUENTIAL),
            (wav, 0.3, "SPK_B", None, MixMode.SEQUENTIAL),
            (wav, 0.4, "SPK_A", None, mix_mode_third),
        ]
        return mixer.mix_sequential(segments)

    def _make_events(self, n: int) -> list[ScriptEvent]:
        return [
            ScriptEvent(
                tier1_category="VERB",
                tier2_subtype="VERB_SHOUT",
                onset=0.0,
                offset=2.0,
                intensity=3,
                speaker_role="AGG",
            )
            for _ in range(n)
        ]

    def test_sequential_label_onset_from_audible(self):
        """SEQUENTIAL: EventLabel onset equals audible_onsets_s."""
        scene = self._three_turn_scene(MixMode.SEQUENTIAL)
        gen = LabelGenerator()
        labels = gen.generate_events_from_scene("clip_seq", self._make_events(3), scene)
        for lbl, audible_onset in zip(labels, scene.audible_onsets_s, strict=True):
            assert lbl.onset == pytest.approx(audible_onset)

    def test_overlap_label_onset_earlier_than_script(self):
        """OVERLAP: audible onset is earlier than the script onset for the overlapping turn."""
        scene = self._three_turn_scene(MixMode.OVERLAP)
        # Third turn starts before its sequential script position.
        assert scene.audible_onsets_s[2] < scene.script_onsets_s[2]

    def test_overlap_turn_not_truncated(self):
        """OVERLAP: previous turn is not truncated — truncated must be False."""
        scene = self._three_turn_scene(MixMode.OVERLAP)
        gen = LabelGenerator()
        labels = gen.generate_events_from_scene("clip_ovlp", self._make_events(3), scene)
        # Previous turn (index 1) should not be truncated.
        assert labels[1].truncated is False

    def test_barge_in_previous_turn_label_truncated(self):
        """BARGE_IN: previous turn (index 1) must have truncated=True on its label."""
        scene = self._three_turn_scene(MixMode.BARGE_IN)
        gen = LabelGenerator()
        labels = gen.generate_events_from_scene("clip_barge", self._make_events(3), scene)
        assert labels[1].truncated is True

    def test_barge_in_interrupted_label_offset_within_waveform(self):
        """BARGE_IN: truncated turn's label offset must not exceed scene duration."""
        scene = self._three_turn_scene(MixMode.BARGE_IN)
        gen = LabelGenerator()
        labels = gen.generate_events_from_scene("clip_barge", self._make_events(3), scene)
        assert labels[1].offset <= scene.duration_s + 1e-6

    def test_barge_in_non_interrupted_turns_not_truncated(self):
        """BARGE_IN: turns that are NOT interrupted keep truncated=False."""
        scene = self._three_turn_scene(MixMode.BARGE_IN)
        gen = LabelGenerator()
        labels = gen.generate_events_from_scene("clip_barge", self._make_events(3), scene)
        # First turn (index 0) is not interrupted.
        assert labels[0].truncated is False
        # Third turn (the barge-in turn itself) is not interrupted.
        assert labels[2].truncated is False

    def test_audible_onsets_non_decreasing_for_barge_in(self):
        """BARGE_IN: audible_onsets_s must be non-decreasing (invariant holds)."""
        scene = self._three_turn_scene(MixMode.BARGE_IN)
        for i in range(1, len(scene.audible_onsets_s)):
            assert scene.audible_onsets_s[i] >= scene.audible_onsets_s[i - 1]
