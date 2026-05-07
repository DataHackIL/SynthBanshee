"""Tests for the #87 effective-prosody runtime cap.

The cap defends against the failure mode where M7 SpeakerState drift
compounds with #51's M15 style_map values to push effective pitch past
+2.5 st (~+15 %) and rate past 1.30× at high-intensity turns — a range
that simultaneously sounds cartoonish to listeners (May-3 listening test)
and trips Whisper-large-v3's silence-detection heuristic, producing the
length-ratio collapse fingerprint documented on PR #87.

These tests exercise the cap in three places:

  1. ``_apply_effective_prosody_cap`` directly — unit-level; no Azure.
  2. ``TTSRenderer.render_utterance`` with a mocked Azure provider — verifies
     the cap fires when state + style would exceed thresholds, and that the
     resulting ``out_cap_events`` list is populated.
  3. ``TTSRenderer.render_scene`` end-to-end — verifies cap events propagate
     to ``DialogueTurn.effective_prosody_caps`` so cli.py can roll them up
     into ``ClipMetadata.generation_metadata.effective_prosody_caps``.
"""

from __future__ import annotations

import io
import struct
import wave
from unittest.mock import MagicMock

from synthbanshee.config.speaker_config import (
    DisfluencyProfile,
    ProsodyBaseline,
    SpeakerConfig,
    StyleEntry,
)
from synthbanshee.script.types import DialogueTurn
from synthbanshee.tts.azure_provider import AzureProvider
from synthbanshee.tts.renderer import (
    _EFFECTIVE_PITCH_MAX_ST,
    _EFFECTIVE_PITCH_MIN_ST,
    _EFFECTIVE_RATE_MAX,
    _EFFECTIVE_RATE_MIN,
    TTSRenderer,
    _apply_effective_prosody_cap,
)
from synthbanshee.tts.speaker_state import SpeakerState


def _make_wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(struct.pack("<24000h", *([0x2000] * 24000)))
    return buf.getvalue()


def _mock_azure_factory(_key, _region):
    synth = MagicMock()
    result = MagicMock()
    result.audio_data = _make_wav_bytes()
    del result.reason
    synth.speak_ssml_async.return_value.get.return_value = result
    return synth


# ---------------------------------------------------------------------------
# 1. Pure helper unit tests
# ---------------------------------------------------------------------------


class TestApplyEffectiveProsodyCapHelper:
    def test_in_range_values_pass_through_unchanged(self):
        events: list[dict] = []
        rate, pitch, vol = _apply_effective_prosody_cap(1.0, 0.0, 0.0, out_cap_events=events)
        assert (rate, pitch, vol) == (1.0, 0.0, 0.0)
        assert events == []

    def test_pitch_above_max_clamps_and_records_event(self):
        events: list[dict] = []
        rate, pitch, vol = _apply_effective_prosody_cap(
            1.0, _EFFECTIVE_PITCH_MAX_ST + 1.5, 0.0, out_cap_events=events
        )
        assert pitch == _EFFECTIVE_PITCH_MAX_ST
        assert len(events) == 1
        assert events[0]["dim"] == "pitch"
        assert events[0]["pre_cap"] == round(_EFFECTIVE_PITCH_MAX_ST + 1.5, 4)
        assert events[0]["post_cap"] == _EFFECTIVE_PITCH_MAX_ST

    def test_pitch_below_min_clamps_and_records_event(self):
        events: list[dict] = []
        _, pitch, _ = _apply_effective_prosody_cap(
            1.0, _EFFECTIVE_PITCH_MIN_ST - 2.0, 0.0, out_cap_events=events
        )
        assert pitch == _EFFECTIVE_PITCH_MIN_ST
        assert len(events) == 1
        assert events[0]["dim"] == "pitch"

    def test_rate_above_max_clamps_and_records_event(self):
        events: list[dict] = []
        rate, _, _ = _apply_effective_prosody_cap(
            _EFFECTIVE_RATE_MAX + 0.15, 0.0, 0.0, out_cap_events=events
        )
        assert rate == _EFFECTIVE_RATE_MAX
        assert len(events) == 1
        assert events[0]["dim"] == "rate"

    def test_rate_below_min_clamps_and_records_event(self):
        events: list[dict] = []
        rate, _, _ = _apply_effective_prosody_cap(
            _EFFECTIVE_RATE_MIN - 0.10, 0.0, 0.0, out_cap_events=events
        )
        assert rate == _EFFECTIVE_RATE_MIN
        assert len(events) == 1
        assert events[0]["dim"] == "rate"

    def test_volume_is_never_capped(self):
        # Per #82's lever probe, Whisper's log-mel internally normalizes
        # loudness — so volume is intentionally NOT in the effective cap.
        events: list[dict] = []
        _, _, vol = _apply_effective_prosody_cap(1.0, 0.0, 999.0, out_cap_events=events)
        assert vol == 999.0
        assert events == []

    def test_multiple_dimensions_record_separate_events(self):
        events: list[dict] = []
        rate, pitch, _ = _apply_effective_prosody_cap(
            _EFFECTIVE_RATE_MAX + 0.1,
            _EFFECTIVE_PITCH_MAX_ST + 0.5,
            0.0,
            out_cap_events=events,
        )
        assert rate == _EFFECTIVE_RATE_MAX
        assert pitch == _EFFECTIVE_PITCH_MAX_ST
        assert {ev["dim"] for ev in events} == {"rate", "pitch"}

    def test_out_cap_events_none_is_silent(self):
        # A caller that does not need the event log can pass None and the
        # cap still fires; nothing is appended (and nothing crashes).
        rate, pitch, vol = _apply_effective_prosody_cap(
            _EFFECTIVE_RATE_MAX + 1.0, _EFFECTIVE_PITCH_MAX_ST + 1.0, 0.0, out_cap_events=None
        )
        assert rate == _EFFECTIVE_RATE_MAX
        assert pitch == _EFFECTIVE_PITCH_MAX_ST

    def test_rate_floor_anchored_to_0_95_for_issue_91(self):
        # #91 R experiment lifts the rate floor 0.85 → 0.95 to test whether
        # VIC's I4/I5 slowdown is what drives the residual Whisper WER gap on
        # sp_it_a_0001 after PR #90.  Anchored as a literal so a silent revert
        # back to 0.85 is caught here (paired listening test is the merge gate
        # for any further movement on this constant).
        assert _EFFECTIVE_RATE_MIN == 0.95


# ---------------------------------------------------------------------------
# 2. render_utterance integration (mocked Azure)
# ---------------------------------------------------------------------------


def _make_high_drift_speaker() -> SpeakerConfig:
    """Speaker with style values that, combined with drifted state, exceed the cap.

    Mirrors the actual sp_it_a_0001 turn 12 (I5 AGG_M) effective prosody we
    measured on the post-#86 main: rate 1.329, pitch +2.90 st, volume +15.8 dB.
    """
    style_map = {i: StyleEntry(style="General", rate_multiplier=1.0) for i in range(1, 6)}
    style_map[5] = StyleEntry(
        style="General",
        rate_multiplier=1.14,
        pitch_delta_st=2.0,
        volume_delta_db=13.0,
    )
    return SpeakerConfig(
        speaker_id="TEST_M_30-45_001",
        role="AGG",
        gender="male",
        age_range="30-45",
        context="she_proves",
        tts_voice_id="he-IL-AvriNeural",
        tts_provider="azure",
        voice_family="he-IL-AvriNeural",
        prosody_baseline=ProsodyBaseline(rate=1.05, pitch_hz=115, volume_db=0),
        style_map=style_map,
        disfluency=DisfluencyProfile(),
        split="train",
    )


class TestRenderUtteranceCapIntegration:
    def setup_method(self):
        self.provider = AzureProvider(sdk_factory=_mock_azure_factory)

    def _make_renderer(self, tmp_path):
        return TTSRenderer(provider=self.provider, cache_dir=tmp_path / "cache")

    def test_cap_fires_on_drifted_high_intensity_turn(self, tmp_path):
        # Construct a SpeakerState that looks like late-arc accumulated drift
        # for an AGG speaker at I5 — pushes the effective pitch past +2.0 st.
        renderer = self._make_renderer(tmp_path)
        speaker = _make_high_drift_speaker()
        state = SpeakerState(
            rate_offset=1.10,  # → 1.14 * 1.05 * 1.10 = 1.317 → caps to 1.20
            pitch_offset_st=1.3,  # → 2.0 + 1.3 = 3.3 → caps to 2.0
            volume_offset_db=4.0,
        )
        events: list[dict] = []
        renderer.render_utterance(
            "test text",
            speaker,
            intensity=5,
            speaker_state=state,
            out_cap_events=events,
        )
        dims = {ev["dim"] for ev in events}
        # Both rate and pitch should have been clamped.
        assert "rate" in dims
        assert "pitch" in dims
        # No volume event — by design, volume is not capped at this layer.
        assert "volume" not in dims

    def test_cap_does_not_fire_on_neutral_low_intensity_turn(self, tmp_path):
        renderer = self._make_renderer(tmp_path)
        speaker = _make_high_drift_speaker()
        events: list[dict] = []
        renderer.render_utterance(
            "test text",
            speaker,
            intensity=1,  # I1 = base style_map default, no drift
            speaker_state=SpeakerState(),
            out_cap_events=events,
        )
        assert events == []


# ---------------------------------------------------------------------------
# 3. render_scene → DialogueTurn propagation
# ---------------------------------------------------------------------------


class TestRenderSceneCapPropagation:
    def setup_method(self):
        self.provider = AzureProvider(sdk_factory=_mock_azure_factory)

    def test_cap_events_attached_to_turn_after_render_scene(self, tmp_path):
        renderer = TTSRenderer(provider=self.provider, cache_dir=tmp_path / "cache")
        speaker = _make_high_drift_speaker()

        # Build a dialogue arc that drifts SpeakerState through high-intensity
        # turns so the cap fires by the last turn.  Mirrors sp_it_a_0001's
        # arc shape ([1,2,3,4,5,...]) on a smaller scale.
        turns = [
            DialogueTurn(speaker_id=speaker.speaker_id, text="t", intensity=i)
            for i in (1, 3, 4, 5, 5)
        ]
        renderer.render_scene(
            turns,
            speakers={speaker.speaker_id: speaker},
            randomize=False,
            quality_gates=False,
        )

        # At least one of the late-arc I5 turns should record a cap event.
        late_turn_caps = [ev for t in turns[-2:] for ev in t.effective_prosody_caps]
        assert late_turn_caps, (
            "Expected cap activations on late high-intensity turns; got none. "
            f"All cap events per turn: {[t.effective_prosody_caps for t in turns]}"
        )
        # Every recorded event must have the documented schema.
        for ev in late_turn_caps:
            assert ev["dim"] in {"rate", "pitch"}
            assert "pre_cap" in ev and "post_cap" in ev
