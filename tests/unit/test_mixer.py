"""Unit tests for synthbanshee.tts.mixer.SceneMixer."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from synthbanshee.tts.mixer import _TARGET_SR, MixMode, SceneMixer, _apply_rms_gain

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sine_wav_bytes(
    freq: float = 440.0,
    duration_s: float = 1.0,
    sample_rate: int = 16000,
    amplitude: float = 0.3,
) -> bytes:
    """Return WAV bytes for a mono sine wave at the given frequency."""
    n = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    samples = (amplitude * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


def _stereo_wav_bytes(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Return WAV bytes for a stereo file (to test downmix)."""
    n = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    ch1 = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    ch2 = (0.3 * np.sin(2 * np.pi * 880 * t) * 32767).astype(np.int16)
    stereo = np.stack([ch1, ch2], axis=1)
    buf = io.BytesIO()
    with wave.open(buf, "w") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(stereo.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSceneMixer:
    def test_empty_segments_returns_zero_length(self):
        mixer = SceneMixer()
        result = mixer.mix_sequential([])
        assert result.duration_s == 0.0
        assert len(result.samples) == 0
        assert result.turn_onsets_s == []
        assert result.turn_offsets_s == []

    def test_single_segment_no_pause(self):
        mixer = SceneMixer()
        wav = _sine_wav_bytes(duration_s=1.0, sample_rate=_TARGET_SR)
        result = mixer.mix_sequential([(wav, 0.0, "SPK_001", None, MixMode.SEQUENTIAL)])

        assert result.sample_rate == _TARGET_SR
        assert len(result.turn_onsets_s) == 1
        assert result.turn_onsets_s[0] == pytest.approx(0.0)
        assert result.turn_offsets_s[0] == pytest.approx(1.0, abs=0.05)
        assert result.duration_s == pytest.approx(1.0, abs=0.05)
        assert result.speaker_ids == ["SPK_001"]

    def test_pause_shifts_onset(self):
        mixer = SceneMixer()
        wav = _sine_wav_bytes(duration_s=0.5, sample_rate=_TARGET_SR)
        pause_s = 0.3
        result = mixer.mix_sequential([(wav, pause_s, "SPK_001", None, MixMode.SEQUENTIAL)])

        assert result.turn_onsets_s[0] == pytest.approx(pause_s, abs=0.01)
        assert result.duration_s == pytest.approx(pause_s + 0.5, abs=0.05)

    def test_two_segments_sequential(self):
        mixer = SceneMixer()
        wav1 = _sine_wav_bytes(freq=440, duration_s=1.0, sample_rate=_TARGET_SR)
        wav2 = _sine_wav_bytes(freq=880, duration_s=0.5, sample_rate=_TARGET_SR)
        result = mixer.mix_sequential(
            [
                (wav1, 0.0, "SPK_A", None, MixMode.SEQUENTIAL),
                (wav2, 0.2, "SPK_B", None, MixMode.SEQUENTIAL),
            ]
        )

        assert len(result.turn_onsets_s) == 2
        # Second turn onset = duration of first + pause
        expected_onset2 = result.turn_offsets_s[0] + 0.2
        assert result.turn_onsets_s[1] == pytest.approx(expected_onset2, abs=0.05)
        assert result.speaker_ids == ["SPK_A", "SPK_B"]

    def test_total_duration_matches_samples(self):
        mixer = SceneMixer()
        segments = [
            (
                _sine_wav_bytes(duration_s=0.8, sample_rate=_TARGET_SR),
                0.1,
                "S1",
                None,
                MixMode.SEQUENTIAL,
            ),
            (
                _sine_wav_bytes(duration_s=0.6, sample_rate=_TARGET_SR),
                0.2,
                "S2",
                None,
                MixMode.SEQUENTIAL,
            ),
            (
                _sine_wav_bytes(duration_s=0.4, sample_rate=_TARGET_SR),
                0.15,
                "S3",
                None,
                MixMode.SEQUENTIAL,
            ),
        ]
        result = mixer.mix_sequential(segments)
        computed_duration = len(result.samples) / result.sample_rate
        assert result.duration_s == pytest.approx(computed_duration, abs=1e-4)

    def test_resamples_24k_to_16k(self):
        """Mixer should downsample 24 kHz input to 16 kHz output."""
        mixer = SceneMixer()
        wav_24k = _sine_wav_bytes(duration_s=0.5, sample_rate=24000)
        result = mixer.mix_sequential([(wav_24k, 0.0, "SPK_001", None, MixMode.SEQUENTIAL)])

        assert result.sample_rate == _TARGET_SR
        # Duration should still be approximately 0.5 s
        assert result.duration_s == pytest.approx(0.5, abs=0.05)

    def test_downmixes_stereo_to_mono(self):
        mixer = SceneMixer()
        stereo_wav = _stereo_wav_bytes(duration_s=1.0, sample_rate=_TARGET_SR)
        result = mixer.mix_sequential([(stereo_wav, 0.0, "SPK_001", None, MixMode.SEQUENTIAL)])

        assert result.samples.ndim == 1
        assert result.duration_s == pytest.approx(1.0, abs=0.05)

    def test_output_samples_are_float32(self):
        mixer = SceneMixer()
        wav = _sine_wav_bytes(duration_s=0.5, sample_rate=_TARGET_SR)
        result = mixer.mix_sequential([(wav, 0.0, "SPK_001", None, MixMode.SEQUENTIAL)])
        assert result.samples.dtype == np.float32

    def test_offsets_greater_than_onsets(self):
        mixer = SceneMixer()
        segments = [
            (
                _sine_wav_bytes(duration_s=0.5, sample_rate=_TARGET_SR),
                0.1,
                "S1",
                None,
                MixMode.SEQUENTIAL,
            ),
            (
                _sine_wav_bytes(duration_s=0.5, sample_rate=_TARGET_SR),
                0.2,
                "S2",
                None,
                MixMode.SEQUENTIAL,
            ),
        ]
        result = mixer.mix_sequential(segments)
        for onset, offset in zip(result.turn_onsets_s, result.turn_offsets_s, strict=True):
            assert offset > onset


class TestApplyRmsGain:
    """Tests for _apply_rms_gain (M3 per-turn gain helper)."""

    def _rms_dbfs(self, arr: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(arr**2)))
        return 20.0 * np.log10(rms)

    def test_gain_reaches_target(self):
        """Output RMS should match the requested target within 0.5 dB."""
        n = 16000
        t = np.linspace(0, 1.0, n, endpoint=False)
        mono = (0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        target_dbfs = -20.0
        result = _apply_rms_gain(mono, target_dbfs)
        assert self._rms_dbfs(result) == pytest.approx(target_dbfs, abs=0.5)

    def test_louder_target_amplifies(self):
        """Requesting a higher RMS target produces higher amplitude."""
        n = 8000
        t = np.linspace(0, 0.5, n, endpoint=False)
        mono = (0.02 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        quiet = _apply_rms_gain(mono, -30.0)
        loud = _apply_rms_gain(mono, -15.0)
        assert self._rms_dbfs(loud) > self._rms_dbfs(quiet) + 10.0

    def test_silence_returned_unchanged(self):
        """Near-silent frames (RMS < 1e-4) must not be amplified."""
        silence = np.zeros(8000, dtype=np.float32)
        result = _apply_rms_gain(silence, -20.0)
        assert np.all(result == 0.0)

    def test_output_is_float32(self):
        mono = (0.1 * np.ones(4000)).astype(np.float32)
        result = _apply_rms_gain(mono, -25.0)
        assert result.dtype == np.float32


class TestRmsGainInMixer:
    """Integration: rms_target_dbfs wired through mix_sequential."""

    def _rms_dbfs(self, arr: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(arr**2)))
        return 20.0 * np.log10(rms)

    def test_none_target_leaves_level_unchanged(self):
        """Passing None for rms_target_dbfs must not alter the signal level."""
        wav = _sine_wav_bytes(amplitude=0.1, duration_s=1.0, sample_rate=_TARGET_SR)
        mixer = SceneMixer()
        result = mixer.mix_sequential([(wav, 0.0, "SPK", None, MixMode.SEQUENTIAL)])
        # RMS of 0.1-amplitude sine = 0.1/sqrt(2) ≈ −23 dBFS; allow ±2 dB
        assert self._rms_dbfs(result.samples) == pytest.approx(-23.0, abs=2.0)

    def test_rms_target_applied(self):
        """Segment with rms_target_dbfs should reach the requested level."""
        wav = _sine_wav_bytes(amplitude=0.02, duration_s=1.0, sample_rate=_TARGET_SR)
        mixer = SceneMixer()
        target = -20.0
        result = mixer.mix_sequential([(wav, 0.0, "SPK", target, MixMode.SEQUENTIAL)])
        assert self._rms_dbfs(result.samples) == pytest.approx(target, abs=0.5)

    def test_escalation_across_two_segments(self):
        """Louder RMS target on second segment produces measurably higher RMS."""
        wav_quiet = _sine_wav_bytes(amplitude=0.02, duration_s=0.5, sample_rate=_TARGET_SR)
        wav_loud = _sine_wav_bytes(amplitude=0.02, duration_s=0.5, sample_rate=_TARGET_SR)
        mixer = SceneMixer()
        result = mixer.mix_sequential(
            [
                (wav_quiet, 0.0, "AGG", -28.0, MixMode.SEQUENTIAL),
                (wav_loud, 0.0, "AGG", -15.0, MixMode.SEQUENTIAL),
            ]
        )
        # Verify combined duration is correct
        assert result.duration_s == pytest.approx(1.0, abs=0.05)
        # Extract the two halves and check RMS ordering
        half = len(result.samples) // 2
        rms_q = self._rms_dbfs(result.samples[:half])
        rms_l = self._rms_dbfs(result.samples[half:])
        assert rms_l > rms_q + 8.0  # ≥ 8 dB gap matches the M3 spec target


# ---------------------------------------------------------------------------
# M8a: Overlap and BARGE_IN mixing tests
# ---------------------------------------------------------------------------


class TestOverlapMixing:
    """Tests for OVERLAP and BARGE_IN MixMode placement (M8a §4.6)."""

    def test_overlap_onset_earlier_than_sequential(self):
        """OVERLAP onset must be earlier than the end of the previous segment."""
        mixer = SceneMixer()
        dur = 1.0
        overlap_s = 0.3
        wav1 = _sine_wav_bytes(freq=440, duration_s=dur, sample_rate=_TARGET_SR)
        wav2 = _sine_wav_bytes(freq=880, duration_s=0.5, sample_rate=_TARGET_SR)
        result = mixer.mix_sequential(
            [
                (wav1, 0.0, "AGG", None, MixMode.SEQUENTIAL),
                (wav2, overlap_s, "VIC", None, MixMode.OVERLAP),
            ]
        )
        # rendered onset of turn 2 should be dur - overlap_s
        assert result.rendered_onsets_s[1] == pytest.approx(dur - overlap_s, abs=0.02)

    def test_overlap_total_duration_shorter_than_sequential(self):
        """Overlapping turns produce a shorter output than strict sequential."""
        mixer = SceneMixer()
        dur = 1.0
        overlap_s = 0.3
        wav1 = _sine_wav_bytes(duration_s=dur, sample_rate=_TARGET_SR)
        wav2 = _sine_wav_bytes(duration_s=dur, sample_rate=_TARGET_SR)

        seq_result = mixer.mix_sequential(
            [
                (wav1, 0.0, "A", None, MixMode.SEQUENTIAL),
                (wav2, 0.0, "B", None, MixMode.SEQUENTIAL),
            ]
        )
        olap_result = mixer.mix_sequential(
            [
                (wav1, 0.0, "A", None, MixMode.SEQUENTIAL),
                (wav2, overlap_s, "B", None, MixMode.OVERLAP),
            ]
        )
        assert olap_result.duration_s < seq_result.duration_s

    def test_overlap_both_turns_audible_in_region(self):
        """In an OVERLAP mix, the overlap region has energy from both turns."""
        mixer = SceneMixer()
        dur = 1.0
        overlap_s = 0.3
        wav1 = _sine_wav_bytes(freq=440, duration_s=dur, sample_rate=_TARGET_SR, amplitude=0.3)
        wav2 = _sine_wav_bytes(freq=880, duration_s=0.5, sample_rate=_TARGET_SR, amplitude=0.3)

        result = mixer.mix_sequential(
            [
                (wav1, 0.0, "A", None, MixMode.SEQUENTIAL),
                (wav2, overlap_s, "B", None, MixMode.OVERLAP),
            ]
        )
        # The overlap region lies between rendered_onsets_s[1] and rendered_offsets_s[0].
        overlap_start = int(result.rendered_onsets_s[1] * _TARGET_SR)
        overlap_end = int(result.rendered_offsets_s[0] * _TARGET_SR)
        assert overlap_end > overlap_start
        # Overlap region peak amplitude should exceed what a single sine of amplitude 0.3 would give.
        overlap_region = result.samples[overlap_start:overlap_end]
        assert float(np.max(np.abs(overlap_region))) > 0.3

    def test_barge_in_previous_turn_truncated(self):
        """BARGE_IN: the previous turn's audio must be zero after the barge-in point."""
        mixer = SceneMixer()
        dur = 1.5
        barge_depth = 0.4
        wav1 = _sine_wav_bytes(freq=440, duration_s=dur, sample_rate=_TARGET_SR, amplitude=0.4)
        wav2 = _sine_wav_bytes(freq=880, duration_s=0.5, sample_rate=_TARGET_SR, amplitude=0.0)

        result = mixer.mix_sequential(
            [
                (wav1, 0.0, "A", None, MixMode.SEQUENTIAL),
                (wav2, barge_depth, "B", None, MixMode.BARGE_IN),
            ]
        )
        # After COPILOT-6: rendered_offsets_s is updated to the truncation point,
        # so audible_ends_s[0] == rendered_offsets_s[0] < script_offsets_s[0].
        assert result.audible_ends_s[0] == pytest.approx(result.rendered_offsets_s[0], abs=1e-4)
        assert result.audible_ends_s[0] < result.script_offsets_s[0]
        # No audio from the first turn must remain after the barge-in point.
        barge_sample = int(result.rendered_onsets_s[1] * _TARGET_SR)
        # Buffer from barge-in point onward is zero (silent wav2 added, prev truncated).
        tail = result.samples[barge_sample : int(result.script_offsets_s[0] * _TARGET_SR)]
        assert np.allclose(tail, 0.0, atol=1e-5)

    def test_barge_in_audible_end_equals_onset_of_next(self):
        """Interrupted turn's audible_end_s == barge-in turn's rendered_onset_s."""
        mixer = SceneMixer()
        wav1 = _sine_wav_bytes(duration_s=1.0, sample_rate=_TARGET_SR)
        wav2 = _sine_wav_bytes(duration_s=0.5, sample_rate=_TARGET_SR)
        result = mixer.mix_sequential(
            [
                (wav1, 0.0, "A", None, MixMode.SEQUENTIAL),
                (wav2, 0.3, "B", None, MixMode.BARGE_IN),
            ]
        )
        assert result.audible_ends_s[0] == pytest.approx(result.rendered_onsets_s[1], abs=1e-4)

    def test_three_timeline_fields_populated(self):
        """MixedScene must expose script / rendered / audible timeline fields."""
        mixer = SceneMixer()
        wav = _sine_wav_bytes(duration_s=0.5, sample_rate=_TARGET_SR)
        result = mixer.mix_sequential(
            [
                (wav, 0.1, "A", None, MixMode.SEQUENTIAL),
                (wav, 0.2, "B", None, MixMode.SEQUENTIAL),
            ]
        )
        for attr in (
            "script_onsets_s",
            "script_offsets_s",
            "rendered_onsets_s",
            "rendered_offsets_s",
            "audible_onsets_s",
            "audible_ends_s",
        ):
            vals = getattr(result, attr)
            assert len(vals) == 2, f"{attr} should have 2 entries"

    def test_sequential_three_timelines_equal(self):
        """For purely sequential mixes, all three timelines must be identical."""
        mixer = SceneMixer()
        wav = _sine_wav_bytes(duration_s=0.5, sample_rate=_TARGET_SR)
        result = mixer.mix_sequential(
            [
                (wav, 0.1, "A", None, MixMode.SEQUENTIAL),
                (wav, 0.2, "B", None, MixMode.SEQUENTIAL),
            ]
        )
        for i in range(2):
            assert result.script_onsets_s[i] == pytest.approx(result.rendered_onsets_s[i], abs=1e-6)
            assert result.audible_onsets_s[i] == pytest.approx(
                result.rendered_onsets_s[i], abs=1e-6
            )
            assert result.audible_ends_s[i] == pytest.approx(result.rendered_offsets_s[i], abs=1e-6)

    def test_turn_onsets_backward_compat(self):
        """turn_onsets_s / turn_offsets_s must mirror the audible timeline (COPILOT-2)."""
        mixer = SceneMixer()
        wav = _sine_wav_bytes(duration_s=0.5, sample_rate=_TARGET_SR)
        result = mixer.mix_sequential(
            [
                (wav, 0.0, "A", None, MixMode.SEQUENTIAL),
                (wav, 0.3, "B", None, MixMode.OVERLAP),
            ]
        )
        assert result.turn_onsets_s == result.audible_onsets_s
        assert result.turn_offsets_s == result.audible_ends_s

    def test_overlap_as_first_segment_clamps_to_zero(self):
        """OVERLAP as first segment (no previous turn in buffer) must clamp onset to ≥ 0."""
        mixer = SceneMixer()
        wav = _sine_wav_bytes(duration_s=1.0, sample_rate=_TARGET_SR)
        # amount_s would push onset to render_cursor_s(0) - 0.5 = -0.5 → clamped to 0.
        result = mixer.mix_sequential([(wav, 0.5, "A", None, MixMode.OVERLAP)])
        assert result.rendered_onsets_s[0] == pytest.approx(0.0, abs=1e-4)
        assert result.duration_s == pytest.approx(1.0, abs=0.05)

    def test_barge_in_as_first_segment_clamps_to_zero(self):
        """BARGE_IN as first segment (no previous turn in buffer) must clamp onset to ≥ 0."""
        mixer = SceneMixer()
        wav = _sine_wav_bytes(duration_s=1.0, sample_rate=_TARGET_SR)
        result = mixer.mix_sequential([(wav, 0.5, "A", None, MixMode.BARGE_IN)])
        assert result.rendered_onsets_s[0] == pytest.approx(0.0, abs=1e-4)

    def test_barge_in_full_depth_truncates_entirely(self):
        """BARGE_IN with overlap depth ≥ previous duration zeroes out the previous turn."""
        mixer = SceneMixer()
        wav1 = _sine_wav_bytes(freq=440, duration_s=0.5, sample_rate=_TARGET_SR, amplitude=0.4)
        wav2 = _sine_wav_bytes(freq=880, duration_s=0.5, sample_rate=_TARGET_SR)
        # overlap depth (1.0) > prev duration (0.5): onset_sample == prev_onset_sample → max_samples == 0.
        result = mixer.mix_sequential(
            [
                (wav1, 0.0, "A", None, MixMode.SEQUENTIAL),
                (wav2, 1.0, "B", None, MixMode.BARGE_IN),
            ]
        )
        # The interrupted turn's audible end should collapse to its onset.
        assert result.audible_ends_s[0] == pytest.approx(result.audible_onsets_s[0], abs=1e-4)
