"""Unit tests for synthbanshee.tts.mixer.SceneMixer."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from synthbanshee.tts.mixer import (
    _TARGET_SR,
    MixMode,
    SceneMixer,
    Segment,
    _apply_edge_fades,
    _apply_lombard_tilt,
    _apply_rms_gain,
    _speech_end_sample,
)

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


def _sine_with_trailing_silence_wav_bytes(
    speech_s: float,
    silence_s: float,
    freq: float = 440.0,
    sample_rate: int = 16000,
    amplitude: float = 0.4,
) -> bytes:
    """Sine of *speech_s* followed by *silence_s* of zeros — repro for #66."""
    n_speech = int(sample_rate * speech_s)
    n_silence = int(sample_rate * silence_s)
    t = np.linspace(0, speech_s, n_speech, endpoint=False)
    speech = (amplitude * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    silence = np.zeros(n_silence, dtype=np.int16)
    samples = np.concatenate([speech, silence])
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
        result = mixer.mix_sequential([Segment(wav, 0.0, "SPK_001", None, MixMode.SEQUENTIAL)])

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
        result = mixer.mix_sequential([Segment(wav, pause_s, "SPK_001", None, MixMode.SEQUENTIAL)])

        assert result.turn_onsets_s[0] == pytest.approx(pause_s, abs=0.01)
        assert result.duration_s == pytest.approx(pause_s + 0.5, abs=0.05)

    def test_two_segments_sequential(self):
        mixer = SceneMixer()
        wav1 = _sine_wav_bytes(freq=440, duration_s=1.0, sample_rate=_TARGET_SR)
        wav2 = _sine_wav_bytes(freq=880, duration_s=0.5, sample_rate=_TARGET_SR)
        result = mixer.mix_sequential(
            [
                Segment(wav1, 0.0, "SPK_A", None, MixMode.SEQUENTIAL),
                Segment(wav2, 0.2, "SPK_B", None, MixMode.SEQUENTIAL),
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
            Segment(
                _sine_wav_bytes(duration_s=0.8, sample_rate=_TARGET_SR),
                0.1,
                "S1",
                None,
                MixMode.SEQUENTIAL,
            ),
            Segment(
                _sine_wav_bytes(duration_s=0.6, sample_rate=_TARGET_SR),
                0.2,
                "S2",
                None,
                MixMode.SEQUENTIAL,
            ),
            Segment(
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
        result = mixer.mix_sequential([Segment(wav_24k, 0.0, "SPK_001", None, MixMode.SEQUENTIAL)])

        assert result.sample_rate == _TARGET_SR
        # Duration should still be approximately 0.5 s
        assert result.duration_s == pytest.approx(0.5, abs=0.05)

    def test_downmixes_stereo_to_mono(self):
        mixer = SceneMixer()
        stereo_wav = _stereo_wav_bytes(duration_s=1.0, sample_rate=_TARGET_SR)
        result = mixer.mix_sequential(
            [Segment(stereo_wav, 0.0, "SPK_001", None, MixMode.SEQUENTIAL)]
        )

        assert result.samples.ndim == 1
        assert result.duration_s == pytest.approx(1.0, abs=0.05)

    def test_output_samples_are_float32(self):
        mixer = SceneMixer()
        wav = _sine_wav_bytes(duration_s=0.5, sample_rate=_TARGET_SR)
        result = mixer.mix_sequential([Segment(wav, 0.0, "SPK_001", None, MixMode.SEQUENTIAL)])
        assert result.samples.dtype == np.float32

    def test_offsets_greater_than_onsets(self):
        mixer = SceneMixer()
        segments = [
            Segment(
                _sine_wav_bytes(duration_s=0.5, sample_rate=_TARGET_SR),
                0.1,
                "S1",
                None,
                MixMode.SEQUENTIAL,
            ),
            Segment(
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
        result = mixer.mix_sequential([Segment(wav, 0.0, "SPK", None, MixMode.SEQUENTIAL)])
        # RMS of 0.1-amplitude sine = 0.1/sqrt(2) ≈ −23 dBFS; allow ±2 dB
        assert self._rms_dbfs(result.samples) == pytest.approx(-23.0, abs=2.0)

    def test_rms_target_applied(self):
        """Segment with rms_target_dbfs should reach the requested level."""
        wav = _sine_wav_bytes(amplitude=0.02, duration_s=1.0, sample_rate=_TARGET_SR)
        mixer = SceneMixer()
        target = -20.0
        result = mixer.mix_sequential([Segment(wav, 0.0, "SPK", target, MixMode.SEQUENTIAL)])
        assert self._rms_dbfs(result.samples) == pytest.approx(target, abs=0.5)

    def test_escalation_across_two_segments(self):
        """Louder RMS target on second segment produces measurably higher RMS."""
        wav_quiet = _sine_wav_bytes(amplitude=0.02, duration_s=0.5, sample_rate=_TARGET_SR)
        wav_loud = _sine_wav_bytes(amplitude=0.02, duration_s=0.5, sample_rate=_TARGET_SR)
        mixer = SceneMixer()
        result = mixer.mix_sequential(
            [
                Segment(wav_quiet, 0.0, "AGG", -28.0, MixMode.SEQUENTIAL),
                Segment(wav_loud, 0.0, "AGG", -15.0, MixMode.SEQUENTIAL),
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
                Segment(wav1, 0.0, "AGG", None, MixMode.SEQUENTIAL),
                Segment(wav2, overlap_s, "VIC", None, MixMode.OVERLAP),
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
                Segment(wav1, 0.0, "A", None, MixMode.SEQUENTIAL),
                Segment(wav2, 0.0, "B", None, MixMode.SEQUENTIAL),
            ]
        )
        olap_result = mixer.mix_sequential(
            [
                Segment(wav1, 0.0, "A", None, MixMode.SEQUENTIAL),
                Segment(wav2, overlap_s, "B", None, MixMode.OVERLAP),
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
                Segment(wav1, 0.0, "A", None, MixMode.SEQUENTIAL),
                Segment(wav2, overlap_s, "B", None, MixMode.OVERLAP),
            ]
        )
        # The overlap region lies between rendered_onsets_s[1] and rendered_offsets_s[0].
        overlap_start = int(result.rendered_onsets_s[1] * _TARGET_SR)
        overlap_end = int(result.rendered_offsets_s[0] * _TARGET_SR)
        assert overlap_end > overlap_start
        # Overlap region peak amplitude should exceed what a single sine of amplitude 0.3 would give.
        overlap_region = result.samples[overlap_start:overlap_end]
        assert float(np.max(np.abs(overlap_region))) > 0.3

    def test_barge_in_previous_turn_truncated_at_speech_end(self):
        """BARGE_IN: the previous turn extends through speech end with a fade-out
        across the overlap region, then is silent past speech end (#66)."""
        mixer = SceneMixer()
        dur = 1.5
        barge_depth = 0.4
        wav1 = _sine_wav_bytes(freq=440, duration_s=dur, sample_rate=_TARGET_SR, amplitude=0.4)
        # Silent wav2 so any non-zero buffer energy comes from wav1 alone.
        wav2 = _sine_wav_bytes(freq=880, duration_s=0.5, sample_rate=_TARGET_SR, amplitude=0.0)

        result = mixer.mix_sequential(
            [
                Segment(wav1, 0.0, "A", None, MixMode.SEQUENTIAL),
                Segment(wav2, barge_depth, "B", None, MixMode.BARGE_IN),
            ]
        )
        # Truncation point sits at wav1's speech end (~ file end here, no trailing
        # silence) — i.e. AT script_offsets_s[0], not at the new onset.
        assert result.audible_ends_s[0] == pytest.approx(result.rendered_offsets_s[0], abs=1e-4)
        assert result.audible_ends_s[0] == pytest.approx(result.script_offsets_s[0], abs=0.04)
        # Prev's audible_end is now AFTER the new onset by ~ barge_depth.
        assert result.audible_ends_s[0] > result.rendered_onsets_s[1]
        # Buffer past the truncation point must be silent (wav2 is silent, prev truncated).
        tail_start = int(result.audible_ends_s[0] * _TARGET_SR)
        tail = result.samples[tail_start:]
        assert np.allclose(tail, 0.0, atol=1e-5)
        # And the overlap region is non-zero (fade-out of wav1 still audible).
        overlap_start = int(result.rendered_onsets_s[1] * _TARGET_SR)
        overlap_end = int(result.audible_ends_s[0] * _TARGET_SR)
        assert overlap_end > overlap_start
        assert float(np.max(np.abs(result.samples[overlap_start:overlap_end]))) > 0.05

    def test_barge_in_audible_end_after_onset_of_next(self):
        """Interrupted turn's audible_end_s lands AFTER the new turn's onset by the
        overlap-region length (#66)."""
        mixer = SceneMixer()
        wav1 = _sine_wav_bytes(duration_s=1.0, sample_rate=_TARGET_SR)
        wav2 = _sine_wav_bytes(duration_s=0.5, sample_rate=_TARGET_SR)
        result = mixer.mix_sequential(
            [
                Segment(wav1, 0.0, "A", None, MixMode.SEQUENTIAL),
                Segment(wav2, 0.3, "B", None, MixMode.BARGE_IN),
            ]
        )
        # Audible end of prev is at its speech end; new onset is barge_depth before that.
        assert result.audible_ends_s[0] > result.rendered_onsets_s[1]
        assert result.audible_ends_s[0] == pytest.approx(
            result.rendered_onsets_s[1] + 0.3, abs=0.05
        )

    def test_three_timeline_fields_populated(self):
        """MixedScene must expose script / rendered / audible timeline fields."""
        mixer = SceneMixer()
        wav = _sine_wav_bytes(duration_s=0.5, sample_rate=_TARGET_SR)
        result = mixer.mix_sequential(
            [
                Segment(wav, 0.1, "A", None, MixMode.SEQUENTIAL),
                Segment(wav, 0.2, "B", None, MixMode.SEQUENTIAL),
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
                Segment(wav, 0.1, "A", None, MixMode.SEQUENTIAL),
                Segment(wav, 0.2, "B", None, MixMode.SEQUENTIAL),
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
                Segment(wav, 0.0, "A", None, MixMode.SEQUENTIAL),
                Segment(wav, 0.3, "B", None, MixMode.OVERLAP),
            ]
        )
        assert result.turn_onsets_s == result.audible_onsets_s
        assert result.turn_offsets_s == result.audible_ends_s

    def test_overlap_as_first_segment_clamps_to_zero(self):
        """OVERLAP as first segment (no previous turn in buffer) must clamp onset to ≥ 0."""
        mixer = SceneMixer()
        wav = _sine_wav_bytes(duration_s=1.0, sample_rate=_TARGET_SR)
        # amount_s would push onset to render_cursor_s(0) - 0.5 = -0.5 → clamped to 0.
        result = mixer.mix_sequential([Segment(wav, 0.5, "A", None, MixMode.OVERLAP)])
        assert result.rendered_onsets_s[0] == pytest.approx(0.0, abs=1e-4)
        assert result.duration_s == pytest.approx(1.0, abs=0.05)

    def test_barge_in_as_first_segment_clamps_to_zero(self):
        """BARGE_IN as first segment (no previous turn in buffer) must clamp onset to ≥ 0."""
        mixer = SceneMixer()
        wav = _sine_wav_bytes(duration_s=1.0, sample_rate=_TARGET_SR)
        result = mixer.mix_sequential([Segment(wav, 0.5, "A", None, MixMode.BARGE_IN)])
        assert result.rendered_onsets_s[0] == pytest.approx(0.0, abs=1e-4)

    def test_barge_in_full_depth_truncates_entirely(self):
        """BARGE_IN with overlap depth ≥ previous duration zeroes out the previous turn."""
        mixer = SceneMixer()
        wav1 = _sine_wav_bytes(freq=440, duration_s=0.5, sample_rate=_TARGET_SR, amplitude=0.4)
        wav2 = _sine_wav_bytes(freq=880, duration_s=0.5, sample_rate=_TARGET_SR)
        # overlap depth (1.0) > prev duration (0.5): onset_sample == prev_onset_sample → max_samples == 0.
        result = mixer.mix_sequential(
            [
                Segment(wav1, 0.0, "A", None, MixMode.SEQUENTIAL),
                Segment(wav2, 1.0, "B", None, MixMode.BARGE_IN),
            ]
        )
        # The interrupted turn's audible end should collapse to its onset.
        assert result.audible_ends_s[0] == pytest.approx(result.audible_onsets_s[0], abs=1e-4)

    def test_barge_in_zero_depth_no_truncation(self):
        """BARGE_IN with amount_s=0 leaves previous turn intact (max_samples >= len(prev_mono))."""
        mixer = SceneMixer()
        dur = 0.5
        wav1 = _sine_wav_bytes(freq=440, duration_s=dur, sample_rate=_TARGET_SR, amplitude=0.4)
        wav2 = _sine_wav_bytes(freq=880, duration_s=dur, sample_rate=_TARGET_SR)
        # amount_s=0 → onset is clamped to prev_offset_s → max_samples == len(prev_mono) → no truncation.
        result = mixer.mix_sequential(
            [
                Segment(wav1, 0.0, "A", None, MixMode.SEQUENTIAL),
                Segment(wav2, 0.0, "B", None, MixMode.BARGE_IN),
            ]
        )
        # Previous turn plays its full duration (audible end == script end).
        assert result.audible_ends_s[0] == pytest.approx(result.script_offsets_s[0], abs=1e-4)
        # New turn starts right where the previous one ended.
        assert result.rendered_onsets_s[1] == pytest.approx(dur, abs=0.02)

    def test_barge_in_overlaps_speech_through_trailing_silence(self):
        """#66: BARGE_IN onset must land in the speech region of the previous turn,
        even when that turn has trailing silence padding (TTS-style)."""
        mixer = SceneMixer()
        speech_s = 1.0
        silence_s = 0.3  # mimics Azure / Google trailing-silence padding
        barge_depth_s = 0.25  # smaller than silence_s — the bug scenario
        wav1 = _sine_with_trailing_silence_wav_bytes(
            speech_s=speech_s,
            silence_s=silence_s,
            freq=440,
            sample_rate=_TARGET_SR,
            amplitude=0.4,
        )
        wav2 = _sine_wav_bytes(freq=880, duration_s=0.6, sample_rate=_TARGET_SR, amplitude=0.4)
        result = mixer.mix_sequential(
            [
                Segment(wav1, 0.0, "A", None, MixMode.SEQUENTIAL),
                Segment(wav2, barge_depth_s, "B", None, MixMode.BARGE_IN),
            ]
        )
        # The new turn must onset *before* the previous turn's speech ended,
        # not somewhere inside its trailing silence.
        new_onset_s = result.rendered_onsets_s[1]
        assert new_onset_s < speech_s, (
            f"BARGE_IN onset {new_onset_s:.3f}s landed at/after speech end "
            f"{speech_s:.3f}s — overlap fell inside trailing silence (#66)."
        )
        # And there must be audible energy from BOTH speakers in the first
        # 100 ms of the overlap region (acceptance criterion).
        n_check = int(0.1 * _TARGET_SR)
        onset_sample = int(new_onset_s * _TARGET_SR)
        # Sum of both turns in the overlap window must clearly exceed wav2 alone:
        # we check that the buffer's RMS in this window is at least ~1.4x wav2's
        # contribution (two uncorrelated sines at the same amplitude → ~sqrt(2)x RMS).
        overlap_window = result.samples[onset_sample : onset_sample + n_check]
        overlap_rms = float(np.sqrt(np.mean(overlap_window**2)))
        # wav2's contribution alone (0.4 amp sine) has RMS ≈ 0.283.
        assert overlap_rms > 0.32, (
            f"Overlap-region RMS {overlap_rms:.3f} is no louder than a single "
            "speaker — previous turn was inaudible at the barge-in point."
        )

    def test_overlap_unaffected_by_speech_end_anchor(self):
        """OVERLAP must continue to anchor against file end, not speech end —
        regression guard for #66 to confirm the fix is BARGE_IN-only."""
        mixer = SceneMixer()
        speech_s = 1.0
        silence_s = 0.3
        overlap_s = 0.25
        wav1 = _sine_with_trailing_silence_wav_bytes(
            speech_s=speech_s, silence_s=silence_s, freq=440, sample_rate=_TARGET_SR
        )
        wav2 = _sine_wav_bytes(freq=880, duration_s=0.5, sample_rate=_TARGET_SR)
        result = mixer.mix_sequential(
            [
                Segment(wav1, 0.0, "A", None, MixMode.SEQUENTIAL),
                Segment(wav2, overlap_s, "B", None, MixMode.OVERLAP),
            ]
        )
        # Anchor is file end (speech_s + silence_s) → onset is at that minus depth.
        expected_onset_s = (speech_s + silence_s) - overlap_s
        assert result.rendered_onsets_s[1] == pytest.approx(expected_onset_s, abs=0.02)


class TestSpeechEndSample:
    """Tests for _speech_end_sample (#66 — TTS trailing-silence trim)."""

    def test_returns_zero_for_empty(self):
        assert _speech_end_sample(np.zeros(0, dtype=np.float32), _TARGET_SR) == 0

    def test_returns_zero_for_silent_array(self):
        mono = np.zeros(_TARGET_SR, dtype=np.float32)
        assert _speech_end_sample(mono, _TARGET_SR) == 0

    def test_skips_trailing_silence(self):
        """An array of speech-level energy followed by silence returns ≈ speech end."""
        sr = _TARGET_SR
        speech = (0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr))).astype(np.float32)
        silence = np.zeros(int(0.3 * sr), dtype=np.float32)
        mono = np.concatenate([speech, silence])
        end = _speech_end_sample(mono, sr)
        # Tolerance: one window (30 ms = 480 samples) plus quantisation slack.
        assert abs(end - len(speech)) <= int(0.04 * sr), (
            f"Detected speech end {end} differs from expected {len(speech)} by "
            f"more than one window."
        )

    def test_full_array_when_no_trailing_silence(self):
        sr = _TARGET_SR
        mono = (0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(0.5 * sr)))).astype(
            np.float32
        )
        # With no trailing silence, the last window is loud → returns len(mono).
        assert _speech_end_sample(mono, sr) == len(mono)


# ---------------------------------------------------------------------------
# M14: Edge fade (crossfade) tests
# ---------------------------------------------------------------------------


class TestApplyEdgeFades:
    """Tests for _apply_edge_fades (M14 — turn boundary click elimination)."""

    def test_fade_in_starts_at_zero(self):
        """First sample after fade-in must be zero."""
        mono = np.ones(1000, dtype=np.float32)
        result = _apply_edge_fades(mono, n=160)
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_fade_out_ends_at_zero(self):
        """Last sample after fade-out must be zero."""
        mono = np.ones(1000, dtype=np.float32)
        result = _apply_edge_fades(mono, n=160)
        assert result[-1] == pytest.approx(0.0, abs=1e-6)

    def test_middle_unchanged(self):
        """Samples outside the fade regions must be unchanged."""
        mono = np.ones(1000, dtype=np.float32) * 0.5
        result = _apply_edge_fades(mono, n=100)
        # Middle region (samples 100 to 899) should be untouched
        np.testing.assert_array_almost_equal(result[100:900], mono[100:900])

    def test_empty_array_returns_empty(self):
        mono = np.zeros(0, dtype=np.float32)
        result = _apply_edge_fades(mono)
        assert len(result) == 0

    def test_short_segment_no_crash(self):
        """A segment shorter than 2*n should get proportionally shorter fades."""
        mono = np.ones(50, dtype=np.float32)
        result = _apply_edge_fades(mono, n=160)
        assert len(result) == 50
        assert result[0] == pytest.approx(0.0, abs=1e-6)
        assert result[-1] == pytest.approx(0.0, abs=1e-6)

    def test_single_sample_returned_unchanged(self):
        """A 1-sample array has fade_len=0 and must be returned unchanged."""
        mono = np.array([0.5], dtype=np.float32)
        result = _apply_edge_fades(mono, n=160)
        assert len(result) == 1
        assert result[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# #65: Lombard tilt (high-shelf at I4/I5)
# ---------------------------------------------------------------------------


# 3500 Hz sits clearly above the 2.5 kHz shelf knee and inside the asymptotic-gain
# region of the high-shelf, so HF/LF band ratios actually reflect the boost.
_HF_LF_SPLIT_HZ = 3500.0


def _hf_lf_band_ratio(samples: np.ndarray, sr: int, split_hz: float = _HF_LF_SPLIT_HZ) -> float:
    """Energy in the >split_hz band divided by energy in the <split_hz band."""
    spectrum = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / sr)
    hf = float(np.sum(spectrum[freqs >= split_hz] ** 2))
    lf = float(np.sum(spectrum[freqs < split_hz] ** 2)) + 1e-12
    return hf / lf


def _white_noise(n: int, seed: int = 0, amp: float = 0.1) -> np.ndarray:
    """Reproducible band-flat test signal — the spectrum is what matters."""
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(n)).astype(np.float32)


def _noise_wav_bytes(seed: int = 1, duration_s: float = 1.0) -> bytes:
    """Return WAV bytes for white noise — broadband enough for spectral tests."""
    n = int(_TARGET_SR * duration_s)
    samples = (_white_noise(n, seed=seed) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_TARGET_SR)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


class TestLombardTilt:
    """Tests for _apply_lombard_tilt (#65)."""

    def test_low_intensity_passes_through(self):
        """I1–I3 must leave the signal bit-exact (object identity, no copy)."""
        signal = _white_noise(8000)
        for intensity in (1, 2, 3):
            out = _apply_lombard_tilt(signal, intensity)
            np.testing.assert_array_equal(out, signal)

    def test_unmapped_intensity_passes_through(self):
        """Out-of-range intensities (e.g. 0, 6, -1) must no-op rather than crash."""
        signal = _white_noise(8000)
        for intensity in (-1, 0, 6, 99):
            out = _apply_lombard_tilt(signal, intensity)
            np.testing.assert_array_equal(out, signal)

    def test_i5_boosts_high_frequencies(self):
        """I5 must materially raise the >3.5 kHz / <3.5 kHz energy ratio."""
        signal = _white_noise(_TARGET_SR)  # 1 second of band-flat noise
        baseline_ratio = _hf_lf_band_ratio(signal, _TARGET_SR)
        i5 = _apply_lombard_tilt(signal, 5)
        i5_ratio = _hf_lf_band_ratio(i5, _TARGET_SR)
        # +3.5 dB asymptotic shelf yields a power factor ≈ 2.24× *deep* in the HF
        # band, blended with partial-gain energy near the 2.5 kHz knee.  The
        # whole-band HF/LF ratio rises by ~1.7× — assert ≥ 1.5× for margin.
        assert i5_ratio > 1.5 * baseline_ratio

    def test_i5_boost_exceeds_i4(self):
        """The I5 shelf must raise HF energy more than the I4 shelf."""
        signal = _white_noise(_TARGET_SR)
        i4 = _apply_lombard_tilt(signal, 4)
        i5 = _apply_lombard_tilt(signal, 5)
        assert _hf_lf_band_ratio(i5, _TARGET_SR) > _hf_lf_band_ratio(i4, _TARGET_SR)

    def test_returns_float32(self):
        signal = _white_noise(4000)
        out = _apply_lombard_tilt(signal, 5)
        assert out.dtype == np.float32

    def test_empty_array_no_crash(self):
        empty = np.zeros(0, dtype=np.float32)
        out = _apply_lombard_tilt(empty, 5)
        assert len(out) == 0

    def test_low_band_largely_preserved(self):
        """A high-shelf must leave low-frequency energy roughly intact."""
        # 200 Hz tone — well below the 2.5 kHz shelf knee.
        n = _TARGET_SR
        t = np.arange(n) / _TARGET_SR
        signal = (0.3 * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)
        out = _apply_lombard_tilt(signal, 5)
        # Skip the filter's startup transient (~50 ms at 16 kHz).
        skip = 800
        in_rms = float(np.sqrt(np.mean(signal[skip:] ** 2)))
        out_rms = float(np.sqrt(np.mean(out[skip:] ** 2)))
        assert out_rms == pytest.approx(in_rms, rel=0.1)


class TestLombardInMixer:
    """Integration: Segment.intensity drives Lombard tilt in mix_sequential."""

    def test_i5_segment_has_more_hf_than_i1(self):
        """A scene rendered at I5 has a higher HF/LF ratio than the same scene at I1."""
        wav = _noise_wav_bytes()
        mixer = SceneMixer()
        i1_scene = mixer.mix_sequential(
            [Segment(wav, 0.0, "SPK", None, MixMode.SEQUENTIAL, intensity=1)]
        )
        i5_scene = mixer.mix_sequential(
            [Segment(wav, 0.0, "SPK", None, MixMode.SEQUENTIAL, intensity=5)]
        )
        i1_ratio = _hf_lf_band_ratio(i1_scene.samples, _TARGET_SR)
        i5_ratio = _hf_lf_band_ratio(i5_scene.samples, _TARGET_SR)
        assert i5_ratio > i1_ratio * 1.6

    def test_default_intensity_does_not_alter_signal(self):
        """Omitting intensity (default I1) must produce the same waveform as explicit I1."""
        wav = _sine_wav_bytes(freq=440, duration_s=0.5, sample_rate=_TARGET_SR)
        mixer = SceneMixer()
        default_scene = mixer.mix_sequential([Segment(wav, 0.0, "SPK", None, MixMode.SEQUENTIAL)])
        i1_scene = mixer.mix_sequential(
            [Segment(wav, 0.0, "SPK", None, MixMode.SEQUENTIAL, intensity=1)]
        )
        np.testing.assert_allclose(default_scene.samples, i1_scene.samples, atol=1e-6)

    def test_i5_does_not_clip_typical_signal(self):
        """A typical-amplitude turn at I5 must stay within the unity-amplitude budget."""
        wav = _sine_wav_bytes(freq=1000, duration_s=0.5, amplitude=0.3, sample_rate=_TARGET_SR)
        mixer = SceneMixer()
        scene = mixer.mix_sequential(
            [Segment(wav, 0.0, "SPK", None, MixMode.SEQUENTIAL, intensity=5)]
        )
        # Lombard adds +3.5 dB above 2.5 kHz; 1 kHz is under the shelf so the
        # peak should remain comfortably below 1.0 — preprocessing's peak
        # limiter handles the few dB of ceiling that may be needed.
        assert float(np.max(np.abs(scene.samples))) < 0.95
