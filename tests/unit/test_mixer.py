"""Unit tests for synthbanshee.tts.mixer.SceneMixer."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from synthbanshee.tts.mixer import _TARGET_SR, SceneMixer, _apply_rms_gain

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
        result = mixer.mix_sequential([(wav, 0.0, "SPK_001", None)])

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
        result = mixer.mix_sequential([(wav, pause_s, "SPK_001", None)])

        assert result.turn_onsets_s[0] == pytest.approx(pause_s, abs=0.01)
        assert result.duration_s == pytest.approx(pause_s + 0.5, abs=0.05)

    def test_two_segments_sequential(self):
        mixer = SceneMixer()
        wav1 = _sine_wav_bytes(freq=440, duration_s=1.0, sample_rate=_TARGET_SR)
        wav2 = _sine_wav_bytes(freq=880, duration_s=0.5, sample_rate=_TARGET_SR)
        result = mixer.mix_sequential(
            [
                (wav1, 0.0, "SPK_A", None),
                (wav2, 0.2, "SPK_B", None),
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
            (_sine_wav_bytes(duration_s=0.8, sample_rate=_TARGET_SR), 0.1, "S1", None),
            (_sine_wav_bytes(duration_s=0.6, sample_rate=_TARGET_SR), 0.2, "S2", None),
            (_sine_wav_bytes(duration_s=0.4, sample_rate=_TARGET_SR), 0.15, "S3", None),
        ]
        result = mixer.mix_sequential(segments)
        computed_duration = len(result.samples) / result.sample_rate
        assert result.duration_s == pytest.approx(computed_duration, abs=1e-4)

    def test_resamples_24k_to_16k(self):
        """Mixer should downsample 24 kHz input to 16 kHz output."""
        mixer = SceneMixer()
        wav_24k = _sine_wav_bytes(duration_s=0.5, sample_rate=24000)
        result = mixer.mix_sequential([(wav_24k, 0.0, "SPK_001", None)])

        assert result.sample_rate == _TARGET_SR
        # Duration should still be approximately 0.5 s
        assert result.duration_s == pytest.approx(0.5, abs=0.05)

    def test_downmixes_stereo_to_mono(self):
        mixer = SceneMixer()
        stereo_wav = _stereo_wav_bytes(duration_s=1.0, sample_rate=_TARGET_SR)
        result = mixer.mix_sequential([(stereo_wav, 0.0, "SPK_001", None)])

        assert result.samples.ndim == 1
        assert result.duration_s == pytest.approx(1.0, abs=0.05)

    def test_output_samples_are_float32(self):
        mixer = SceneMixer()
        wav = _sine_wav_bytes(duration_s=0.5, sample_rate=_TARGET_SR)
        result = mixer.mix_sequential([(wav, 0.0, "SPK_001", None)])
        assert result.samples.dtype == np.float32

    def test_offsets_greater_than_onsets(self):
        mixer = SceneMixer()
        segments = [
            (_sine_wav_bytes(duration_s=0.5, sample_rate=_TARGET_SR), 0.1, "S1", None),
            (_sine_wav_bytes(duration_s=0.5, sample_rate=_TARGET_SR), 0.2, "S2", None),
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
        result = mixer.mix_sequential([(wav, 0.0, "SPK", None)])
        # RMS of 0.1-amplitude sine = 0.1/sqrt(2) ≈ −23 dBFS; allow ±2 dB
        assert self._rms_dbfs(result.samples) == pytest.approx(-23.0, abs=2.0)

    def test_rms_target_applied(self):
        """Segment with rms_target_dbfs should reach the requested level."""
        wav = _sine_wav_bytes(amplitude=0.02, duration_s=1.0, sample_rate=_TARGET_SR)
        mixer = SceneMixer()
        target = -20.0
        result = mixer.mix_sequential([(wav, 0.0, "SPK", target)])
        assert self._rms_dbfs(result.samples) == pytest.approx(target, abs=0.5)

    def test_escalation_across_two_segments(self):
        """Louder RMS target on second segment produces measurably higher RMS."""
        wav_quiet = _sine_wav_bytes(amplitude=0.02, duration_s=0.5, sample_rate=_TARGET_SR)
        wav_loud = _sine_wav_bytes(amplitude=0.02, duration_s=0.5, sample_rate=_TARGET_SR)
        mixer = SceneMixer()
        result = mixer.mix_sequential(
            [
                (wav_quiet, 0.0, "AGG", -28.0),
                (wav_loud, 0.0, "AGG", -15.0),
            ]
        )
        # Verify combined duration is correct
        assert result.duration_s == pytest.approx(1.0, abs=0.05)
        # Extract the two halves and check RMS ordering
        half = len(result.samples) // 2
        rms_q = self._rms_dbfs(result.samples[:half])
        rms_l = self._rms_dbfs(result.samples[half:])
        assert rms_l > rms_q + 8.0  # ≥ 8 dB gap matches the M3 spec target
