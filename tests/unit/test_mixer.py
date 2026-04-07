"""Unit tests for synthbanshee.tts.mixer.SceneMixer."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from synthbanshee.tts.mixer import _TARGET_SR, SceneMixer

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
        result = mixer.mix_sequential([(wav, 0.0, "SPK_001")])

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
        result = mixer.mix_sequential([(wav, pause_s, "SPK_001")])

        assert result.turn_onsets_s[0] == pytest.approx(pause_s, abs=0.01)
        assert result.duration_s == pytest.approx(pause_s + 0.5, abs=0.05)

    def test_two_segments_sequential(self):
        mixer = SceneMixer()
        wav1 = _sine_wav_bytes(freq=440, duration_s=1.0, sample_rate=_TARGET_SR)
        wav2 = _sine_wav_bytes(freq=880, duration_s=0.5, sample_rate=_TARGET_SR)
        result = mixer.mix_sequential(
            [
                (wav1, 0.0, "SPK_A"),
                (wav2, 0.2, "SPK_B"),
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
            (_sine_wav_bytes(duration_s=0.8, sample_rate=_TARGET_SR), 0.1, "S1"),
            (_sine_wav_bytes(duration_s=0.6, sample_rate=_TARGET_SR), 0.2, "S2"),
            (_sine_wav_bytes(duration_s=0.4, sample_rate=_TARGET_SR), 0.15, "S3"),
        ]
        result = mixer.mix_sequential(segments)
        computed_duration = len(result.samples) / result.sample_rate
        assert result.duration_s == pytest.approx(computed_duration, abs=1e-4)

    def test_resamples_24k_to_16k(self):
        """Mixer should downsample 24 kHz input to 16 kHz output."""
        mixer = SceneMixer()
        wav_24k = _sine_wav_bytes(duration_s=0.5, sample_rate=24000)
        result = mixer.mix_sequential([(wav_24k, 0.0, "SPK_001")])

        assert result.sample_rate == _TARGET_SR
        # Duration should still be approximately 0.5 s
        assert result.duration_s == pytest.approx(0.5, abs=0.05)

    def test_downmixes_stereo_to_mono(self):
        mixer = SceneMixer()
        stereo_wav = _stereo_wav_bytes(duration_s=1.0, sample_rate=_TARGET_SR)
        result = mixer.mix_sequential([(stereo_wav, 0.0, "SPK_001")])

        assert result.samples.ndim == 1
        assert result.duration_s == pytest.approx(1.0, abs=0.05)

    def test_output_samples_are_float32(self):
        mixer = SceneMixer()
        wav = _sine_wav_bytes(duration_s=0.5, sample_rate=_TARGET_SR)
        result = mixer.mix_sequential([(wav, 0.0, "SPK_001")])
        assert result.samples.dtype == np.float32

    def test_offsets_greater_than_onsets(self):
        mixer = SceneMixer()
        segments = [
            (_sine_wav_bytes(duration_s=0.5, sample_rate=_TARGET_SR), 0.1, "S1"),
            (_sine_wav_bytes(duration_s=0.5, sample_rate=_TARGET_SR), 0.2, "S2"),
        ]
        result = mixer.mix_sequential(segments)
        for onset, offset in zip(result.turn_onsets_s, result.turn_offsets_s, strict=True):
            assert offset > onset
