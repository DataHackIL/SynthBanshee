"""Unit tests for the preprocessing pipeline (Phase 0.4)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf

from synthbanshee.augment.preprocessing import (
    _PEAK_DBFS,
    _SILENCE_PAD_S,
    _TARGET_SR,
    _peak_limit,
    preprocess,
    validate_audio,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_sine_wav(
    path: Path,
    freq_hz: float = 440.0,
    duration_s: float = 5.0,
    sample_rate: int = 44100,
    channels: int = 1,
    amplitude: float = 0.5,
) -> Path:
    """Write a pure sine WAV to path and return path."""
    n = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n, endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
    if channels == 2:
        wave = np.stack([wave, wave], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wave, sample_rate, subtype="PCM_16")
    return path


def _peak_dbfs(path: Path) -> float:
    data, _ = sf.read(str(path), dtype="float32")
    peak = float(np.max(np.abs(data)))
    if peak == 0:
        return -math.inf
    return 20.0 * math.log10(peak)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPreprocess:
    def test_output_sample_rate(self, tmp_path):
        src = _write_sine_wav(tmp_path / "src.wav", sample_rate=44100)
        dst = tmp_path / "out.wav"
        result = preprocess(src, dst)
        assert result.sample_rate == _TARGET_SR

        data, sr = sf.read(str(dst))
        assert sr == _TARGET_SR

    def test_stereo_to_mono(self, tmp_path):
        src = _write_sine_wav(tmp_path / "src_stereo.wav", channels=2)
        dst = tmp_path / "out.wav"
        result = preprocess(src, dst)
        assert result.channels == 1

        data, _ = sf.read(str(dst))
        assert data.ndim == 1

    def test_peak_limiter_loud_clip_clamped(self, tmp_path):
        """A clip whose peak exceeds the ceiling must be attenuated to ≤ ceiling."""
        # amplitude=0.99 → peak ≈ −0.09 dBFS, above the −1.0 dBFS ceiling
        src = _write_sine_wav(tmp_path / "src.wav", amplitude=0.99)
        dst = tmp_path / "out.wav"
        preprocess(src, dst)
        peak = _peak_dbfs(dst)
        assert peak <= _PEAK_DBFS + 0.5, f"Peak {peak:.2f} dBFS exceeds ceiling {_PEAK_DBFS} dBFS"

    def test_peak_limiter_quiet_clip_not_scaled_up(self, tmp_path):
        """A clip already below the ceiling must NOT be scaled up."""
        # amplitude=0.05 → peak ≈ −26 dBFS, well below the −1.0 dBFS ceiling
        src = _write_sine_wav(tmp_path / "src.wav", amplitude=0.05, sample_rate=_TARGET_SR)
        dst = tmp_path / "out.wav"
        preprocess(src, dst)
        peak = _peak_dbfs(dst)
        # Must remain well below the ceiling — not normalized up to −1.0 dBFS
        assert peak < _PEAK_DBFS - 5.0, (
            f"Quiet clip was scaled up: peak {peak:.2f} dBFS (expected << {_PEAK_DBFS} dBFS)"
        )

    def test_silence_padding_present(self, tmp_path):
        src = _write_sine_wav(tmp_path / "src.wav", duration_s=4.0)
        dst = tmp_path / "out.wav"
        preprocess(src, dst)

        data, sr = sf.read(str(dst))
        pad_n = int(_SILENCE_PAD_S * sr)
        # Output must be longer than input due to added padding
        assert len(data) > int(4.0 * sr)
        # Head and tail should be silence
        assert float(np.max(np.abs(data[:pad_n]))) < 1e-6
        assert float(np.max(np.abs(data[-pad_n:]))) < 1e-6

    def test_dirty_file_retained(self, tmp_path):
        src = _write_sine_wav(tmp_path / "src.wav")
        dst = tmp_path / "out.wav"
        dirty_dir = tmp_path / "dirty"
        result = preprocess(src, dst, dirty_dir=dirty_dir)
        assert result.dirty_path is not None
        assert result.dirty_path.exists()
        assert result.dirty_path != dst

    def test_steps_applied_list(self, tmp_path):
        src = _write_sine_wav(tmp_path / "src.wav", sample_rate=44100)
        dst = tmp_path / "out.wav"
        result = preprocess(src, dst)
        steps_str = " ".join(result.steps_applied)
        assert "resample" in steps_str
        # "mono" step only appears when source has >1 channels
        assert "lowpass" in steps_str
        assert "peak_limit" in steps_str
        assert "silence_pad" in steps_str

    def test_short_clip_warning(self, tmp_path):
        """A 1-second clip should produce a duration warning (below 3 s minimum)."""
        src = _write_sine_wav(tmp_path / "src.wav", duration_s=1.0)
        dst = tmp_path / "out.wav"
        result = preprocess(src, dst)
        # With silence padding added, output is 1 + 2*0.5 = 2 s — still < 3 s
        assert any("below minimum" in w for w in result.warnings)

    def test_already_16khz_no_resample_step(self, tmp_path):
        src = _write_sine_wav(tmp_path / "src.wav", sample_rate=_TARGET_SR)
        dst = tmp_path / "out.wav"
        result = preprocess(src, dst)
        assert not any("resample" in s for s in result.steps_applied)


class TestValidateAudio:
    def test_valid_clip_passes(self, tmp_path):
        src = _write_sine_wav(tmp_path / "src.wav", duration_s=4.0)
        dst = tmp_path / "valid.wav"
        preprocess(src, dst)
        ok, errors = validate_audio(dst)
        assert ok, f"Validation failed: {errors}"

    def test_missing_file(self, tmp_path):
        ok, errors = validate_audio(tmp_path / "nonexistent.wav")
        assert not ok
        assert any("not found" in e for e in errors)

    def test_wrong_sample_rate(self, tmp_path):
        path = tmp_path / "bad_sr.wav"
        _write_sine_wav(path, sample_rate=8000, duration_s=4.0)
        ok, errors = validate_audio(path)
        assert not ok
        assert any("sample_rate" in e for e in errors)

    def test_stereo_fails(self, tmp_path):
        path = tmp_path / "stereo.wav"
        _write_sine_wav(path, sample_rate=_TARGET_SR, channels=2, duration_s=4.0)
        ok, errors = validate_audio(path)
        assert not ok
        assert any("channels" in e for e in errors)

    def test_too_short(self, tmp_path):
        """After preprocessing a very short clip should fail duration check."""
        src = _write_sine_wav(tmp_path / "src.wav", duration_s=1.0)
        dst = tmp_path / "short.wav"
        preprocess(src, dst)
        ok, errors = validate_audio(dst)
        # 1 s + 2*0.5 s padding = 2 s — still below 3 s minimum
        assert not ok
        assert any("duration" in e for e in errors)

    def test_quiet_clip_passes_validator(self, tmp_path):
        """A valid clip whose peak is well below −1.0 dBFS must pass validation.

        With a limiter (not normalizer) preprocess() does not scale up quiet
        clips.  validate_audio() must accept them — only clips that exceed the
        ceiling should be rejected.
        """
        src = _write_sine_wav(
            tmp_path / "src.wav", amplitude=0.05, duration_s=4.0, sample_rate=_TARGET_SR
        )
        dst = tmp_path / "quiet.wav"
        preprocess(src, dst)
        ok, errors = validate_audio(dst)
        assert ok, f"Quiet clip incorrectly rejected: {errors}"

    def test_over_ceiling_clip_fails_validator(self, tmp_path):
        """A WAV written with peak above the ceiling must fail validation."""
        # Write a loud clip directly (bypassing preprocess) so the peak is > ceiling
        path = tmp_path / "loud.wav"
        loud = np.ones(int(4.0 * _TARGET_SR), dtype=np.float32) * 0.99
        sf.write(str(path), loud, _TARGET_SR, subtype="PCM_16")
        # Reload to get the actual stored peak (PCM_16 quantisation)
        data, _ = sf.read(str(path), dtype="float32")
        # Manually check it is above ceiling before asserting validator catches it
        assert float(np.max(np.abs(data))) > 10 ** (_PEAK_DBFS / 20.0) + 0.01
        ok, errors = validate_audio(path)
        assert not ok
        assert any("exceeds ceiling" in e for e in errors)


class TestPeakLimit:
    """Unit tests for _peak_limit() helper (M3b)."""

    def test_loud_signal_clamped_to_ceiling(self):
        loud = np.ones(1000, dtype=np.float32) * 0.99  # peak ≈ −0.09 dBFS
        ceiling_dbfs = -1.0
        result = _peak_limit(loud, ceiling_dbfs)
        peak = float(np.max(np.abs(result)))
        assert peak <= 10 ** (ceiling_dbfs / 20.0) + 1e-6

    def test_quiet_signal_unchanged(self):
        quiet = np.ones(1000, dtype=np.float32) * 0.1  # peak ≈ −20 dBFS
        original = quiet.copy()
        result = _peak_limit(quiet, -1.0)
        np.testing.assert_array_equal(result, original)

    def test_silence_unchanged(self):
        silence = np.zeros(500, dtype=np.float32)
        result = _peak_limit(silence, -1.0)
        np.testing.assert_array_equal(result, silence)

    def test_exactly_at_ceiling_unchanged(self):
        ceiling_linear = 10 ** (-1.0 / 20.0)
        at_ceiling = np.full(500, ceiling_linear, dtype=np.float32)
        result = _peak_limit(at_ceiling, -1.0)
        np.testing.assert_array_almost_equal(result, at_ceiling)
