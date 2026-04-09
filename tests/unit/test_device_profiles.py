"""Unit tests for synthbanshee.augment.device_profiles."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from synthbanshee.augment.device_profiles import _PROFILES, DeviceProfiler

_SR = 16_000
_N = _SR  # 1 s


def _make_samples(n: int = _N) -> np.ndarray:
    """Broadband white-ish signal that exercises all frequency bands."""
    rng = np.random.default_rng(0)
    return rng.standard_normal(n).astype(np.float32) * 0.3


# ---------------------------------------------------------------------------
# _PROFILES sanity checks
# ---------------------------------------------------------------------------


def test_all_expected_devices_present():
    expected = {"phone_in_hand", "phone_in_pocket", "phone_on_table", "pi_budget_mic"}
    assert set(_PROFILES.keys()) == expected


def test_profile_keys_complete():
    required = {"highpass_hz", "lowpass_hz", "hum_hz", "hum_dbfs", "level_db"}
    for device, profile in _PROFILES.items():
        assert required.issubset(profile.keys()), f"{device} missing keys"


def test_phone_in_pocket_has_lower_lowpass_than_in_hand():
    assert _PROFILES["phone_in_pocket"]["lowpass_hz"] < _PROFILES["phone_in_hand"]["lowpass_hz"]


def test_phone_in_pocket_has_negative_level():
    assert _PROFILES["phone_in_pocket"]["level_db"] < 0.0


def test_pi_budget_mic_has_hum():
    assert _PROFILES["pi_budget_mic"]["hum_hz"] is not None
    assert _PROFILES["pi_budget_mic"]["hum_hz"] == 50.0
    assert _PROFILES["pi_budget_mic"]["hum_dbfs"] is not None


def test_phone_devices_no_hum():
    for device in ("phone_in_hand", "phone_in_pocket", "phone_on_table"):
        assert _PROFILES[device]["hum_hz"] is None


# ---------------------------------------------------------------------------
# DeviceProfiler.apply — output shape and dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", list(_PROFILES.keys()))
def test_output_length_matches_input(device):
    profiler = DeviceProfiler()
    samples = _make_samples()
    out = profiler.apply(samples, _SR, device, rng_seed=0)
    assert out.shape == samples.shape


@pytest.mark.parametrize("device", list(_PROFILES.keys()))
def test_output_dtype_is_float32(device):
    profiler = DeviceProfiler()
    out = profiler.apply(_make_samples(), _SR, device, rng_seed=0)
    assert out.dtype == np.float32


def test_unknown_device_raises():
    profiler = DeviceProfiler()
    with pytest.raises(KeyError):
        profiler.apply(_make_samples(), _SR, "does_not_exist")


# ---------------------------------------------------------------------------
# Filtering behaviour
# ---------------------------------------------------------------------------


def _power_in_band(signal: np.ndarray, sr: int, lo_hz: float, hi_hz: float) -> float:
    """Return fraction of signal power in [lo_hz, hi_hz]."""
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sr)
    spectrum = np.abs(np.fft.rfft(signal)) ** 2
    mask = (freqs >= lo_hz) & (freqs <= hi_hz)
    total = float(spectrum.sum())
    if total == 0:
        return 0.0
    return float(spectrum[mask].sum()) / total


def test_phone_in_pocket_attenuates_high_frequencies():
    """phone_in_pocket LPF at 2500 Hz should reduce energy above 4 kHz."""
    profiler = DeviceProfiler()
    samples = _make_samples()
    out = profiler.apply(samples, _SR, "phone_in_pocket")
    # High-frequency power should be much lower after filtering
    hi_power_before = _power_in_band(samples, _SR, 4000, 8000)
    hi_power_after = _power_in_band(out, _SR, 4000, 8000)
    assert hi_power_after < hi_power_before * 0.5, (
        f"High-freq attenuation insufficient: before={hi_power_before:.3f}, after={hi_power_after:.3f}"
    )


def test_phone_in_pocket_reduces_overall_level():
    """phone_in_pocket has −6 dB gain → RMS should drop."""
    profiler = DeviceProfiler()
    samples = _make_samples()
    out_pocket = profiler.apply(samples, _SR, "phone_in_pocket")
    out_hand = profiler.apply(samples, _SR, "phone_in_hand")
    rms_pocket = float(np.sqrt(np.mean(out_pocket.astype(np.float64) ** 2)))
    rms_hand = float(np.sqrt(np.mean(out_hand.astype(np.float64) ** 2)))
    assert rms_pocket < rms_hand


def test_pi_budget_mic_injects_50hz_hum():
    """pi_budget_mic should add detectable 50 Hz energy."""
    profiler = DeviceProfiler()
    # Use a signal with no energy at 50 Hz (pure 440 Hz tone)
    t = np.arange(_N, dtype=np.float32) / _SR
    clean = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    out = profiler.apply(clean, _SR, "pi_budget_mic")

    # 50 Hz power should be detectable in output but not in input
    power_before = _power_in_band(clean, _SR, 45, 55)
    power_after = _power_in_band(out, _SR, 45, 55)
    assert power_after > power_before * 2


def test_phone_in_hand_zero_level_db_preserves_rms():
    """phone_in_hand has level_db=0 → RMS should be roughly preserved after filtering."""
    profiler = DeviceProfiler()
    samples = _make_samples()
    out = profiler.apply(samples, _SR, "phone_in_hand")
    rms_in = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    rms_out = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    # Within a factor of 2 (filtering redistributes energy but shouldn't be huge)
    assert 0.2 < rms_out / rms_in < 5.0


def test_silent_input_stays_silent_for_non_hum_device():
    profiler = DeviceProfiler()
    silence = np.zeros(_N, dtype=np.float32)
    out = profiler.apply(silence, _SR, "phone_in_hand")
    assert np.max(np.abs(out)) < 1e-9


def test_pi_budget_mic_adds_hum_to_silence():
    """Hum is injected even when speech is silent."""
    profiler = DeviceProfiler()
    silence = np.zeros(_N, dtype=np.float32)
    out = profiler.apply(silence, _SR, "pi_budget_mic")
    assert np.max(np.abs(out)) > 0.0


def test_highpass_skipped_when_hz_is_zero():
    """Cover the False branch of 'if hp_hz > 0' using a patched profile."""
    profiler = DeviceProfiler()
    fake_profile = {
        "highpass_hz": 0,
        "lowpass_hz": 8_000,
        "hum_hz": None,
        "hum_dbfs": None,
        "level_db": 0.0,
    }
    with patch.dict("synthbanshee.augment.device_profiles._PROFILES", {"zero_hp": fake_profile}):
        out = profiler.apply(_make_samples(), _SR, "zero_hp")
    assert out.shape == (_N,)
    assert out.dtype == np.float32
