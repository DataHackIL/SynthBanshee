"""Unit tests for M15 turn-level quality gates."""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from synthbanshee.tts.quality_gates import (
    GateResult,
    check_clicks,
    check_f0_guardrails,
    check_sustained_vowel,
    run_quality_gates,
)

SR = 16_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sine(freq_hz: float, duration_s: float, amplitude: float = 0.5) -> np.ndarray:
    """Generate a mono float32 sine wave."""
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    return (np.sin(2 * np.pi * freq_hz * t) * amplitude).astype(np.float32)


def _silence(duration_s: float) -> np.ndarray:
    return np.zeros(int(SR * duration_s), dtype=np.float32)


def _to_wav_bytes(samples: np.ndarray, sr: int = SR) -> bytes:
    """Encode float32 samples as WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Sustained vowel gate
# ---------------------------------------------------------------------------


class TestSustainedVowelGate:
    def test_short_voiced_passes(self) -> None:
        # 1.0s sine — well under 2.8s threshold
        samples = _sine(200.0, 1.0)
        result = check_sustained_vowel(samples, SR)
        assert result.passed

    def test_long_sustained_fails(self) -> None:
        # 3.5s continuous sine — exceeds 2.8s threshold
        samples = _sine(200.0, 3.5)
        result = check_sustained_vowel(samples, SR)
        assert not result.passed
        assert result.gate_name == "sustained_vowel"

    def test_interrupted_voiced_passes(self) -> None:
        # Two 1.5s voiced segments separated by silence — each under threshold
        seg1 = _sine(200.0, 1.5)
        gap = _silence(0.5)
        seg2 = _sine(200.0, 1.5)
        samples = np.concatenate([seg1, gap, seg2])
        result = check_sustained_vowel(samples, SR)
        assert result.passed

    def test_empty_audio_passes(self) -> None:
        result = check_sustained_vowel(np.array([], dtype=np.float32), SR)
        assert result.passed

    def test_silence_passes(self) -> None:
        samples = _silence(5.0)
        result = check_sustained_vowel(samples, SR)
        assert result.passed


# ---------------------------------------------------------------------------
# F0 guardrails gate
# ---------------------------------------------------------------------------


class TestF0GuardrailsGate:
    def test_male_in_range_passes(self) -> None:
        # 120 Hz sine — within male [80, 180] Hz
        samples = _sine(120.0, 1.0)
        result = check_f0_guardrails(samples, SR, "male")
        assert result.passed

    def test_female_in_range_passes(self) -> None:
        # 200 Hz sine — within female [150, 290] Hz
        samples = _sine(200.0, 1.0)
        result = check_f0_guardrails(samples, SR, "female")
        assert result.passed

    def test_male_too_high_fails(self) -> None:
        # 250 Hz — above male max (180 Hz)
        samples = _sine(250.0, 1.0)
        result = check_f0_guardrails(samples, SR, "male")
        assert not result.passed
        assert result.gate_name == "f0_guardrails"

    def test_female_too_low_fails(self) -> None:
        # 100 Hz — below female min (150 Hz)
        samples = _sine(100.0, 1.0)
        result = check_f0_guardrails(samples, SR, "female")
        assert not result.passed
        assert result.gate_name == "f0_guardrails"

    def test_silence_passes(self) -> None:
        # No voiced frames → pass by default
        samples = _silence(2.0)
        result = check_f0_guardrails(samples, SR, "male")
        assert result.passed


# ---------------------------------------------------------------------------
# Click detection gate
# ---------------------------------------------------------------------------


class TestClickDetectionGate:
    def test_clean_sine_passes(self) -> None:
        samples = _sine(200.0, 1.0)
        result = check_clicks(samples, SR)
        assert result.passed

    def test_isolated_dc_jumps_fail(self) -> None:
        # Low-amplitude signal with isolated DC jumps (simulating SSML boundary clicks)
        samples = np.zeros(16000, dtype=np.float32)
        # Insert isolated spikes well-separated (>CLICK_ISOLATION_RADIUS apart)
        for pos in [1000, 3000, 5000, 7000]:
            samples[pos] = 0.8  # Isolated single-sample spike
        result = check_clicks(samples, SR)
        assert not result.passed
        assert result.gate_name == "click_detection"

    def test_plosive_burst_passes(self) -> None:
        # A plosive burst spans many consecutive high-diff samples — not isolated
        samples = np.zeros(16000, dtype=np.float32)
        # Simulate a plosive: multiple consecutive high-amplitude samples
        samples[5000:5020] = 0.5  # 20-sample burst — NOT an isolated click
        result = check_clicks(samples, SR)
        assert result.passed

    def test_few_clicks_passes(self) -> None:
        # Only 1-2 isolated clicks — below count threshold of 3
        samples = np.zeros(16000, dtype=np.float32)
        samples[5000] = 0.8
        result = check_clicks(samples, SR)
        assert result.passed

    def test_empty_passes(self) -> None:
        result = check_clicks(np.array([], dtype=np.float32), SR)
        assert result.passed


# ---------------------------------------------------------------------------
# Composite gate runner
# ---------------------------------------------------------------------------


class TestRunQualityGates:
    def test_good_male_audio_passes(self) -> None:
        # 1s at 120 Hz — passes all gates
        samples = _sine(120.0, 1.0)
        wav_bytes = _to_wav_bytes(samples)
        result = run_quality_gates(wav_bytes, "male")
        assert result.passed

    def test_good_female_audio_passes(self) -> None:
        # 1s at 200 Hz — passes all gates
        samples = _sine(200.0, 1.0)
        wav_bytes = _to_wav_bytes(samples)
        result = run_quality_gates(wav_bytes, "female")
        assert result.passed

    def test_sustained_vowel_fails_first(self) -> None:
        # 4s sustained sine at valid frequency — sustained vowel gate fires
        samples = _sine(120.0, 4.0)
        wav_bytes = _to_wav_bytes(samples)
        result = run_quality_gates(wav_bytes, "male")
        assert not result.passed
        assert result.gate_name == "sustained_vowel"

    def test_f0_out_of_range_fails(self) -> None:
        # Short audio at invalid frequency for male
        # Use 2.5s so it stays under sustained threshold, but fails F0
        samples = _sine(250.0, 2.5)
        wav_bytes = _to_wav_bytes(samples)
        result = run_quality_gates(wav_bytes, "male")
        assert not result.passed
        assert result.gate_name == "f0_guardrails"


# ---------------------------------------------------------------------------
# GateResult dataclass
# ---------------------------------------------------------------------------


class TestGateResult:
    def test_default_is_passing(self) -> None:
        r = GateResult(passed=True)
        assert r.passed
        assert r.gate_name is None
        assert r.detail is None

    def test_failure_has_info(self) -> None:
        r = GateResult(passed=False, gate_name="test_gate", detail="something bad")
        assert not r.passed
        assert r.gate_name == "test_gate"
