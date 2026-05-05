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

    def test_dc_offset_steps_fail(self) -> None:
        # Three sustained baseline shifts — the failure mode the gate targets.
        # Each shift holds for 50 ms (>> the 20 ms detector window) so both
        # the rising and falling edge register as step events.
        samples = np.zeros(16000, dtype=np.float32)
        for start, amp in [(2000, 0.20), (6000, -0.18), (10000, 0.25)]:
            samples[start : start + 800] = amp
        result = check_clicks(samples, SR)
        assert not result.passed
        assert result.gate_name == "click_detection"

    def test_isolated_single_sample_spikes_pass(self) -> None:
        # Single-sample spikes are NOT DC clicks — they're impulse noise that
        # leaves the running mean unchanged.  Pre-#80 the gate caught these
        # but also caught any high-frequency content (Hebrew sibilants); the
        # step-shift detector correctly ignores both.
        samples = np.zeros(16000, dtype=np.float32)
        for pos in [1000, 3000, 5000, 7000]:
            samples[pos] = 0.8
        result = check_clicks(samples, SR)
        assert result.passed

    def test_plosive_burst_passes(self) -> None:
        # A 20-sample plosive burst at 0.5 leaves running mean ~0 either side
        # of the burst; not a baseline shift.
        samples = np.zeros(16000, dtype=np.float32)
        samples[5000:5020] = 0.5
        result = check_clicks(samples, SR)
        assert result.passed

    def test_sibilant_like_noise_passes(self) -> None:
        # Regression for #80: zero-mean high-frequency noise with an
        # envelope (mimicking Hebrew /ʃ/ /s/ fricatives) used to flood the
        # old per-sample-diff detector with thousands of isolated "clicks".
        # The step-shift detector must pass it because the baseline mean is
        # unchanged on either side of any single sample.
        rng = np.random.default_rng(42)
        # 1.5 s signal at -18 dBFS RMS, comparable to in-clip speech level.
        n = int(1.5 * SR)
        envelope = np.zeros(n, dtype=np.float32)
        # Two sibilant-like bursts of 200 ms each
        for start_s in (0.3, 0.9):
            i = int(start_s * SR)
            burst_n = int(0.2 * SR)
            ramp = np.hanning(burst_n).astype(np.float32)
            envelope[i : i + burst_n] = ramp
        noise = rng.standard_normal(n).astype(np.float32)
        samples = (noise * envelope * 0.30).astype(np.float32)
        result = check_clicks(samples, SR)
        assert result.passed, f"sibilant-like noise wrongly flagged: {result.detail}"

    def test_few_steps_pass(self) -> None:
        # Only 1 sustained step — below count threshold of 3.
        samples = np.zeros(16000, dtype=np.float32)
        samples[5000:5800] = 0.20  # one shift, two edges (rising + falling) = 2 events
        result = check_clicks(samples, SR)
        assert result.passed

    def test_empty_passes(self) -> None:
        result = check_clicks(np.array([], dtype=np.float32), SR)
        assert result.passed

    def test_short_audio_passes(self) -> None:
        # Audio shorter than 2 * window — no detection window, pass by default.
        samples = np.zeros(100, dtype=np.float32)
        result = check_clicks(samples, SR)
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
