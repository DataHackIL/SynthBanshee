"""Unit tests for M15 turn-level quality gates."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from synthbanshee.tts.quality_gates import (
    GateResult,
    check_clicks,
    check_f0_guardrails,
    check_sustained_vowel,
    run_quality_gates,
)

SR = 16_000

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "clicks"


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

    @pytest.mark.parametrize("sr", [16_000, 24_000])
    def test_dc_offset_steps_fail(self, sr: int) -> None:
        # Three sustained baseline shifts — the failure mode the gate targets.
        # Each shift holds 50 ms (longer than the 40 ms detector window) so
        # the rising and falling edge of every shift register as separate
        # step events.  Six events expected (3 shifts × 2 edges); we accept
        # 5–8 to allow for find_peaks dropping a single boundary edge.
        samples = np.zeros(sr, dtype=np.float32)
        shift_n = int(0.05 * sr)  # 50 ms
        for start_s, amp in [(0.10, 0.20), (0.35, -0.18), (0.60, 0.25)]:
            i = int(start_s * sr)
            samples[i : i + shift_n] = amp
        result = check_clicks(samples, sr)
        assert not result.passed
        assert result.gate_name == "click_detection"
        # Tight bound on the count locks in detector behaviour, not just outcome.
        # The detail string is "...step events (threshold: 3); first at t=[...]s".
        assert result.detail is not None
        n = int(result.detail.split()[1])
        assert 5 <= n <= 8, f"expected 5-8 step events, got {n}: {result.detail}"

    def test_isolated_single_sample_spikes_pass(self) -> None:
        # Single-sample spikes are NOT DC clicks — they're impulse noise that
        # leaves the running mean unchanged.  Pre-#80 the gate caught these
        # alongside any high-frequency content (Hebrew sibilants).  This is
        # an intentional capability change: the new detector targets only
        # sustained baseline shifts, since Azure he-IL output does not
        # produce isolated impulse noise in practice (see #80 diagnosis).
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

    @pytest.mark.parametrize("sr", [16_000, 24_000])
    def test_sibilant_like_noise_passes(self, sr: int) -> None:
        # Synthetic sibilant proxy: envelope-shaped white Gaussian noise.
        # Real /ʃ/ /s/ have spectral peaks at 5–8 kHz; this fixture is
        # weaker than that but exercises the zero-mean property the
        # step detector relies on.  The real regression coverage lives
        # in test_real_tts_turn_passes (Azure-rendered fixture).
        rng = np.random.default_rng(42)
        n = int(1.5 * sr)
        envelope = np.zeros(n, dtype=np.float32)
        for start_s in (0.3, 0.9):
            i = int(start_s * sr)
            burst_n = int(0.2 * sr)
            ramp = np.hanning(burst_n).astype(np.float32)
            envelope[i : i + burst_n] = ramp
        noise = rng.standard_normal(n).astype(np.float32)
        samples = (noise * envelope * 0.30).astype(np.float32)
        result = check_clicks(samples, sr)
        assert result.passed, f"sibilant-like noise wrongly flagged: {result.detail}"

    @pytest.mark.parametrize(
        "fixture_name",
        [
            # 16 kHz pair (post-mixer rate)
            "sp_neu_a_0001_turn_03_female_2s_16k.wav",
            "sp_neu_a_0001_turn_08_male_2s_16k.wav",
            # 24 kHz pair (Azure provider native rate — what the gate sees in production)
            "sp_neu_a_0001_turn_03_female_2s_24k.wav",
            "sp_neu_a_0001_turn_08_male_2s_24k.wav",
        ],
    )
    def test_real_tts_turn_passes(self, fixture_name: str) -> None:
        # Regression guard for #80: 2 s segments captured from real Azure
        # he-IL output for sp_neu_a_0001 (one VIC/F, one AGG/M) at the
        # window with the highest pre-fix click density.  Both the 24 kHz
        # native (gate-time) rate and the 16 kHz post-mixer rate are
        # exercised because behaviour depends on win = win_ms * sr / 1000.
        # Pre-#80 the per-sample-diff detector logged 100s of false events
        # on these segments; the step-shift detector must report passed=True.
        fixture_path = FIXTURES_DIR / fixture_name
        samples, sr = sf.read(fixture_path, dtype="float32")
        if samples.ndim > 1:
            samples = samples[:, 0]
        result = check_clicks(samples, sr)
        assert result.passed, f"real Hebrew TTS turn wrongly flagged: {result.detail}"

    def test_few_steps_pass(self) -> None:
        # One sustained shift produces 2 edge events (rising + falling) —
        # below the threshold of 3.
        samples = np.zeros(16000, dtype=np.float32)
        samples[5000:5800] = 0.20
        result = check_clicks(samples, SR)
        assert result.passed

    def test_empty_passes(self) -> None:
        result = check_clicks(np.array([], dtype=np.float32), SR)
        assert result.passed

    def test_short_audio_passes(self) -> None:
        # All-zero short audio — virtual zero padding gives step ≈ 0 everywhere.
        samples = np.zeros(100, dtype=np.float32)
        result = check_clicks(samples, SR)
        assert result.passed

    def test_dc_offset_at_signal_start_caught(self) -> None:
        # Three sustained offsets; the first starts at sample 0.  Without
        # the boundary fix the t=0 rising edge is silently dropped and we
        # see 5 events; with the fix we see 6.  Asserting count ≥ 6 and
        # that the first detected peak is at t≈0 actually proves the
        # boundary edge was counted.
        samples = np.zeros(int(0.5 * SR), dtype=np.float32)
        samples[: int(0.06 * SR)] = 0.20  # offset starts AT sample 0
        samples[int(0.15 * SR) : int(0.20 * SR)] = -0.18
        samples[int(0.30 * SR) : int(0.35 * SR)] = 0.22
        result = check_clicks(samples, SR)
        assert not result.passed
        assert result.gate_name == "click_detection"
        assert result.detail is not None
        n = int(result.detail.split()[1])
        assert n >= 6, f"sample-0 edge dropped: only {n} events; expected ≥6"
        # Detail format: "...; first at t=[0.00, 0.06, 0.15]s" — the boundary
        # edge must appear as the first detected peak (within one sample of t=0).
        first_t_str = result.detail.split("[")[1].split(",")[0]
        assert float(first_t_str) < 1.0 / SR, f"sample-0 edge not the first peak: {result.detail}"

    def test_dc_offset_at_signal_end_caught(self) -> None:
        # Symmetric: a DC offset that persists through the final sample.
        # Without the boundary fix the trailing edge at sample n-1 is
        # dropped (5 events); with the fix it's caught (6 events) and the
        # last detected peak time lands within one sample of the final.
        n_samples = int(0.5 * SR)
        samples = np.zeros(n_samples, dtype=np.float32)
        samples[int(0.10 * SR) : int(0.15 * SR)] = 0.20
        samples[int(0.25 * SR) : int(0.30 * SR)] = -0.18
        samples[int(0.44 * SR) :] = 0.22  # offset persists to last sample
        result = check_clicks(samples, SR)
        assert not result.passed
        assert result.gate_name == "click_detection"
        assert result.detail is not None
        n = int(result.detail.split()[1])
        assert n >= 6, f"sample-(n-1) edge dropped: only {n} events; expected ≥6"
        # The detail surfaces the FIRST 3 peak timestamps, not the last —
        # so we re-run with the same rationale and check the count alone
        # proves the boundary edge was counted (5 events without fix, 6 with).


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
