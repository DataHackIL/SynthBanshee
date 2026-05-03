"""Unit tests for voice texture augmentation — VIC breathiness (M12)."""

from __future__ import annotations

import numpy as np
import pytest

from synthbanshee.augment.voice_texture import _MAX_NOISE_GAIN, add_breathiness

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sine_wave(freq_hz: float = 440.0, duration_s: float = 1.0, sr: int = 16000) -> np.ndarray:
    """Generate a sine wave at the given frequency."""
    t = np.arange(int(sr * duration_s), dtype=np.float32) / sr
    return (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_level_below_zero_raises(self) -> None:
        samples = _sine_wave()
        with pytest.raises(ValueError, match="level must be in"):
            add_breathiness(samples, 16000, -0.1)

    def test_level_above_one_raises(self) -> None:
        samples = _sine_wave()
        with pytest.raises(ValueError, match="level must be in"):
            add_breathiness(samples, 16000, 1.5)

    def test_sample_rate_too_low_raises(self) -> None:
        samples = _sine_wave(sr=12000)
        with pytest.raises(ValueError, match="too low"):
            add_breathiness(samples, 12000, 0.5)


# ---------------------------------------------------------------------------
# No-op cases
# ---------------------------------------------------------------------------


class TestNoOp:
    def test_level_zero_returns_copy(self) -> None:
        samples = _sine_wave()
        result = add_breathiness(samples, 16000, 0.0)
        np.testing.assert_array_equal(result, samples)
        # Must be a copy, not same buffer
        assert result is not samples

    def test_silence_input_returns_copy(self) -> None:
        silence = np.zeros(16000, dtype=np.float32)
        result = add_breathiness(silence, 16000, 0.8)
        np.testing.assert_array_equal(result, silence)


# ---------------------------------------------------------------------------
# Functional behavior
# ---------------------------------------------------------------------------


class TestFunctionalBehavior:
    def test_output_differs_from_input_at_nonzero_level(self) -> None:
        samples = _sine_wave()
        result = add_breathiness(samples, 16000, 0.5)
        assert not np.array_equal(result, samples)

    def test_output_same_length(self) -> None:
        samples = _sine_wave(duration_s=2.0)
        result = add_breathiness(samples, 16000, 0.7)
        assert len(result) == len(samples)

    def test_output_dtype_is_float32(self) -> None:
        samples = _sine_wave()
        result = add_breathiness(samples, 16000, 1.0)
        assert result.dtype == np.float32

    def test_higher_level_adds_more_noise(self) -> None:
        samples = _sine_wave()
        low = add_breathiness(samples, 16000, 0.2, rng_seed=42)
        high = add_breathiness(samples, 16000, 0.9, rng_seed=42)
        # Difference from original should be larger for higher level
        diff_low = float(np.sqrt(np.mean((low - samples) ** 2)))
        diff_high = float(np.sqrt(np.mean((high - samples) ** 2)))
        assert diff_high > diff_low

    def test_noise_gain_bounded_by_max(self) -> None:
        """Added noise RMS should not exceed _MAX_NOISE_GAIN * signal RMS."""
        samples = _sine_wave()
        sig_rms = float(np.sqrt(np.mean(samples**2)))
        result = add_breathiness(samples, 16000, 1.0, rng_seed=0)
        noise_only = result - samples
        noise_rms = float(np.sqrt(np.mean(noise_only**2)))
        # Allow 5% tolerance for filter edge effects
        assert noise_rms <= sig_rms * _MAX_NOISE_GAIN * 1.05

    def test_deterministic_with_same_seed(self) -> None:
        samples = _sine_wave()
        r1 = add_breathiness(samples, 16000, 0.5, rng_seed=123)
        r2 = add_breathiness(samples, 16000, 0.5, rng_seed=123)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seed_gives_different_result(self) -> None:
        samples = _sine_wave()
        r1 = add_breathiness(samples, 16000, 0.5, rng_seed=1)
        r2 = add_breathiness(samples, 16000, 0.5, rng_seed=2)
        assert not np.array_equal(r1, r2)


# ---------------------------------------------------------------------------
# Integration with SpeakerState breathiness_level
# ---------------------------------------------------------------------------


class TestSpeakerStateIntegration:
    def test_vic_i3_sets_breathiness(self) -> None:
        from synthbanshee.tts.speaker_state import SpeakerState

        state = SpeakerState(compute_breathiness=True)
        state.update(3, "VIC")
        assert state.breathiness_level > 0.0

    def test_vic_i5_higher_than_i3(self) -> None:
        from synthbanshee.tts.speaker_state import SpeakerState

        state_i3 = SpeakerState(compute_breathiness=True)
        state_i3.update(3, "VIC")

        state_i5 = SpeakerState(compute_breathiness=True)
        state_i5.update(5, "VIC")

        assert state_i5.breathiness_level > state_i3.breathiness_level

    def test_vic_i1_does_not_set_breathiness(self) -> None:
        from synthbanshee.tts.speaker_state import SpeakerState

        state = SpeakerState(compute_breathiness=True)
        state.update(1, "VIC")
        assert state.breathiness_level == 0.0

    def test_agg_does_not_set_breathiness(self) -> None:
        from synthbanshee.tts.speaker_state import SpeakerState

        state = SpeakerState(compute_breathiness=True)
        state.update(5, "AGG")
        assert state.breathiness_level == 0.0

    def test_breathiness_decays_on_deescalation(self) -> None:
        from synthbanshee.tts.speaker_state import SpeakerState

        state = SpeakerState(compute_breathiness=True)
        state.update(5, "VIC")
        high_level = state.breathiness_level
        state.update(1, "VIC")
        assert state.breathiness_level < high_level

    def test_breathiness_not_computed_when_disabled(self) -> None:
        from synthbanshee.tts.speaker_state import SpeakerState

        state = SpeakerState(compute_breathiness=False)
        state.update(5, "VIC")
        assert state.breathiness_level == 0.0


# ---------------------------------------------------------------------------
# RunConfig flag
# ---------------------------------------------------------------------------


class TestRunConfigFlag:
    def test_enable_breathiness_defaults_false(self) -> None:
        from synthbanshee.config.run_config import RunConfig, TypologyTarget

        rc = RunConfig(
            run_id="test",
            project="she_proves",
            targets=[TypologyTarget(violence_typology="SV", count=1)],
        )
        assert rc.enable_breathiness is False

    def test_enable_breathiness_can_be_set_true(self) -> None:
        from synthbanshee.config.run_config import RunConfig, TypologyTarget

        rc = RunConfig(
            run_id="test",
            project="she_proves",
            targets=[TypologyTarget(violence_typology="SV", count=1)],
            enable_breathiness=True,
        )
        assert rc.enable_breathiness is True
