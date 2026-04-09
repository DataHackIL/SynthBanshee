"""Unit tests for synthbanshee.augment.room_sim."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from synthbanshee.augment.room_sim import _ROOM_PRESETS, RoomSimulator
from synthbanshee.config.acoustic_config import AcousticSceneConfig

# Short 0.5-s synthetic speech signal at 16 kHz
_SR = 16_000
_N = _SR // 2  # 0.5 s


def _make_samples(n: int = _N, sr: int = _SR) -> np.ndarray:
    """Return a deterministic float32 sine wave."""
    t = np.arange(n, dtype=np.float32) / sr
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _make_config(
    room_type: str = "small_bedroom",
    speaker_distance_meters: float = 1.0,
) -> AcousticSceneConfig:
    return AcousticSceneConfig(
        room_type=room_type,
        device="phone_in_hand",
        speaker_distance_meters=speaker_distance_meters,
        victim_distance_meters=1.0,
    )


# ---------------------------------------------------------------------------
# _ROOM_PRESETS sanity checks
# ---------------------------------------------------------------------------


def test_all_preset_room_types_present():
    expected = {
        "small_bedroom",
        "apartment_kitchen",
        "living_room",
        "clinic_office",
        "welfare_office",
        "open_office_corridor",
    }
    assert set(_ROOM_PRESETS.keys()) == expected


def test_preset_dims_consistent():
    """Each preset's min dims must be < max dims for all three axes."""
    for room_type, (dims_min, dims_max, _) in _ROOM_PRESETS.items():
        for axis, (lo, hi) in enumerate(zip(dims_min, dims_max, strict=False)):
            assert lo < hi, f"{room_type} axis {axis}: min={lo} >= max={hi}"


def test_preset_rt60_ranges_positive():
    for room_type, (_, _, (rt60_lo, rt60_hi)) in _ROOM_PRESETS.items():
        assert 0 < rt60_lo < rt60_hi, f"{room_type} RT60 range invalid"


# ---------------------------------------------------------------------------
# RoomSimulator.apply — output shape and dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("room_type", list(_ROOM_PRESETS.keys()))
def test_output_length_matches_input(room_type):
    sim = RoomSimulator()
    samples = _make_samples()
    config = _make_config(room_type=room_type)
    out = sim.apply(samples, _SR, config, rng_seed=0)
    assert out.shape == samples.shape, f"{room_type}: shape mismatch"


@pytest.mark.parametrize("room_type", list(_ROOM_PRESETS.keys()))
def test_output_dtype_is_float32(room_type):
    sim = RoomSimulator()
    samples = _make_samples()
    config = _make_config(room_type=room_type)
    out = sim.apply(samples, _SR, config, rng_seed=0)
    assert out.dtype == np.float32, f"{room_type}: dtype {out.dtype}"


def test_output_is_not_identical_to_input():
    """Room sim should change the signal (not a no-op)."""
    sim = RoomSimulator()
    samples = _make_samples()
    config = _make_config()
    out = sim.apply(samples, _SR, config, rng_seed=0)
    assert not np.allclose(out, samples), "RoomSimulator returned unmodified input"


def test_reproducible_with_same_seed():
    sim = RoomSimulator()
    samples = _make_samples()
    config = _make_config()
    out1 = sim.apply(samples, _SR, config, rng_seed=42)
    out2 = sim.apply(samples, _SR, config, rng_seed=42)
    np.testing.assert_array_equal(out1, out2)


def test_different_seeds_differ():
    sim = RoomSimulator()
    samples = _make_samples()
    config = _make_config()
    out1 = sim.apply(samples, _SR, config, rng_seed=0)
    out2 = sim.apply(samples, _SR, config, rng_seed=99)
    assert not np.allclose(out1, out2)


def test_unknown_room_type_raises():
    sim = RoomSimulator()
    config = AcousticSceneConfig.model_construct(
        room_type="nonexistent_room",
        room_dimensions_range=None,
        rt60_range=None,
        speaker_distance_meters=1.0,
    )
    with pytest.raises(KeyError):
        sim.apply(_make_samples(), _SR, config, rng_seed=0)


def test_custom_room_dimensions_used():
    """Explicit room_dimensions_range should be accepted without error."""
    from synthbanshee.config.acoustic_config import RoomDimensionsRange

    config = AcousticSceneConfig(
        room_type="living_room",
        device="phone_in_hand",
        speaker_distance_meters=1.0,
        victim_distance_meters=1.0,
        room_dimensions_range=RoomDimensionsRange(
            min=[5.0, 4.0, 2.5],
            max=[5.5, 4.5, 3.0],
        ),
    )
    sim = RoomSimulator()
    out = sim.apply(_make_samples(), _SR, config, rng_seed=0)
    assert out.shape == (_N,)


def test_low_ceiling_room_clamps_mic_src_z():
    """Room height < 1.2 m: mic/src z must be clamped inside the room bounds."""
    from synthbanshee.config.acoustic_config import RoomDimensionsRange

    # Room height of 1.0 m is below both default mic (1.2 m) and src (1.6 m)
    config = AcousticSceneConfig(
        room_type="small_bedroom",
        device="phone_in_hand",
        speaker_distance_meters=0.3,
        victim_distance_meters=0.3,
        room_dimensions_range=RoomDimensionsRange(
            min=[3.0, 3.0, 1.0],
            max=[3.1, 3.1, 1.0],
        ),
    )
    sim = RoomSimulator()
    # Should not raise — mic/src z are clamped to [0.1, 0.9]
    out = sim.apply(_make_samples(), _SR, config, rng_seed=0)
    assert out.shape == (_N,)


def test_custom_rt60_range_used():
    config = AcousticSceneConfig(
        room_type="clinic_office",
        device="phone_in_hand",
        speaker_distance_meters=1.0,
        victim_distance_meters=1.0,
        rt60_range=[0.10, 0.15],
    )
    sim = RoomSimulator()
    out = sim.apply(_make_samples(), _SR, config, rng_seed=0)
    assert out.shape == (_N,)


def test_silent_input_produces_silent_output():
    sim = RoomSimulator()
    silence = np.zeros(_N, dtype=np.float32)
    config = _make_config()
    out = sim.apply(silence, _SR, config, rng_seed=0)
    # After convolving silence with any RIR the result must still be (near) silence
    assert np.max(np.abs(out)) < 1e-6


def test_output_zero_padded_when_reverbed_shorter_than_input():
    """Cover the zero-pad branch when the simulated signal is shorter than input."""

    sim = RoomSimulator()
    samples = _make_samples()
    config = _make_config()

    # Patch ShoeBox so mic_array.signals[0] is shorter than the input
    short_reverb = np.zeros(_N // 2, dtype=np.float32)
    mock_room = MagicMock()
    mock_room.mic_array.signals = [short_reverb]

    with (
        patch("pyroomacoustics.ShoeBox", return_value=mock_room),
        patch("pyroomacoustics.inverse_sabine", return_value=(0.5, 8)),
        patch("pyroomacoustics.Material", return_value=MagicMock()),
    ):
        out = sim.apply(samples, _SR, config, rng_seed=0)

    assert len(out) == _N
    assert out.dtype == np.float32
    # Tail should be zero-padded
    assert np.all(out[_N // 2 :] == 0.0)
