"""Room impulse response simulation using pyroomacoustics (ShoeBox / ISM).

For each clip, a random room geometry is drawn within the bounds specified in
AcousticSceneConfig.  If no bounds are given, per-room-type preset defaults are
used.  The speech signal is convolved with the resulting RIR and the output is
truncated to the original clip length.

Spec reference: docs/spec.md §3.1 (Stage 3 acoustic augmentation)
"""

from __future__ import annotations

import numpy as np

from synthbanshee.config.acoustic_config import AcousticSceneConfig

_ISM_ORDER_MAX = 12  # cap to keep simulation tractable

# Preset fallback bounds: (dims_min, dims_max, rt60_range)
# dims = [width_m, length_m, height_m]
_ROOM_PRESETS: dict[str, tuple[list[float], list[float], tuple[float, float]]] = {
    "small_bedroom": (
        [2.5, 2.5, 2.3],
        [3.5, 3.5, 2.7],
        (0.20, 0.40),
    ),
    "apartment_kitchen": (
        [2.5, 2.0, 2.3],
        [4.0, 3.5, 2.7],
        (0.15, 0.35),
    ),
    "living_room": (
        [4.0, 3.5, 2.4],
        [6.5, 5.5, 3.0],
        (0.30, 0.60),
    ),
    "clinic_office": (
        [3.0, 2.5, 2.4],
        [5.0, 4.0, 3.0],
        (0.20, 0.40),
    ),
    "welfare_office": (
        [3.5, 3.0, 2.4],
        [6.0, 5.0, 3.0],
        (0.25, 0.50),
    ),
    "open_office_corridor": (
        [6.0, 2.5, 2.4],
        [12.0, 4.0, 3.5],
        (0.30, 0.70),
    ),
}


class RoomSimulator:
    """Apply room acoustics to a mono float32 signal via pyroomacoustics ShoeBox."""

    def apply(
        self,
        samples: np.ndarray,
        sr: int,
        config: AcousticSceneConfig,
        rng_seed: int = 0,
    ) -> np.ndarray:
        """Convolve speech with a room impulse response derived from config.

        Args:
            samples: Float32 mono audio at ``sr`` Hz.
            sr: Sample rate in Hz (should be 16 000).
            config: AcousticSceneConfig controlling room geometry, RT60, and
                    source distance.  Unknown room_type raises KeyError.
            rng_seed: Seed for reproducible room geometry / placement sampling.

        Returns:
            Float32 mono array of the same length as ``samples``.

        Raises:
            ImportError: if pyroomacoustics is not installed.
            KeyError: if config.room_type is not in _ROOM_PRESETS.
        """
        import pyroomacoustics as pra  # Phase 2 dependency — deferred import

        rng = np.random.default_rng(rng_seed)
        dims_min, dims_max, preset_rt60 = _ROOM_PRESETS[config.room_type]

        # Resolve room dimensions
        if config.room_dimensions_range is not None:
            lo = config.room_dimensions_range.min
            hi = config.room_dimensions_range.max
        else:
            lo, hi = dims_min, dims_max
        dims = [float(rng.uniform(lo[i], hi[i])) for i in range(3)]

        # Resolve RT60
        if config.rt60_range is not None:
            rt60_lo, rt60_hi = config.rt60_range
        else:
            rt60_lo, rt60_hi = preset_rt60
        rt60 = float(rng.uniform(rt60_lo, rt60_hi))

        # Absorption coefficient from Sabine's formula via pyroomacoustics helper
        e_absorption, order = pra.inverse_sabine(rt60, dims)
        order = min(int(order), _ISM_ORDER_MAX)

        room = pra.ShoeBox(
            dims,
            fs=sr,
            materials=pra.Material(e_absorption),
            max_order=order,
            ray_tracing=False,
            air_absorption=True,
        )

        # Microphone at room centre, ear height (1.2 m)
        mic = np.array([dims[0] / 2.0, dims[1] / 2.0, 1.2])

        # Source at speaker_distance_meters from mic (random azimuth, mouth height)
        # Clamp so source stays inside the room (10 cm margin from walls)
        max_dist = min(
            config.speaker_distance_meters,
            min(dims[0], dims[1]) * 0.85,
        )
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        src = np.array(
            [
                float(np.clip(mic[0] + max_dist * np.cos(angle), 0.1, dims[0] - 0.1)),
                float(np.clip(mic[1] + max_dist * np.sin(angle), 0.1, dims[1] - 0.1)),
                1.6,  # mouth height
            ]
        )

        room.add_source(src.tolist(), signal=samples.astype(np.float32))
        room.add_microphone(mic.reshape(3, 1))
        room.simulate()

        reverbed = room.mic_array.signals[0].astype(np.float32)

        # Truncate or zero-pad to match input length
        n = len(samples)
        if len(reverbed) >= n:
            return reverbed[:n]
        return np.pad(reverbed, (0, n - len(reverbed))).astype(np.float32)
