"""Tier B acoustic augmentation pipeline orchestrator (M16).

Chains room simulation → device profiling → noise mixing into a single
``augment_scene()`` call.  Also provides SNR band sampling and variant
generation for producing 2–4 Tier B variants per Tier A scene.

Spec reference: docs/audio_generation_v3_design.md §M16
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from synthbanshee.augment.device_profiles import DeviceProfiler
from synthbanshee.augment.noise_mixer import NoiseMixer
from synthbanshee.augment.room_sim import RoomSimulator
from synthbanshee.augment.types import AugmentationResult
from synthbanshee.config.acoustic_config import AcousticSceneConfig

# SNR band distribution (M16 spec):
#   50% at 18–30 dB, 30% at 10–18 dB, 10% at 5–10 dB, 10% at 30–40 dB
_SNR_BANDS: list[tuple[float, float, float]] = [
    # (probability, snr_lo, snr_hi)
    (0.50, 18.0, 30.0),
    (0.30, 10.0, 18.0),
    (0.10, 5.0, 10.0),
    (0.10, 30.0, 40.0),
]


def sample_snr_db(rng: np.random.Generator) -> float:
    """Sample an SNR target from the M16 distribution bands.

    Distribution: 50% at 18–30 dB, 30% at 10–18 dB,
    10% at 5–10 dB, 10% at 30–40 dB.
    """
    probs = [band[0] for band in _SNR_BANDS]
    idx = int(rng.choice(len(_SNR_BANDS), p=probs))
    _, lo, hi = _SNR_BANDS[idx]
    return float(rng.uniform(lo, hi))


def augment_scene(
    samples: np.ndarray,
    sr: int,
    config: AcousticSceneConfig,
    rng_seed: int = 0,
    assets_dir: Path = Path("assets"),
    phase_boundaries: dict[str, float] | None = None,
) -> AugmentationResult:
    """Apply the full Tier B augmentation chain: room → device → noise.

    Args:
        samples: Float32 mono audio at *sr* Hz (post-preprocessing, with
            silence padding already applied).
        sr: Sample rate (should be 16000).
        config: Acoustic scene configuration controlling room, device, and
            noise parameters.
        rng_seed: Seed for reproducible augmentation.
        assets_dir: Root of the assets tree for noise/SFX file loading.
        phase_boundaries: Optional phase name → onset_s mapping for SFX
            onset resolution.

    Returns:
        AugmentationResult with the augmented samples and event log.
    """
    # Room simulation
    reverbed = RoomSimulator().apply(samples, sr, config, rng_seed=rng_seed)

    # Device profiling
    device_colored = DeviceProfiler().apply(reverbed, sr, config.device, rng_seed=rng_seed)

    # Noise mixing
    aug_samples, aug_events, snr_actual = NoiseMixer(assets_dir=assets_dir).mix(
        device_colored,
        sr,
        config,
        rng_seed=rng_seed,
        phase_boundaries=phase_boundaries,
    )

    return AugmentationResult(
        samples=aug_samples,
        sample_rate=sr,
        room_type=config.room_type,
        device=config.device,
        ir_source=config.ir_source,
        speaker_distance_meters=config.speaker_distance_meters,
        snr_db_actual=snr_actual,
        events=aug_events,
    )


def generate_variant_configs(
    base_config: AcousticSceneConfig,
    n_variants: int,
    rng_seed: int,
    preferred_devices: list[str] | None = None,
    preferred_room_types: list[str] | None = None,
) -> list[AcousticSceneConfig]:
    """Generate N variant acoustic configs for Tier B augmentation.

    Each variant gets a different combination of room type, device profile,
    speaker distance, and SNR target sampled from the M16 distribution.

    Args:
        base_config: The base acoustic scene config to vary.
        n_variants: Number of variants to generate (typically 2–4).
        rng_seed: Seed for reproducible variant generation.
        preferred_devices: If provided, sample devices from this list instead
            of all available devices.
        preferred_room_types: If provided, sample room types from this list
            instead of all available.

    Returns:
        List of *n_variants* AcousticSceneConfig objects, each with a
        different room/device/SNR combination.
    """
    from synthbanshee.config.acoustic_config import _VALID_DEVICES, _VALID_ROOM_TYPES

    rng = np.random.default_rng(rng_seed)

    devices = list(preferred_devices) if preferred_devices else sorted(_VALID_DEVICES)
    rooms = list(preferred_room_types) if preferred_room_types else sorted(_VALID_ROOM_TYPES)

    # Speaker distance range: near (0.5 m), mid (1.5 m), far (3.5 m)
    distance_options = [0.5, 1.5, 3.5]

    variants: list[AcousticSceneConfig] = []
    for _ in range(n_variants):
        room = rooms[int(rng.integers(len(rooms)))]
        device = devices[int(rng.integers(len(devices)))]
        distance = distance_options[int(rng.integers(len(distance_options)))]
        snr = sample_snr_db(rng)

        variant = base_config.model_copy(
            update={
                "room_type": room,
                "device": device,
                "speaker_distance_meters": distance,
                "snr_target_db": round(snr, 1),
            }
        )
        variants.append(variant)

    return variants
