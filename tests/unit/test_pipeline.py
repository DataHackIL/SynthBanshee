"""Unit tests for synthbanshee.augment.pipeline (M16 orchestrator)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from synthbanshee.augment.pipeline import (
    _SNR_BANDS,
    augment_scene,
    generate_variant_configs,
    sample_snr_db,
)
from synthbanshee.augment.types import AugmentationResult
from synthbanshee.config.acoustic_config import AcousticSceneConfig, BackgroundEvent

_SR = 16_000
_N = _SR * 2  # 2 s


def _sine(n: int = _N, sr: int = _SR) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / sr
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _make_config(
    room_type: str = "small_bedroom",
    device: str = "phone_in_hand",
    snr_target_db: float = 20.0,
    background_events: list[BackgroundEvent] | None = None,
) -> AcousticSceneConfig:
    return AcousticSceneConfig(
        room_type=room_type,
        device=device,
        speaker_distance_meters=1.0,
        victim_distance_meters=1.0,
        snr_target_db=snr_target_db,
        background_events=background_events or [],
    )


# ---------------------------------------------------------------------------
# sample_snr_db tests
# ---------------------------------------------------------------------------


class TestSampleSnrDb:
    def test_returns_float(self):
        rng = np.random.default_rng(0)
        snr = sample_snr_db(rng)
        assert isinstance(snr, float)

    def test_within_overall_range(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            snr = sample_snr_db(rng)
            assert 5.0 <= snr <= 40.0

    def test_distribution_roughly_correct(self):
        """With enough samples, the distribution should match the spec bands."""
        rng = np.random.default_rng(123)
        n_samples = 10_000
        band_counts = [0] * len(_SNR_BANDS)
        for _ in range(n_samples):
            snr = sample_snr_db(rng)
            for i, (_, lo, hi) in enumerate(_SNR_BANDS):
                if lo <= snr <= hi:
                    band_counts[i] += 1
                    break

        # Check each band is within ±5% of expected
        for i, (prob, _, _) in enumerate(_SNR_BANDS):
            actual_frac = band_counts[i] / n_samples
            assert abs(actual_frac - prob) < 0.05, (
                f"Band {i} ({_SNR_BANDS[i]}): expected {prob:.0%}, got {actual_frac:.0%}"
            )

    def test_reproducible_with_same_seed(self):
        v1 = sample_snr_db(np.random.default_rng(7))
        v2 = sample_snr_db(np.random.default_rng(7))
        assert v1 == v2


# ---------------------------------------------------------------------------
# augment_scene tests
# ---------------------------------------------------------------------------


class TestAugmentScene:
    def test_returns_augmentation_result(self):
        samples = _sine()
        config = _make_config()
        result = augment_scene(samples, _SR, config, rng_seed=0)
        assert isinstance(result, AugmentationResult)

    def test_output_shape_matches_input(self):
        samples = _sine()
        config = _make_config()
        result = augment_scene(samples, _SR, config, rng_seed=0)
        assert result.samples.shape == samples.shape

    def test_output_dtype_float32(self):
        samples = _sine()
        config = _make_config()
        result = augment_scene(samples, _SR, config, rng_seed=0)
        assert result.samples.dtype == np.float32

    def test_output_differs_from_input(self):
        samples = _sine()
        config = _make_config()
        result = augment_scene(samples, _SR, config, rng_seed=0)
        assert not np.allclose(result.samples, samples)

    def test_metadata_populated(self):
        samples = _sine()
        config = _make_config(room_type="clinic_office", device="pi_budget_mic")
        result = augment_scene(samples, _SR, config, rng_seed=0)
        assert result.room_type == "clinic_office"
        assert result.device == "pi_budget_mic"
        assert result.ir_source == "pyroomacoustics_ism"
        assert result.sample_rate == _SR
        assert isinstance(result.snr_db_actual, float)

    def test_reproducible_with_same_seed(self):
        samples = _sine()
        config = _make_config()
        r1 = augment_scene(samples, _SR, config, rng_seed=42)
        r2 = augment_scene(samples, _SR, config, rng_seed=42)
        np.testing.assert_array_equal(r1.samples, r2.samples)

    def test_different_seeds_differ(self):
        samples = _sine()
        config = _make_config()
        r1 = augment_scene(samples, _SR, config, rng_seed=0)
        r2 = augment_scene(samples, _SR, config, rng_seed=99)
        assert not np.allclose(r1.samples, r2.samples)

    def test_with_background_events(self):
        samples = _sine()
        ev = BackgroundEvent(type="tv_ambient", loop=True, onset_seconds=0.0)
        config = _make_config(background_events=[ev])
        result = augment_scene(
            samples,
            _SR,
            config,
            rng_seed=0,
            assets_dir=Path("/tmp/nonexistent"),
        )
        assert len(result.events) >= 1

    def test_acoustic_scene_dict(self):
        samples = _sine()
        config = _make_config()
        result = augment_scene(samples, _SR, config, rng_seed=0)
        d = result.acoustic_scene_dict
        assert "room_type" in d
        assert "device" in d
        assert "snr_db_actual" in d


# ---------------------------------------------------------------------------
# generate_variant_configs tests
# ---------------------------------------------------------------------------


class TestGenerateVariantConfigs:
    def test_returns_correct_count(self):
        base = _make_config()
        variants = generate_variant_configs(base, n_variants=3, rng_seed=0)
        assert len(variants) == 3

    def test_all_variants_are_valid_configs(self):
        base = _make_config()
        variants = generate_variant_configs(base, n_variants=4, rng_seed=0)
        for v in variants:
            assert isinstance(v, AcousticSceneConfig)

    def test_variants_differ_from_each_other(self):
        base = _make_config()
        variants = generate_variant_configs(base, n_variants=4, rng_seed=0)
        # At least some variants should differ (room, device, distance, or SNR)
        configs_set = set()
        for v in variants:
            configs_set.add((v.room_type, v.device, v.speaker_distance_meters, v.snr_target_db))
        assert len(configs_set) > 1

    def test_snr_within_m16_range(self):
        base = _make_config()
        variants = generate_variant_configs(base, n_variants=10, rng_seed=42)
        for v in variants:
            assert 5.0 <= v.snr_target_db <= 40.0

    def test_speaker_distance_from_options(self):
        base = _make_config()
        variants = generate_variant_configs(base, n_variants=20, rng_seed=0)
        distances = {v.speaker_distance_meters for v in variants}
        assert distances.issubset({0.5, 1.5, 3.5})

    def test_preferred_devices_respected(self):
        base = _make_config()
        variants = generate_variant_configs(
            base,
            n_variants=10,
            rng_seed=0,
            preferred_devices=["phone_in_pocket", "phone_on_table"],
        )
        for v in variants:
            assert v.device in ("phone_in_pocket", "phone_on_table")

    def test_preferred_rooms_respected(self):
        base = _make_config()
        variants = generate_variant_configs(
            base,
            n_variants=10,
            rng_seed=0,
            preferred_room_types=["clinic_office", "welfare_office"],
        )
        for v in variants:
            assert v.room_type in ("clinic_office", "welfare_office")

    def test_reproducible_with_same_seed(self):
        base = _make_config()
        v1 = generate_variant_configs(base, n_variants=3, rng_seed=7)
        v2 = generate_variant_configs(base, n_variants=3, rng_seed=7)
        for a, b in zip(v1, v2, strict=True):
            assert a.room_type == b.room_type
            assert a.device == b.device
            assert a.snr_target_db == b.snr_target_db

    def test_background_events_preserved(self):
        ev = BackgroundEvent(type="tv_ambient", loop=True, onset_seconds=0.0)
        base = _make_config(background_events=[ev])
        variants = generate_variant_configs(base, n_variants=2, rng_seed=0)
        for v in variants:
            assert len(v.background_events) == 1
            assert v.background_events[0].type == "tv_ambient"
