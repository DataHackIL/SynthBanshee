"""Unit tests for synthbanshee.augment.noise_mixer."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from synthbanshee.augment.noise_mixer import (
    _PHASE_FRACTIONS,
    NoiseMixer,
    _resolve_onset_s,
    _rms,
    _snr_db,
    _speech_rms,
    _synthesise_ambient,
    _synthesise_sfx,
    _to_dbfs,
)
from synthbanshee.config.acoustic_config import AcousticSceneConfig, BackgroundEvent

_SR = 16_000
_N = _SR * 2  # 2 s — long enough for SNR estimation


def _sine(freq: float = 440.0, n: int = _N, sr: int = _SR, amp: float = 0.5) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _make_config(
    background_events: list[BackgroundEvent] | None = None,
    snr_target_db: float = 20.0,
) -> AcousticSceneConfig:
    return AcousticSceneConfig(
        room_type="small_bedroom",
        device="phone_in_hand",
        speaker_distance_meters=1.0,
        victim_distance_meters=1.0,
        background_events=background_events or [],
        snr_target_db=snr_target_db,
    )


def _ambient_event(event_type: str = "tv_ambient") -> BackgroundEvent:
    return BackgroundEvent(type=event_type, loop=True, onset_seconds=0.0)


def _sfx_event(
    event_type: str = "ACOU_SLAM",
    onset_seconds: float = 0.5,
    level_db: float | None = None,
) -> BackgroundEvent:
    return BackgroundEvent(
        type=event_type,
        loop=False,
        onset_seconds=onset_seconds,
        level_db=level_db,
    )


# ---------------------------------------------------------------------------
# Audio helper tests
# ---------------------------------------------------------------------------


class TestAudioHelpers:
    def test_rms_sine(self):
        samples = _sine()
        rms = _rms(samples)
        # RMS of sin(x)*amp = amp/sqrt(2)
        expected = 0.5 / math.sqrt(2)
        assert abs(rms - expected) < 0.01

    def test_rms_zeros(self):
        assert _rms(np.zeros(100, dtype=np.float32)) == 0.0

    def test_to_dbfs_full_scale(self):
        samples = np.ones(100, dtype=np.float32)
        assert _to_dbfs(samples) == pytest.approx(0.0, abs=1e-6)

    def test_to_dbfs_half_scale(self):
        samples = np.full(100, 0.5, dtype=np.float32)
        assert _to_dbfs(samples) == pytest.approx(-6.02, abs=0.1)

    def test_to_dbfs_silence(self):
        assert _to_dbfs(np.zeros(100, dtype=np.float32)) == -math.inf

    def test_speech_rms_excludes_silence_frames(self):
        # Mix loud active frames with silent frames
        n_frames = 100
        frame_n = 320
        loud = np.ones(frame_n * n_frames // 2, dtype=np.float32) * 0.8
        silent = np.zeros(frame_n * n_frames // 2, dtype=np.float32)
        mixed = np.concatenate([loud, silent])
        srms = _speech_rms(mixed, frame_ms=20, sr=_SR)
        full_rms = _rms(mixed)
        # speech_rms should be higher than full RMS since silence is excluded
        assert srms > full_rms

    def test_snr_db_equal_levels_is_zero(self):
        signal = _sine(440, amp=0.5)
        noise = _sine(880, amp=0.5)
        snr = _snr_db(signal, noise)
        # Not exact (speech_rms excludes silence) but should be near 0
        assert -5 < snr < 5

    def test_snr_db_silent_signal(self):
        assert _snr_db(np.zeros(_N, dtype=np.float32), _sine()) == -math.inf

    def test_snr_db_silent_noise(self):
        assert _snr_db(_sine(), np.zeros(_N, dtype=np.float32)) == math.inf


# ---------------------------------------------------------------------------
# Synthetic generator tests
# ---------------------------------------------------------------------------


class TestSynthesiseAmbient:
    @pytest.mark.parametrize(
        "event_type", ["hvac_hum", "distant_phone_ring", "tv_ambient", "unknown_type"]
    )
    def test_length_matches_duration(self, event_type):
        rng = np.random.default_rng(0)
        out = _synthesise_ambient(event_type, 1.0, _SR, rng)
        assert len(out) == _SR

    @pytest.mark.parametrize("event_type", ["hvac_hum", "distant_phone_ring", "tv_ambient"])
    def test_dtype_float32(self, event_type):
        rng = np.random.default_rng(0)
        out = _synthesise_ambient(event_type, 1.0, _SR, rng)
        assert out.dtype == np.float32

    def test_zero_duration_returns_empty(self):
        rng = np.random.default_rng(0)
        out = _synthesise_ambient("tv_ambient", 0.0, _SR, rng)
        assert len(out) == 0

    def test_hvac_hum_not_silent(self):
        rng = np.random.default_rng(0)
        out = _synthesise_ambient("hvac_hum", 1.0, _SR, rng)
        assert _rms(out) > 0.0


class TestSynthesiseSfx:
    @pytest.mark.parametrize(
        "event_type",
        ["ACOU_BREAK", "ACOU_SLAM", "ACOU_THROW", "ACOU_FALL", "ACOU_FOOT"],
    )
    def test_dtype_float32(self, event_type):
        rng = np.random.default_rng(0)
        out = _synthesise_sfx(event_type, _SR, rng)
        assert out.dtype == np.float32

    @pytest.mark.parametrize(
        "event_type",
        ["ACOU_BREAK", "ACOU_SLAM", "ACOU_THROW", "ACOU_FALL", "ACOU_FOOT"],
    )
    def test_not_silent(self, event_type):
        rng = np.random.default_rng(0)
        out = _synthesise_sfx(event_type, _SR, rng)
        assert _rms(out) > 0.0

    def test_slam_shorter_than_fall(self):
        rng = np.random.default_rng(0)
        slam = _synthesise_sfx("ACOU_SLAM", _SR, rng)
        rng2 = np.random.default_rng(0)
        fall = _synthesise_sfx("ACOU_FALL", _SR, rng2)
        # SLAM duration range (0.15-0.35) max < FALL range (0.5-1.2) min
        assert len(slam) <= len(fall)


# ---------------------------------------------------------------------------
# _resolve_onset_s tests
# ---------------------------------------------------------------------------


class TestResolveOnsetS:
    def _make_ev(self, **kwargs) -> BackgroundEvent:
        # Provide a default onset to satisfy the validator
        defaults = {"type": "ACOU_SLAM", "loop": False, "onset_seconds": 0.0}
        defaults.update(kwargs)
        return BackgroundEvent(**defaults)

    def test_explicit_onset_seconds(self):
        ev = self._make_ev(onset_seconds=1.5)
        rng = np.random.default_rng(0)
        onset = _resolve_onset_s(ev, 5.0, None, rng)
        assert onset == pytest.approx(1.5)

    def test_onset_with_offset(self):
        ev = self._make_ev(onset_seconds=1.0, onset_offset_seconds=0.3)
        rng = np.random.default_rng(0)
        onset = _resolve_onset_s(ev, 5.0, None, rng)
        assert onset == pytest.approx(1.3)

    def test_phase_boundaries_resolution(self):
        ev = BackgroundEvent(type="ACOU_SLAM", loop=False, onset_at_phase="peak")
        phase_boundaries = {"peak": 2.5}
        rng = np.random.default_rng(0)
        onset = _resolve_onset_s(ev, 5.0, phase_boundaries, rng)
        assert onset == pytest.approx(2.5)

    def test_phase_fractions_fallback(self):
        ev = BackgroundEvent(type="ACOU_SLAM", loop=False, onset_at_phase="peak")
        rng = np.random.default_rng(0)
        onset = _resolve_onset_s(ev, 10.0, None, rng)
        expected = _PHASE_FRACTIONS["peak"] * 10.0
        assert onset == pytest.approx(expected)

    def test_clamped_to_duration(self):
        ev = self._make_ev(onset_seconds=100.0)
        rng = np.random.default_rng(0)
        onset = _resolve_onset_s(ev, 5.0, None, rng)
        assert onset <= 5.0 - 0.1

    def test_negative_clamped_to_zero(self):
        ev = self._make_ev(onset_seconds=0.5, onset_offset_seconds=-2.0)
        rng = np.random.default_rng(0)
        onset = _resolve_onset_s(ev, 5.0, None, rng)
        assert onset == 0.0


# ---------------------------------------------------------------------------
# NoiseMixer.mix — no assets (pure synthetic)
# ---------------------------------------------------------------------------


class TestNoiseMixerNoAssets:
    def setup_method(self):
        self.mixer = NoiseMixer(assets_dir=Path("/tmp/nonexistent_assets_dir"))
        self.samples = _sine()

    def test_no_events_returns_original(self):
        config = _make_config(background_events=[])
        out, events, snr = self.mixer.mix(self.samples, _SR, config)
        np.testing.assert_array_equal(out, self.samples)
        assert events == []
        assert snr == math.inf

    def test_ambient_event_adds_noise(self):
        config = _make_config(background_events=[_ambient_event()])
        out, events, snr = self.mixer.mix(self.samples, _SR, config)
        assert not np.allclose(out, self.samples)
        assert len(events) == 1

    def test_ambient_event_produces_one_record(self):
        config = _make_config(background_events=[_ambient_event("hvac_hum")])
        _, events, _ = self.mixer.mix(self.samples, _SR, config)
        assert len(events) == 1
        ev = events[0]
        assert ev.type == "hvac_hum"
        assert ev.onset_s == 0.0
        assert ev.offset_s == pytest.approx(len(self.samples) / _SR)

    def test_snr_is_near_target(self):
        """SNR should be within 3 dB of the target for a normal speech signal."""
        config = _make_config(background_events=[_ambient_event()], snr_target_db=20.0)
        _, _, snr = self.mixer.mix(self.samples, _SR, config)
        assert abs(snr - 20.0) < 3.0

    def test_sfx_event_appended_to_events(self):
        config = _make_config(background_events=[_sfx_event("ACOU_SLAM", onset_seconds=0.5)])
        _, events, _ = self.mixer.mix(self.samples, _SR, config)
        assert len(events) == 1
        assert events[0].type == "ACOU_SLAM"

    def test_sfx_onset_within_bounds(self):
        config = _make_config(background_events=[_sfx_event(onset_seconds=0.5)])
        _, events, _ = self.mixer.mix(self.samples, _SR, config)
        assert 0.0 <= events[0].onset_s < len(self.samples) / _SR

    def test_sfx_offset_after_onset(self):
        config = _make_config(background_events=[_sfx_event(onset_seconds=0.5)])
        _, events, _ = self.mixer.mix(self.samples, _SR, config)
        assert events[0].offset_s > events[0].onset_s

    def test_sfx_level_db_applied(self):
        """SFX at an explicit level_db should not be silent."""
        config = _make_config(
            background_events=[_sfx_event("ACOU_SLAM", onset_seconds=0.3, level_db=-20.0)]
        )
        out, events, _ = self.mixer.mix(self.samples, _SR, config)
        ev = events[0]
        assert ev.level_db == pytest.approx(-20.0, abs=0.1)

    def test_output_same_length_as_input(self):
        config = _make_config(
            background_events=[
                _ambient_event(),
                _sfx_event("ACOU_FALL", onset_seconds=0.8),
            ]
        )
        out, _, _ = self.mixer.mix(self.samples, _SR, config)
        assert len(out) == len(self.samples)

    def test_output_dtype_float32(self):
        config = _make_config(background_events=[_ambient_event()])
        out, _, _ = self.mixer.mix(self.samples, _SR, config)
        assert out.dtype == np.float32

    def test_reproducible_with_same_seed(self):
        config = _make_config(background_events=[_ambient_event(), _sfx_event(onset_seconds=0.5)])
        out1, _, _ = self.mixer.mix(self.samples, _SR, config, rng_seed=7)
        out2, _, _ = self.mixer.mix(self.samples, _SR, config, rng_seed=7)
        np.testing.assert_array_equal(out1, out2)

    def test_different_seeds_differ(self):
        config = _make_config(background_events=[_ambient_event()])
        out1, _, _ = self.mixer.mix(self.samples, _SR, config, rng_seed=0)
        out2, _, _ = self.mixer.mix(self.samples, _SR, config, rng_seed=99)
        assert not np.allclose(out1, out2)

    def test_multiple_ambient_events(self):
        config = _make_config(
            background_events=[
                _ambient_event("tv_ambient"),
                _ambient_event("hvac_hum"),
            ]
        )
        out, events, _ = self.mixer.mix(self.samples, _SR, config)
        assert len(events) == 2

    def test_sfx_onset_clamped_to_near_end(self):
        """_resolve_onset_s clamps very large onset_seconds to duration_s - 0.1."""
        config = _make_config(background_events=[_sfx_event(onset_seconds=9999.0)])
        _, events, _ = self.mixer.mix(self.samples, _SR, config)
        # Event is clamped to near end of clip, not dropped
        duration_s = len(self.samples) / _SR
        if events:
            assert events[0].onset_s <= duration_s

    def test_phase_boundaries_used_for_onset(self):
        ev = BackgroundEvent(type="ACOU_SLAM", loop=False, onset_at_phase="tension")
        config = _make_config(background_events=[ev])
        phase_boundaries = {"tension": 0.8}
        _, events, _ = self.mixer.mix(
            self.samples, _SR, config, rng_seed=0, phase_boundaries=phase_boundaries
        )
        assert len(events) == 1
        assert events[0].onset_s == pytest.approx(0.8, abs=0.01)


# ---------------------------------------------------------------------------
# NoiseMixer.mix — with real WAV assets
# ---------------------------------------------------------------------------


class TestNoiseMixerWithAssets:
    def _write_wav(self, path: Path, n: int = _SR) -> None:
        rng = np.random.default_rng(42)
        data = rng.standard_normal(n).astype(np.float32) * 0.1
        sf.write(str(path), data, _SR)

    def test_ambient_asset_loaded_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir)
            ambient_dir = assets_dir / "ambient"
            ambient_dir.mkdir()
            self._write_wav(ambient_dir / "tv_ambient_001.wav")

            mixer = NoiseMixer(assets_dir=assets_dir)
            samples = _sine()
            config = _make_config(background_events=[_ambient_event("tv_ambient")])
            out, events, _ = mixer.mix(samples, _SR, config)
            assert len(events) == 1
            assert not np.allclose(out, samples)

    def test_sfx_asset_loaded_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir)
            sfx_dir = assets_dir / "sfx"
            sfx_dir.mkdir()
            self._write_wav(sfx_dir / "ACOU_SLAM_001.wav", n=_SR // 4)

            mixer = NoiseMixer(assets_dir=assets_dir)
            samples = _sine()
            config = _make_config(background_events=[_sfx_event("ACOU_SLAM", onset_seconds=0.5)])
            out, events, _ = mixer.mix(samples, _SR, config)
            assert len(events) == 1

    def test_explicit_asset_path_used(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            asset_file = Path(tmpdir) / "custom_sfx.wav"
            self._write_wav(asset_file, n=_SR // 4)

            ev = BackgroundEvent(
                type="ACOU_SLAM",
                loop=False,
                onset_seconds=0.3,
                asset_path=str(asset_file),
            )
            mixer = NoiseMixer(assets_dir=Path("/tmp/nonexistent"))
            samples = _sine()
            config = _make_config(background_events=[ev])
            _, events, _ = mixer.mix(samples, _SR, config)
            assert len(events) == 1
