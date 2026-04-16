"""Unit tests for synthbanshee.labels.prosody_metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from synthbanshee.cli import cli
from synthbanshee.labels.prosody_metrics import (
    AGG_ESCALATION_MIN_DB,
    VIC_I1_F0_MAX_HZ,
    VIC_I4_F0_MAX_HZ,
    VIC_I5_F0_MAX_HZ,
    RoleIntensityStats,
    TurnMetrics,
    _measure_segment,
    aggregate_metrics,
    measure_clip,
    run_threshold_checks,
)

SR = 16_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sine(freq_hz: float, duration_s: float, sr: int = SR) -> np.ndarray:
    """Generate a mono int16 sine wave at freq_hz."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    wave = (np.sin(2 * np.pi * freq_hz * t) * 32767 * 0.5).astype(np.int16)
    return wave


def _silence(duration_s: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(sr * duration_s), dtype=np.int16)


def _write_wav(path: Path, samples: np.ndarray, sr: int = SR) -> None:
    sf.write(str(path), samples, sr, subtype="PCM_16")


def _write_jsonl(path: Path, events: list[dict]) -> None:
    lines = [json.dumps(e) for e in events]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _event(
    clip_id: str,
    onset: float,
    offset: float,
    intensity: int,
    speaker_role: str,
    event_id: str = "ev_001",
) -> dict:
    return {
        "event_id": event_id,
        "clip_id": clip_id,
        "onset": onset,
        "offset": offset,
        "tier1_category": "NONE",
        "tier2_subtype": "NONE_AMBIENT",
        "intensity": intensity,
        "speaker_id": "AGG_001",
        "speaker_role": speaker_role,
        "emotional_state": "neutral",
        "confidence": 1.0,
        "label_source": "auto",
        "iaa_reviewed": False,
    }


# ---------------------------------------------------------------------------
# _measure_segment
# ---------------------------------------------------------------------------


class TestMeasureSegment:
    def test_rms_silent_segment(self):
        samples = _silence(1.0)
        _, _, rms_db = _measure_segment(samples, SR, 0.0, 1.0)
        assert rms_db < -80.0

    def test_rms_tone_louder_than_silence(self):
        tone = _sine(220.0, 1.0)
        silent = _silence(1.0)
        _, _, rms_tone = _measure_segment(tone, SR, 0.0, 1.0)
        _, _, rms_silent = _measure_segment(silent, SR, 0.0, 1.0)
        assert rms_tone > rms_silent

    def test_segment_offset_respected(self):
        """Only the [onset, offset] window is measured."""
        # First half loud, second half silent
        loud = _sine(440.0, 0.5)
        quiet = _silence(0.5)
        samples = np.concatenate([loud, quiet])
        _, _, rms_loud_half = _measure_segment(samples, SR, 0.0, 0.5)
        _, _, rms_quiet_half = _measure_segment(samples, SR, 0.5, 1.0)
        assert rms_loud_half > rms_quiet_half + 20  # at least 20 dB difference

    def test_too_short_returns_sentinel(self):
        """Segments shorter than 50 ms return (None, None, -96.0)."""
        samples = _sine(220.0, 0.02)  # 20 ms
        f0, f0_std, rms_db = _measure_segment(samples, SR, 0.0, 0.02)
        assert f0 is None
        assert f0_std is None
        assert rms_db == pytest.approx(-96.0)

    def test_out_of_bounds_onset_offset_clamped(self):
        """Onset/offset beyond array length must not raise."""
        samples = _sine(220.0, 0.5)
        # offset beyond array
        f0, f0_std, rms_db = _measure_segment(samples, SR, 0.0, 10.0)
        assert rms_db < 0  # just no crash; some rms value returned

    def test_f0_none_when_librosa_unavailable(self, monkeypatch):
        """If librosa raises ImportError, F0 is None but RMS is still returned."""
        import builtins

        real_import = builtins.__import__

        def _block_librosa(name, *args, **kwargs):
            if name == "librosa":
                raise ImportError("librosa not available")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_librosa)
        samples = _sine(220.0, 0.5)
        f0, f0_std, rms_db = _measure_segment(samples, SR, 0.0, 0.5)
        assert f0 is None
        assert f0_std is None
        assert rms_db < 0  # rms still computed


# ---------------------------------------------------------------------------
# measure_clip
# ---------------------------------------------------------------------------


class TestMeasureClip:
    def test_missing_jsonl_returns_empty(self, tmp_path):
        wav = tmp_path / "clip_001.wav"
        _write_wav(wav, _sine(220.0, 1.0))
        assert measure_clip(wav) == []

    def test_empty_jsonl_returns_empty(self, tmp_path):
        wav = tmp_path / "clip_001.wav"
        _write_wav(wav, _sine(220.0, 1.0))
        (tmp_path / "clip_001.jsonl").write_text("", encoding="utf-8")
        assert measure_clip(wav) == []

    def test_event_without_speaker_role_skipped(self, tmp_path):
        wav = tmp_path / "clip_001.wav"
        _write_wav(wav, _sine(220.0, 1.0))
        ev = _event("clip_001", 0.1, 0.9, 1, "AGG")
        ev["speaker_role"] = None
        _write_jsonl(tmp_path / "clip_001.jsonl", [ev])
        assert measure_clip(wav) == []

    def test_one_event_returns_one_metric(self, tmp_path):
        wav = tmp_path / "clip_001.wav"
        _write_wav(wav, _sine(220.0, 1.0))
        _write_jsonl(
            tmp_path / "clip_001.jsonl",
            [_event("clip_001", 0.1, 0.9, 2, "AGG")],
        )
        result = measure_clip(wav)
        assert len(result) == 1
        assert result[0].speaker_role == "AGG"
        assert result[0].intensity == 2
        assert result[0].rms_db < 0

    def test_two_events_different_roles(self, tmp_path):
        wav = tmp_path / "clip_001.wav"
        _write_wav(wav, _silence(2.0))
        events = [
            _event("clip_001", 0.0, 0.9, 1, "AGG", "ev_001"),
            _event("clip_001", 1.0, 1.9, 3, "VIC", "ev_002"),
        ]
        _write_jsonl(tmp_path / "clip_001.jsonl", events)
        result = measure_clip(wav)
        assert len(result) == 2
        roles = {t.speaker_role for t in result}
        assert roles == {"AGG", "VIC"}

    def test_stereo_wav_downmixed(self, tmp_path):
        """Stereo WAV should not crash — first channel used."""
        wav = tmp_path / "clip_001.wav"
        stereo = np.stack([_sine(220.0, 1.0), _sine(440.0, 1.0)], axis=1)
        sf.write(str(wav), stereo, SR, subtype="PCM_16")
        _write_jsonl(
            tmp_path / "clip_001.jsonl",
            [_event("clip_001", 0.1, 0.9, 1, "AGG")],
        )
        result = measure_clip(wav)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# aggregate_metrics
# ---------------------------------------------------------------------------


class TestAggregateMetrics:
    def _make_turns(self) -> list[TurnMetrics]:
        return [
            TurnMetrics("c1", "AGG", 1, 130.0, 10.0, -25.0),
            TurnMetrics("c1", "AGG", 1, 135.0, 12.0, -24.0),
            TurnMetrics("c1", "AGG", 5, 140.0, 15.0, -15.0),
            TurnMetrics("c1", "VIC", 1, 190.0, 8.0, -26.0),
            TurnMetrics("c1", "VIC", 5, 230.0, 20.0, -20.0),
        ]

    def test_bucket_count(self):
        stats = aggregate_metrics(self._make_turns())
        assert len(stats) == 4  # AGG/1, AGG/5, VIC/1, VIC/5

    def test_n_turns_correct(self):
        stats = aggregate_metrics(self._make_turns())
        agg1 = next(s for s in stats if s.role == "AGG" and s.intensity == 1)
        assert agg1.n_turns == 2

    def test_f0_median_of_medians(self):
        stats = aggregate_metrics(self._make_turns())
        agg1 = next(s for s in stats if s.role == "AGG" and s.intensity == 1)
        assert agg1.f0_median_hz == pytest.approx(np.median([130.0, 135.0]))

    def test_rms_mean(self):
        stats = aggregate_metrics(self._make_turns())
        agg1 = next(s for s in stats if s.role == "AGG" and s.intensity == 1)
        assert agg1.rms_db_mean == pytest.approx(np.mean([-25.0, -24.0]))

    def test_sorted_by_role_then_intensity(self):
        stats = aggregate_metrics(self._make_turns())
        keys = [(s.role, s.intensity) for s in stats]
        assert keys == sorted(keys)

    def test_none_f0_excluded_from_median(self):
        turns = [
            TurnMetrics("c1", "AGG", 1, None, None, -25.0),
            TurnMetrics("c1", "AGG", 1, 130.0, 10.0, -24.0),
        ]
        stats = aggregate_metrics(turns)
        assert stats[0].f0_median_hz == pytest.approx(130.0)

    def test_all_none_f0_gives_none_median(self):
        turns = [TurnMetrics("c1", "VIC", 1, None, None, -25.0)]
        stats = aggregate_metrics(turns)
        assert stats[0].f0_median_hz is None

    def test_empty_input_returns_empty(self):
        assert aggregate_metrics([]) == []


# ---------------------------------------------------------------------------
# run_threshold_checks
# ---------------------------------------------------------------------------


class TestRunThresholdChecks:
    def _stats(
        self,
        vic_i1_f0=185.0,
        vic_i4_f0=220.0,
        vic_i5_f0=230.0,
        agg_i1_rms=-25.0,
        agg_i5_rms=-14.0,
    ) -> list[RoleIntensityStats]:
        # RoleIntensityStats fields: role, intensity, n_turns, f0_median_hz, f0_std_hz_mean, rms_db_mean
        return [
            RoleIntensityStats("AGG", 1, 10, 128.0, 12.0, agg_i1_rms),
            RoleIntensityStats("AGG", 5, 10, 140.0, 15.0, agg_i5_rms),
            RoleIntensityStats("VIC", 1, 10, vic_i1_f0, 8.0, -26.0),
            RoleIntensityStats("VIC", 4, 10, vic_i4_f0, 10.0, -22.0),
            RoleIntensityStats("VIC", 5, 10, vic_i5_f0, 12.0, -20.0),
        ]

    def test_all_pass(self):
        checks = run_threshold_checks(self._stats())
        assert all(passed for _, passed, _ in checks)

    def test_vic_i1_f0_fail(self):
        checks = run_threshold_checks(self._stats(vic_i1_f0=VIC_I1_F0_MAX_HZ + 1))
        vic_i1 = next(c for c in checks if "I1" in c[0] and "VIC" in c[0])
        assert not vic_i1[1]

    def test_vic_i4_f0_fail(self):
        checks = run_threshold_checks(self._stats(vic_i4_f0=VIC_I4_F0_MAX_HZ + 1))
        vic_i4 = next(c for c in checks if "I4" in c[0])
        assert not vic_i4[1]

    def test_vic_i5_f0_fail(self):
        checks = run_threshold_checks(self._stats(vic_i5_f0=VIC_I5_F0_MAX_HZ + 1))
        vic_i5 = next(c for c in checks if "I5" in c[0])
        assert not vic_i5[1]

    def test_agg_escalation_fail(self):
        # 6 dB delta — below the 8 dB minimum
        checks = run_threshold_checks(self._stats(agg_i1_rms=-25.0, agg_i5_rms=-19.0))
        agg_check = next(c for c in checks if "AGG" in c[0])
        assert not agg_check[1]

    def test_agg_escalation_exact_threshold_passes(self):
        checks = run_threshold_checks(
            self._stats(agg_i1_rms=-25.0, agg_i5_rms=-25.0 + AGG_ESCALATION_MIN_DB)
        )
        agg_check = next(c for c in checks if "AGG" in c[0])
        assert agg_check[1]

    def test_missing_role_reports_no_data(self):
        # No VIC entries at all
        stats = [
            RoleIntensityStats("AGG", 1, 5, 128.0, 12.0, -25.0),
            RoleIntensityStats("AGG", 5, 5, 140.0, 15.0, -14.0),  # noqa: E241
        ]
        checks = run_threshold_checks(stats)
        no_data = [c for c in checks if "no data" in c[2]]
        assert len(no_data) == 3  # VIC I1, I4, I5 all missing

    def test_returns_four_checks(self):
        checks = run_threshold_checks(self._stats())
        assert len(checks) == 4

    def test_vic_i1_exactly_at_threshold_passes(self):
        # I1 uses ≤ so exactly 200 Hz passes
        checks = run_threshold_checks(self._stats(vic_i1_f0=VIC_I1_F0_MAX_HZ))
        vic_i1 = next(c for c in checks if "I1" in c[0] and "VIC" in c[0])
        assert vic_i1[1]

    def test_vic_i4_exactly_at_threshold_fails(self):
        # I4 uses strict < so exactly 250 Hz must FAIL
        checks = run_threshold_checks(self._stats(vic_i4_f0=VIC_I4_F0_MAX_HZ))
        vic_i4 = next(c for c in checks if "I4" in c[0])
        assert not vic_i4[1]

    def test_vic_i5_exactly_at_threshold_fails(self):
        # I5 uses strict < so exactly 250 Hz must FAIL
        checks = run_threshold_checks(self._stats(vic_i5_f0=VIC_I5_F0_MAX_HZ))
        vic_i5 = next(c for c in checks if "I5" in c[0])
        assert not vic_i5[1]

    def test_stats_with_none_f0_reports_no_data(self):
        # Stats entry exists but f0_median_hz is None (all turns unvoiced)
        stats = [RoleIntensityStats("VIC", 1, 5, None, None, -25.0)]
        checks = run_threshold_checks(stats)
        vic_i1 = next(c for c in checks if "I1" in c[0] and "VIC" in c[0])
        assert not vic_i1[1]
        assert "no data" in vic_i1[2]

    def test_include_roles_skips_vic_checks(self):
        """include_roles={'AGG'} suppresses all VIC threshold checks."""
        checks = run_threshold_checks(self._stats(), include_roles={"AGG"})
        assert all("VIC" not in label for label, _, _ in checks)
        assert any("AGG" in label for label, _, _ in checks)

    def test_include_roles_skips_agg_check(self):
        """include_roles={'VIC'} suppresses the AGG escalation check."""
        checks = run_threshold_checks(self._stats(), include_roles={"VIC"})
        assert all("AGG" not in label for label, _, _ in checks)
        assert any("VIC" in label for label, _, _ in checks)

    def test_include_roles_none_runs_all_checks(self):
        """include_roles=None (default) runs all four checks."""
        checks = run_threshold_checks(self._stats(), include_roles=None)
        assert len(checks) == 4


# ---------------------------------------------------------------------------
# _measure_segment — int16 normalization and librosa F0 path
# ---------------------------------------------------------------------------


class TestMeasureSegmentCoverage:
    def test_rms_int16_and_float32_equivalent(self):
        """int16 and float32 of the same signal should give the same RMS dBFS."""
        samples_i16 = _sine(220.0, 0.5)
        samples_f32 = samples_i16.astype(np.float32) / 32768.0
        _, _, rms_i16 = _measure_segment(samples_i16, SR, 0.0, 0.5)
        _, _, rms_f32 = _measure_segment(samples_f32, SR, 0.0, 0.5)
        assert rms_i16 == pytest.approx(rms_f32, abs=0.5)

    def _make_librosa_mock(self, f0_values: np.ndarray, voiced_flag: np.ndarray) -> MagicMock:
        """Return a sys.modules-injectable librosa mock with a stubbed pyin."""
        mock_lib = MagicMock()
        mock_lib.pyin.return_value = (f0_values, voiced_flag, None)
        mock_lib.note_to_hz.side_effect = lambda note: {"C2": 65.4, "C7": 2093.0}[note]
        return mock_lib

    def test_f0_returned_when_librosa_returns_voiced_frames(self, monkeypatch):
        """Cover the successful librosa path (lines 122–133): pyin returns voiced F0."""
        n_frames = 50
        f0_values = np.full(n_frames, 220.0)
        voiced_flag = np.ones(n_frames, dtype=bool)
        mock_lib = self._make_librosa_mock(f0_values, voiced_flag)

        monkeypatch.setitem(sys.modules, "librosa", mock_lib)
        samples = _sine(220.0, 1.0)
        f0, f0_std, rms_db = _measure_segment(samples, SR, 0.0, 1.0)
        assert f0 == pytest.approx(220.0)
        assert f0_std is not None
        assert rms_db < 0

    def test_f0_none_when_all_frames_unvoiced(self, monkeypatch):
        """Cover line 131-132: pyin runs but returns no voiced frames."""
        n_frames = 50
        f0_values = np.full(n_frames, np.nan)
        voiced_flag = np.zeros(n_frames, dtype=bool)
        mock_lib = self._make_librosa_mock(f0_values, voiced_flag)

        monkeypatch.setitem(sys.modules, "librosa", mock_lib)
        samples = _sine(220.0, 1.0)
        f0, f0_std, rms_db = _measure_segment(samples, SR, 0.0, 1.0)
        assert f0 is None
        assert f0_std is None
        assert rms_db < 0


# ---------------------------------------------------------------------------
# measure_clip — stereo downmix explicit coverage
# ---------------------------------------------------------------------------


class TestMeasureClipStereo:
    def test_stereo_downmix_uses_first_channel(self, tmp_path):
        """Cover line 167: stereo WAV is downmixed to first channel."""
        wav = tmp_path / "clip_001.wav"
        # Channel 0: loud 220 Hz; channel 1: silence — verifies first channel is used
        loud = _sine(220.0, 1.0).astype(np.float32) / 32768.0
        silent = np.zeros(len(loud), dtype=np.float32)
        stereo = np.stack([loud, silent], axis=1)
        sf.write(str(wav), stereo, SR)
        _write_jsonl(
            tmp_path / "clip_001.jsonl",
            [_event("clip_001", 0.1, 0.9, 1, "AGG")],
        )
        result = measure_clip(wav)
        assert len(result) == 1
        # Should have a non-trivial RMS (from the loud channel, not silence)
        assert result[0].rms_db > -60.0


# ---------------------------------------------------------------------------
# measure-prosody CLI command
# ---------------------------------------------------------------------------


class TestMeasureProsodyCLI:
    def _make_clip(self, tmp_path: Path, clip_id: str, role: str, intensity: int) -> None:
        """Write a minimal WAV + JSONL pair."""
        wav = tmp_path / f"{clip_id}.wav"
        _write_wav(wav, _sine(220.0, 1.0))
        _write_jsonl(
            tmp_path / f"{clip_id}.jsonl",
            [_event(clip_id, 0.1, 0.9, intensity, role)],
        )

    def test_empty_dir_exits_cleanly(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["measure-prosody", str(tmp_path)])
        assert result.exit_code == 0
        assert "No .wav files" in result.output

    def test_missing_jsonl_skipped(self, tmp_path):
        wav = tmp_path / "clip_001.wav"
        _write_wav(wav, _sine(220.0, 1.0))
        runner = CliRunner()
        result = runner.invoke(cli, ["measure-prosody", str(tmp_path)])
        assert result.exit_code == 0
        assert "No speaker-role events" in result.output

    def test_prints_table_with_clips(self, tmp_path):
        self._make_clip(tmp_path, "clip_agg", "AGG", 1)
        self._make_clip(tmp_path, "clip_vic", "VIC", 1)
        runner = CliRunner()
        result = runner.invoke(cli, ["measure-prosody", str(tmp_path)])
        assert "AGG" in result.output
        assert "VIC" in result.output
        assert "threshold" in result.output.lower()

    def test_csv_output_written(self, tmp_path):
        self._make_clip(tmp_path, "clip_agg", "AGG", 1)
        csv_out = tmp_path / "out.csv"
        runner = CliRunner()
        runner.invoke(
            cli,
            ["measure-prosody", str(tmp_path), "--output", str(csv_out), "--roles", "AGG"],
        )
        assert csv_out.exists()
        content = csv_out.read_text()
        assert "clip_id" in content
        assert "clip_agg" in content

    def test_roles_filter_excludes_other_roles(self, tmp_path):
        """--roles AGG excludes VIC from table and checks; exit 0 when AGG checks pass."""
        # AGG I1 (very quiet) and I5 (loud) so RMS escalation check passes (>8 dB delta)
        wav_i1 = tmp_path / "clip_agg_i1.wav"
        _write_wav(wav_i1, (_sine(220.0, 1.0).astype(np.float32) * 0.01).astype(np.int16))
        _write_jsonl(tmp_path / "clip_agg_i1.jsonl", [_event("clip_agg_i1", 0.1, 0.9, 1, "AGG")])

        wav_i5 = tmp_path / "clip_agg_i5.wav"
        _write_wav(wav_i5, _sine(220.0, 1.0))  # default amplitude (~40 dB louder than I1)
        _write_jsonl(tmp_path / "clip_agg_i5.jsonl", [_event("clip_agg_i5", 0.1, 0.9, 5, "AGG")])

        self._make_clip(tmp_path, "clip_vic", "VIC", 1)  # excluded by --roles AGG

        runner = CliRunner()
        result = runner.invoke(cli, ["measure-prosody", str(tmp_path), "--roles", "AGG"])
        assert result.exit_code == 0  # VIC checks are skipped; AGG checks pass
        assert "AGG" in result.output
        assert "VIC" not in result.output  # excluded from table and threshold check labels

    def test_malformed_jsonl_line_skipped(self, tmp_path):
        """Malformed JSONL lines are skipped with a warning; valid events still returned."""
        import warnings

        wav = tmp_path / "clip_001.wav"
        _write_wav(wav, _sine(220.0, 1.0))
        jsonl = tmp_path / "clip_001.jsonl"
        valid = json.dumps(_event("clip_001", 0.1, 0.9, 1, "AGG"))
        jsonl.write_text(valid + "\nnot valid json\n", encoding="utf-8")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = measure_clip(wav)

        assert len(result) == 1  # valid event returned
        assert any("malformed" in str(w.message).lower() for w in caught)

    def test_empty_jsonl_lines_skipped(self, tmp_path):
        """Empty / blank lines in JSONL are silently skipped."""
        wav = tmp_path / "clip_001.wav"
        _write_wav(wav, _sine(220.0, 1.0))
        jsonl = tmp_path / "clip_001.jsonl"
        valid = json.dumps(_event("clip_001", 0.1, 0.9, 1, "AGG"))
        # Intersperse blank lines
        jsonl.write_text(f"\n{valid}\n\n", encoding="utf-8")

        result = measure_clip(wav)
        assert len(result) == 1

    def test_threshold_failure_sets_exit_code_1(self, tmp_path):
        """A failing check (here: VIC missing with --roles AGG,VIC) exits with code 1."""
        self._make_clip(tmp_path, "clip_agg", "AGG", 1)
        runner = CliRunner()
        result = runner.invoke(cli, ["measure-prosody", str(tmp_path), "--roles", "AGG,VIC"])
        # VIC data absent → VIC checks emit "no data" → passed=False → exit 1
        assert result.exit_code == 1
        assert "VIC" in result.output
