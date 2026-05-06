"""Unit tests for the automated QA suite."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from synthbanshee.labels.prosody_metrics import TurnMetrics, parse_jsonl_events
from synthbanshee.package.qa import (
    DatasetStats,
    RunSummary,
    _check_acoustic_warnings,
    _check_emotion_downgrade,
    _compute_run_warnings,
    _detect_outliers,
    _has_overlap,
    run_qa,
)

# ---------------------------------------------------------------------------
# #87 prosody-cap roll-up tests
# ---------------------------------------------------------------------------


class TestProsodyCapRollup:
    """qa-report's static surfacing of #87 effective-prosody cap activations."""

    def test_clip_with_no_cap_events_does_not_appear_in_report(self, tmp_path):
        _write_valid_clip(tmp_path / "agg_m_30-45_001", "clip_001_00")
        report = run_qa(tmp_path)
        assert report.prosody_cap_activations == {}
        assert report.stats.clips_with_prosody_cap_activations == 0
        assert report.stats.prosody_cap_activations_total == 0

    def test_clip_with_cap_events_recorded(self, tmp_path):
        events = [
            {"turn_index": 12, "intensity": 5, "dim": "rate", "pre_cap": 1.33, "post_cap": 1.20},
            {"turn_index": 12, "intensity": 5, "dim": "pitch", "pre_cap": 2.9, "post_cap": 2.0},
        ]
        _write_valid_clip(tmp_path / "agg_m_30-45_001", "clip_001_00", prosody_cap_events=events)
        report = run_qa(tmp_path)
        assert report.prosody_cap_activations == {"clip_001_00": 2}
        assert report.stats.clips_with_prosody_cap_activations == 1
        assert report.stats.prosody_cap_activations_total == 2

    def test_multiple_clips_aggregated(self, tmp_path):
        _write_valid_clip(tmp_path / "agg_m_30-45_001", "clip_clean_00")
        _write_valid_clip(
            tmp_path / "agg_m_30-45_002",
            "clip_one_cap_00",
            prosody_cap_events=[
                {"turn_index": 0, "intensity": 5, "dim": "rate", "pre_cap": 1.5, "post_cap": 1.2},
            ],
        )
        _write_valid_clip(
            tmp_path / "agg_m_30-45_003",
            "clip_three_caps_00",
            prosody_cap_events=[
                {"turn_index": 0, "intensity": 5, "dim": "rate", "pre_cap": 1.4, "post_cap": 1.2},
                {"turn_index": 1, "intensity": 5, "dim": "pitch", "pre_cap": 3.0, "post_cap": 2.0},
                {"turn_index": 2, "intensity": 4, "dim": "pitch", "pre_cap": 2.5, "post_cap": 2.0},
            ],
        )
        report = run_qa(tmp_path)
        assert report.stats.clips_with_prosody_cap_activations == 2
        assert report.stats.prosody_cap_activations_total == 4
        assert report.prosody_cap_activations == {
            "clip_one_cap_00": 1,
            "clip_three_caps_00": 3,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_valid_clip(
    parent: Path,
    clip_id: str,
    *,
    project: str = "she_proves",
    violence_typology: str = "NEU",
    tier: str = "A",
    duration: float = 4.0,
    speaker_id: str = "AGG_M_30-45_001",
    quality_flags: list[str] | None = None,
    tts_engine: str = "azure_he_IL",
    tts_voice_id: str = "he-IL-AvriNeural",
    gender: str = "male",
    prosody_cap_events: list[dict] | None = None,
) -> Path:
    """Write a minimal valid WAV + TXT + JSON triplet and return the WAV path."""
    parent.mkdir(parents=True, exist_ok=True)
    wav_path = parent / f"{clip_id}.wav"
    txt_path = parent / f"{clip_id}.txt"
    json_path = parent / f"{clip_id}.json"

    sr = 16000
    n = int(sr * duration)
    pad = int(0.5 * sr)
    samples = np.zeros(n, dtype=np.float32)
    t = np.linspace(0, duration, n, endpoint=False)
    samples[pad : n - pad] = 0.5 * np.sin(2 * np.pi * 440 * t[pad : n - pad])
    peak = float(np.max(np.abs(samples)))
    samples = (samples / peak * (10 ** (-1.0 / 20))).astype(np.float32)
    sf.write(str(wav_path), samples, sr, subtype="PCM_16")

    txt_path.write_text("shalom", encoding="utf-8")

    metadata = {
        "clip_id": clip_id,
        "project": project,
        "language": "he",
        "violence_typology": violence_typology,
        "tier": tier,
        "duration_seconds": duration,
        "sample_rate": 16000,
        "channels": 1,
        "generation_date": datetime.date.today().isoformat(),
        "generator_version": "0.1.0",
        "is_synthetic": True,
        "tts_engine": tts_engine,
        "acoustic_scene": {},
        "speakers": [
            {
                "speaker_id": speaker_id,
                "role": "AGG",
                "gender": gender,
                "age_range": "30-45",
                "tts_voice_id": tts_voice_id,
            }
        ],
        "weak_label": {
            "has_violence": False,
            "violence_categories": [],
            "max_intensity": 1,
            "violence_typology": violence_typology,
        },
        "preprocessing_applied": {},
        "quality_flags": quality_flags or [],
        "annotator_confidence": 1.0,
        "iaa_reviewed": False,
    }
    if prosody_cap_events is not None:
        metadata["generation_metadata"] = {
            "pipeline_version": "0.1.0",
            "effective_prosody_caps": prosody_cap_events,
        }
    json_path.write_text(json.dumps(metadata), encoding="utf-8")
    return wav_path


# ---------------------------------------------------------------------------
# Empty directory
# ---------------------------------------------------------------------------


class TestRunQAEmptyDirectory:
    def test_empty_dir_passes(self, tmp_path):
        report = run_qa(tmp_path)
        assert report.passed is True
        assert report.stats.total_clips == 0
        assert report.stats.failed_clips == 0
        assert report.failure_rate == 0.0

    def test_empty_dir_no_failed_ids(self, tmp_path):
        report = run_qa(tmp_path)
        assert report.failed_clip_ids == []


# ---------------------------------------------------------------------------
# Valid clips
# ---------------------------------------------------------------------------


class TestRunQAValidClips:
    def test_single_valid_clip_passes(self, tmp_path):
        _write_valid_clip(tmp_path / "spk", "clip_001_00")
        report = run_qa(tmp_path)

        assert report.passed is True
        assert report.stats.total_clips == 1
        assert report.stats.failed_clips == 0
        assert report.failure_rate == 0.0

    def test_duration_accumulated(self, tmp_path):
        _write_valid_clip(tmp_path / "spk_a", "clip_001_00", duration=4.0)
        _write_valid_clip(tmp_path / "spk_b", "clip_002_00", duration=4.0)
        report = run_qa(tmp_path)

        assert report.stats.total_clips == 2
        assert report.stats.total_duration_seconds > 7.0  # ≥ 2 × 4 s (minus padding trim)

    def test_clips_by_typology_counted(self, tmp_path):
        for i, typ in enumerate(["IT", "IT", "NEU"]):
            _write_valid_clip(tmp_path / f"spk_{i}", f"clip_{i:03d}_00", violence_typology=typ)
        report = run_qa(tmp_path)

        assert report.stats.clips_by_typology.get("IT") == 2
        assert report.stats.clips_by_typology.get("NEU") == 1

    def test_speaker_count(self, tmp_path):
        _write_valid_clip(tmp_path / "spk_a", "clip_001_00", speaker_id="SPK_A")
        _write_valid_clip(tmp_path / "spk_b", "clip_002_00", speaker_id="SPK_B")
        report = run_qa(tmp_path)

        assert report.stats.speaker_count == 2

    def test_quality_flagged_clips_tracked(self, tmp_path):
        _write_valid_clip(tmp_path / "spk_a", "clip_good_00", quality_flags=[])
        _write_valid_clip(tmp_path / "spk_b", "clip_bad_00", quality_flags=["low_snr"])
        report = run_qa(tmp_path)

        assert report.stats.quality_flagged_clips == 1
        assert "clip_bad_00" in report.quality_flagged
        assert report.quality_flagged["clip_bad_00"] == ["low_snr"]

    def test_clips_by_split_with_splits_dict(self, tmp_path):
        _write_valid_clip(tmp_path / "spk_a", "clip_001_00")
        _write_valid_clip(tmp_path / "spk_b", "clip_002_00")
        splits = {"clip_001_00": "train", "clip_002_00": "val"}
        report = run_qa(tmp_path, splits=splits)

        assert report.stats.clips_by_split.get("train") == 1
        assert report.stats.clips_by_split.get("val") == 1

    def test_unassigned_clips_in_unassigned_bucket(self, tmp_path):
        _write_valid_clip(tmp_path / "spk", "clip_001_00")
        report = run_qa(tmp_path, splits={})

        assert report.stats.clips_by_split.get("unassigned") == 1

    def test_dirty_wavs_excluded(self, tmp_path):
        """WAVs whose stems contain '_dirty' are not processed."""
        spk_dir = tmp_path / "spk"
        _write_valid_clip(spk_dir, "clip_001_00")
        # Create a dirty wav (just copy content — format doesn't matter since we exclude)
        (spk_dir / "clip_001_00_dirty.wav").write_bytes(b"FAKE")
        report = run_qa(tmp_path)

        assert report.stats.total_clips == 1


# ---------------------------------------------------------------------------
# Failed clips
# ---------------------------------------------------------------------------


class TestRunQAFailedClips:
    def test_invalid_wav_counted_as_failed(self, tmp_path):
        """A WAV that fails validate_clip() contributes to failed_clips."""
        spk_dir = tmp_path / "spk"
        spk_dir.mkdir()
        wav_path = spk_dir / "bad_clip_00.wav"
        txt_path = spk_dir / "bad_clip_00.txt"
        json_path = spk_dir / "bad_clip_00.json"
        wav_path.write_bytes(b"not a real wav file")
        txt_path.write_text("shalom", encoding="utf-8")
        json_path.write_text("{}", encoding="utf-8")

        report = run_qa(tmp_path)

        assert report.stats.failed_clips >= 1
        assert "bad_clip_00" in report.failed_clip_ids

    def test_failure_rate_computed(self, tmp_path):
        # 1 valid, 1 bad
        _write_valid_clip(tmp_path / "spk_good", "good_clip_00")
        spk_dir = tmp_path / "spk_bad"
        spk_dir.mkdir()
        (spk_dir / "bad_clip_00.wav").write_bytes(b"garbage")
        (spk_dir / "bad_clip_00.txt").write_text("x", encoding="utf-8")
        (spk_dir / "bad_clip_00.json").write_text("{}", encoding="utf-8")

        report = run_qa(tmp_path)

        # 1 success + 1 failure = 50% failure rate
        assert abs(report.failure_rate - 0.5) < 0.01

    def test_exceeding_max_failure_rate_fails_report(self, tmp_path):
        """failure_rate > max_failure_rate → report.passed is False."""
        spk_dir = tmp_path / "spk_bad"
        spk_dir.mkdir()
        (spk_dir / "bad_clip_00.wav").write_bytes(b"garbage")
        (spk_dir / "bad_clip_00.txt").write_text("x", encoding="utf-8")
        (spk_dir / "bad_clip_00.json").write_text("{}", encoding="utf-8")

        report = run_qa(tmp_path, max_failure_rate=0.0)

        assert report.passed is False

    def test_within_failure_budget_passes(self, tmp_path):
        """failure_rate ≤ max_failure_rate → report.passed is True."""
        for i in range(99):
            _write_valid_clip(tmp_path / f"spk_{i:03d}", f"good_clip_{i:03d}_00")
        spk_dir = tmp_path / "spk_bad"
        spk_dir.mkdir()
        (spk_dir / "bad_clip_00.wav").write_bytes(b"garbage")
        (spk_dir / "bad_clip_00.txt").write_text("x", encoding="utf-8")
        (spk_dir / "bad_clip_00.json").write_text("{}", encoding="utf-8")

        # 99 valid + 1 bad = 1% failure rate; budget is 2%
        report = run_qa(tmp_path, max_failure_rate=0.02)

        assert report.passed is True


# ---------------------------------------------------------------------------
# QAReport dataclass
# ---------------------------------------------------------------------------


class TestQAReportStructure:
    def test_report_has_data_dir(self, tmp_path):
        report = run_qa(tmp_path)
        assert report.data_dir == str(tmp_path)

    def test_report_stats_is_dataset_stats(self, tmp_path):
        report = run_qa(tmp_path)
        assert isinstance(report.stats, DatasetStats)

    def test_report_failed_clip_ids_is_list(self, tmp_path):
        report = run_qa(tmp_path)
        assert isinstance(report.failed_clip_ids, list)

    def test_report_quality_flagged_is_dict(self, tmp_path):
        report = run_qa(tmp_path)
        assert isinstance(report.quality_flagged, dict)


# ---------------------------------------------------------------------------
# qa.py lines 101-104: validate_clip passes but ClipMetadata re-parse fails
# ---------------------------------------------------------------------------


class TestRunQAMetadataReparseFails:
    def test_valid_clip_but_metadata_reparse_raises_counted_as_failed(self, tmp_path):
        """validate_clip succeeds but ClipMetadata.model_validate_json in run_qa raises.

        This covers qa.py lines 101-104: the except block after the second
        ClipMetadata parse attempt.  validate_clip uses its own import of
        ClipMetadata (synthbanshee.package.validator.ClipMetadata) while
        run_qa uses synthbanshee.package.qa.ClipMetadata — patching only the
        qa module's reference leaves validate_clip unaffected.
        """
        from unittest.mock import patch

        _write_valid_clip(tmp_path / "spk", "clip_001_00")

        with patch("synthbanshee.package.qa.ClipMetadata") as MockMeta:
            MockMeta.model_validate_json.side_effect = ValueError("schema changed")
            report = run_qa(tmp_path)

        # validate_clip passed (uses validator.ClipMetadata, not patched)
        # but run_qa's re-parse failed → clip counted as failed, not as valid
        assert report.stats.failed_clips == 1
        assert report.stats.total_clips == 0
        assert "clip_001_00" in report.failed_clip_ids

    def test_clips_missing_strong_labels_counted(self, tmp_path):
        """Clips without a sibling .jsonl increment clips_missing_strong_labels."""
        _write_valid_clip(tmp_path / "spk", "clip_001_00")
        # No .jsonl written alongside the clip

        report = run_qa(tmp_path)

        assert report.stats.clips_missing_strong_labels == 1
        # Missing JSONL is a warning, not a failure — clip still passes
        assert report.stats.total_clips == 1
        assert report.stats.failed_clips == 0

    def test_clips_with_strong_labels_not_counted(self, tmp_path):
        """Clips with a sibling .jsonl do NOT increment clips_missing_strong_labels."""
        wav = _write_valid_clip(tmp_path / "spk", "clip_001_00")
        wav.with_suffix(".jsonl").write_text(
            '{"clip_id":"clip_001_00","onset":0.5,"offset":1.0}\n', encoding="utf-8"
        )

        report = run_qa(tmp_path)

        assert report.stats.clips_missing_strong_labels == 0


# ---------------------------------------------------------------------------
# Helpers for M10a tests
# ---------------------------------------------------------------------------


def _write_jsonl_events(wav_path: Path, events: list[dict]) -> None:
    """Write a JSONL file alongside the WAV with the given events."""
    jsonl_path = wav_path.with_suffix(".jsonl")
    lines = [json.dumps(e) for e in events]
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_event(
    clip_id: str,
    onset: float,
    offset: float,
    intensity: int,
    speaker_role: str,
    event_id: str = "ev_001",
    notes: str | None = None,
    emotional_state: str = "neutral",
) -> dict:
    """Return a minimal valid EventLabel dict."""
    d = {
        "event_id": event_id,
        "clip_id": clip_id,
        "onset": onset,
        "offset": offset,
        "tier1_category": "NONE",
        "tier2_subtype": "NONE_AMBIENT",
        "intensity": intensity,
        "speaker_id": "AGG_001",
        "speaker_role": speaker_role,
        "emotional_state": emotional_state,
        "confidence": 1.0,
        "label_source": "auto",
        "iaa_reviewed": False,
    }
    if notes is not None:
        d["notes"] = notes
    return d


# ---------------------------------------------------------------------------
# M10a: _check_acoustic_warnings
# ---------------------------------------------------------------------------


class TestCheckAcousticWarnings:
    def test_no_warnings_for_normal_clip(self):
        turns = [
            TurnMetrics("c1", "VIC", 4, 200.0, 10.0, -20.0),
            TurnMetrics("c1", "AGG", 1, 130.0, 10.0, -25.0),
            TurnMetrics("c1", "AGG", 5, 140.0, 15.0, -15.0),
        ]
        assert _check_acoustic_warnings(turns) == []

    def test_warn_vic_f0_high_triggered(self):
        turns = [
            TurnMetrics("c1", "VIC", 5, 260.0, 10.0, -20.0),
        ]
        warnings = _check_acoustic_warnings(turns)
        assert "vic_f0_high" in warnings

    def test_warn_vic_f0_high_not_triggered_at_low_intensity(self):
        """VIC F0 warning only applies to I4–I5, not lower intensities."""
        turns = [
            TurnMetrics("c1", "VIC", 2, 300.0, 10.0, -20.0),
        ]
        assert _check_acoustic_warnings(turns) == []

    def test_warn_vic_f0_high_not_triggered_at_250(self):
        """Exactly 250 Hz should NOT trigger (threshold is >250)."""
        turns = [TurnMetrics("c1", "VIC", 4, 250.0, 10.0, -20.0)]
        assert _check_acoustic_warnings(turns) == []

    def test_warn_agg_no_escalation_triggered(self):
        """AGG I5 − I1 < 6 dB should trigger warning."""
        turns = [
            TurnMetrics("c1", "AGG", 1, 130.0, 10.0, -20.0),
            TurnMetrics("c1", "AGG", 5, 140.0, 15.0, -16.0),  # 4 dB delta
        ]
        warnings = _check_acoustic_warnings(turns)
        assert "agg_no_escalation" in warnings

    def test_warn_agg_no_escalation_not_triggered_at_threshold(self):
        """AGG I5 − I1 = 6 dB exactly should NOT trigger."""
        turns = [
            TurnMetrics("c1", "AGG", 1, 130.0, 10.0, -26.0),
            TurnMetrics("c1", "AGG", 5, 140.0, 15.0, -20.0),  # 6 dB delta
        ]
        assert _check_acoustic_warnings(turns) == []

    def test_warn_agg_no_escalation_not_triggered_without_both_intensities(self):
        """AGG warning requires both I1 and I5 turns."""
        turns = [TurnMetrics("c1", "AGG", 1, 130.0, 10.0, -20.0)]
        assert _check_acoustic_warnings(turns) == []

    def test_empty_turns_no_warnings(self):
        assert _check_acoustic_warnings([]) == []

    def test_multiple_warnings_can_fire(self):
        turns = [
            TurnMetrics("c1", "VIC", 5, 260.0, 10.0, -20.0),
            TurnMetrics("c1", "AGG", 1, 130.0, 10.0, -20.0),
            TurnMetrics("c1", "AGG", 5, 140.0, 15.0, -18.0),  # 2 dB delta
        ]
        warnings = _check_acoustic_warnings(turns)
        assert "vic_f0_high" in warnings
        assert "agg_no_escalation" in warnings

    def test_vic_f0_none_skipped(self):
        """VIC turns with None F0 should not trigger the warning."""
        turns = [TurnMetrics("c1", "VIC", 5, None, None, -20.0)]
        assert _check_acoustic_warnings(turns) == []


# ---------------------------------------------------------------------------
# M10a: run_qa integration — acoustic warnings
# ---------------------------------------------------------------------------


class TestRunQAAcousticWarnings:
    def test_acoustic_warnings_empty_without_jsonl(self, tmp_path):
        """No JSONL → no acoustic warnings raised."""
        _write_valid_clip(tmp_path / "spk", "clip_001_00")
        report = run_qa(tmp_path)
        assert report.stats.clips_with_acoustic_warnings == 0
        assert report.acoustic_warnings == {}

    def test_new_stats_fields_present(self, tmp_path):
        """DatasetStats includes M10a fields."""
        _write_valid_clip(tmp_path / "spk", "clip_001_00")
        report = run_qa(tmp_path)
        assert hasattr(report.stats, "acoustic_warnings")
        assert hasattr(report.stats, "clips_with_acoustic_warnings")

    def test_qa_report_acoustic_warnings_field(self, tmp_path):
        """QAReport has acoustic_warnings dict."""
        report = run_qa(tmp_path)
        assert isinstance(report.acoustic_warnings, dict)

    def test_run_qa_with_jsonl_runs_acoustic_checks(self, tmp_path):
        """Clips with JSONL should have acoustic checks run (no warnings for clean clip)."""
        wav = _write_valid_clip(tmp_path / "spk", "clip_001_00")
        ev = _make_event("clip_001_00", 0.5, 3.0, 1, "AGG")
        _write_jsonl_events(wav, [ev])
        report = run_qa(tmp_path)
        assert report.stats.total_clips == 1
        # No warnings expected for a clean tone clip at I1
        assert report.stats.clips_with_acoustic_warnings == 0

    def test_run_qa_warning_accumulated_in_stats(self, tmp_path):
        """When measure_events returns a turn that triggers a warning, stats are updated."""
        from unittest.mock import patch

        wav = _write_valid_clip(tmp_path / "spk", "clip_001_00")
        ev = _make_event("clip_001_00", 0.5, 3.0, 5, "VIC")
        _write_jsonl_events(wav, [ev])

        high_f0_turn = TurnMetrics("clip_001_00", "VIC", 5, 260.0, 10.0, -20.0)
        with patch("synthbanshee.package.qa.measure_events", return_value=[high_f0_turn]):
            report = run_qa(tmp_path)

        assert report.stats.clips_with_acoustic_warnings == 1
        assert "clip_001_00" in report.acoustic_warnings
        assert "vic_f0_high" in report.acoustic_warnings["clip_001_00"]
        assert report.stats.acoustic_warnings.get("vic_f0_high") == 1

    def test_run_qa_acoustic_measurement_exception_handled(self, tmp_path):
        """If measure_events raises, the clip still passes QA (graceful fallback)."""
        from unittest.mock import patch

        wav = _write_valid_clip(tmp_path / "spk", "clip_001_00")
        ev = _make_event("clip_001_00", 0.5, 3.0, 1, "AGG")
        _write_jsonl_events(wav, [ev])

        with patch("synthbanshee.package.qa.measure_events", side_effect=RuntimeError("boom")):
            report = run_qa(tmp_path)

        assert report.stats.total_clips == 1
        assert report.stats.failed_clips == 0


# ---------------------------------------------------------------------------
# M10b: parse_jsonl_events
# ---------------------------------------------------------------------------


class TestParseJsonlEvents:
    def test_parses_valid_events(self, tmp_path):
        jsonl = tmp_path / "clip.jsonl"
        ev = _make_event("c1", 0.5, 1.0, 1, "AGG")
        jsonl.write_text(json.dumps(ev) + "\n", encoding="utf-8")
        events = parse_jsonl_events(jsonl)
        assert len(events) == 1
        assert events[0].clip_id == "c1"

    def test_skips_malformed_lines(self, tmp_path):
        jsonl = tmp_path / "clip.jsonl"
        ev = _make_event("c1", 0.5, 1.0, 1, "AGG")
        jsonl.write_text("not valid json\n" + json.dumps(ev) + "\n", encoding="utf-8")
        events = parse_jsonl_events(jsonl)
        assert len(events) == 1

    def test_skips_blank_lines(self, tmp_path):
        jsonl = tmp_path / "clip.jsonl"
        ev = _make_event("c1", 0.5, 1.0, 1, "AGG")
        jsonl.write_text("\n\n" + json.dumps(ev) + "\n\n", encoding="utf-8")
        events = parse_jsonl_events(jsonl)
        assert len(events) == 1

    def test_empty_file(self, tmp_path):
        jsonl = tmp_path / "clip.jsonl"
        jsonl.write_text("", encoding="utf-8")
        events = parse_jsonl_events(jsonl)
        assert events == []


# ---------------------------------------------------------------------------
# M10b: _has_overlap
# ---------------------------------------------------------------------------


class TestHasOverlap:
    def test_no_overlap_sequential_events(self):
        from synthbanshee.labels.schema import EventLabel

        events = [
            EventLabel(
                event_id="ev_001",
                clip_id="c1",
                onset=0.5,
                offset=1.0,
                tier1_category="NONE",
                tier2_subtype="NONE_AMBIENT",
                intensity=1,
                speaker_role="AGG",
                emotional_state="neutral",
            ),
            EventLabel(
                event_id="ev_002",
                clip_id="c1",
                onset=1.5,
                offset=2.0,
                tier1_category="NONE",
                tier2_subtype="NONE_AMBIENT",
                intensity=1,
                speaker_role="VIC",
                emotional_state="neutral",
            ),
        ]
        assert _has_overlap(events) is False

    def test_overlap_detected(self):
        from synthbanshee.labels.schema import EventLabel

        events = [
            EventLabel(
                event_id="ev_001",
                clip_id="c1",
                onset=0.5,
                offset=1.5,
                tier1_category="NONE",
                tier2_subtype="NONE_AMBIENT",
                intensity=1,
                speaker_role="AGG",
                emotional_state="neutral",
            ),
            EventLabel(
                event_id="ev_002",
                clip_id="c1",
                onset=1.0,
                offset=2.0,
                tier1_category="NONE",
                tier2_subtype="NONE_AMBIENT",
                intensity=1,
                speaker_role="VIC",
                emotional_state="neutral",
            ),
        ]
        assert _has_overlap(events) is True

    def test_adjacent_events_no_overlap(self):
        """Events touching at boundaries (onset == offset) do not overlap."""
        from synthbanshee.labels.schema import EventLabel

        events = [
            EventLabel(
                event_id="ev_001",
                clip_id="c1",
                onset=0.5,
                offset=1.0,
                tier1_category="NONE",
                tier2_subtype="NONE_AMBIENT",
                intensity=1,
                speaker_role="AGG",
                emotional_state="neutral",
            ),
            EventLabel(
                event_id="ev_002",
                clip_id="c1",
                onset=1.0,
                offset=2.0,
                tier1_category="NONE",
                tier2_subtype="NONE_AMBIENT",
                intensity=1,
                speaker_role="VIC",
                emotional_state="neutral",
            ),
        ]
        assert _has_overlap(events) is False

    def test_empty_events(self):
        assert _has_overlap([]) is False

    def test_single_event(self):
        from synthbanshee.labels.schema import EventLabel

        events = [
            EventLabel(
                event_id="ev_001",
                clip_id="c1",
                onset=0.5,
                offset=1.0,
                tier1_category="NONE",
                tier2_subtype="NONE_AMBIENT",
                intensity=1,
                speaker_role="AGG",
                emotional_state="neutral",
            ),
        ]
        assert _has_overlap(events) is False

    def test_unsorted_events_overlap_detected(self):
        """Overlap detection works even when events arrive out of order."""
        from synthbanshee.labels.schema import EventLabel

        events = [
            EventLabel(
                event_id="ev_002",
                clip_id="c1",
                onset=1.0,
                offset=2.0,
                tier1_category="NONE",
                tier2_subtype="NONE_AMBIENT",
                intensity=1,
                speaker_role="VIC",
                emotional_state="neutral",
            ),
            EventLabel(
                event_id="ev_001",
                clip_id="c1",
                onset=0.5,
                offset=1.5,
                tier1_category="NONE",
                tier2_subtype="NONE_AMBIENT",
                intensity=1,
                speaker_role="AGG",
                emotional_state="neutral",
            ),
        ]
        assert _has_overlap(events) is True


# ---------------------------------------------------------------------------
# M10b: _check_emotion_downgrade
# ---------------------------------------------------------------------------


class TestCheckEmotionDowngrade:
    def test_no_downgrade_when_neutral_matches_low_intensity(self):
        """neutral count == low-intensity count → not flagged."""
        from synthbanshee.labels.schema import EventLabel

        events = [
            EventLabel(
                event_id="ev_001",
                clip_id="c1",
                onset=0.5,
                offset=1.0,
                tier1_category="NONE",
                tier2_subtype="NONE_AMBIENT",
                intensity=1,
                speaker_role="AGG",
                emotional_state="neutral",
            ),
            EventLabel(
                event_id="ev_002",
                clip_id="c1",
                onset=1.5,
                offset=2.0,
                tier1_category="NONE",
                tier2_subtype="NONE_AMBIENT",
                intensity=4,
                speaker_role="AGG",
                emotional_state="anger",
            ),
        ]
        # 1 neutral, 1 low-intensity (I1) — equal, not flagged
        assert _check_emotion_downgrade(events) is False

    def test_downgrade_when_neutral_exceeds_low_intensity(self):
        from synthbanshee.labels.schema import EventLabel

        events = [
            EventLabel(
                event_id="ev_001",
                clip_id="c1",
                onset=0.5,
                offset=1.0,
                tier1_category="NONE",
                tier2_subtype="NONE_AMBIENT",
                intensity=3,
                speaker_role="AGG",
                emotional_state="neutral",
            ),
            EventLabel(
                event_id="ev_002",
                clip_id="c1",
                onset=1.5,
                offset=2.0,
                tier1_category="NONE",
                tier2_subtype="NONE_AMBIENT",
                intensity=4,
                speaker_role="VIC",
                emotional_state="neutral",
            ),
        ]
        # 2 neutral, 0 low-intensity → flagged
        assert _check_emotion_downgrade(events) is True

    def test_no_downgrade_all_emotional(self):
        from synthbanshee.labels.schema import EventLabel

        events = [
            EventLabel(
                event_id="ev_001",
                clip_id="c1",
                onset=0.5,
                offset=1.0,
                tier1_category="NONE",
                tier2_subtype="NONE_AMBIENT",
                intensity=4,
                speaker_role="AGG",
                emotional_state="anger",
            ),
        ]
        # 0 neutral → not flagged
        assert _check_emotion_downgrade(events) is False

    def test_empty_events(self):
        assert _check_emotion_downgrade([]) is False


# ---------------------------------------------------------------------------
# M10b: _detect_outliers
# ---------------------------------------------------------------------------


class TestDetectOutliers:
    def test_no_outliers_uniform_data(self):
        turns = [
            TurnMetrics("c1", "AGG", 1, 130.0, 10.0, -20.0),
            TurnMetrics("c2", "AGG", 1, 131.0, 10.0, -20.5),
            TurnMetrics("c3", "AGG", 1, 129.0, 10.0, -19.5),
        ]
        assert _detect_outliers(turns) == []

    def test_f0_outlier_detected(self):
        # 9 normal + 1 extreme outlier — enough normal samples so σ stays tight
        turns = [TurnMetrics(f"c{i}", "AGG", 1, 130.0 + i * 0.1, 10.0, -20.0) for i in range(9)] + [
            TurnMetrics("c_outlier", "AGG", 1, 500.0, 10.0, -20.0),
        ]
        outliers = _detect_outliers(turns)
        assert "c_outlier" in outliers

    def test_rms_outlier_detected(self):
        turns = [TurnMetrics(f"c{i}", "AGG", 1, 130.0, 10.0, -20.0 + i * 0.1) for i in range(9)] + [
            TurnMetrics("c_outlier", "AGG", 1, 130.0, 10.0, 0.0),  # outlier
        ]
        outliers = _detect_outliers(turns)
        assert "c_outlier" in outliers

    def test_single_turn_no_outliers(self):
        """Can't compute σ with a single sample."""
        turns = [TurnMetrics("c1", "AGG", 1, 130.0, 10.0, -20.0)]
        assert _detect_outliers(turns) == []

    def test_empty_turns(self):
        assert _detect_outliers([]) == []


# ---------------------------------------------------------------------------
# M10b: _compute_run_warnings
# ---------------------------------------------------------------------------


class TestComputeRunWarnings:
    def test_low_voice_diversity_male(self):
        summary = RunSummary(voices_by_gender={"male": 2, "female": 3})
        warnings = _compute_run_warnings(summary, total_clips=10, clips_with_acoustic_warnings=0)
        assert "low_voice_diversity_male" in warnings
        assert "low_voice_diversity_female" not in warnings

    def test_low_voice_diversity_female(self):
        summary = RunSummary(voices_by_gender={"male": 3, "female": 1})
        warnings = _compute_run_warnings(summary, total_clips=10, clips_with_acoustic_warnings=0)
        assert "low_voice_diversity_female" in warnings
        assert "low_voice_diversity_male" not in warnings

    def test_single_backend(self):
        summary = RunSummary(backend_count=1, voices_by_gender={"male": 3, "female": 3})
        warnings = _compute_run_warnings(summary, total_clips=10, clips_with_acoustic_warnings=0)
        assert "single_backend" in warnings

    def test_no_single_backend_warning_with_two(self):
        summary = RunSummary(backend_count=2, voices_by_gender={"male": 3, "female": 3})
        warnings = _compute_run_warnings(summary, total_clips=10, clips_with_acoustic_warnings=0)
        assert "single_backend" not in warnings

    def test_zero_overlap_with_i4_clips(self):
        """zero_overlap fires when there are I4+ clips but none overlap."""
        summary = RunSummary(
            overlap_ratio=0.0,
            clips_with_i4_plus=5,
            voices_by_gender={"male": 3, "female": 3},
            backend_count=2,
        )
        warnings = _compute_run_warnings(summary, total_clips=10, clips_with_acoustic_warnings=0)
        assert "zero_overlap" in warnings

    def test_no_zero_overlap_without_i4_clips(self):
        """zero_overlap does NOT fire when there are no I4+ clips."""
        summary = RunSummary(
            overlap_ratio=0.0,
            clips_with_i4_plus=0,
            voices_by_gender={"male": 3, "female": 3},
            backend_count=2,
        )
        warnings = _compute_run_warnings(summary, total_clips=10, clips_with_acoustic_warnings=0)
        assert "zero_overlap" not in warnings

    def test_no_zero_overlap_with_some_overlap(self):
        summary = RunSummary(
            overlap_ratio=0.5,
            clips_with_i4_plus=4,
            voices_by_gender={"male": 3, "female": 3},
            backend_count=2,
        )
        warnings = _compute_run_warnings(summary, total_clips=10, clips_with_acoustic_warnings=0)
        assert "zero_overlap" not in warnings

    def test_high_emotion_downgrade(self):
        summary = RunSummary(
            emotion_downgrade_ratio=0.06,
            voices_by_gender={"male": 3, "female": 3},
            backend_count=2,
            overlap_ratio=0.5,
            clips_with_i4_plus=5,
        )
        warnings = _compute_run_warnings(summary, total_clips=10, clips_with_acoustic_warnings=0)
        assert "high_emotion_downgrade" in warnings

    def test_no_high_emotion_downgrade_at_threshold(self):
        summary = RunSummary(
            emotion_downgrade_ratio=0.05,
            voices_by_gender={"male": 3, "female": 3},
            backend_count=2,
            overlap_ratio=0.5,
            clips_with_i4_plus=5,
        )
        warnings = _compute_run_warnings(summary, total_clips=10, clips_with_acoustic_warnings=0)
        assert "high_emotion_downgrade" not in warnings

    def test_high_warning_rate(self):
        summary = RunSummary(
            voices_by_gender={"male": 3, "female": 3},
            backend_count=2,
            overlap_ratio=0.5,
            clips_with_i4_plus=5,
        )
        warnings = _compute_run_warnings(summary, total_clips=10, clips_with_acoustic_warnings=2)
        assert "high_warning_rate" in warnings

    def test_no_high_warning_rate_at_threshold(self):
        summary = RunSummary(
            voices_by_gender={"male": 3, "female": 3},
            backend_count=2,
            overlap_ratio=0.5,
            clips_with_i4_plus=5,
        )
        warnings = _compute_run_warnings(summary, total_clips=10, clips_with_acoustic_warnings=1)
        assert "high_warning_rate" not in warnings

    def test_no_warnings_for_empty_run(self):
        summary = RunSummary()
        warnings = _compute_run_warnings(summary, total_clips=0, clips_with_acoustic_warnings=0)
        assert warnings == []


# ---------------------------------------------------------------------------
# M10b: run_qa with run_summary=True
# ---------------------------------------------------------------------------


class TestRunQARunSummary:
    def test_run_summary_none_by_default(self, tmp_path):
        _write_valid_clip(tmp_path / "spk", "clip_001_00")
        report = run_qa(tmp_path)
        assert report.run_summary is None

    def test_run_summary_populated_when_flag_set(self, tmp_path):
        wav = _write_valid_clip(tmp_path / "spk", "clip_001_00")
        ev = _make_event("clip_001_00", 0.5, 3.0, 1, "AGG")
        _write_jsonl_events(wav, [ev])
        report = run_qa(tmp_path, run_summary=True)
        assert report.run_summary is not None
        assert isinstance(report.run_summary, RunSummary)

    def test_run_summary_backend_diversity(self, tmp_path):
        _write_valid_clip(tmp_path / "spk_a", "clip_001_00")
        _write_valid_clip(tmp_path / "spk_b", "clip_002_00")
        report = run_qa(tmp_path, run_summary=True)
        rs = report.run_summary
        assert rs is not None
        # Both clips use azure_he_IL by default
        assert rs.backend_count == 1
        assert rs.clips_by_tts_engine.get("azure_he_IL") == 2

    def test_run_summary_voice_diversity(self, tmp_path):
        _write_valid_clip(tmp_path / "spk_a", "clip_001_00", speaker_id="SPK_A")
        _write_valid_clip(tmp_path / "spk_b", "clip_002_00", speaker_id="SPK_B")
        report = run_qa(tmp_path, run_summary=True)
        rs = report.run_summary
        assert rs is not None
        # Both speakers use same tts_voice_id (he-IL-AvriNeural) in the helper
        assert rs.voices_by_gender.get("male") == 1

    def test_run_summary_multiple_voices(self, tmp_path):
        """Different tts_voice_id values are counted as distinct voices."""
        _write_valid_clip(
            tmp_path / "spk_a",
            "clip_001_00",
            tts_voice_id="he-IL-AvriNeural",
            gender="male",
        )
        _write_valid_clip(
            tmp_path / "spk_b",
            "clip_002_00",
            tts_voice_id="he-IL-DanNeural",
            gender="male",
        )
        _write_valid_clip(
            tmp_path / "spk_c",
            "clip_003_00",
            tts_voice_id="he-IL-HilaNeural",
            gender="female",
        )
        report = run_qa(tmp_path, run_summary=True)
        rs = report.run_summary
        assert rs is not None
        assert rs.voices_by_gender.get("male") == 2
        assert rs.voices_by_gender.get("female") == 1

    def test_structural_warnings_fire_without_run_summary(self, tmp_path):
        """M10b per-clip warnings fire even when run_summary=False."""
        from unittest.mock import patch

        wav = _write_valid_clip(tmp_path / "spk", "clip_001_00")
        events = [
            _make_event("clip_001_00", 0.5, 1.5, 4, "AGG", event_id="ev_001"),
            _make_event("clip_001_00", 2.0, 3.0, 4, "VIC", event_id="ev_002"),
        ]
        _write_jsonl_events(wav, events)

        with patch("synthbanshee.package.qa.measure_events", return_value=[]):
            report = run_qa(tmp_path, run_summary=False)

        # Structural warnings fire without --run-summary
        assert "warn_no_overlap" in report.structural_warnings.get("clip_001_00", [])
        assert "warn_emotion_downgrade" in report.structural_warnings.get("clip_001_00", [])
        # But run_summary is still None
        assert report.run_summary is None

    def test_run_summary_warn_no_overlap(self, tmp_path):
        """Clip with I4+ turns and no overlap gets warn_no_overlap."""
        from unittest.mock import patch

        wav = _write_valid_clip(tmp_path / "spk", "clip_001_00")
        events = [
            _make_event("clip_001_00", 0.5, 1.5, 4, "AGG", event_id="ev_001"),
            _make_event("clip_001_00", 2.0, 3.0, 4, "VIC", event_id="ev_002"),
        ]
        _write_jsonl_events(wav, events)

        with patch("synthbanshee.package.qa.measure_events", return_value=[]):
            report = run_qa(tmp_path, run_summary=True)

        assert "warn_no_overlap" in report.structural_warnings.get("clip_001_00", [])

    def test_run_summary_no_warn_no_overlap_when_overlap_present(self, tmp_path):
        """Clip with I4+ turns and overlap does NOT get warn_no_overlap."""
        from unittest.mock import patch

        wav = _write_valid_clip(tmp_path / "spk", "clip_001_00")
        events = [
            _make_event("clip_001_00", 0.5, 2.0, 4, "AGG", event_id="ev_001"),
            _make_event("clip_001_00", 1.5, 3.0, 4, "VIC", event_id="ev_002"),
        ]
        _write_jsonl_events(wav, events)

        with patch("synthbanshee.package.qa.measure_events", return_value=[]):
            report = run_qa(tmp_path, run_summary=True)

        clip_warnings = report.structural_warnings.get("clip_001_00", [])
        assert "warn_no_overlap" not in clip_warnings

    def test_run_summary_warn_emotion_downgrade(self, tmp_path):
        """Clip with more neutrals than low-intensity turns gets warn_emotion_downgrade."""
        from unittest.mock import patch

        wav = _write_valid_clip(tmp_path / "spk", "clip_001_00")
        events = [
            _make_event("clip_001_00", 0.5, 1.5, 4, "AGG", event_id="ev_001"),
            _make_event("clip_001_00", 2.0, 3.0, 4, "VIC", event_id="ev_002"),
        ]
        # Both events are I4 with emotional_state="neutral" (default in _make_event)
        # 2 neutral, 0 low-intensity → flagged
        _write_jsonl_events(wav, events)

        with patch("synthbanshee.package.qa.measure_events", return_value=[]):
            report = run_qa(tmp_path, run_summary=True)

        assert "warn_emotion_downgrade" in report.structural_warnings.get("clip_001_00", [])

    def test_run_summary_no_emotion_downgrade_when_expected(self, tmp_path):
        """Clip where neutral count matches low-intensity count is not flagged."""
        from unittest.mock import patch

        wav = _write_valid_clip(tmp_path / "spk", "clip_001_00")
        events = [
            _make_event("clip_001_00", 0.5, 1.5, 1, "AGG", event_id="ev_001"),  # low + neutral
            _make_event(
                "clip_001_00", 2.0, 3.0, 4, "VIC", event_id="ev_002", emotional_state="fear"
            ),
        ]
        _write_jsonl_events(wav, events)

        with patch("synthbanshee.package.qa.measure_events", return_value=[]):
            report = run_qa(tmp_path, run_summary=True)

        clip_warnings = report.structural_warnings.get("clip_001_00", [])
        assert "warn_emotion_downgrade" not in clip_warnings

    def test_run_summary_overlap_ratio(self, tmp_path):
        """overlap_ratio reflects fraction of I4+ clips with overlap."""
        from unittest.mock import patch

        # Clip 1: I4, no overlap
        wav1 = _write_valid_clip(tmp_path / "spk_a", "clip_001_00")
        _write_jsonl_events(
            wav1,
            [
                _make_event("clip_001_00", 0.5, 1.5, 4, "AGG", event_id="ev_001"),
                _make_event("clip_001_00", 2.0, 3.0, 4, "VIC", event_id="ev_002"),
            ],
        )
        # Clip 2: I4, with overlap
        wav2 = _write_valid_clip(tmp_path / "spk_b", "clip_002_00")
        _write_jsonl_events(
            wav2,
            [
                _make_event("clip_002_00", 0.5, 2.0, 4, "AGG", event_id="ev_001"),
                _make_event("clip_002_00", 1.5, 3.0, 4, "VIC", event_id="ev_002"),
            ],
        )

        with patch("synthbanshee.package.qa.measure_events", return_value=[]):
            report = run_qa(tmp_path, run_summary=True)

        rs = report.run_summary
        assert rs is not None
        assert rs.overlap_ratio == 0.5  # 1 out of 2 I4+ clips

    def test_run_summary_empty_dir(self, tmp_path):
        """Empty directory → run_summary is None even with flag."""
        report = run_qa(tmp_path, run_summary=True)
        assert report.run_summary is None

    def test_run_summary_no_jsonl_no_m10b_warnings(self, tmp_path):
        """Clips without JSONL don't produce M10b warnings."""
        _write_valid_clip(tmp_path / "spk", "clip_001_00")
        report = run_qa(tmp_path, run_summary=True)
        assert report.run_summary is not None
        assert report.structural_warnings == {}

    def test_structural_warnings_separate_from_acoustic(self, tmp_path):
        """Structural and acoustic warnings go to separate dicts."""
        from unittest.mock import patch

        wav = _write_valid_clip(tmp_path / "spk", "clip_001_00")
        events = [
            _make_event("clip_001_00", 0.5, 1.5, 5, "VIC", event_id="ev_001"),
            _make_event("clip_001_00", 2.0, 3.0, 5, "AGG", event_id="ev_002"),
        ]
        _write_jsonl_events(wav, events)

        high_f0_turn = TurnMetrics("clip_001_00", "VIC", 5, 260.0, 10.0, -20.0)
        with patch("synthbanshee.package.qa.measure_events", return_value=[high_f0_turn]):
            report = run_qa(tmp_path)

        # Acoustic warning in acoustic_warnings
        assert "vic_f0_high" in report.acoustic_warnings.get("clip_001_00", [])
        # Structural warning in structural_warnings
        assert "warn_no_overlap" in report.structural_warnings.get("clip_001_00", [])
        # They don't leak into each other
        assert "warn_no_overlap" not in report.acoustic_warnings.get("clip_001_00", [])
        assert "vic_f0_high" not in report.structural_warnings.get("clip_001_00", [])


# ---------------------------------------------------------------------------
# M10b: CLI --run-summary smoke tests
# ---------------------------------------------------------------------------


class TestQAReportCLIRunSummary:
    def test_cli_run_summary_no_crash(self, tmp_path):
        """qa-report --run-summary runs without error on a simple dataset."""
        from click.testing import CliRunner

        from synthbanshee.cli import qa_report

        _write_valid_clip(tmp_path / "spk", "clip_001_00")
        runner = CliRunner()
        result = runner.invoke(qa_report, [str(tmp_path), "--run-summary"])
        assert result.exit_code == 0
        assert "Run-Level Summary" in result.output

    def test_cli_run_summary_json_output(self, tmp_path):
        """qa-report --run-summary --output writes run_summary to JSON."""
        from click.testing import CliRunner

        from synthbanshee.cli import qa_report

        _write_valid_clip(tmp_path / "spk", "clip_001_00")
        out_file = tmp_path / "report.json"
        runner = CliRunner()
        result = runner.invoke(qa_report, [str(tmp_path), "--run-summary", "-o", str(out_file)])
        assert result.exit_code == 0
        report_data = json.loads(out_file.read_text(encoding="utf-8"))
        assert "run_summary" in report_data
        rs = report_data["run_summary"]
        assert "voices_by_gender" in rs
        assert "backend_count" in rs
        assert "overlap_ratio" in rs
        assert "clips_with_i4_plus" in rs
        assert "run_warnings" in rs

    def test_cli_without_run_summary_no_section(self, tmp_path):
        """qa-report without --run-summary does not show the section."""
        from click.testing import CliRunner

        from synthbanshee.cli import qa_report

        _write_valid_clip(tmp_path / "spk", "clip_001_00")
        runner = CliRunner()
        result = runner.invoke(qa_report, [str(tmp_path)])
        assert result.exit_code == 0
        assert "Run-Level Summary" not in result.output

    def test_cli_json_without_run_summary_no_key(self, tmp_path):
        """JSON output without --run-summary has no run_summary key."""
        from click.testing import CliRunner

        from synthbanshee.cli import qa_report

        _write_valid_clip(tmp_path / "spk", "clip_001_00")
        out_file = tmp_path / "report.json"
        runner = CliRunner()
        result = runner.invoke(qa_report, [str(tmp_path), "-o", str(out_file)])
        assert result.exit_code == 0
        report_data = json.loads(out_file.read_text(encoding="utf-8"))
        assert "run_summary" not in report_data
