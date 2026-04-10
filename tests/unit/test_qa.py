"""Unit tests for the automated QA suite."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from synthbanshee.package.qa import DatasetStats, run_qa

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
        "tts_engine": "azure_he_IL",
        "acoustic_scene": {},
        "speakers": [
            {
                "speaker_id": speaker_id,
                "role": "AGG",
                "gender": "male",
                "age_range": "30-45",
                "tts_voice_id": "he-IL-AvriNeural",
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
