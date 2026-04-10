"""Unit tests for manifest CSV generation."""

from __future__ import annotations

import csv
import datetime
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from synthbanshee.package.manifest import _MANIFEST_COLUMNS, ManifestRow, generate_manifest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_valid_clip(
    parent: Path,
    clip_id: str,
    *,
    project: str = "she_proves",
    violence_typology: str = "IT",
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
            "has_violence": True,
            "violence_categories": ["VERB"],
            "max_intensity": 3,
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
# ManifestRow
# ---------------------------------------------------------------------------


class TestManifestRow:
    def test_is_dataclass(self):
        row = ManifestRow(
            clip_id="test_clip",
            project="she_proves",
            violence_typology="IT",
            tier="A",
            duration_seconds=3.5,
            speaker_ids="AGG_M_30-45_001",
            has_violence=True,
            max_intensity=3,
            quality_flags="",
            split="train",
            wav_path="data/he/test_clip.wav",
            strong_labels_path="",
        )
        assert row.clip_id == "test_clip"
        assert row.split == "train"


# ---------------------------------------------------------------------------
# generate_manifest
# ---------------------------------------------------------------------------


class TestGenerateManifest:
    def test_empty_directory_produces_empty_manifest(self, tmp_path):
        out = tmp_path / "manifest.csv"
        rows = generate_manifest(tmp_path, out)
        assert rows == []
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "clip_id" in content  # header still written

    def test_single_clip_produces_one_row(self, tmp_path):
        speaker_dir = tmp_path / "agg_m_30-45_001"
        _write_valid_clip(speaker_dir, "sp_it_a_0001_00")
        out = tmp_path / "manifest.csv"

        rows = generate_manifest(tmp_path, out)

        assert len(rows) == 1
        assert rows[0].clip_id == "sp_it_a_0001_00"
        assert rows[0].project == "she_proves"
        assert rows[0].violence_typology == "IT"
        assert rows[0].tier == "A"
        assert rows[0].has_violence is True
        assert rows[0].max_intensity == 3

    def test_multiple_clips_produce_multiple_rows(self, tmp_path):
        for i in range(5):
            _write_valid_clip(tmp_path / f"spk_{i:03d}", f"clip_{i:03d}_00")
        out = tmp_path / "manifest.csv"

        rows = generate_manifest(tmp_path, out)

        assert len(rows) == 5

    def test_splits_column_populated(self, tmp_path):
        _write_valid_clip(tmp_path / "spk_000", "clip_000_00")
        _write_valid_clip(tmp_path / "spk_001", "clip_001_00")
        splits = {"clip_000_00": "train", "clip_001_00": "val"}
        out = tmp_path / "manifest.csv"

        rows = generate_manifest(tmp_path, out, splits=splits)

        row_by_id = {r.clip_id: r for r in rows}
        assert row_by_id["clip_000_00"].split == "train"
        assert row_by_id["clip_001_00"].split == "val"

    def test_unassigned_split_is_empty_string(self, tmp_path):
        _write_valid_clip(tmp_path / "spk_000", "clip_000_00")
        out = tmp_path / "manifest.csv"
        rows = generate_manifest(tmp_path, out, splits={})
        assert rows[0].split == ""

    def test_csv_has_all_expected_columns(self, tmp_path):
        _write_valid_clip(tmp_path / "spk_000", "clip_000_00")
        out = tmp_path / "manifest.csv"
        generate_manifest(tmp_path, out)

        with out.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            assert set(reader.fieldnames or []) == set(_MANIFEST_COLUMNS)

    def test_dirty_files_excluded(self, tmp_path):
        """JSON files whose stems contain '_dirty' must not appear in the manifest."""
        clip_dir = tmp_path / "spk_000"
        clip_dir.mkdir()
        # Write a real clip
        _write_valid_clip(clip_dir, "clip_000_00")
        # Write a fake dirty JSON that would fail parsing anyway
        (clip_dir / "clip_000_00_dirty.json").write_text('{"clip_id": "dirty"}', encoding="utf-8")
        out = tmp_path / "manifest.csv"

        rows = generate_manifest(tmp_path, out)

        assert len(rows) == 1
        assert rows[0].clip_id == "clip_000_00"

    def test_invalid_json_skipped(self, tmp_path):
        """Malformed JSON files are silently skipped."""
        clip_dir = tmp_path / "spk_000"
        clip_dir.mkdir()
        (clip_dir / "bad_file.json").write_text("not json {{{{", encoding="utf-8")
        _write_valid_clip(clip_dir, "good_clip_00")
        out = tmp_path / "manifest.csv"

        rows = generate_manifest(tmp_path, out)

        assert len(rows) == 1
        assert rows[0].clip_id == "good_clip_00"

    def test_speaker_ids_pipe_separated(self, tmp_path):
        """speaker_ids column uses '|' as separator."""
        # The fixture writes a single speaker, so verify format
        _write_valid_clip(tmp_path / "spk_000", "clip_000_00", speaker_id="AGG_M_30-45_001")
        out = tmp_path / "manifest.csv"
        rows = generate_manifest(tmp_path, out)

        assert rows[0].speaker_ids == "AGG_M_30-45_001"

    def test_quality_flags_comma_separated(self, tmp_path):
        _write_valid_clip(
            tmp_path / "spk_000", "clip_000_00", quality_flags=["low_snr", "clipping"]
        )
        out = tmp_path / "manifest.csv"
        rows = generate_manifest(tmp_path, out)

        assert rows[0].quality_flags == "low_snr,clipping"

    def test_output_directory_created_if_missing(self, tmp_path):
        _write_valid_clip(tmp_path / "spk_000", "clip_000_00")
        out = tmp_path / "nested" / "deep" / "manifest.csv"

        generate_manifest(tmp_path, out)

        assert out.exists()

    def test_rows_sorted_by_clip_id(self, tmp_path):
        """Rows are in lexicographic order by clip_id (via sorted rglob)."""
        for clip_id in ["clip_c_00", "clip_a_00", "clip_b_00"]:
            _write_valid_clip(tmp_path / "spk", clip_id)
        out = tmp_path / "manifest.csv"
        rows = generate_manifest(tmp_path, out)

        assert [r.clip_id for r in rows] == sorted(r.clip_id for r in rows)

    def test_clip_ids_filter_excludes_unlisted_clips(self, tmp_path):
        """clip_ids allow-list restricts manifest to only those clips."""
        for clip_id in ["clip_a_00", "clip_b_00", "clip_c_00"]:
            _write_valid_clip(tmp_path / "spk", clip_id)
        out = tmp_path / "manifest.csv"

        rows = generate_manifest(tmp_path, out, clip_ids={"clip_a_00", "clip_c_00"})

        ids = {r.clip_id for r in rows}
        assert ids == {"clip_a_00", "clip_c_00"}
        assert "clip_b_00" not in ids

    def test_clip_ids_none_includes_all_clips(self, tmp_path):
        """When clip_ids is None all clips are included (default behaviour)."""
        for clip_id in ["clip_a_00", "clip_b_00"]:
            _write_valid_clip(tmp_path / "spk", clip_id)
        out = tmp_path / "manifest.csv"

        rows = generate_manifest(tmp_path, out, clip_ids=None)

        assert {r.clip_id for r in rows} == {"clip_a_00", "clip_b_00"}

    def test_strong_labels_path_populated_when_jsonl_exists(self, tmp_path):
        """strong_labels_path is the .jsonl path when the file exists alongside the clip."""
        clip_dir = tmp_path / "spk_000"
        wav = _write_valid_clip(clip_dir, "clip_000_00")
        jsonl_path = wav.with_suffix(".jsonl")
        jsonl_path.write_text('{"clip_id":"clip_000_00"}\n', encoding="utf-8")
        out = tmp_path / "manifest.csv"

        rows = generate_manifest(tmp_path, out)

        assert rows[0].strong_labels_path == str(jsonl_path)

    def test_strong_labels_path_empty_when_jsonl_absent(self, tmp_path):
        """strong_labels_path is empty string when no .jsonl exists alongside the clip."""
        _write_valid_clip(tmp_path / "spk_000", "clip_000_00")
        out = tmp_path / "manifest.csv"

        rows = generate_manifest(tmp_path, out)

        assert rows[0].strong_labels_path == ""

    def test_json_without_sibling_wav_is_skipped(self, tmp_path):
        """A metadata JSON whose sibling .wav does not exist is excluded from the manifest."""
        clip_dir = tmp_path / "spk_000"
        clip_dir.mkdir()
        # Write a complete valid clip
        _write_valid_clip(clip_dir, "good_clip_00")
        # Write a JSON-only orphan (no matching .wav)
        (clip_dir / "orphan_clip_00.json").write_text(
            json.dumps(
                {
                    "clip_id": "orphan_clip_00",
                    "project": "she_proves",
                    "language": "he",
                    "violence_typology": "IT",
                    "tier": "A",
                    "duration_seconds": 4.0,
                    "sample_rate": 16000,
                    "channels": 1,
                    "generation_date": datetime.date.today().isoformat(),
                    "generator_version": "0.1.0",
                    "is_synthetic": True,
                    "tts_engine": "azure_he_IL",
                    "acoustic_scene": {},
                    "speakers": [
                        {
                            "speaker_id": "AGG_M_30-45_001",
                            "role": "AGG",
                            "gender": "male",
                            "age_range": "30-45",
                            "tts_voice_id": "he-IL-AvriNeural",
                        }
                    ],
                    "weak_label": {
                        "has_violence": True,
                        "violence_categories": ["VERB"],
                        "max_intensity": 3,
                        "violence_typology": "IT",
                    },
                    "preprocessing_applied": {},
                    "quality_flags": [],
                    "annotator_confidence": 1.0,
                    "iaa_reviewed": False,
                }
            ),
            encoding="utf-8",
        )
        out = tmp_path / "manifest.csv"

        rows = generate_manifest(tmp_path, out)

        assert len(rows) == 1
        assert rows[0].clip_id == "good_clip_00"
