"""Unit tests for the SynthBanshee CLI entry points."""

from __future__ import annotations

import io
import json
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
from click.testing import CliRunner

from synthbanshee.cli import cli

SCENES_DIR = Path(__file__).parent.parent.parent / "configs" / "scenes"
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "configs" / "examples"


# ---------------------------------------------------------------------------
# Helpers shared with integration tests
# ---------------------------------------------------------------------------


def _make_wav_bytes(sample_rate: int = 24000, duration_s: float = 5.0) -> bytes:
    n = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        t = np.linspace(0, duration_s, n, endpoint=False)
        samples = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


def _write_valid_clip(tmp_path: Path, clip_id: str = "test_clip_01") -> Path:
    """Write a minimal valid WAV + TXT + JSON triplet and return the WAV path."""
    wav_path = tmp_path / f"{clip_id}.wav"
    txt_path = tmp_path / f"{clip_id}.txt"
    json_path = tmp_path / f"{clip_id}.json"

    sr = 16000
    duration = 4.0
    n = int(sr * duration)
    pad = int(0.5 * sr)
    samples = np.zeros(n, dtype=np.float32)
    t = np.linspace(0, duration, n, endpoint=False)
    samples[pad : n - pad] = 0.5 * np.sin(2 * np.pi * 440 * t[pad : n - pad])
    peak = float(np.max(np.abs(samples)))
    samples = (samples / peak * (10 ** (-1.0 / 20))).astype(np.float32)
    sf.write(str(wav_path), samples, sr, subtype="PCM_16")

    txt_path.write_text("shalom", encoding="utf-8")

    import datetime

    metadata = {
        "clip_id": clip_id,
        "project": "she_proves",
        "language": "he",
        "violence_typology": "NEU",
        "tier": "A",
        "duration_seconds": duration,
        "sample_rate": 16000,
        "channels": 1,
        "generation_date": datetime.date.today().isoformat(),
        "generator_version": "0.1.0",
        "is_synthetic": True,
        "tts_engine": "azure_he_IL",
        "acoustic_scene": {},
        "speakers": [],
        "weak_label": {
            "has_violence": False,
            "violence_categories": [],
            "max_intensity": 1,
            "violence_typology": "NEU",
        },
        "preprocessing_applied": {},
        "quality_flags": [],
        "annotator_confidence": 1.0,
        "iaa_reviewed": False,
    }
    json_path.write_text(json.dumps(metadata), encoding="utf-8")
    return wav_path


# ---------------------------------------------------------------------------
# generate command
# ---------------------------------------------------------------------------


class TestGenerateCommand:
    def test_dry_run_valid_config(self):
        """--dry-run loads and validates the config then exits cleanly."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["generate", "--config", str(SCENES_DIR / "test_scene_001.yaml"), "--dry-run"],
        )
        assert result.exit_code == 0, result.output
        assert "Dry run" in result.output

    def test_dry_run_missing_config(self):
        """--dry-run exits non-zero when the config file does not exist."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["generate", "--config", "no_such_file.yaml", "--dry-run"],
        )
        assert result.exit_code != 0

    def test_full_generate_with_mocked_tts(self, tmp_path):
        """Full generate (no dry-run) wires TTS → preprocessing → labels → output files."""

        def _fake_render_to_file(text, speaker, output_path, intensity=1):
            """Write synthetic 24 kHz WAV to output_path as the TTS stub would."""
            wav_bytes = _make_wav_bytes()
            Path(output_path).write_bytes(wav_bytes)
            return Path(output_path)

        runner = CliRunner()
        with patch("synthbanshee.tts.renderer.TTSRenderer") as MockRenderer:
            MockRenderer.return_value.render_utterance_to_file.side_effect = _fake_render_to_file
            result = runner.invoke(
                cli,
                [
                    "generate",
                    "--config",
                    str(SCENES_DIR / "test_scene_001.yaml"),
                    "--output-dir",
                    str(tmp_path / "out"),
                    "--cache-dir",
                    str(tmp_path / "cache"),
                    "--dirty-dir",
                    str(tmp_path / "dirty"),
                ],
            )

        assert result.exit_code == 0, result.output
        # At least one WAV file was written
        wav_files = list((tmp_path / "out").rglob("*.wav"))
        assert len(wav_files) >= 1


# ---------------------------------------------------------------------------
# validate command
# ---------------------------------------------------------------------------


class TestValidateCommand:
    def test_validate_valid_clip(self, tmp_path):
        """validate exits 0 and prints VALID for a spec-compliant clip."""
        wav_path = _write_valid_clip(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", str(wav_path)])
        assert result.exit_code == 0, result.output
        assert "VALID" in result.output

    def test_validate_invalid_clip_exits_nonzero(self, tmp_path):
        """validate exits non-zero and prints INVALID for a bad WAV."""
        wav_path = tmp_path / "bad_clip_01.wav"
        txt_path = tmp_path / "bad_clip_01.txt"
        json_path = tmp_path / "bad_clip_01.json"
        # Write an empty (invalid) WAV
        wav_path.write_bytes(b"not a wav")
        txt_path.write_text("x", encoding="utf-8")
        json_path.write_text("{}", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(cli, ["validate", str(wav_path)])
        assert result.exit_code != 0
        assert "INVALID" in result.output


# ---------------------------------------------------------------------------
# generate-batch command
# ---------------------------------------------------------------------------

_BATCH_RUN_CONFIG_TEMPLATE = """\
run_id: test_batch_run
project: she_proves
tier: A
language: he
random_seed: 42
output_dir: {output_dir}
scene_configs_dir: {scene_configs_dir}
targets:
  - violence_typology: NEU
    count: 2
splits:
  train: 0.70
  val: 0.15
  test: 0.15
max_retries: 1
fail_fast: false
"""


class TestGenerateBatchCommand:
    def test_dry_run_with_missing_scene_configs_dir(self, tmp_path):
        """--dry-run exits non-zero when scene_configs_dir does not exist."""
        run_cfg_path = tmp_path / "run.yaml"
        run_cfg_path.write_text(
            _BATCH_RUN_CONFIG_TEMPLATE.format(
                output_dir=str(tmp_path / "out"),
                scene_configs_dir=str(tmp_path / "no_such_dir"),
            ),
            encoding="utf-8",
        )
        runner = CliRunner()
        result = runner.invoke(
            cli, ["generate-batch", "--run-config", str(run_cfg_path), "--dry-run"]
        )
        assert result.exit_code != 0

    def test_dry_run_with_empty_scene_configs_dir(self, tmp_path):
        """--dry-run passes cleanly when scene_configs_dir exists but is empty."""
        scene_dir = tmp_path / "scenes"
        scene_dir.mkdir()
        run_cfg_path = tmp_path / "run.yaml"
        run_cfg_path.write_text(
            _BATCH_RUN_CONFIG_TEMPLATE.format(
                output_dir=str(tmp_path / "out"),
                scene_configs_dir=str(scene_dir),
            ),
            encoding="utf-8",
        )
        runner = CliRunner()
        result = runner.invoke(
            cli, ["generate-batch", "--run-config", str(run_cfg_path), "--dry-run"]
        )
        assert result.exit_code == 0, result.output
        assert "Dry run" in result.output

    def test_dry_run_shows_selection_summary(self, tmp_path):
        """--dry-run prints the number of selected scene configs."""
        scene_dir = tmp_path / "scenes"
        scene_dir.mkdir()
        run_cfg_path = tmp_path / "run.yaml"
        run_cfg_path.write_text(
            _BATCH_RUN_CONFIG_TEMPLATE.format(
                output_dir=str(tmp_path / "out"),
                scene_configs_dir=str(scene_dir),
            ),
            encoding="utf-8",
        )
        runner = CliRunner()
        result = runner.invoke(
            cli, ["generate-batch", "--run-config", str(run_cfg_path), "--dry-run"]
        )
        assert result.exit_code == 0, result.output
        assert "selected" in result.output

    def test_full_batch_with_mocked_tts(self, tmp_path):
        """Full generate-batch renders each scene config and writes a manifest CSV."""
        import yaml as _yaml

        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir()

        # Write two minimal Tier A NEU scene configs pointing at the example speaker
        for i in range(2):
            scene = {
                "scene_id": f"SP_NEU_A_000{i}",
                "project": "she_proves",
                "language": "he",
                "violence_typology": "NEU",
                "tier": "A",
                "random_seed": i,
                "speakers": [{"speaker_id": "AGG_M_30-45_001", "role": "AGG"}],
                "script_template": "synthbanshee/script/templates/she_proves/neutral_domestic_routine.j2",
                "script_slots": {},
                "intensity_arc": [1, 1, 1],
                "target_duration_minutes": 3.0,
                "output_dir": str(tmp_path / "out"),
            }
            (scenes_dir / f"scene_{i:03d}.yaml").write_text(_yaml.dump(scene), encoding="utf-8")

        run_cfg_path = tmp_path / "run.yaml"
        run_cfg_path.write_text(
            _BATCH_RUN_CONFIG_TEMPLATE.format(
                output_dir=str(tmp_path / "out"),
                scene_configs_dir=str(scenes_dir),
            ),
            encoding="utf-8",
        )

        def _fake_render_to_file(text, speaker, output_path, intensity=1):
            Path(output_path).write_bytes(_make_wav_bytes())
            return Path(output_path)

        manifest_path = tmp_path / "manifest.csv"
        runner = CliRunner()
        with patch("synthbanshee.tts.renderer.TTSRenderer") as MockRenderer:
            MockRenderer.return_value.render_utterance_to_file.side_effect = _fake_render_to_file
            result = runner.invoke(
                cli,
                [
                    "generate-batch",
                    "--run-config",
                    str(run_cfg_path),
                    "--manifest-out",
                    str(manifest_path),
                    "--cache-dir",
                    str(tmp_path / "cache"),
                    "--dirty-dir",
                    str(tmp_path / "dirty"),
                ],
            )

        assert result.exit_code == 0, result.output
        assert "Manifest written" in result.output
        # Manifest CSV must exist with at least a header row
        assert manifest_path.exists()


# ---------------------------------------------------------------------------
# qa-report command
# ---------------------------------------------------------------------------


class TestQAReportCommand:
    def test_empty_dir_passes(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["qa-report", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "QA PASSED" in result.output

    def test_valid_clip_passes(self, tmp_path):
        _write_valid_clip(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["qa-report", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "QA PASSED" in result.output

    def test_invalid_clip_fails_report(self, tmp_path):
        """A single invalid clip exceeds the 0% failure budget → exit 1."""
        bad_dir = tmp_path / "spk"
        bad_dir.mkdir()
        (bad_dir / "bad_clip_00.wav").write_bytes(b"garbage")
        (bad_dir / "bad_clip_00.txt").write_text("x", encoding="utf-8")
        (bad_dir / "bad_clip_00.json").write_text("{}", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(cli, ["qa-report", str(tmp_path), "--max-failure-rate", "0.0"])
        assert result.exit_code != 0
        assert "QA FAILED" in result.output

    def test_json_output_written(self, tmp_path):
        """--output writes a JSON report file."""
        _write_valid_clip(tmp_path)
        out_json = tmp_path / "report.json"
        runner = CliRunner()
        result = runner.invoke(cli, ["qa-report", str(tmp_path), "--output", str(out_json)])
        assert result.exit_code == 0, result.output
        assert out_json.exists()
        report = json.loads(out_json.read_text(encoding="utf-8"))
        assert "stats" in report
        assert "passed" in report
        assert report["passed"] is True
