"""Unit tests for the SynthBanshee CLI entry points."""

from __future__ import annotations

import io
import json
import pathlib
import textwrap
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
from click.testing import CliRunner

from synthbanshee.cli import (
    _derive_event_type,
    _discover_scene_configs,
    _print_batch_summary,
    _print_selection_summary,
    _run_generate_pipeline,
    _select_configs_by_typology,
    cli,
)
from synthbanshee.package.validator import ValidationResult

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


def _make_mixed_scene(
    duration_s: float = 5.0, n_turns: int = 1, speaker_id: str = "AGG_M_30-45_001"
):
    """Return a MixedScene for mocking render_scene.

    Onsets/offsets are clamped to [0, duration_s] and each offset > its onset,
    matching SceneMixer.mix_sequential() semantics.
    """
    from synthbanshee.script.types import MixedScene

    sr = 16000
    n = int(sr * duration_s)
    samples = (0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, duration_s, n))).astype(np.float32)
    turn_dur = (duration_s * 0.8) / n_turns  # leave 20% headroom so offsets stay in range
    onsets = [i * turn_dur + 0.3 for i in range(n_turns)]
    offsets = [min(o + turn_dur - 0.05, duration_s) for o in onsets]
    offsets = [max(off, on + 0.05) for on, off in zip(onsets, offsets, strict=False)]
    return MixedScene(
        samples=samples,
        sample_rate=sr,
        turn_onsets_s=onsets,
        turn_offsets_s=offsets,
        duration_s=duration_s,
        speaker_ids=[speaker_id] * n_turns,
    )


def _make_dialogue_turns(n: int = 1, speaker_id: str = "AGG_M_30-45_001", intensity: int = 1):
    """Return a list of DialogueTurn stubs for mocking ScriptGenerator.generate."""
    from synthbanshee.script.types import DialogueTurn

    return [DialogueTurn(speaker_id=speaker_id, text="שלום", intensity=intensity) for _ in range(n)]


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
        """Full generate (no dry-run) wires ScriptGenerator → TTS → preprocessing → output files."""
        turns = _make_dialogue_turns(n=1)
        mixed = _make_mixed_scene(n_turns=1)

        runner = CliRunner()
        with (
            patch("synthbanshee.script.generator.ScriptGenerator") as MockGen,
            patch("synthbanshee.tts.renderer.TTSRenderer") as MockRenderer,
        ):
            MockGen.return_value.generate.return_value = turns
            MockRenderer.return_value.render_scene.return_value = mixed
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
                    "--script-cache-dir",
                    str(tmp_path / "scripts"),
                ],
            )

        assert result.exit_code == 0, result.output
        # At least one WAV file was written
        wav_files = list((tmp_path / "out").rglob("*.wav"))
        assert len(wav_files) >= 1

    def test_full_generate_verbose(self, tmp_path):
        """--verbose flag exercises the vlog code path (cli.py lines 100-101)."""
        turns = _make_dialogue_turns(n=1)
        mixed = _make_mixed_scene(n_turns=1)

        runner = CliRunner()
        with (
            patch("synthbanshee.script.generator.ScriptGenerator") as MockGen,
            patch("synthbanshee.tts.renderer.TTSRenderer") as MockRenderer,
        ):
            MockGen.return_value.generate.return_value = turns
            MockRenderer.return_value.render_scene.return_value = mixed
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
                    "--script-cache-dir",
                    str(tmp_path / "scripts"),
                    "--verbose",
                ],
            )

        assert result.exit_code == 0, result.output
        assert "Stage" in result.output


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

        turns = _make_dialogue_turns(n=1)
        mixed = _make_mixed_scene(n_turns=1)

        manifest_path = tmp_path / "manifest.csv"
        runner = CliRunner()
        with (
            patch("synthbanshee.script.generator.ScriptGenerator") as MockGen,
            patch("synthbanshee.tts.renderer.TTSRenderer") as MockRenderer,
        ):
            MockGen.return_value.generate.return_value = turns
            MockRenderer.return_value.render_scene.return_value = mixed
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
                    "--script-cache-dir",
                    str(tmp_path / "scripts"),
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

    def test_more_than_20_failed_clips_output_truncated(self, tmp_path):
        """When >20 clips fail, the output shows only 20 IDs plus a truncation message."""
        # Create 21 invalid WAV files (bad content, will fail validate_clip)
        for i in range(21):
            clip_dir = tmp_path / f"spk_{i:03d}"
            clip_dir.mkdir()
            (clip_dir / f"bad_clip_{i:03d}_00.wav").write_bytes(b"garbage")
            (clip_dir / f"bad_clip_{i:03d}_00.txt").write_text("x", encoding="utf-8")
            (clip_dir / f"bad_clip_{i:03d}_00.json").write_text("{}", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(cli, ["qa-report", str(tmp_path), "--max-failure-rate", "1.0"])
        # With max_failure_rate=1.0 all failures are acceptable, so exit 0
        assert result.exit_code == 0, result.output
        assert "and" in result.output  # truncation message "... and N more"


# ---------------------------------------------------------------------------
# _run_generate_pipeline — direct unit tests for error branches
# ---------------------------------------------------------------------------


class TestRunGeneratePipeline:
    """Cover error and alternate branches in _run_generate_pipeline."""

    def test_invalid_config_returns_error(self, tmp_path):
        """A YAML that fails SceneConfig validation returns (None, [config error])."""
        bad_cfg = tmp_path / "bad.yaml"
        bad_cfg.write_text("scene_id: x\nproject: not_a_valid_project\n", encoding="utf-8")
        wav, errors = _run_generate_pipeline(bad_cfg, tmp_path, tmp_path, tmp_path, tmp_path)
        assert wav is None
        assert any("Config parse error" in e for e in errors)

    def test_unknown_speaker_returns_error(self, tmp_path):
        """Scene config with a speaker ID that has no YAML anywhere returns (None, errors)."""
        scene_yaml = tmp_path / "scene.yaml"
        scene_yaml.write_text(
            textwrap.dedent("""\
                scene_id: SP_NEU_A_TEST01
                project: she_proves
                language: he
                violence_typology: NEU
                tier: A
                random_seed: 0
                speakers:
                  - speaker_id: NO_SUCH_SPEAKER_XYZ_99999
                    role: AGG
                script_template: synthbanshee/script/templates/she_proves/neutral_domestic_routine.j2
                intensity_arc: [1]
                target_duration_minutes: 3.0
                output_dir: data/he
            """),
            encoding="utf-8",
        )
        wav, errors = _run_generate_pipeline(scene_yaml, tmp_path, tmp_path, tmp_path, tmp_path)
        assert wav is None
        assert any("Speaker config not found" in e for e in errors)

    def test_speaker_parse_error_returns_error(self, tmp_path):
        """SpeakerConfig.from_yaml raising is caught and returned as parse error."""
        with patch(
            "synthbanshee.config.speaker_config.SpeakerConfig.from_yaml",
            side_effect=ValueError("corrupted speaker file"),
        ):
            wav, errors = _run_generate_pipeline(
                SCENES_DIR / "test_scene_001.yaml", tmp_path, tmp_path, tmp_path, tmp_path
            )
        assert wav is None
        assert any("Speaker config parse error" in e for e in errors)

    def test_script_generation_error_returns_error(self, tmp_path):
        """ScriptGenerator.generate() raising is caught and returned as a script error."""
        with (
            patch("synthbanshee.script.generator.ScriptGenerator") as MockGen,
        ):
            MockGen.return_value.generate.side_effect = RuntimeError("LLM quota exceeded")
            wav, errors = _run_generate_pipeline(
                SCENES_DIR / "test_scene_001.yaml",
                tmp_path / "out",
                tmp_path / "cache",
                tmp_path / "dirty",
                tmp_path / "scripts",
            )
        assert wav is None
        assert any("Script generation error" in e for e in errors)

    def test_tts_render_scene_error_returns_error(self, tmp_path):
        """TTSRenderer.render_scene() raising is caught and returned as a TTS render error."""
        turns = _make_dialogue_turns(n=1)
        with (
            patch("synthbanshee.script.generator.ScriptGenerator") as MockGen,
            patch("synthbanshee.tts.renderer.TTSRenderer") as MockRenderer,
        ):
            MockGen.return_value.generate.return_value = turns
            MockRenderer.return_value.render_scene.side_effect = RuntimeError(
                "TTS service unavailable"
            )
            wav, errors = _run_generate_pipeline(
                SCENES_DIR / "test_scene_001.yaml",
                tmp_path / "out",
                tmp_path / "cache",
                tmp_path / "dirty",
                tmp_path / "scripts",
            )
        assert wav is None
        assert any("TTS render error" in e for e in errors)

    def test_validation_failure_returns_error_list(self, tmp_path):
        """A clip that fails validate_clip returns (None, validation errors)."""
        turns = _make_dialogue_turns(n=1)
        mixed = _make_mixed_scene(n_turns=1)

        with (
            patch("synthbanshee.script.generator.ScriptGenerator") as MockGen,
            patch("synthbanshee.tts.renderer.TTSRenderer") as MockRenderer,
            patch(
                "synthbanshee.package.validator.validate_clip",
                return_value=ValidationResult(is_valid=False, errors=["bad sample rate"]),
            ),
        ):
            MockGen.return_value.generate.return_value = turns
            MockRenderer.return_value.render_scene.return_value = mixed
            wav, errors = _run_generate_pipeline(
                SCENES_DIR / "test_scene_001.yaml",
                tmp_path / "out",
                tmp_path / "cache",
                tmp_path / "dirty",
                tmp_path / "scripts",
            )
        assert wav is None
        assert "bad sample rate" in errors

    def test_success_with_warnings_returns_wav_and_warnings(self, tmp_path):
        """Successful pipeline with validation warnings returns (wav_path, warnings)."""
        turns = _make_dialogue_turns(n=1)
        mixed = _make_mixed_scene(n_turns=1)

        with (
            patch("synthbanshee.script.generator.ScriptGenerator") as MockGen,
            patch("synthbanshee.tts.renderer.TTSRenderer") as MockRenderer,
            patch(
                "synthbanshee.package.validator.validate_clip",
                return_value=ValidationResult(is_valid=True, warnings=["minor issue"]),
            ),
        ):
            MockGen.return_value.generate.return_value = turns
            MockRenderer.return_value.render_scene.return_value = mixed
            wav, messages = _run_generate_pipeline(
                SCENES_DIR / "test_scene_001.yaml",
                tmp_path / "out",
                tmp_path / "cache",
                tmp_path / "dirty",
                tmp_path / "scripts",
            )
        assert wav is not None
        assert "minor issue" in messages


# ---------------------------------------------------------------------------
# generate command — additional failure and warning branches
# ---------------------------------------------------------------------------


class TestGenerateCommandBranches:
    """Cover the generate command's failure path and warning loop."""

    def test_pipeline_failure_exits_nonzero_and_prints_failed(self, tmp_path):
        """generate exits non-zero and prints FAILED when pipeline returns (None, errors)."""
        with patch(
            "synthbanshee.cli._run_generate_pipeline",
            return_value=(None, ["TTS quota exceeded"]),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "generate",
                    "--config",
                    str(SCENES_DIR / "test_scene_001.yaml"),
                    "--output-dir",
                    str(tmp_path),
                ],
            )
        assert result.exit_code != 0
        assert "FAILED" in result.output
        assert "TTS quota exceeded" in result.output

    def test_pipeline_success_with_warnings_prints_them(self, tmp_path):
        """generate prints validation warnings when the pipeline returns them."""
        fake_wav = tmp_path / "fake.wav"
        fake_wav.touch()
        with patch(
            "synthbanshee.cli._run_generate_pipeline",
            return_value=(fake_wav, ["clip_id mismatch"]),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "generate",
                    "--config",
                    str(SCENES_DIR / "test_scene_001.yaml"),
                    "--output-dir",
                    str(tmp_path),
                ],
            )
        assert result.exit_code == 0, result.output
        assert "clip_id mismatch" in result.output


# ---------------------------------------------------------------------------
# Internal CLI helpers — direct unit tests
# ---------------------------------------------------------------------------


class TestDiscoverSceneConfigs:
    def test_invalid_yaml_is_skipped(self, tmp_path):
        """YAMLs that fail SceneConfig validation are silently skipped."""
        (tmp_path / "bad.yaml").write_text("project: not_a_valid_project\n", encoding="utf-8")
        result = _discover_scene_configs(tmp_path, "she_proves", "A")
        assert result == []

    def test_valid_matching_config_is_returned(self):
        """test_scene_001.yaml is returned when project/tier match."""
        result = _discover_scene_configs(SCENES_DIR, "she_proves", "A")
        names = [p.name for p in result]
        assert "test_scene_001.yaml" in names

    def test_wrong_project_is_filtered_out(self):
        """Configs with a different project are excluded."""
        result = _discover_scene_configs(SCENES_DIR, "elephant_in_the_room", "A")
        assert all("she_proves" not in p.name for p in result), (
            "she_proves configs should not appear for elephant project"
        )


class TestSelectConfigsByTypology:
    def test_unparseable_config_is_skipped(self, tmp_path):
        """Configs that fail SceneConfig.from_yaml inside _select are silently skipped."""
        bad = tmp_path / "bad.yaml"
        bad.write_text("project: not_a_valid_project\n", encoding="utf-8")
        result = _select_configs_by_typology([bad], {"IT": 10}, rng_seed=0)
        assert result == []

    def test_count_cap_applied(self):
        """Only up to `count` configs per typology are returned."""
        configs = list(SCENES_DIR.rglob("*.yaml"))
        result = _select_configs_by_typology(configs, {"IT": 1}, rng_seed=0)
        assert len(result) <= 1

    def test_reproducible_with_same_seed(self):
        configs = list(SCENES_DIR.rglob("*.yaml"))
        r1 = _select_configs_by_typology(configs, {"IT": 1}, rng_seed=99)
        r2 = _select_configs_by_typology(configs, {"IT": 1}, rng_seed=99)
        assert r1 == r2


class TestPrintSelectionSummary:
    def test_with_valid_config_does_not_raise(self):
        """_print_selection_summary with a valid config renders without error."""
        _print_selection_summary([SCENES_DIR / "test_scene_001.yaml"])

    def test_with_invalid_config_silently_passes(self, tmp_path):
        """An unparseable config is caught and skipped (covers the except:pass branch)."""
        bad = tmp_path / "bad.yaml"
        bad.write_text("project: not_a_valid_project\n", encoding="utf-8")
        _print_selection_summary([bad])  # must not raise

    def test_with_empty_list_does_not_raise(self):
        _print_selection_summary([])


class TestPrintBatchSummary:
    def test_with_failures_prints_without_error(self, tmp_path):
        """_print_batch_summary with non-empty failed list covers the failed-configs block."""
        failed_cfg = tmp_path / "bad_scene.yaml"
        failed_cfg.touch()
        _print_batch_summary(
            succeeded=[],
            failed=[(failed_cfg, ["render timeout", "speaker not found"])],
            splits={},
            failure_rate=1.0,
        )

    def test_with_no_failures_prints_without_error(self, tmp_path):
        wav = tmp_path / "clip.wav"
        wav.touch()
        _print_batch_summary(
            succeeded=[wav],
            failed=[],
            splits={"clip": "train"},
            failure_rate=0.0,
        )


# ---------------------------------------------------------------------------
# generate-batch — fail_fast, failure exit, split-assignment JSON error
# ---------------------------------------------------------------------------

_BATCH_RUN_CONFIG_FAIL_FAST = """\
run_id: test_batch_fail_fast
project: she_proves
tier: A
language: he
random_seed: 42
output_dir: {output_dir}
scene_configs_dir: {scene_configs_dir}
targets:
  - violence_typology: IT
    count: 2
splits:
  train: 0.70
  val: 0.15
  test: 0.15
max_retries: 1
fail_fast: true
"""


class TestGenerateBatchAdvanced:
    """Cover generate-batch branches: fail_fast, failure exit, JSON parse fallback."""

    def _write_it_scene_config(self, scenes_dir: Path, idx: int) -> Path:
        """Write a minimal IT scene config to scenes_dir and return its path."""
        import yaml as _yaml

        scene = {
            "scene_id": f"SP_IT_A_{idx:04d}",
            "project": "she_proves",
            "language": "he",
            "violence_typology": "IT",
            "tier": "A",
            "random_seed": idx,
            "speakers": [{"speaker_id": "AGG_M_30-45_001", "role": "AGG"}],
            "script_template": (
                "synthbanshee/script/templates/she_proves/intimate_terror_coercive_control.j2"
            ),
            "script_slots": {},
            "intensity_arc": [1, 2, 3],
            "target_duration_minutes": 3.0,
            "output_dir": "data/he",
        }
        p = scenes_dir / f"scene_{idx:03d}.yaml"
        p.write_text(_yaml.dump(scene), encoding="utf-8")
        return p

    def test_fail_fast_aborts_on_first_failure(self, tmp_path):
        """fail_fast=true exits immediately after the first pipeline failure."""
        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir()
        self._write_it_scene_config(scenes_dir, 0)
        self._write_it_scene_config(scenes_dir, 1)

        run_cfg = tmp_path / "run.yaml"
        run_cfg.write_text(
            _BATCH_RUN_CONFIG_FAIL_FAST.format(
                output_dir=str(tmp_path / "out"),
                scene_configs_dir=str(scenes_dir),
            ),
            encoding="utf-8",
        )
        with patch(
            "synthbanshee.cli._run_generate_pipeline",
            return_value=(None, ["render failed"]),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "generate-batch",
                    "--run-config",
                    str(run_cfg),
                    "--cache-dir",
                    str(tmp_path / "cache"),
                    "--dirty-dir",
                    str(tmp_path / "dirty"),
                ],
            )
        assert result.exit_code != 0
        assert "fail_fast" in result.output.lower() or "aborting" in result.output.lower()

    def test_batch_with_failures_exits_nonzero(self, tmp_path):
        """generate-batch exits 1 when fail_fast=false but some clips fail."""
        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir()
        self._write_it_scene_config(scenes_dir, 0)

        run_cfg = tmp_path / "run.yaml"
        # Use the standard template (fail_fast: false) from the other test class
        run_cfg.write_text(
            _BATCH_RUN_CONFIG_TEMPLATE.format(
                output_dir=str(tmp_path / "out"),
                scene_configs_dir=str(scenes_dir),
            ).replace("violence_typology: NEU", "violence_typology: IT"),
            encoding="utf-8",
        )

        manifest_path = tmp_path / "manifest.csv"
        with patch(
            "synthbanshee.cli._run_generate_pipeline",
            return_value=(None, ["render error"]),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "generate-batch",
                    "--run-config",
                    str(run_cfg),
                    "--manifest-out",
                    str(manifest_path),
                    "--cache-dir",
                    str(tmp_path / "cache"),
                    "--dirty-dir",
                    str(tmp_path / "dirty"),
                ],
            )
        assert result.exit_code != 0
        # Failed configs section should appear in output
        assert "render error" in result.output

    def test_json_parse_error_in_split_assignment_uses_stem(self, tmp_path):
        """When a succeeded clip's JSON is unreadable, clip_id falls back to wav stem."""
        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir()
        self._write_it_scene_config(scenes_dir, 0)

        run_cfg = tmp_path / "run.yaml"
        run_cfg.write_text(
            _BATCH_RUN_CONFIG_TEMPLATE.format(
                output_dir=str(tmp_path / "out"),
                scene_configs_dir=str(scenes_dir),
            ).replace("violence_typology: NEU", "violence_typology: IT"),
            encoding="utf-8",
        )

        # Create a fake WAV path that has no accompanying JSON
        fake_wav = tmp_path / "out" / "orphan_clip.wav"
        fake_wav.parent.mkdir(parents=True, exist_ok=True)
        fake_wav.write_bytes(b"fake")

        manifest_path = tmp_path / "manifest.csv"
        # Pipeline "succeeds" but returns a path with no JSON
        with patch(
            "synthbanshee.cli._run_generate_pipeline",
            return_value=(fake_wav, []),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "generate-batch",
                    "--run-config",
                    str(run_cfg),
                    "--manifest-out",
                    str(manifest_path),
                    "--cache-dir",
                    str(tmp_path / "cache"),
                    "--dirty-dir",
                    str(tmp_path / "dirty"),
                ],
            )
        # Should complete (even though split assignment falls back to stem)
        assert "Manifest written" in result.output

    def test_retry_succeeds_on_second_attempt(self, tmp_path):
        """A clip that fails once but succeeds on the second attempt is counted as succeeded."""
        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir()
        self._write_it_scene_config(scenes_dir, 0)

        # Use max_retries: 2 so the retry loop runs twice
        run_cfg_yaml = (
            _BATCH_RUN_CONFIG_TEMPLATE.format(
                output_dir=str(tmp_path / "out"),
                scene_configs_dir=str(scenes_dir),
            )
            .replace("max_retries: 1", "max_retries: 2")
            .replace("violence_typology: NEU", "violence_typology: IT")
        )
        run_cfg = tmp_path / "run.yaml"
        run_cfg.write_text(run_cfg_yaml, encoding="utf-8")

        fake_wav = tmp_path / "out" / "clip_ok.wav"
        fake_wav.parent.mkdir(parents=True, exist_ok=True)
        fake_wav.write_bytes(b"fake")
        call_count = [0]

        def _fail_then_succeed(config, out_dir, cache_dir, dirty_dir, script_cache_dir):
            call_count[0] += 1
            if call_count[0] == 1:
                return None, ["transient render error"]
            return fake_wav, []

        manifest_path = tmp_path / "manifest.csv"
        with patch("synthbanshee.cli._run_generate_pipeline", side_effect=_fail_then_succeed):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "generate-batch",
                    "--run-config",
                    str(run_cfg),
                    "--manifest-out",
                    str(manifest_path),
                    "--cache-dir",
                    str(tmp_path / "cache"),
                    "--dirty-dir",
                    str(tmp_path / "dirty"),
                ],
            )

        assert result.exit_code == 0, result.output
        assert call_count[0] == 2  # tried twice
        assert "Manifest written" in result.output


# ---------------------------------------------------------------------------
# generate-batch — parallel workers (--workers / -j)
# ---------------------------------------------------------------------------


class TestGenerateBatchParallel:
    """Tests for --workers N parallel rendering path."""

    def _write_neu_scene(self, scenes_dir: Path, idx: int) -> Path:
        import yaml as _yaml

        scene = {
            "scene_id": f"SP_NEU_A_{idx:04d}",
            "project": "she_proves",
            "language": "he",
            "violence_typology": "NEU",
            "tier": "A",
            "random_seed": idx,
            "speakers": [{"speaker_id": "AGG_M_30-45_001", "role": "AGG"}],
            "script_template": "synthbanshee/script/templates/she_proves/neutral_domestic_routine.j2",
            "script_slots": {},
            "intensity_arc": [1, 1, 1],
            "target_duration_minutes": 3.0,
            "output_dir": "data/he",
        }
        p = scenes_dir / f"scene_{idx:03d}.yaml"
        p.write_text(_yaml.dump(scene), encoding="utf-8")
        return p

    def test_workers_flag_renders_all_clips(self, tmp_path):
        """--workers 2 renders all clips and writes a manifest, same as sequential."""
        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir()
        for i in range(4):
            self._write_neu_scene(scenes_dir, i)

        run_cfg = tmp_path / "run.yaml"
        run_cfg.write_text(
            _BATCH_RUN_CONFIG_TEMPLATE.format(
                output_dir=str(tmp_path / "out"),
                scene_configs_dir=str(scenes_dir),
            ),
            encoding="utf-8",
        )

        turns = _make_dialogue_turns(n=1)
        mixed = _make_mixed_scene(n_turns=1)
        manifest_path = tmp_path / "manifest.csv"

        with (
            patch("synthbanshee.script.generator.ScriptGenerator") as MockGen,
            patch("synthbanshee.tts.renderer.TTSRenderer") as MockRenderer,
        ):
            MockGen.return_value.generate.return_value = turns
            MockRenderer.return_value.render_scene.return_value = mixed
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "generate-batch",
                    "--run-config",
                    str(run_cfg),
                    "--manifest-out",
                    str(manifest_path),
                    "--cache-dir",
                    str(tmp_path / "cache"),
                    "--dirty-dir",
                    str(tmp_path / "dirty"),
                    "--script-cache-dir",
                    str(tmp_path / "scripts"),
                    "--workers",
                    "2",
                ],
            )

        assert result.exit_code == 0, result.output
        assert "Manifest written" in result.output
        assert manifest_path.exists()

    def test_workers_shortflag(self, tmp_path):
        """-j 2 is accepted as an alias for --workers 2."""
        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir()
        self._write_neu_scene(scenes_dir, 0)

        run_cfg = tmp_path / "run.yaml"
        run_cfg.write_text(
            _BATCH_RUN_CONFIG_TEMPLATE.format(
                output_dir=str(tmp_path / "out"),
                scene_configs_dir=str(scenes_dir),
            ),
            encoding="utf-8",
        )

        turns = _make_dialogue_turns(n=1)
        mixed = _make_mixed_scene(n_turns=1)

        with (
            patch("synthbanshee.script.generator.ScriptGenerator") as MockGen,
            patch("synthbanshee.tts.renderer.TTSRenderer") as MockRenderer,
        ):
            MockGen.return_value.generate.return_value = turns
            MockRenderer.return_value.render_scene.return_value = mixed
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "generate-batch",
                    "--run-config",
                    str(run_cfg),
                    "--cache-dir",
                    str(tmp_path / "cache"),
                    "--dirty-dir",
                    str(tmp_path / "dirty"),
                    "--script-cache-dir",
                    str(tmp_path / "scripts"),
                    "-j",
                    "2",
                ],
            )

        assert result.exit_code == 0, result.output
        assert "Manifest written" in result.output

    def test_parallel_workers_log_message(self, tmp_path):
        """--workers N > 1 prints a parallel rendering notice."""
        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir()
        self._write_neu_scene(scenes_dir, 0)

        run_cfg = tmp_path / "run.yaml"
        run_cfg.write_text(
            _BATCH_RUN_CONFIG_TEMPLATE.format(
                output_dir=str(tmp_path / "out"),
                scene_configs_dir=str(scenes_dir),
            ),
            encoding="utf-8",
        )

        turns = _make_dialogue_turns(n=1)
        mixed = _make_mixed_scene(n_turns=1)

        with (
            patch("synthbanshee.script.generator.ScriptGenerator") as MockGen,
            patch("synthbanshee.tts.renderer.TTSRenderer") as MockRenderer,
        ):
            MockGen.return_value.generate.return_value = turns
            MockRenderer.return_value.render_scene.return_value = mixed
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "generate-batch",
                    "--run-config",
                    str(run_cfg),
                    "--cache-dir",
                    str(tmp_path / "cache"),
                    "--dirty-dir",
                    str(tmp_path / "dirty"),
                    "--script-cache-dir",
                    str(tmp_path / "scripts"),
                    "--workers",
                    "3",
                ],
            )

        assert result.exit_code == 0, result.output
        assert "3 workers" in result.output
        assert "Manifest written" in result.output

    def test_workers_invalid_zero_rejected(self, tmp_path):
        """--workers 0 is rejected by Click (IntRange min=1)."""
        run_cfg = tmp_path / "run.yaml"
        run_cfg.write_text(
            _BATCH_RUN_CONFIG_TEMPLATE.format(
                output_dir=str(tmp_path / "out"),
                scene_configs_dir=str(tmp_path / "scenes"),
            ),
            encoding="utf-8",
        )
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "generate-batch",
                "--run-config",
                str(run_cfg),
                "--workers",
                "0",
            ],
        )
        assert result.exit_code != 0

    def test_fail_fast_parallel_aborts(self, tmp_path):
        """fail_fast=true with --workers 2 aborts after first failure."""
        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir()
        for i in range(3):
            self._write_neu_scene(scenes_dir, i)

        run_cfg_yaml = _BATCH_RUN_CONFIG_TEMPLATE.format(
            output_dir=str(tmp_path / "out"),
            scene_configs_dir=str(scenes_dir),
        ).replace("fail_fast: false", "fail_fast: true")
        run_cfg = tmp_path / "run.yaml"
        run_cfg.write_text(run_cfg_yaml, encoding="utf-8")

        with patch(
            "synthbanshee.cli._run_generate_pipeline",
            return_value=(None, ["render failed"]),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "generate-batch",
                    "--run-config",
                    str(run_cfg),
                    "--cache-dir",
                    str(tmp_path / "cache"),
                    "--dirty-dir",
                    str(tmp_path / "dirty"),
                    "--workers",
                    "2",
                ],
            )

        assert result.exit_code != 0
        assert "fail_fast" in result.output.lower() or "aborting" in result.output.lower()

    def test_parallel_failure_without_fail_fast(self, tmp_path):
        """Parallel batch generation reports a failing clip when fail_fast is disabled."""
        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir()
        self._write_neu_scene(scenes_dir, 0)  # only 1 scene → 1 future

        run_cfg = tmp_path / "run.yaml"
        run_cfg.write_text(
            _BATCH_RUN_CONFIG_TEMPLATE.format(
                output_dir=str(tmp_path / "out"),
                scene_configs_dir=str(scenes_dir),
            ),
            encoding="utf-8",
        )  # fail_fast: false (default in template)

        with patch(
            "synthbanshee.cli._run_generate_pipeline",
            return_value=(None, ["render error"]),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "generate-batch",
                    "--run-config",
                    str(run_cfg),
                    "--cache-dir",
                    str(tmp_path / "cache"),
                    "--dirty-dir",
                    str(tmp_path / "dirty"),
                    "--workers",
                    "2",
                ],
            )

        assert result.exit_code != 0

    def test_parallel_subsequent_failures_after_stop_event(self, tmp_path):
        """With fail_fast=true and multiple parallel failures, clips that complete
        after the first failure is recorded are silently skipped rather than
        double-counted in the failed list."""
        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir()
        for i in range(3):
            self._write_neu_scene(scenes_dir, i)

        # count: 3 so all three scenes are selected and submitted as futures
        run_cfg_yaml = (
            _BATCH_RUN_CONFIG_TEMPLATE.format(
                output_dir=str(tmp_path / "out"),
                scene_configs_dir=str(scenes_dir),
            )
            .replace("count: 2", "count: 3")
            .replace("fail_fast: false", "fail_fast: true")
        )
        run_cfg = tmp_path / "run.yaml"
        run_cfg.write_text(run_cfg_yaml, encoding="utf-8")

        # Patch _render_one directly so all three futures return immediately and
        # deterministically with a failure, regardless of stop_event state.
        with patch(
            "synthbanshee.cli._render_one",
            return_value=(None, ["render failed"]),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "generate-batch",
                    "--run-config",
                    str(run_cfg),
                    "--cache-dir",
                    str(tmp_path / "cache"),
                    "--dirty-dir",
                    str(tmp_path / "dirty"),
                    "--workers",
                    "3",
                ],
            )

        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# _render_one — direct unit tests
# ---------------------------------------------------------------------------


class TestRenderOne:
    """Direct unit tests for the _render_one() helper."""

    def test_stop_event_returns_cancelled_immediately(self, tmp_path):
        """When stop_event is already set, _render_one returns (None, ['cancelled'])
        without ever calling _run_generate_pipeline."""
        import threading

        from synthbanshee.cli import _render_one

        stop = threading.Event()
        stop.set()

        with patch("synthbanshee.cli._run_generate_pipeline") as mock_pipeline:
            wav, messages = _render_one(
                tmp_path / "dummy.yaml",
                tmp_path / "out",
                tmp_path / "cache",
                tmp_path / "dirty",
                tmp_path / "scripts",
                max_retries=3,
                stop_event=stop,
            )

        assert wav is None
        assert messages == ["cancelled"]
        mock_pipeline.assert_not_called()

    def test_success_on_first_attempt(self, tmp_path):
        """_render_one returns the wav path immediately on a successful render."""
        import threading

        from synthbanshee.cli import _render_one

        fake_wav = tmp_path / "clip.wav"
        fake_wav.write_bytes(b"fake")
        stop = threading.Event()

        with patch("synthbanshee.cli._run_generate_pipeline", return_value=(fake_wav, [])):
            wav, messages = _render_one(
                tmp_path / "scene.yaml",
                tmp_path / "out",
                tmp_path / "cache",
                tmp_path / "dirty",
                tmp_path / "scripts",
                max_retries=3,
                stop_event=stop,
            )

        assert wav == fake_wav
        assert messages == []

    def test_exhausted_retries_returns_none(self, tmp_path):
        """_render_one returns (None, messages) after exhausting all retries."""
        import threading

        from synthbanshee.cli import _render_one

        stop = threading.Event()

        with patch(
            "synthbanshee.cli._run_generate_pipeline",
            return_value=(None, ["render error"]),
        ):
            wav, messages = _render_one(
                tmp_path / "scene.yaml",
                tmp_path / "out",
                tmp_path / "cache",
                tmp_path / "dirty",
                tmp_path / "scripts",
                max_retries=2,
                stop_event=stop,
            )

        assert wav is None
        assert messages == ["render error"]


# ---------------------------------------------------------------------------
# _run_generate_pipeline — primary speaker path (line 61 False branch)
# ---------------------------------------------------------------------------


class TestRunGeneratePipelinePrimaryPath:
    def test_primary_speaker_config_used_when_exists(self, tmp_path):
        """Covers primary path: configs/speakers/{id}.yaml found → fallback skipped."""
        from synthbanshee.config.speaker_config import SpeakerConfig

        example_speaker = SpeakerConfig.from_yaml(EXAMPLES_DIR / "speaker_AGG_M_30-45_001.yaml")
        _orig_exists = pathlib.Path.exists
        primary_path = str(Path("configs/speakers/AGG_M_30-45_001.yaml"))

        def _fake_exists(self: pathlib.Path) -> bool:
            if str(self) == primary_path:
                return True
            return _orig_exists(self)

        turns = _make_dialogue_turns(n=1)
        mixed = _make_mixed_scene(n_turns=1)

        with (
            patch.object(pathlib.Path, "exists", _fake_exists),
            patch(
                "synthbanshee.config.speaker_config.SpeakerConfig.from_yaml",
                return_value=example_speaker,
            ),
            patch("synthbanshee.script.generator.ScriptGenerator") as MockGen,
            patch("synthbanshee.tts.renderer.TTSRenderer") as MockRenderer,
        ):
            MockGen.return_value.generate.return_value = turns
            MockRenderer.return_value.render_scene.return_value = mixed
            wav, errors = _run_generate_pipeline(
                SCENES_DIR / "test_scene_001.yaml",
                tmp_path / "out",
                tmp_path / "cache",
                tmp_path / "dirty",
                tmp_path / "scripts",
            )

        assert wav is not None, errors


# ---------------------------------------------------------------------------
# _derive_event_type — unit tests
# ---------------------------------------------------------------------------


class TestDeriveEventType:
    def test_neu_always_ambient(self):
        for intensity in range(1, 6):
            t1, t2 = _derive_event_type("NEU", intensity)
            assert t1 == "NONE"
            assert t2 == "NONE_AMBIENT"

    def test_neg_always_argu(self):
        for intensity in range(1, 6):
            t1, t2 = _derive_event_type("NEG", intensity)
            assert t1 == "NONE"
            assert t2 == "NONE_ARGU"

    def test_it_low_intensity_gaslit(self):
        t1, t2 = _derive_event_type("IT", 1)
        assert t1 == "EMOT" and t2 == "EMOT_GASLIT"
        t1, t2 = _derive_event_type("IT", 2)
        assert t1 == "EMOT" and t2 == "EMOT_GASLIT"

    def test_it_mid_intensity_isol(self):
        t1, t2 = _derive_event_type("IT", 3)
        assert t1 == "EMOT" and t2 == "EMOT_ISOL"

    def test_it_high_intensity_threat(self):
        t1, t2 = _derive_event_type("IT", 4)
        assert t1 == "VERB" and t2 == "VERB_THREAT"
        t1, t2 = _derive_event_type("IT", 5)
        assert t1 == "VERB" and t2 == "VERB_THREAT"

    def test_sv_low_intensity_shout(self):
        t1, t2 = _derive_event_type("SV", 1)
        assert t1 == "VERB" and t2 == "VERB_SHOUT"
        t1, t2 = _derive_event_type("SV", 2)
        assert t1 == "VERB" and t2 == "VERB_SHOUT"

    def test_sv_mid_intensity_threat(self):
        t1, t2 = _derive_event_type("SV", 3)
        assert t1 == "VERB" and t2 == "VERB_THREAT"

    def test_sv_high_intensity_scream(self):
        t1, t2 = _derive_event_type("SV", 4)
        assert t1 == "DIST" and t2 == "DIST_SCREAM"

    def test_sv_max_intensity_phys(self):
        t1, t2 = _derive_event_type("SV", 5)
        assert t1 == "PHYS" and t2 == "PHYS_HARD"

    def test_unknown_typology_defaults_to_ambient(self):
        t1, t2 = _derive_event_type("UNKNOWN", 3)
        assert t1 == "NONE" and t2 == "NONE_AMBIENT"


# ---------------------------------------------------------------------------
# Tier B pipeline — Stage 3 acoustic augmentation branch
# ---------------------------------------------------------------------------

_TIER_B_SCENE_YAML = """\
scene_id: SP_SV_B_TEST01
project: she_proves
language: he
violence_typology: SV
tier: B
random_seed: 0
speakers:
  - speaker_id: AGG_M_30-45_001
    role: AGG
script_template: synthbanshee/script/templates/she_proves/neutral_domestic_routine.j2
script_slots: {}
intensity_arc: [2, 3, 5]
target_duration_minutes: 3.0
acoustic_scene:
  room_type: small_bedroom
  device: phone_in_hand
  speaker_distance_meters: 1.5
  victim_distance_meters: 1.0
  snr_target_db: 15
output_dir: data/he
"""


def _make_tier_b_scene_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "tier_b_scene.yaml"
    p.write_text(_TIER_B_SCENE_YAML, encoding="utf-8")
    return p


_TIER_C_SCENE_YAML = """\
scene_id: SP_NEG_C_TEST01
project: she_proves
language: he
violence_typology: NEG
tier: C
random_seed: 0
speakers:
  - speaker_id: AGG_M_30-45_001
    role: AGG
script_template: synthbanshee/script/templates/she_proves/confusor_heated_but_resolved.j2
script_slots: {}
intensity_arc: [1, 2, 3, 2, 1]
target_duration_minutes: 3.0
acoustic_scene:
  room_type: apartment_kitchen
  device: phone_on_table
  speaker_distance_meters: 1.5
  victim_distance_meters: 1.2
  snr_target_db: 14
output_dir: data/he
"""


def _make_tier_c_scene_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "tier_c_scene.yaml"
    p.write_text(_TIER_C_SCENE_YAML, encoding="utf-8")
    return p


class TestRunGeneratePipelineTierB:
    """Cover the Stage 3 acoustic augmentation branch in _run_generate_pipeline."""

    def _run_tier_b(
        self,
        tmp_path: Path,
        *,
        aug_events=None,
        snr_actual: float = 12.0,
        aug_side_effect=None,
        aug_all_zeros: bool = False,
        aug_audio_override: np.ndarray | None = None,
    ):
        """Helper: run Tier B pipeline with mocked TTS + mocked augmentation stack.

        Returns (wav_path, errors).
        """
        from synthbanshee.augment.types import AugmentedEvent

        if aug_events is None:
            aug_events = []

        turns = _make_dialogue_turns(n=1, intensity=3)
        # MixedScene duration is 5 s; preprocess() adds 0.5 s pad at each end → 6 s WAV.
        mixed = _make_mixed_scene(duration_s=5.0, n_turns=1)
        sr = 16_000
        pad = int(0.5 * sr)
        # Build aug_audio to match the preprocessed WAV length (6 s = 5 s + 2 × 0.5 s pad).
        total = mixed.samples.shape[0] + 2 * pad  # 6 × 16 000 = 96 000 samples
        if aug_audio_override is not None:
            aug_audio = aug_audio_override
        elif aug_all_zeros:
            aug_audio = np.zeros(total, dtype=np.float32)
        else:
            aug_audio = np.zeros(total, dtype=np.float32)
            t = np.arange(total - 2 * pad, dtype=np.float32) / sr
            aug_audio[pad : total - pad] = 0.5 * np.sin(2 * np.pi * 440 * t)
            peak = float(np.max(np.abs(aug_audio)))
            if peak > 0.0:
                aug_audio = (aug_audio * (10.0 ** (-1.0 / 20.0) / peak)).astype(np.float32)

        mock_aug_events = [
            AugmentedEvent(
                type=ev["type"],
                onset_s=ev["onset_s"],
                offset_s=ev["offset_s"],
                level_db=ev["level_db"],
            )
            for ev in aug_events
        ]

        with (
            patch("synthbanshee.script.generator.ScriptGenerator") as MockGen,
            patch("synthbanshee.tts.renderer.TTSRenderer") as MockRenderer,
            patch("synthbanshee.augment.room_sim.RoomSimulator") as MockRoom,
            patch("synthbanshee.augment.device_profiles.DeviceProfiler") as MockDevice,
            patch("synthbanshee.augment.noise_mixer.NoiseMixer") as MockMixer,
        ):
            MockGen.return_value.generate.return_value = turns
            MockRenderer.return_value.render_scene.return_value = mixed
            MockRoom.return_value.apply.return_value = aug_audio
            MockDevice.return_value.apply.return_value = aug_audio
            if aug_side_effect is not None:
                MockMixer.return_value.mix.side_effect = aug_side_effect
            else:
                MockMixer.return_value.mix.return_value = (aug_audio, mock_aug_events, snr_actual)

            wav, errors = _run_generate_pipeline(
                _make_tier_b_scene_yaml(tmp_path),
                tmp_path / "out",
                tmp_path / "cache",
                tmp_path / "dirty",
                tmp_path / "scripts",
            )
        return wav, errors

    def test_tier_b_pipeline_succeeds(self, tmp_path):
        """Tier B pipeline with mocked augmentation stack completes without error."""
        wav, errors = self._run_tier_b(tmp_path)
        assert wav is not None, errors

    def test_tier_b_acoustic_scene_in_metadata(self, tmp_path):
        """acoustic_scene block in metadata JSON contains room_type, device, ir_source."""
        wav, errors = self._run_tier_b(tmp_path)
        assert wav is not None, errors
        json_path = wav.with_suffix(".json")
        assert json_path.exists()
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        scene = meta["acoustic_scene"]
        assert scene["room_type"] == "small_bedroom"
        assert scene["device"] == "phone_in_hand"
        assert scene["ir_source"] == "pyroomacoustics_ism"
        assert scene["speaker_distance_meters"] == 1.5

    def test_tier_b_snr_db_actual_in_metadata(self, tmp_path):
        """snr_db_actual is populated in acoustic_scene metadata for Tier B clips."""
        wav, errors = self._run_tier_b(tmp_path, snr_actual=14.75)
        assert wav is not None, errors
        meta = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["acoustic_scene"]["snr_db_actual"] == 14.75

    def test_tier_b_acou_sfx_appear_in_weak_label(self, tmp_path):
        """ACOU_* SFX from Stage 3 contribute 'ACOU' to weak_label.violence_categories."""
        aug_events = [
            {"type": "ACOU_SLAM", "onset_s": 2.5, "offset_s": 2.9, "level_db": -18.0},
        ]
        wav, errors = self._run_tier_b(tmp_path, aug_events=aug_events)
        assert wav is not None, errors
        meta = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
        assert "ACOU" in meta["weak_label"]["violence_categories"]

    def test_tier_b_ambient_events_not_in_weak_label(self, tmp_path):
        """Non-ACOU ambient events (tv_ambient) are NOT added as ScriptEvents."""
        # tv_ambient is not an ACOU_* event so it should not appear in violence_categories
        aug_events = [
            {"type": "tv_ambient", "onset_s": 0.0, "offset_s": 5.0, "level_db": -30.0},
        ]
        wav, errors = self._run_tier_b(tmp_path, aug_events=aug_events)
        assert wav is not None, errors
        meta = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
        # ACOU should NOT appear (only VERB from the SV speech turns at intensity 3)
        assert "ACOU" not in meta["weak_label"]["violence_categories"]

    def test_tier_b_augmentation_error_returns_error(self, tmp_path):
        """When Stage 3 augmentation raises, pipeline returns (None, [augmentation error])."""
        wav, errors = self._run_tier_b(
            tmp_path,
            aug_side_effect=RuntimeError("pyroomacoustics simulation failed"),
        )
        assert wav is None
        assert any("Acoustic augmentation error" in e for e in errors)

    def test_tier_b_silent_augmented_output_skips_normalization(self, tmp_path):
        """When NoiseMixer returns all-zero audio, peak==0 branch is skipped gracefully."""
        # Covers the `if peak > 0.0` False branch (line 209 in cli.py)
        wav, errors = self._run_tier_b(tmp_path, aug_all_zeros=True)
        assert wav is not None, errors
        # WAV must still exist and be the correct length (not crash on zero-peak input)
        import soundfile as _sf

        data, sr = _sf.read(str(wav))
        assert len(data) > 0

    def test_tier_b_unknown_acou_type_is_dropped(self, tmp_path):
        """ACOU_* events with unrecognised tier2 codes are dropped, not propagated as labels."""
        aug_events = [
            {"type": "ACOU_UNKNOWN_XYZ", "onset_s": 1.0, "offset_s": 1.5, "level_db": -20.0},
        ]
        wav, errors = self._run_tier_b(tmp_path, aug_events=aug_events)
        assert wav is not None, errors
        meta = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
        assert "ACOU" not in meta["weak_label"]["violence_categories"]

    def test_tier_b_output_wav_length_preserved(self, tmp_path):
        """Stage 3 write-back keeps the WAV at exactly the preprocessed length."""
        import soundfile as _sf

        wav, errors = self._run_tier_b(tmp_path)
        assert wav is not None, errors
        data, sr = _sf.read(str(wav))
        # Preprocessed duration = MixedScene 5 s + 2 × 0.5 s pad = 6 s
        assert abs(len(data) / sr - 6.0) < 0.05

    def test_tier_b_aug_samples_trimmed_when_too_long(self, tmp_path):
        """aug_samples longer than the input WAV are trimmed to n_orig (line 215-216)."""
        import soundfile as _sf

        sr = 16_000
        # 7 s > 6 s expected preprocessed length — exercises the > n_orig trim branch
        long_audio = np.zeros(sr * 7, dtype=np.float32)
        wav, errors = self._run_tier_b(tmp_path, aug_audio_override=long_audio)
        assert wav is not None, errors
        data, _ = _sf.read(str(wav))
        assert abs(len(data) / sr - 6.0) < 0.05

    def test_tier_b_aug_samples_padded_when_too_short(self, tmp_path):
        """aug_samples shorter than the input WAV are zero-padded to n_orig (line 217-220)."""
        import soundfile as _sf

        sr = 16_000
        # 3 s < 6 s expected preprocessed length — exercises the < n_orig pad branch
        short_audio = np.zeros(sr * 3, dtype=np.float32)
        wav, errors = self._run_tier_b(tmp_path, aug_audio_override=short_audio)
        assert wav is not None, errors
        data, _ = _sf.read(str(wav))
        assert abs(len(data) / sr - 6.0) < 0.05

    def test_tier_b_strong_labels_jsonl_written(self, tmp_path):
        """Stage 7b writes a .jsonl strong-labels file alongside the clip WAV."""
        wav, errors = self._run_tier_b(tmp_path)
        assert wav is not None, errors
        jsonl_path = wav.with_suffix(".jsonl")
        assert jsonl_path.exists(), f"Expected strong labels JSONL at {jsonl_path}"
        # Each line must be valid JSON
        import json as _json

        lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            _json.loads(line)  # raises on malformed JSON

    def test_tier_b_pad_regions_zeroed_after_augmentation(self, tmp_path):
        """Ambient energy in silence-pad regions is zeroed before write-back.

        Covers the validate_audio() head/tail silence requirement: even when
        NoiseMixer fills the entire clip (including padding) with ambient noise,
        the written WAV must have silence in the first/last 0.5 s.
        """
        import soundfile as _sf

        sr = 16_000
        pad = int(0.5 * sr)
        total = sr * 6  # 6 s preprocessed length
        # Build audio where the pad regions are NOT silent — simulating ambient bleed
        noisy_pad_audio = np.full(total, 0.1, dtype=np.float32)
        wav, errors = self._run_tier_b(tmp_path, aug_audio_override=noisy_pad_audio)
        assert wav is not None, errors
        data, _ = _sf.read(str(wav))
        # First and last 0.5 s must be silent after Stage 3 pad-zeroing
        assert np.all(data[:pad] == 0.0), "head pad should be zeroed"
        assert np.all(data[-pad:] == 0.0), "tail pad should be zeroed"


class TestRunGeneratePipelineTierC:
    """Tier C (hard-negative confusor) pipeline uses Stage 3b augmentation like Tier B."""

    def _run_tier_c(self, tmp_path: Path):
        turns = _make_dialogue_turns(n=1, intensity=2)
        mixed = _make_mixed_scene(duration_s=5.0, n_turns=1)
        sr = 16_000
        pad = int(0.5 * sr)
        total = mixed.samples.shape[0] + 2 * pad
        aug_audio = np.zeros(total, dtype=np.float32)
        t = np.arange(mixed.samples.shape[0], dtype=np.float32) / sr
        aug_audio[pad : total - pad] = 0.3 * np.sin(2 * np.pi * 440 * t)
        peak = float(np.max(np.abs(aug_audio)))
        aug_audio = (aug_audio * (10.0 ** (-1.0 / 20.0) / peak)).astype(np.float32)

        with (
            patch("synthbanshee.script.generator.ScriptGenerator") as MockGen,
            patch("synthbanshee.tts.renderer.TTSRenderer") as MockRenderer,
            patch("synthbanshee.augment.room_sim.RoomSimulator") as MockRoom,
            patch("synthbanshee.augment.device_profiles.DeviceProfiler") as MockDevice,
            patch("synthbanshee.augment.noise_mixer.NoiseMixer") as MockMixer,
        ):
            MockGen.return_value.generate.return_value = turns
            MockRenderer.return_value.render_scene.return_value = mixed
            MockRoom.return_value.apply.return_value = aug_audio
            MockDevice.return_value.apply.return_value = aug_audio
            MockMixer.return_value.mix.return_value = (aug_audio, [], 16.0)

            return _run_generate_pipeline(
                _make_tier_c_scene_yaml(tmp_path),
                tmp_path / "out",
                tmp_path / "cache",
                tmp_path / "dirty",
                tmp_path / "scripts",
            )

    def test_tier_c_pipeline_succeeds(self, tmp_path):
        """Tier C pipeline runs Stage 3b augmentation and completes without error."""
        wav, errors = self._run_tier_c(tmp_path)
        assert wav is not None, errors

    def test_tier_c_acoustic_scene_in_metadata(self, tmp_path):
        """Tier C clips carry an acoustic_scene block in metadata (Stage 3b ran)."""
        wav, errors = self._run_tier_c(tmp_path)
        assert wav is not None, errors
        meta = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["acoustic_scene"]["room_type"] == "apartment_kitchen"
        assert meta["acoustic_scene"]["ir_source"] == "pyroomacoustics_ism"

    def test_tier_c_violence_typology_is_neg(self, tmp_path):
        """Tier C confusor clips carry violence_typology=NEG and no violence categories."""
        wav, errors = self._run_tier_c(tmp_path)
        assert wav is not None, errors
        meta = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["violence_typology"] == "NEG"
        assert meta["weak_label"]["has_violence"] is False

    def test_tier_c_strong_labels_jsonl_written(self, tmp_path):
        """Stage 4b JSONL is written for Tier C clips just as for Tier B."""
        wav, errors = self._run_tier_c(tmp_path)
        assert wav is not None, errors
        assert wav.with_suffix(".jsonl").exists()

    def test_tier_c_augmentation_called(self, tmp_path):
        """RoomSimulator is invoked for Tier C scenes (Stage 3b is not Tier B exclusive)."""
        turns = _make_dialogue_turns(n=1, intensity=2)
        mixed = _make_mixed_scene(duration_s=5.0, n_turns=1)
        sr = 16_000
        pad = int(0.5 * sr)
        total = mixed.samples.shape[0] + 2 * pad
        aug_audio = np.zeros(total, dtype=np.float32)

        with (
            patch("synthbanshee.script.generator.ScriptGenerator") as MockGen,
            patch("synthbanshee.tts.renderer.TTSRenderer") as MockRenderer,
            patch("synthbanshee.augment.room_sim.RoomSimulator") as MockRoom,
            patch("synthbanshee.augment.device_profiles.DeviceProfiler") as MockDevice,
            patch("synthbanshee.augment.noise_mixer.NoiseMixer") as MockMixer,
        ):
            MockGen.return_value.generate.return_value = turns
            MockRenderer.return_value.render_scene.return_value = mixed
            MockRoom.return_value.apply.return_value = aug_audio
            MockDevice.return_value.apply.return_value = aug_audio
            MockMixer.return_value.mix.return_value = (aug_audio, [], 16.0)

            wav, errors = _run_generate_pipeline(
                _make_tier_c_scene_yaml(tmp_path),
                tmp_path / "out",
                tmp_path / "cache",
                tmp_path / "dirty",
                tmp_path / "scripts",
            )
            assert wav is not None, errors


# ---------------------------------------------------------------------------
# iaa-report command
# ---------------------------------------------------------------------------

# Five-category JSONL (one event per category): PHYS, VERB, DIST, ACOU, EMOT.
# Using this for both annotators in "agreement" clips gives a mix of
# present/absent across category groups when combined with empty clips,
# which avoids the degenerate p_e=1 case in cohen_kappa.
_ALL_CATEGORIES_JSONL = "\n".join(
    [
        '{"event_id":"evt_001","clip_id":"clip_X","onset":0.5,"offset":1.5,'
        '"tier1_category":"PHYS","tier2_subtype":"PHYS_HARD","intensity":3,'
        '"speaker_id":null,"speaker_role":null,"emotional_state":null,'
        '"confidence":1.0,"label_source":"auto","iaa_reviewed":false,"notes":null}',
        '{"event_id":"evt_002","clip_id":"clip_X","onset":1.5,"offset":2.5,'
        '"tier1_category":"VERB","tier2_subtype":"VERB_THREAT","intensity":3,'
        '"speaker_id":null,"speaker_role":null,"emotional_state":null,'
        '"confidence":1.0,"label_source":"auto","iaa_reviewed":false,"notes":null}',
        '{"event_id":"evt_003","clip_id":"clip_X","onset":2.5,"offset":3.5,'
        '"tier1_category":"DIST","tier2_subtype":"DIST_SCREAM","intensity":3,'
        '"speaker_id":null,"speaker_role":null,"emotional_state":null,'
        '"confidence":1.0,"label_source":"auto","iaa_reviewed":false,"notes":null}',
        '{"event_id":"evt_004","clip_id":"clip_X","onset":3.5,"offset":4.5,'
        '"tier1_category":"ACOU","tier2_subtype":"ACOU_SLAM","intensity":3,'
        '"speaker_id":null,"speaker_role":null,"emotional_state":null,'
        '"confidence":1.0,"label_source":"auto","iaa_reviewed":false,"notes":null}',
        '{"event_id":"evt_005","clip_id":"clip_X","onset":4.5,"offset":5.5,'
        '"tier1_category":"EMOT","tier2_subtype":"EMOT_ECON","intensity":3,'
        '"speaker_id":null,"speaker_role":null,"emotional_state":null,'
        '"confidence":1.0,"label_source":"auto","iaa_reviewed":false,"notes":null}',
    ]
)
# Single-category JSONL (PHYS only) for targeted disagreement tests.
_PHYS_EVENT_JSON = (
    '{"event_id":"evt_001","clip_id":"clip_001","onset":1.0,"offset":2.0,'
    '"tier1_category":"PHYS","tier2_subtype":"PHYS_HARD","intensity":3,'
    '"speaker_id":null,"speaker_role":null,"emotional_state":null,'
    '"confidence":1.0,"label_source":"auto","iaa_reviewed":false,"notes":null}'
)


def _make_annotation_dir(
    base: Path,
    clip_dirs: dict[str, tuple[str, str]],
) -> Path:
    """Create annotations_dir/<clip_id>/annotator_{a,b}.jsonl fixture tree.

    clip_dirs maps clip_id -> (jsonl_content_a, jsonl_content_b).
    """
    ann_dir = base / "annotations"
    ann_dir.mkdir()
    for clip_id, (content_a, content_b) in clip_dirs.items():
        clip_dir = ann_dir / clip_id
        clip_dir.mkdir()
        (clip_dir / "annotator_a.jsonl").write_text(content_a, encoding="utf-8")
        (clip_dir / "annotator_b.jsonl").write_text(content_b, encoding="utf-8")
    return ann_dir


def _passing_clips(n_event: int = 10, n_empty: int = 10) -> dict[str, tuple[str, str]]:
    """Return clip_dirs dict with a mix of all-category and empty clips.

    Half-and-half present/absent for every category keeps p_e=0.5, so
    perfect agreement gives kappa=1.0 for all categories (not degenerate).
    """
    clips: dict[str, tuple[str, str]] = {}
    for i in range(n_event):
        clips[f"clip_evt_{i:03d}"] = (_ALL_CATEGORIES_JSONL, _ALL_CATEGORIES_JSONL)
    for i in range(n_empty):
        clips[f"clip_empty_{i:03d}"] = ("", "")
    return clips


class TestIAAReport:
    """Tests for the iaa-report CLI command."""

    def test_passing_report(self, tmp_path):
        """Identical mixed-category pairs produce kappa=1.0 for all categories — exits 0."""
        ann_dir = _make_annotation_dir(tmp_path, _passing_clips())
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["iaa-report", str(ann_dir), "--total-clips", "20"],
        )
        assert result.exit_code == 0, result.output
        assert "PASS" in result.output

    def test_failing_report_exits_1(self, tmp_path):
        """Mismatched annotations drive kappa below minimum thresholds — command exits 1."""
        # A sees all categories; B sees nothing — half the clips disagree.
        clips = {f"clip_{i:03d}": (_ALL_CATEGORIES_JSONL, "") for i in range(10)}
        # Include some "agree on nothing" clips to ensure no degenerate coverage issue.
        for i in range(10):
            clips[f"clip_empty_{i:03d}"] = ("", "")
        ann_dir = _make_annotation_dir(tmp_path, clips)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["iaa-report", str(ann_dir), "--total-clips", "20"],
        )
        assert result.exit_code == 1

    def test_no_pairs_exits_1(self, tmp_path):
        """An annotations_dir with no valid clip subdirs exits 1."""
        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()
        # Put a file (not a dir) and an empty dir (0 jsonl files) — both skipped.
        (ann_dir / "not_a_dir.txt").write_text("x")
        (ann_dir / "empty_clip").mkdir()
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["iaa-report", str(ann_dir), "--total-clips", "10"],
        )
        assert result.exit_code == 1
        assert "No valid annotation pairs" in result.output

    def test_skip_clip_dir_with_wrong_jsonl_count(self, tmp_path):
        """Clip subdirs with ≠2 JSONL files are skipped with a warning."""
        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()
        # One dir with 3 JSONL files (wrong count) — should be warned and skipped.
        bad_dir = ann_dir / "clip_bad"
        bad_dir.mkdir()
        for name in ("a.jsonl", "b.jsonl", "c.jsonl"):
            (bad_dir / name).write_text(_PHYS_EVENT_JSON)
        # Fill in the rest of the clips so we have valid pairs to compute IAA on.
        for clip_id, (ca, cb) in _passing_clips().items():
            d = ann_dir / clip_id
            d.mkdir()
            (d / "annotator_a.jsonl").write_text(ca)
            (d / "annotator_b.jsonl").write_text(cb)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["iaa-report", str(ann_dir), "--total-clips", "20"],
        )
        assert "clip_bad" in result.output  # warning printed for skipped dir
        assert result.exit_code == 0  # _passing_clips() still satisfy thresholds

    def test_skip_clip_dir_with_parse_error(self, tmp_path):
        """Clip subdirs whose JSONL cannot be parsed are skipped with a warning."""
        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()
        bad_dir = ann_dir / "clip_bad"
        bad_dir.mkdir()
        (bad_dir / "annotator_a.jsonl").write_text("not valid json\n")
        (bad_dir / "annotator_b.jsonl").write_text(_PHYS_EVENT_JSON)
        for clip_id, (ca, cb) in _passing_clips().items():
            d = ann_dir / clip_id
            d.mkdir()
            (d / "annotator_a.jsonl").write_text(ca)
            (d / "annotator_b.jsonl").write_text(cb)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["iaa-report", str(ann_dir), "--total-clips", "20"],
        )
        assert "clip_bad" in result.output  # warning printed for skipped dir
        assert result.exit_code == 0  # _passing_clips() still satisfy thresholds

    def test_output_json_written(self, tmp_path):
        """--output writes a valid JSON report file."""
        ann_dir = _make_annotation_dir(tmp_path, _passing_clips())
        out_path = tmp_path / "reports" / "iaa.json"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "iaa-report",
                str(ann_dir),
                "--total-clips",
                "20",
                "--output",
                str(out_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_path.exists()
        report_data = json.loads(out_path.read_text())
        assert "category_results" in report_data
        assert report_data["n_clips_reviewed"] == 20
        assert "IAA report written" in result.output


# ---------------------------------------------------------------------------
# dataset-card and package-dataset commands
# ---------------------------------------------------------------------------


def _make_empty_data_dir(base: Path) -> Path:
    """Return an empty data directory (QA passes with 0 clips)."""
    data_dir = base / "data"
    data_dir.mkdir()
    return data_dir


class TestDatasetCardCommand:
    def test_prints_card_to_stdout(self, tmp_path):
        data_dir = _make_empty_data_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["dataset-card", str(data_dir), "--version", "v1.0"])
        assert result.exit_code == 0, result.output
        assert "AVDP Synthetic Dataset" in result.output
        assert "v1.0" in result.output

    def test_writes_card_to_file(self, tmp_path):
        data_dir = _make_empty_data_dir(tmp_path)
        out = tmp_path / "card.md"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["dataset-card", str(data_dir), "--version", "v1.0", "--output", str(out)],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
        assert "AVDP Synthetic Dataset" in out.read_text()
        assert "Dataset card written" in result.output

    def test_short_version_flag(self, tmp_path):
        data_dir = _make_empty_data_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["dataset-card", str(data_dir), "-v", "v2.0"])
        assert result.exit_code == 0, result.output
        assert "v2.0" in result.output

    def test_invalid_version_rejected(self, tmp_path):
        """Version strings with path separators or spaces are rejected."""
        data_dir = _make_empty_data_dir(tmp_path)
        runner = CliRunner()
        for bad in ("../evil", "v1 0", "v1:0"):
            result = runner.invoke(cli, ["dataset-card", str(data_dir), "-v", bad])
            assert result.exit_code != 0, f"Expected failure for version {bad!r}"


class TestPackageDatasetCommand:
    def test_creates_archive_and_card(self, tmp_path):
        data_dir = _make_empty_data_dir(tmp_path)
        out_dir = tmp_path / "releases"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["package-dataset", str(data_dir), str(out_dir), "--version", "v1.0"],
        )
        assert result.exit_code == 0, result.output
        assert (out_dir / "avdp_synth_v1.0.tar.gz").exists()
        assert (out_dir / "DATASET_CARD.md").exists()
        assert (out_dir / "avdp_synth_v1.0_SHA256SUMS.txt").exists()
        assert "Archive written" in result.output

    def test_qa_fail_aborts_without_force(self, tmp_path):
        """QA failure (failure_rate > 2%) causes exit 1 without --force."""
        data_dir = _make_empty_data_dir(tmp_path)
        out_dir = tmp_path / "releases"
        from unittest.mock import patch

        from synthbanshee.package.qa import DatasetStats, QAReport

        failing_report = QAReport(
            data_dir=str(data_dir),
            stats=DatasetStats(total_clips=10, failed_clips=5),
            passed=False,
            failure_rate=0.5,
        )
        with patch("synthbanshee.package.qa.run_qa", return_value=failing_report):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["package-dataset", str(data_dir), str(out_dir), "--version", "v1.0"],
            )
        assert result.exit_code == 1
        assert "Aborting" in result.output
        assert not (out_dir / "avdp_synth_v1.0.tar.gz").exists()

    def test_force_flag_packages_despite_qa_failure(self, tmp_path):
        """--force creates the archive even when QA fails."""
        data_dir = _make_empty_data_dir(tmp_path)
        out_dir = tmp_path / "releases"
        from unittest.mock import patch

        from synthbanshee.package.qa import DatasetStats, QAReport

        failing_report = QAReport(
            data_dir=str(data_dir),
            stats=DatasetStats(total_clips=10, failed_clips=5),
            passed=False,
            failure_rate=0.5,
        )
        with patch("synthbanshee.package.qa.run_qa", return_value=failing_report):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "package-dataset",
                    str(data_dir),
                    str(out_dir),
                    "--version",
                    "v1.0",
                    "--force",
                ],
            )
        assert result.exit_code == 0, result.output
        assert (out_dir / "avdp_synth_v1.0.tar.gz").exists()

    def test_sha256_sidecar_written(self, tmp_path):
        data_dir = _make_empty_data_dir(tmp_path)
        out_dir = tmp_path / "releases"
        runner = CliRunner()
        runner.invoke(
            cli,
            ["package-dataset", str(data_dir), str(out_dir), "--version", "v1.0"],
        )
        assert (out_dir / "avdp_synth_v1.0.tar.gz.sha256").exists()

    def test_invalid_version_rejected(self, tmp_path):
        """Version strings with path separators or spaces are rejected before QA runs."""
        data_dir = _make_empty_data_dir(tmp_path)
        out_dir = tmp_path / "releases"
        runner = CliRunner()
        for bad_version in ("../evil", "v1 0", "v1/0"):
            result = runner.invoke(
                cli,
                ["package-dataset", str(data_dir), str(out_dir), "--version", bad_version],
            )
            assert result.exit_code != 0, f"Expected failure for version {bad_version!r}"
            assert not (out_dir / f"avdp_synth_{bad_version}.tar.gz").exists()
