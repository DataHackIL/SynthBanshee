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


class TestRunGeneratePipelineTierB:
    """Cover the Stage 3 acoustic augmentation branch in _run_generate_pipeline."""

    def _run_tier_b(
        self,
        tmp_path: Path,
        *,
        aug_events=None,
        snr_actual: float = 12.0,
        aug_side_effect=None,
    ):
        """Helper: run Tier B pipeline with mocked TTS + mocked augmentation stack.

        Returns (wav_path, errors).
        """
        from synthbanshee.augment.types import AugmentedEvent

        if aug_events is None:
            aug_events = []

        sr = 16_000
        pad = int(0.5 * sr)
        total = sr * 5  # 5 seconds
        aug_audio = np.zeros(total, dtype=np.float32)
        # Place a sine tone in the middle, leaving ≥ 0.5 s silence at each end
        t = np.arange(total - 2 * pad, dtype=np.float32) / sr
        aug_audio[pad : total - pad] = 0.5 * np.sin(2 * np.pi * 440 * t)
        # Peak-normalize to −1.0 dBFS (as the pipeline does before writing)
        peak = float(np.max(np.abs(aug_audio)))
        if peak > 0.0:
            aug_audio = (aug_audio * (10.0 ** (-1.0 / 20.0) / peak)).astype(np.float32)

        turns = _make_dialogue_turns(n=1, intensity=3)
        mixed = _make_mixed_scene(n_turns=1)

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
