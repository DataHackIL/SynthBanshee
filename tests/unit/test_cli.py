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
