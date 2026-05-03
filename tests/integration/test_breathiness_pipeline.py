"""Integration test: M12 VIC breathiness augmentation pipeline path.

Exercises the full _run_generate_pipeline() with enable_breathiness=True,
verifying that breathiness noise is applied to VIC turns at I3+ and the
resulting metadata records the augmentation.

No real TTS credentials are required — ScriptGenerator and TTSRenderer are
mocked; the breathiness augmentation itself runs for real.

Issue: #60
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from synthbanshee.cli import _run_generate_pipeline
from synthbanshee.config.scene_config import SceneConfig
from synthbanshee.package.validator import validate_clip

# Reuse the existing Tier A test scene which has AGG + VIC speakers and
# intensity_arc [1, 2, 3, 4, 5].
SCENE_YAML = Path(__file__).parent.parent.parent / "configs" / "scenes" / "test_scene_001.yaml"

# Per-turn frequencies so each segment has distinct spectral content / RMS,
# exercising the per-segment noise amplitude calculation in add_breathiness.
_TURN_FREQS_HZ = [440, 330, 520, 280, 600]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mixed_scene(duration_s: float = 5.0):
    """Return a MixedScene with AGG and VIC turns for mocking render_scene.

    Turn layout (VIC at I3, I4, I5 to exercise full breathiness range):
      0: AGG I1  |  1: AGG I2  |  2: VIC I3  |  3: VIC I4  |  4: VIC I5
    """
    from synthbanshee.script.types import MixedScene

    sr = 16_000
    n = int(sr * duration_s)
    turn_dur = duration_s / 5
    turn_n = int(sr * turn_dur)

    # Build audio with distinct frequency per turn segment.
    samples = np.zeros(n, dtype=np.float32)
    for i, freq in enumerate(_TURN_FREQS_HZ):
        start = i * turn_n
        end = min((i + 1) * turn_n, n)
        t = np.arange(end - start, dtype=np.float32) / sr
        samples[start:end] = 0.3 * np.sin(2 * np.pi * freq * t)

    onsets = [i * turn_dur for i in range(5)]
    offsets = [(i + 1) * turn_dur - 0.05 for i in range(5)]
    speaker_ids = [
        "AGG_M_30-45_001",
        "AGG_M_30-45_001",
        "VIC_F_25-40_002",
        "VIC_F_25-40_002",
        "VIC_F_25-40_002",
    ]
    return MixedScene(
        samples=samples,
        sample_rate=sr,
        turn_onsets_s=onsets,
        turn_offsets_s=offsets,
        duration_s=duration_s,
        speaker_ids=speaker_ids,
    )


def _make_dialogue_turns():
    """Return 5 DialogueTurns: AGG at I1–I2, VIC at I3–I5.

    This ensures three VIC turns receive breathiness (I3, I4, I5), covering
    the full non-zero range of _VIC_BREATHINESS_TARGETS.
    """
    from synthbanshee.script.types import DialogueTurn

    return [
        DialogueTurn(speaker_id="AGG_M_30-45_001", text="מה עשית?", intensity=1),
        DialogueTurn(speaker_id="AGG_M_30-45_001", text="תשלם לי!", intensity=2),
        DialogueTurn(speaker_id="VIC_F_25-40_002", text="לא עשיתי כלום.", intensity=3),
        DialogueTurn(speaker_id="VIC_F_25-40_002", text="בבקשה, תרגע.", intensity=4),
        DialogueTurn(speaker_id="VIC_F_25-40_002", text="עזוב אותי.", intensity=5),
    ]


def _run_pipeline(tmp_path: Path, *, enable_breathiness: bool):
    """Run _run_generate_pipeline with breathiness on or off."""
    mixed = _make_mixed_scene()
    turns = _make_dialogue_turns()

    with (
        patch("synthbanshee.script.generator.ScriptGenerator") as MockGen,
        patch("synthbanshee.tts.renderer.TTSRenderer") as MockRenderer,
    ):
        MockGen.return_value.generate.return_value = turns
        MockRenderer.return_value.render_scene.return_value = mixed

        return _run_generate_pipeline(
            SCENE_YAML,
            tmp_path / "out",
            tmp_path / "cache",
            tmp_path / "dirty",
            tmp_path / "scripts",
            enable_breathiness=enable_breathiness,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBreathinessPipeline:
    """M12 breathiness augmentation integration tests."""

    def test_fixture_has_expected_shape(self) -> None:
        """Guard: the shared scene YAML still has the shape this test expects."""
        scene = SceneConfig.from_yaml(SCENE_YAML)
        roles = {s.role for s in scene.speakers}
        assert "VIC" in roles, "Fixture must include a VIC speaker"
        assert "AGG" in roles, "Fixture must include an AGG speaker"
        assert any(i >= 3 for i in scene.intensity_arc), "Fixture intensity_arc must include I3+"

    def test_pipeline_succeeds_with_breathiness(self, tmp_path: Path) -> None:
        """Pipeline completes without errors when breathiness is enabled."""
        wav, errors = _run_pipeline(tmp_path, enable_breathiness=True)
        assert wav is not None, errors
        assert errors == []

    def test_vic_segments_differ_from_baseline(self, tmp_path: Path) -> None:
        """VIC turn segments have measurably higher RMS with breathiness enabled."""
        wav_on, errors_on = _run_pipeline(tmp_path / "on", enable_breathiness=True)
        wav_off, errors_off = _run_pipeline(tmp_path / "off", enable_breathiness=False)
        assert wav_on is not None, errors_on
        assert wav_off is not None, errors_off

        audio_on, sr_on = sf.read(str(wav_on))
        audio_off, sr_off = sf.read(str(wav_off))
        assert sr_on == sr_off == 16_000

        # VIC turns are indices 2, 3, 4 in our 5-turn layout.
        # preprocess() adds silence padding (≥ 0.5 s) at head, so shift onsets.
        pad_samples = int(0.5 * sr_on)
        turn_dur_samples = len(audio_off) // 5  # approximate

        vic_rms_diffs: list[float] = []
        for turn_idx in (2, 3, 4):
            start = pad_samples + turn_idx * turn_dur_samples
            end = min(start + turn_dur_samples, len(audio_on), len(audio_off))
            if end <= start:
                continue
            seg_on = audio_on[start:end]
            seg_off = audio_off[start:end]
            rms_on = float(np.sqrt(np.mean(seg_on**2)))
            rms_off = float(np.sqrt(np.mean(seg_off**2)))
            if rms_off > 1e-10:
                vic_rms_diffs.append(abs(rms_on - rms_off) / rms_off)

        assert len(vic_rms_diffs) > 0, "Should have measured at least one VIC segment"
        assert any(d > 0.001 for d in vic_rms_diffs), (
            f"Breathiness should measurably change VIC segment RMS, got diffs: {vic_rms_diffs}"
        )

    def test_breathiness_applied_in_metadata(self, tmp_path: Path) -> None:
        """Clip metadata marks breathiness_applied=True."""
        wav, errors = _run_pipeline(tmp_path, enable_breathiness=True)
        assert wav is not None, errors
        meta = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
        gen = meta.get("generation_metadata", {})
        assert gen.get("breathiness_applied") is True

    def test_baseline_has_no_breathiness_flag(self, tmp_path: Path) -> None:
        """Without breathiness, the flag remains False."""
        wav, errors = _run_pipeline(tmp_path, enable_breathiness=False)
        assert wav is not None, errors
        meta = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
        gen = meta.get("generation_metadata", {})
        assert gen.get("breathiness_applied") is False

    def test_clip_passes_validator(self, tmp_path: Path) -> None:
        """Output clip with breathiness passes the full project validator."""
        wav, errors = _run_pipeline(tmp_path, enable_breathiness=True)
        assert wav is not None, errors
        result = validate_clip(wav)
        assert result.is_valid, f"Validation errors: {result.errors}"
