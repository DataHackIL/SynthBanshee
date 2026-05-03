"""Integration test: M12 VIC breathiness augmentation pipeline path.

Exercises the full _run_generate_pipeline() with enable_breathiness=True,
verifying that breathiness noise is applied to VIC turns at I3+ and the
resulting metadata records the augmentation.

No real TTS credentials are required — ScriptGenerator and TTSRenderer are
mocked; the breathiness augmentation itself runs for real.

Issue: #60
"""

from __future__ import annotations

import io
import json
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from synthbanshee.cli import _run_generate_pipeline

# Reuse the existing Tier A test scene which has AGG + VIC speakers and
# intensity_arc [1, 2, 3, 4, 5] — intensities 3–5 trigger breathiness
# on VIC turns.
SCENE_YAML = Path(__file__).parent.parent.parent / "configs" / "scenes" / "test_scene_001.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav_bytes(sample_rate: int = 24000, duration_s: float = 5.0) -> bytes:
    """Return valid WAV bytes (sine wave) for mocking TTS synthesis."""
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


def _make_mixed_scene(duration_s: float = 5.0):
    """Return a MixedScene with VIC + AGG turns for mocking render_scene."""
    from synthbanshee.script.types import MixedScene

    sr = 16_000
    n = int(sr * duration_s)
    samples = (0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, duration_s, n))).astype(np.float32)
    turn_dur = duration_s / 5  # 5 turns matching intensity_arc length
    onsets = [i * turn_dur for i in range(5)]
    offsets = [(i + 1) * turn_dur - 0.05 for i in range(5)]
    # Alternate speakers: AGG, VIC, AGG, VIC, AGG
    speaker_ids = [
        "AGG_M_30-45_001",
        "VIC_F_25-40_002",
        "AGG_M_30-45_001",
        "VIC_F_25-40_002",
        "AGG_M_30-45_001",
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
    """Return 5 DialogueTurns with escalating intensity (I1–I5)."""
    from synthbanshee.script.types import DialogueTurn

    return [
        DialogueTurn(speaker_id="AGG_M_30-45_001", text="מה עשית?", intensity=1),
        DialogueTurn(speaker_id="VIC_F_25-40_002", text="לא עשיתי כלום.", intensity=2),
        DialogueTurn(speaker_id="AGG_M_30-45_001", text="תשלם לי!", intensity=3),
        DialogueTurn(speaker_id="VIC_F_25-40_002", text="בבקשה, תרגע.", intensity=4),
        DialogueTurn(speaker_id="AGG_M_30-45_001", text="אני אזהיר אותך!", intensity=5),
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

    def test_pipeline_succeeds_with_breathiness(self, tmp_path: Path) -> None:
        """Pipeline completes without errors when breathiness is enabled."""
        wav, errors = _run_pipeline(tmp_path, enable_breathiness=True)
        assert wav is not None, errors
        assert errors == []

    def test_output_differs_from_baseline(self, tmp_path: Path) -> None:
        """Output WAV with breathiness differs from a baseline run without it."""
        wav_on, errors_on = _run_pipeline(tmp_path / "on", enable_breathiness=True)
        wav_off, errors_off = _run_pipeline(tmp_path / "off", enable_breathiness=False)
        assert wav_on is not None, errors_on
        assert wav_off is not None, errors_off

        audio_on, sr_on = sf.read(str(wav_on))
        audio_off, sr_off = sf.read(str(wav_off))
        assert sr_on == sr_off == 16_000

        # The two arrays should differ due to breathiness noise injection.
        # They may have different lengths due to padding, so compare overlapping
        # region.
        min_len = min(len(audio_on), len(audio_off))
        assert not np.array_equal(audio_on[:min_len], audio_off[:min_len]), (
            "Breathiness should change the audio samples"
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

    def test_wav_is_spec_compliant(self, tmp_path: Path) -> None:
        """Output WAV is 16 kHz mono with non-zero audio."""
        wav, errors = _run_pipeline(tmp_path, enable_breathiness=True)
        assert wav is not None, errors
        data, sr = sf.read(str(wav))
        assert sr == 16_000
        assert data.ndim == 1  # mono
        assert float(np.max(np.abs(data))) > 0.0
