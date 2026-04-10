"""Integration test: Tier C hard-negative (confusor) pipeline end-to-end.

Verifies that:
- Tier C scenes run Stage 3b acoustic augmentation (same as Tier B)
- Output clips carry violence_typology=NEG with has_violence=False
- No violence categories appear in weak labels
- All four output files (.wav, .txt, .json, .jsonl) are written
- validate_clip passes with no errors
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from synthbanshee.cli import _run_generate_pipeline
from synthbanshee.package.validator import validate_clip

SCENES_DIR = Path(__file__).parent.parent.parent / "configs" / "scenes"
TIER_C_SCENE = SCENES_DIR / "she_proves_tier_c" / "sp_neg_c_0001.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mixed_scene(duration_s: float = 5.0):
    from synthbanshee.script.types import MixedScene

    sr = 16_000
    n = int(sr * duration_s)
    samples = (0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, duration_s, n))).astype(np.float32)
    turn_dur = duration_s * 0.4
    return MixedScene(
        samples=samples,
        sample_rate=sr,
        turn_onsets_s=[0.3, 0.3 + turn_dur],
        turn_offsets_s=[0.3 + turn_dur - 0.05, 0.3 + 2 * turn_dur - 0.05],
        duration_s=duration_s,
        speaker_ids=["AGG_M_30-45_001", "VIC_F_25-40_002"],
    )


def _make_dialogue_turns():
    from synthbanshee.script.types import DialogueTurn

    return [
        DialogueTurn(speaker_id="AGG_M_30-45_001", text="אני לא מוכן לקבל את זה!", intensity=3),
        DialogueTurn(speaker_id="VIC_F_25-40_002", text="בוא נדבר על זה בשקט.", intensity=1),
    ]


def _build_aug_audio(mixed, pad_s: float = 0.5) -> np.ndarray:
    sr = 16_000
    pad_n = int(pad_s * sr)
    total = mixed.samples.shape[0] + 2 * pad_n
    audio = np.zeros(total, dtype=np.float32)
    t = np.arange(mixed.samples.shape[0], dtype=np.float32) / sr
    audio[pad_n : total - pad_n] = 0.4 * np.sin(2 * np.pi * 440 * t)
    peak = float(np.max(np.abs(audio)))
    if peak > 0.0:
        audio = (audio * (10.0 ** (-1.0 / 20.0) / peak)).astype(np.float32)
    return audio


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestTierCPipeline:
    def _run(self, tmp_path: Path):
        mixed = _make_mixed_scene()
        turns = _make_dialogue_turns()
        aug_audio = _build_aug_audio(mixed)

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
            MockMixer.return_value.mix.return_value = (aug_audio, [], 17.0)

            return _run_generate_pipeline(
                TIER_C_SCENE,
                tmp_path / "out",
                tmp_path / "cache",
                tmp_path / "dirty",
                tmp_path / "scripts",
            )

    def test_pipeline_succeeds(self, tmp_path):
        """Tier C pipeline with mocked Stage 3b completes without errors."""
        wav, errors = self._run(tmp_path)
        assert wav is not None, errors
        assert errors == []

    def test_all_four_output_files_present(self, tmp_path):
        """WAV, TXT, JSON, and JSONL are written for Tier C clips."""
        wav, errors = self._run(tmp_path)
        assert wav is not None, errors
        stem, parent = wav.stem, wav.parent
        assert wav.exists()
        assert (parent / f"{stem}.txt").exists()
        assert (parent / f"{stem}.json").exists()
        assert (parent / f"{stem}.jsonl").exists()

    def test_clip_passes_validator(self, tmp_path):
        """Tier C output clip passes validate_clip with no errors and no JSONL warning."""
        wav, errors = self._run(tmp_path)
        assert wav is not None, errors
        result = validate_clip(wav)
        assert result.is_valid, f"Validation errors: {result.errors}"
        assert not any("Strong labels JSONL missing" in w for w in result.warnings)

    def test_metadata_typology_is_neg(self, tmp_path):
        """Tier C clip metadata carries violence_typology=NEG and tier=C."""
        wav, errors = self._run(tmp_path)
        assert wav is not None, errors
        meta = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["violence_typology"] == "NEG"
        assert meta["tier"] == "C"

    def test_weak_label_has_no_violence(self, tmp_path):
        """Tier C confusor clips have has_violence=False and empty violence_categories."""
        wav, errors = self._run(tmp_path)
        assert wav is not None, errors
        meta = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["weak_label"]["has_violence"] is False
        assert meta["weak_label"]["violence_categories"] == []

    def test_acoustic_scene_populated_from_stage_3b(self, tmp_path):
        """acoustic_scene block in metadata confirms Stage 3b ran for Tier C."""
        wav, errors = self._run(tmp_path)
        assert wav is not None, errors
        meta = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
        scene = meta["acoustic_scene"]
        assert scene["room_type"] == "apartment_kitchen"
        assert scene["device"] == "phone_on_table"
        assert scene["ir_source"] == "pyroomacoustics_ism"
        assert scene["snr_db_actual"] == 17.0

    def test_wav_is_spec_compliant(self, tmp_path):
        """Output WAV is 16 kHz mono."""
        wav, errors = self._run(tmp_path)
        assert wav is not None, errors
        data, sr = sf.read(str(wav))
        assert sr == 16_000
        assert data.ndim == 1
