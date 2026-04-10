"""Integration test: Tier B end-to-end pipeline (strong labels + acoustic augmentation).

Wires all four stages together for a Tier B scene config:
  SceneConfig → ScriptGenerator (mocked) → TTSRenderer (mocked)
  → preprocess() → RoomSimulator / DeviceProfiler / NoiseMixer (mocked)
  → strong-labels JSONL → ClipMetadata JSON → validate_clip

No real TTS or room-simulation credentials are required.
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
from synthbanshee.package.validator import validate_clip

SCENES_DIR = Path(__file__).parent.parent.parent / "configs" / "scenes"
TIER_B_SCENE = SCENES_DIR / "she_proves_tier_b" / "sp_sv_b_0001.yaml"


# ---------------------------------------------------------------------------
# Helpers
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


def _make_mixed_scene(duration_s: float = 5.0):
    """Return a MixedScene stub with two speaker turns for mocking render_scene."""
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
        DialogueTurn(speaker_id="AGG_M_30-45_001", text="תשלם לי!", intensity=4),
        DialogueTurn(speaker_id="VIC_F_25-40_002", text="בבקשה, תרגע.", intensity=2),
    ]


def _build_aug_audio(mixed, pad_s: float = 0.5) -> np.ndarray:
    """Build a valid augmented-audio array: sine in speech region, zeros in pads."""
    sr = 16_000
    pad_n = int(pad_s * sr)
    total = mixed.samples.shape[0] + 2 * pad_n
    audio = np.zeros(total, dtype=np.float32)
    t = np.arange(mixed.samples.shape[0], dtype=np.float32) / sr
    audio[pad_n : total - pad_n] = 0.5 * np.sin(2 * np.pi * 440 * t)
    peak = float(np.max(np.abs(audio)))
    if peak > 0.0:
        audio = (audio * (10.0 ** (-1.0 / 20.0) / peak)).astype(np.float32)
    return audio


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestTierBPipeline:
    def _run(self, tmp_path: Path, *, acou_events=None):
        """Run the full Tier B pipeline and return (wav_path, errors)."""
        from synthbanshee.augment.types import AugmentedEvent

        mixed = _make_mixed_scene()
        turns = _make_dialogue_turns()
        aug_audio = _build_aug_audio(mixed)

        if acou_events is None:
            acou_events = []
        aug_event_objs = [
            AugmentedEvent(
                type=ev["type"],
                onset_s=ev["onset_s"],
                offset_s=ev["offset_s"],
                level_db=ev["level_db"],
            )
            for ev in acou_events
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
            MockMixer.return_value.mix.return_value = (aug_audio, aug_event_objs, 18.5)

            return _run_generate_pipeline(
                TIER_B_SCENE,
                tmp_path / "out",
                tmp_path / "cache",
                tmp_path / "dirty",
                tmp_path / "scripts",
            )

    def test_pipeline_succeeds(self, tmp_path):
        """Tier B pipeline with mocked Stage 3 completes without errors."""
        wav, errors = self._run(tmp_path)
        assert wav is not None, errors
        assert errors == []

    def test_all_four_output_files_present(self, tmp_path):
        """WAV, TXT, JSON, and JSONL are all written alongside each other."""
        wav, errors = self._run(tmp_path)
        assert wav is not None, errors
        stem = wav.stem
        parent = wav.parent
        assert wav.exists(), "WAV missing"
        assert (parent / f"{stem}.txt").exists(), "Transcript missing"
        assert (parent / f"{stem}.json").exists(), "Metadata JSON missing"
        assert (parent / f"{stem}.jsonl").exists(), "Strong labels JSONL missing"

    def test_clip_passes_validator(self, tmp_path):
        """The output clip (with JSONL present) passes validate_clip with no errors."""
        wav, errors = self._run(tmp_path)
        assert wav is not None, errors
        result = validate_clip(wav)
        assert result.is_valid, f"Validation errors: {result.errors}"
        # JSONL present → no 'Strong labels JSONL missing' warning
        jsonl_warnings = [w for w in result.warnings if "Strong labels JSONL missing" in w]
        assert jsonl_warnings == [], f"Unexpected JSONL warning: {jsonl_warnings}"

    def test_metadata_is_synthetic_and_tier_b(self, tmp_path):
        """Clip metadata marks is_synthetic=True and tier='B'."""
        wav, errors = self._run(tmp_path)
        assert wav is not None, errors
        meta = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["is_synthetic"] is True
        assert meta["tier"] == "B"

    def test_acoustic_scene_in_metadata(self, tmp_path):
        """acoustic_scene block contains room_type, device, ir_source, snr_db_actual."""
        wav, errors = self._run(tmp_path)
        assert wav is not None, errors
        meta = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
        scene = meta["acoustic_scene"]
        assert scene["room_type"] == "living_room"
        assert scene["device"] == "phone_on_table"
        assert scene["ir_source"] == "pyroomacoustics_ism"
        assert scene["snr_db_actual"] == 18.5

    def test_acou_events_appear_in_weak_label(self, tmp_path):
        """ACOU_* events from Stage 3 contribute 'ACOU' to weak_label.violence_categories."""
        acou_events = [
            {"type": "ACOU_THROW", "onset_s": 3.0, "offset_s": 3.4, "level_db": -20.0},
        ]
        wav, errors = self._run(tmp_path, acou_events=acou_events)
        assert wav is not None, errors
        meta = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
        assert "ACOU" in meta["weak_label"]["violence_categories"]

    def test_strong_labels_jsonl_is_valid_json_lines(self, tmp_path):
        """Every line in the JSONL is parseable JSON with required event fields."""
        wav, errors = self._run(tmp_path)
        assert wav is not None, errors
        jsonl_path = wav.with_suffix(".jsonl")
        assert jsonl_path.exists()
        lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) > 0, "JSONL must not be empty"
        for line in lines:
            record = json.loads(line)
            assert "clip_id" in record
            assert "onset" in record
            assert "offset" in record
            assert "tier1_category" in record

    def test_wav_is_spec_compliant(self, tmp_path):
        """Output WAV is 16 kHz, mono, with non-zero peak-normalized audio."""
        wav, errors = self._run(tmp_path)
        assert wav is not None, errors
        data, sr = sf.read(str(wav))
        assert sr == 16_000
        assert data.ndim == 1  # mono
        assert float(np.max(np.abs(data))) > 0.0
