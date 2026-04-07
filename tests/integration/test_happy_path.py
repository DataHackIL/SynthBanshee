"""Integration test: end-to-end happy path for Phase 0.6.

Wires all stages together:
  config → TTS render (mocked) → preprocessing → label generation → output files
  → validate_clip

The Azure TTS call is mocked with synthetic audio so no credentials are needed.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

from synthbanshee.augment.preprocessing import preprocess
from synthbanshee.config.scene_config import SceneConfig
from synthbanshee.config.speaker_config import SpeakerConfig
from synthbanshee.labels.generator import LabelGenerator, ScriptEvent
from synthbanshee.labels.schema import PreprocessingApplied, SpeakerInfo
from synthbanshee.package.validator import validate_clip
from synthbanshee.tts.azure_provider import AzureProvider
from synthbanshee.tts.renderer import TTSRenderer

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "configs" / "examples"
SCENES_DIR = Path(__file__).parent.parent.parent / "configs" / "scenes"


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
        # Pure sine wave so normalization has something to work with
        t = np.linspace(0, duration_s, n, endpoint=False)
        samples = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


def _mock_azure_factory(key, region):
    synth = MagicMock()
    mock_result = MagicMock()
    mock_result.audio_data = _make_wav_bytes()
    del mock_result.reason
    synth.speak_ssml_async.return_value.get.return_value = mock_result
    return synth


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestHappyPath:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp_path = tmp_path
        self.provider = AzureProvider(sdk_factory=_mock_azure_factory)
        self.renderer = TTSRenderer(
            provider=self.provider,
            cache_dir=tmp_path / "cache",
        )
        self.label_gen = LabelGenerator()

    def _run_pipeline(self, scene_id: str = "sp_it_a_0001_00") -> Path:
        """Run the full pipeline and return the path to the output .wav file."""
        scene = SceneConfig.from_yaml(SCENES_DIR / "test_scene_001.yaml")
        speaker = SpeakerConfig.from_yaml(EXAMPLES_DIR / "speaker_AGG_M_30-45_001.yaml")

        # Stage 1: TTS render
        raw_wav = self.tmp_path / "raw.wav"
        self.renderer.render_utterance_to_file(
            "hello", speaker, raw_wav, intensity=scene.intensity_arc[0]
        )

        # Stage 2: Preprocessing
        speaker_dir = self.tmp_path / "data" / "he" / speaker.speaker_id.lower()
        clip_wav = speaker_dir / f"{scene_id}.wav"
        clip_txt = speaker_dir / f"{scene_id}.txt"
        clip_json = speaker_dir / f"{scene_id}.json"

        result = preprocess(raw_wav, clip_wav, dirty_dir=self.tmp_path / "dirty")

        # Stage 3: Transcript
        clip_txt.parent.mkdir(parents=True, exist_ok=True)
        clip_txt.write_text(
            f"[CLIP_ID: {scene_id}]\n"
            f"[SPEAKER: {speaker.speaker_id} | ROLE: AGG | ONSET: 0.5 | OFFSET: 4.5]\n"
            "hello\n",
            encoding="utf-8",
        )

        # Stage 4: Labels
        events = [
            ScriptEvent(
                tier1_category="NONE",
                tier2_subtype="NONE_AMBIENT",
                onset=0.5,
                offset=max(result.duration_seconds - 0.5, 1.0),
                intensity=1,
                speaker_id=speaker.speaker_id,
                speaker_role="AGG",
            )
        ]
        event_labels = self.label_gen.generate_event_labels(scene_id, events)

        speakers_meta = [
            SpeakerInfo(
                speaker_id=speaker.speaker_id,
                role=speaker.role,
                gender=speaker.gender,
                age_range=speaker.age_range,
                tts_voice_id=speaker.tts_voice_id,
            )
        ]
        preprocessing_meta = PreprocessingApplied(
            resampled_to_16k=True,
            downmixed_to_mono=True,
            spectral_filtered=True,
            denoised=True,
            normalized_dbfs=-1.0,
            silence_padded=True,
        )
        metadata = self.label_gen.generate_clip_metadata(
            clip_id=scene_id,
            project=scene.project,
            violence_typology=scene.violence_typology,
            tier=scene.tier,
            duration_seconds=result.duration_seconds,
            events=event_labels,
            speakers=speakers_meta,
            scene_config_path=str(SCENES_DIR / "test_scene_001.yaml"),
            random_seed=scene.random_seed,
            preprocessing=preprocessing_meta,
            dirty_file_path=str(result.dirty_path) if result.dirty_path else None,
            transcript_path=str(clip_txt),
        )
        self.label_gen.write_clip_metadata_json(metadata, clip_json)

        return clip_wav

    def test_all_three_files_produced(self):
        clip_wav = self._run_pipeline()
        stem = clip_wav.stem
        parent = clip_wav.parent
        assert clip_wav.exists(), "WAV file missing"
        assert (parent / f"{stem}.txt").exists(), "Transcript missing"
        assert (parent / f"{stem}.json").exists(), "Metadata JSON missing"

    def test_clip_passes_validator(self):
        clip_wav = self._run_pipeline()
        result = validate_clip(clip_wav)
        assert result.is_valid, f"Validation errors: {result.errors}"

    def test_metadata_is_synthetic(self):
        import json

        clip_wav = self._run_pipeline()
        json_path = clip_wav.with_suffix(".json")
        data = json.loads(json_path.read_text())
        assert data["is_synthetic"] is True

    def test_wav_spec_compliant(self):
        clip_wav = self._run_pipeline()
        data, sr = sf.read(str(clip_wav))
        assert sr == 16000
        assert data.ndim == 1  # mono

    def test_dirty_file_retained(self):
        self._run_pipeline()
        dirty_dir = self.tmp_path / "dirty"
        assert dirty_dir.exists()
        dirty_files = list(dirty_dir.iterdir())
        assert len(dirty_files) > 0

    def test_label_jsonl_round_trips(self):
        scene_id = "sp_it_a_0001_00"
        self._run_pipeline(scene_id)

        events = [
            ScriptEvent(
                tier1_category="VERB",
                tier2_subtype="VERB_SHOUT",
                onset=1.0,
                offset=3.0,
                intensity=3,
                speaker_role="AGG",
                emotional_state="anger",
            )
        ]
        labels = self.label_gen.generate_event_labels(scene_id, events)
        jsonl_path = self.tmp_path / "labels.jsonl"
        self.label_gen.write_strong_labels_jsonl(labels, jsonl_path)
        loaded = self.label_gen.read_strong_labels_jsonl(jsonl_path)
        assert loaded == labels
