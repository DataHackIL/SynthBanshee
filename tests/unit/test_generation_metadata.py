"""Unit tests for GenerationMetadata (M11 — spec §4.11)."""

from __future__ import annotations

import json
from collections import Counter

from synthbanshee.labels.schema import (
    ClipMetadata,
    GenerationMetadata,
)


def _minimal_clip_metadata(**overrides) -> dict:
    """Return a minimal valid ClipMetadata dict for testing."""
    base = {
        "clip_id": "test_clip_00",
        "project": "she_proves",
        "language": "he",
        "violence_typology": "NEU",
        "tier": "A",
        "duration_seconds": 5.0,
        "sample_rate": 16000,
        "channels": 1,
        "generation_date": "2026-05-01",
        "generator_version": "0.1.0",
        "is_synthetic": True,
        "weak_label": {
            "has_violence": False,
            "violence_categories": [],
            "max_intensity": 1,
            "violence_typology": "NEU",
        },
    }
    base.update(overrides)
    return base


class TestGenerationMetadata:
    """Tests for the GenerationMetadata Pydantic model."""

    def test_minimal_construction(self) -> None:
        gm = GenerationMetadata(pipeline_version="v3.0")
        assert gm.pipeline_version == "v3.0"
        assert gm.tts_backend == {}
        assert gm.voice_family == {}
        assert gm.text_normalization_version is None
        assert gm.prosody_controller_version is None
        assert gm.timing_controller_version is None
        assert gm.breathiness_applied is False
        assert gm.mix_mode_used == "sequential"
        assert gm.normalization_strategy == "per_turn_rms_v1"
        assert gm.speaker_state_serialized == {}

    def test_full_construction(self) -> None:
        states = {
            "spk_a": {
                "rate_offset": 1.05,
                "pitch_offset_st": 1.2,
                "volume_offset_db": 3.0,
                "breathiness_level": 0.0,
            }
        }
        gm = GenerationMetadata(
            pipeline_version="v3.0",
            tts_backend={"spk_a": "google", "spk_b": "azure"},
            voice_family={"spk_a": "he-IL-Chirp3-HD-Aoede", "spk_b": "he-IL-AvriNeural"},
            text_normalization_version="1.0",
            prosody_controller_version="1.0",
            timing_controller_version="1.0",
            mix_mode_used="overlap",
            normalization_strategy="per_turn_rms_v1",
            breathiness_applied=True,
            speaker_state_serialized=states,
        )
        assert gm.tts_backend == {"spk_a": "google", "spk_b": "azure"}
        assert gm.breathiness_applied is True
        assert gm.speaker_state_serialized["spk_a"]["pitch_offset_st"] == 1.2

    def test_per_speaker_backend_and_voice(self) -> None:
        """Per-speaker maps capture mixed-provider scenes (M9a)."""
        gm = GenerationMetadata(
            pipeline_version="v3.0",
            tts_backend={"agg_m": "azure", "vic_f": "google"},
            voice_family={"agg_m": "he-IL-AvriNeural", "vic_f": "he-IL-Chirp3-HD-Aoede"},
        )
        assert gm.tts_backend["agg_m"] == "azure"
        assert gm.tts_backend["vic_f"] == "google"
        assert gm.voice_family["vic_f"] == "he-IL-Chirp3-HD-Aoede"

    def test_version_fields_default_to_none(self) -> None:
        """Version fields default to None (not empty string) to distinguish
        'not tracked' from 'tracked as empty'."""
        gm = GenerationMetadata(pipeline_version="v3.0")
        assert gm.text_normalization_version is None
        assert gm.prosody_controller_version is None
        assert gm.timing_controller_version is None

    def test_serialization_roundtrip(self) -> None:
        gm = GenerationMetadata(
            pipeline_version="v3.0",
            tts_backend={"spk_a": "azure"},
            voice_family={"spk_a": "he-IL-AvriNeural"},
            speaker_state_serialized={"spk_a": {"rate_offset": 0.1}},
        )
        raw = gm.model_dump_json()
        restored = GenerationMetadata.model_validate_json(raw)
        assert restored == gm


class TestClipMetadataWithGenerationMetadata:
    """Tests for ClipMetadata backward compatibility with generation_metadata."""

    def test_backward_compat_no_generation_metadata(self) -> None:
        """Existing V1 clips without generation_metadata must still validate."""
        data = _minimal_clip_metadata()
        cm = ClipMetadata.model_validate(data)
        assert cm.generation_metadata is None

    def test_with_generation_metadata(self) -> None:
        gm_data = {
            "pipeline_version": "v3.0",
            "tts_backend": {"spk_a": "azure"},
            "voice_family": {"spk_a": "he-IL-AvriNeural"},
        }
        data = _minimal_clip_metadata(generation_metadata=gm_data)
        cm = ClipMetadata.model_validate(data)
        assert cm.generation_metadata is not None
        assert cm.generation_metadata.pipeline_version == "v3.0"
        assert cm.generation_metadata.tts_backend == {"spk_a": "azure"}

    def test_json_roundtrip_with_generation_metadata(self) -> None:
        gm = GenerationMetadata(
            pipeline_version="v3.0",
            tts_backend={"spk_a": "azure"},
            voice_family={"spk_a": "he-IL-AvriNeural"},
            breathiness_applied=False,
            speaker_state_serialized={"spk_a": {"rate_offset": 0.05}},
        )
        data = _minimal_clip_metadata(generation_metadata=gm.model_dump())
        cm = ClipMetadata.model_validate(data)
        raw_json = cm.model_dump_json(indent=2)
        restored = ClipMetadata.model_validate_json(raw_json)
        assert restored.generation_metadata is not None
        assert restored.generation_metadata.pipeline_version == "v3.0"
        assert restored.generation_metadata.speaker_state_serialized == {
            "spk_a": {"rate_offset": 0.05}
        }

    def test_json_output_contains_generation_metadata_key(self) -> None:
        gm = GenerationMetadata(
            pipeline_version="v3.0",
            tts_backend={"spk_a": "azure"},
            voice_family={"spk_a": "he-IL-AvriNeural"},
        )
        data = _minimal_clip_metadata(generation_metadata=gm.model_dump())
        cm = ClipMetadata.model_validate(data)
        output = json.loads(cm.model_dump_json())
        assert "generation_metadata" in output
        assert output["generation_metadata"]["pipeline_version"] == "v3.0"

    def test_generation_metadata_null_when_not_provided(self) -> None:
        """When generation_metadata is not set, the key serializes as null."""
        data = _minimal_clip_metadata()
        cm = ClipMetadata.model_validate(data)
        output = json.loads(cm.model_dump_json())
        assert "generation_metadata" in output
        assert output["generation_metadata"] is None


class TestMixedSceneMixModes:
    """Tests for MixedScene.mix_modes populated by SceneMixer."""

    def test_mixer_populates_mix_modes(self) -> None:
        """SceneMixer.mix_sequential must populate mix_modes on MixedScene."""
        import io

        import numpy as np
        import soundfile as sf

        from synthbanshee.tts.mix_mode import MixMode
        from synthbanshee.tts.mixer import SceneMixer, Segment

        mono = np.zeros(1600, dtype=np.float32)  # 0.1s at 16kHz
        buf = io.BytesIO()
        sf.write(buf, mono, 16000, format="WAV", subtype="FLOAT")
        wav_bytes = buf.getvalue()

        mixer = SceneMixer()
        segments = [
            Segment(wav_bytes, 0.0, "spk_a", None, MixMode.SEQUENTIAL),
            Segment(wav_bytes, 0.05, "spk_b", None, MixMode.OVERLAP),
            Segment(wav_bytes, 0.0, "spk_a", None, MixMode.SEQUENTIAL),
        ]
        scene = mixer.mix_sequential(segments)
        assert scene.mix_modes == ["sequential", "overlap", "sequential"]

    def test_dominant_mix_mode_from_counter(self) -> None:
        """Counter.most_common gives deterministic dominant mode."""
        modes = ["sequential", "overlap", "sequential", "barge_in"]
        dominant = Counter(modes).most_common(1)[0][0]
        assert dominant == "sequential"

        modes2 = ["overlap", "overlap", "barge_in"]
        dominant2 = Counter(modes2).most_common(1)[0][0]
        assert dominant2 == "overlap"


class TestGenerateClipMetadataWithGenMeta:
    """Tests that LabelGenerator.generate_clip_metadata passes through generation_metadata."""

    def test_passthrough(self) -> None:
        from synthbanshee.labels.generator import LabelGenerator, ScriptEvent

        gen = LabelGenerator()
        events = [
            ScriptEvent(
                tier1_category="NONE",
                tier2_subtype="NONE_AMBIENT",
                onset=0.0,
                offset=1.0,
                intensity=1,
            ),
        ]
        labels = gen.generate_event_labels("test_00", events)
        gm = GenerationMetadata(
            pipeline_version="v3.0",
            tts_backend={"spk_a": "azure"},
            voice_family={"spk_a": "he-IL-AvriNeural"},
            mix_mode_used="overlap",
        )
        metadata = gen.generate_clip_metadata(
            clip_id="test_00",
            project="she_proves",
            violence_typology="NEU",
            tier="A",
            duration_seconds=5.0,
            events=labels,
            generation_metadata=gm,
        )
        assert metadata.generation_metadata is not None
        assert metadata.generation_metadata.pipeline_version == "v3.0"
        assert metadata.generation_metadata.mix_mode_used == "overlap"
        assert metadata.generation_metadata.tts_backend == {"spk_a": "azure"}

    def test_none_by_default(self) -> None:
        from synthbanshee.labels.generator import LabelGenerator, ScriptEvent

        gen = LabelGenerator()
        events = [
            ScriptEvent(
                tier1_category="NONE",
                tier2_subtype="NONE_AMBIENT",
                onset=0.0,
                offset=1.0,
                intensity=1,
            ),
        ]
        labels = gen.generate_event_labels("test_00", events)
        metadata = gen.generate_clip_metadata(
            clip_id="test_00",
            project="she_proves",
            violence_typology="NEU",
            tier="A",
            duration_seconds=5.0,
            events=labels,
        )
        assert metadata.generation_metadata is None
