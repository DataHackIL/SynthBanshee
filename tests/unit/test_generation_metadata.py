"""Unit tests for GenerationMetadata (M11 — spec §4.11)."""

from __future__ import annotations

import json

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
        "tts_engine": "azure_he_IL",
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
        gm = GenerationMetadata(
            pipeline_version="v3.0",
            tts_backend="azure",
            voice_family="he-IL-AvriNeural",
        )
        assert gm.pipeline_version == "v3.0"
        assert gm.tts_backend == "azure"
        assert gm.voice_family == "he-IL-AvriNeural"
        assert gm.breathiness_applied is False
        assert gm.mix_mode_used == "SEQUENTIAL"
        assert gm.normalization_strategy == "per_turn_rms_v1"
        assert gm.speaker_state_serialized == {}

    def test_full_construction(self) -> None:
        states = {"spk_a": {"rate_offset": 0.05, "pitch_offset": 1.2, "volume_offset": 3.0}}
        gm = GenerationMetadata(
            pipeline_version="v3.0",
            tts_backend="google",
            voice_family="he-IL-Chirp3-HD-Aoede",
            text_normalization_version="1.0",
            prosody_controller_version="1.0",
            timing_controller_version="1.0",
            mix_mode_used="OVERLAP",
            normalization_strategy="per_turn_rms_v1",
            breathiness_applied=True,
            speaker_state_serialized=states,
        )
        assert gm.tts_backend == "google"
        assert gm.breathiness_applied is True
        assert gm.speaker_state_serialized["spk_a"]["pitch_offset"] == 1.2

    def test_serialization_roundtrip(self) -> None:
        gm = GenerationMetadata(
            pipeline_version="v3.0",
            tts_backend="azure",
            voice_family="he-IL-AvriNeural",
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
            "tts_backend": "azure",
            "voice_family": "he-IL-AvriNeural",
        }
        data = _minimal_clip_metadata(generation_metadata=gm_data)
        cm = ClipMetadata.model_validate(data)
        assert cm.generation_metadata is not None
        assert cm.generation_metadata.pipeline_version == "v3.0"
        assert cm.generation_metadata.tts_backend == "azure"

    def test_json_roundtrip_with_generation_metadata(self) -> None:
        gm = GenerationMetadata(
            pipeline_version="v3.0",
            tts_backend="azure",
            voice_family="he-IL-AvriNeural",
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
            tts_backend="azure",
            voice_family="he-IL-AvriNeural",
        )
        data = _minimal_clip_metadata(generation_metadata=gm.model_dump())
        cm = ClipMetadata.model_validate(data)
        output = json.loads(cm.model_dump_json())
        assert "generation_metadata" in output
        assert output["generation_metadata"]["pipeline_version"] == "v3.0"

    def test_json_output_omits_generation_metadata_when_none(self) -> None:
        data = _minimal_clip_metadata()
        cm = ClipMetadata.model_validate(data)
        output = json.loads(cm.model_dump_json())
        # When None, the key should still appear in the output (Pydantic default)
        # but the value should be null — backward compat is about parsing, not omission.
        assert output.get("generation_metadata") is None
