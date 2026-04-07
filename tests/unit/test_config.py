"""Unit tests for config schema models (Phase 0.2)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from avdp.config.acoustic_config import AcousticSceneConfig, BackgroundEvent, RoomDimensionsRange
from avdp.config.scene_config import SceneConfig, SheProvesConfig
from avdp.config.speaker_config import SpeakerConfig

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "configs" / "examples"
SCENES_DIR = Path(__file__).parent.parent.parent / "configs" / "scenes"


# ---------------------------------------------------------------------------
# SceneConfig tests
# ---------------------------------------------------------------------------


class TestSceneConfig:
    def test_load_test_scene(self):
        cfg = SceneConfig.from_yaml(SCENES_DIR / "test_scene_001.yaml")
        assert cfg.scene_id == "SP_IT_A_0001"
        assert cfg.project == "she_proves"
        assert cfg.violence_typology == "IT"
        assert cfg.tier == "A"
        assert cfg.random_seed == 42

    def test_load_she_proves_example(self):
        cfg = SceneConfig.from_yaml(EXAMPLES_DIR / "scene_she_proves_IT_example.yaml")
        assert cfg.project == "she_proves"
        assert cfg.she_proves is not None
        assert 0 < cfg.she_proves.incident_window_start_fraction < 1

    def test_load_elephant_example(self):
        cfg = SceneConfig.from_yaml(EXAMPLES_DIR / "scene_elephant_SV_example.yaml")
        assert cfg.project == "elephant_in_the_room"
        assert cfg.elephant is not None
        assert cfg.acoustic_scene is not None

    def test_invalid_violence_typology(self):
        with pytest.raises(ValidationError, match="violence_typology"):
            SceneConfig(
                scene_id="X",
                project="she_proves",
                language="he",
                violence_typology="INVALID",
                tier="A",
                random_seed=0,
                speakers=[{"speaker_id": "AGG_M_30-45_001", "role": "AGG"}],
                script_template="x.j2",
                intensity_arc=[1],
                target_duration_minutes=3.0,
            )

    def test_invalid_project(self):
        with pytest.raises(ValidationError, match="project"):
            SceneConfig(
                scene_id="X",
                project="unknown_project",
                language="he",
                violence_typology="IT",
                tier="A",
                random_seed=0,
                speakers=[{"speaker_id": "AGG_M_30-45_001", "role": "AGG"}],
                script_template="x.j2",
                intensity_arc=[1],
                target_duration_minutes=3.0,
            )

    def test_invalid_tier(self):
        with pytest.raises(ValidationError, match="tier"):
            SceneConfig(
                scene_id="X",
                project="she_proves",
                language="he",
                violence_typology="IT",
                tier="D",
                random_seed=0,
                speakers=[{"speaker_id": "AGG_M_30-45_001", "role": "AGG"}],
                script_template="x.j2",
                intensity_arc=[1],
                target_duration_minutes=3.0,
            )

    def test_intensity_arc_out_of_range(self):
        with pytest.raises(ValidationError, match="intensity_arc"):
            SceneConfig(
                scene_id="X",
                project="she_proves",
                language="he",
                violence_typology="IT",
                tier="A",
                random_seed=0,
                speakers=[{"speaker_id": "AGG_M_30-45_001", "role": "AGG"}],
                script_template="x.j2",
                intensity_arc=[1, 3, 6],  # 6 is out of range
                target_duration_minutes=3.0,
            )

    def test_tier_b_requires_acoustic_scene(self):
        with pytest.raises(ValidationError):
            SceneConfig(
                scene_id="X",
                project="she_proves",
                language="he",
                violence_typology="IT",
                tier="B",
                random_seed=0,
                speakers=[{"speaker_id": "AGG_M_30-45_001", "role": "AGG"}],
                script_template="x.j2",
                intensity_arc=[1, 2],
                target_duration_minutes=3.0,
                # acoustic_scene omitted — should fail
            )

    def test_she_proves_invalid_phase(self):
        with pytest.raises(ValidationError, match="phases"):
            SheProvesConfig(
                incident_window_start_fraction=0.3,
                pre_incident_phases=["not_a_real_phase"],
            )

    def test_speakers_have_valid_roles(self):
        with pytest.raises(ValidationError, match="role"):
            SceneConfig(
                scene_id="X",
                project="she_proves",
                language="he",
                violence_typology="IT",
                tier="A",
                random_seed=0,
                speakers=[{"speaker_id": "AGG_M_30-45_001", "role": "BADROL"}],
                script_template="x.j2",
                intensity_arc=[1],
                target_duration_minutes=3.0,
            )


# ---------------------------------------------------------------------------
# SpeakerConfig tests
# ---------------------------------------------------------------------------


class TestSpeakerConfig:
    def test_load_aggressor_example(self):
        cfg = SpeakerConfig.from_yaml(EXAMPLES_DIR / "speaker_AGG_M_30-45_001.yaml")
        assert cfg.speaker_id == "AGG_M_30-45_001"
        assert cfg.role == "AGG"
        assert cfg.gender == "male"
        assert cfg.tts_voice_id == "he-IL-AvriNeural"
        assert cfg.split == "train"

    def test_style_map_populated(self):
        cfg = SpeakerConfig.from_yaml(EXAMPLES_DIR / "speaker_AGG_M_30-45_001.yaml")
        assert 1 in cfg.style_map
        assert 5 in cfg.style_map
        assert cfg.style_map[5].style == "angry"

    def test_style_for_intensity_exact(self):
        cfg = SpeakerConfig.from_yaml(EXAMPLES_DIR / "speaker_AGG_M_30-45_001.yaml")
        entry = cfg.style_for_intensity(3)
        assert entry.style == "angry"

    def test_style_for_intensity_fallback(self):
        cfg = SpeakerConfig.from_yaml(EXAMPLES_DIR / "speaker_AGG_M_30-45_001.yaml")
        # intensity 5 exists; intensity 6 should fall back to 5
        entry = cfg.style_for_intensity(5)
        assert entry is not None

    def test_invalid_speaker_id_format(self):
        with pytest.raises(ValidationError, match="speaker_id"):
            SpeakerConfig(
                speaker_id="bad_id",  # wrong format
                role="AGG",
                gender="male",
                age_range="30-45",
                context="she_proves",
                tts_voice_id="he-IL-AvriNeural",
            )

    def test_invalid_role(self):
        with pytest.raises(ValidationError, match="role"):
            SpeakerConfig(
                speaker_id="AGG_M_30-45_001",
                role="VILLAIN",  # not in taxonomy
                gender="male",
                age_range="30-45",
                context="she_proves",
                tts_voice_id="he-IL-AvriNeural",
            )

    def test_style_map_key_out_of_range(self):
        with pytest.raises(ValidationError):
            SpeakerConfig(
                speaker_id="AGG_M_30-45_001",
                role="AGG",
                gender="male",
                age_range="30-45",
                context="she_proves",
                tts_voice_id="he-IL-AvriNeural",
                style_map={6: {"style": "angry"}},  # 6 is out of range
            )

    def test_round_trip_serialization(self):
        cfg = SpeakerConfig.from_yaml(EXAMPLES_DIR / "speaker_AGG_M_30-45_001.yaml")
        data = cfg.model_dump()
        cfg2 = SpeakerConfig.model_validate(data)
        assert cfg2.speaker_id == cfg.speaker_id
        assert cfg2.style_map == cfg.style_map


# ---------------------------------------------------------------------------
# AcousticSceneConfig tests
# ---------------------------------------------------------------------------


class TestAcousticSceneConfig:
    def test_valid_config(self):
        cfg = AcousticSceneConfig(
            room_type="apartment_kitchen",
            device="phone_in_pocket",
            speaker_distance_meters=2.0,
            victim_distance_meters=1.5,
        )
        assert cfg.room_type == "apartment_kitchen"

    def test_invalid_room_type(self):
        with pytest.raises(ValidationError, match="room_type"):
            AcousticSceneConfig(
                room_type="moon_base",
                device="phone_in_pocket",
                speaker_distance_meters=2.0,
                victim_distance_meters=1.5,
            )

    def test_invalid_device(self):
        with pytest.raises(ValidationError, match="device"):
            AcousticSceneConfig(
                room_type="clinic_office",
                device="walkie_talkie",
                speaker_distance_meters=1.0,
                victim_distance_meters=1.0,
            )

    def test_room_dimensions_range_valid(self):
        rd = RoomDimensionsRange(min=[3.0, 2.5, 2.4], max=[5.0, 4.0, 2.8])
        assert rd.min[0] < rd.max[0]

    def test_room_dimensions_range_inverted(self):
        with pytest.raises(ValidationError, match="min dimension"):
            RoomDimensionsRange(min=[6.0, 2.5, 2.4], max=[5.0, 4.0, 2.8])

    def test_background_event_requires_onset(self):
        with pytest.raises(ValidationError):
            BackgroundEvent(type="tv_ambient", level_db=-30)

    def test_background_event_onset_seconds(self):
        evt = BackgroundEvent(type="tv_ambient", onset_seconds=0.0, level_db=-30, loop=True)
        assert evt.onset_seconds == 0.0
