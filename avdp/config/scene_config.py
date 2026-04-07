"""Pydantic models for scene configuration (the primary pipeline input)."""

from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from avdp.config.acoustic_config import AcousticSceneConfig
from avdp.config.taxonomy import (
    scene_phase_values,
    speaker_role_codes,
    violence_typology_codes,
)

_VALID_PROJECTS = {"she_proves", "elephant_in_the_room"}


class SpeakerRef(BaseModel):
    """Inline speaker reference within a scene config."""

    speaker_id: str
    role: str

    @field_validator("role")
    @classmethod
    def valid_role(cls, v: str) -> str:
        valid = speaker_role_codes()
        if v not in valid:
            raise ValueError(f"role {v!r} not in taxonomy. Valid: {sorted(valid)}")
        return v


class ProsodyBounds(BaseModel):
    rate_range: Annotated[list[float], Field(min_length=2, max_length=2)] = Field(
        default_factory=lambda: [0.9, 1.1]
    )
    pitch_shift_range: Annotated[list[float], Field(min_length=2, max_length=2)] = Field(
        default_factory=lambda: [-2.0, 2.0]
    )
    volume_range: Annotated[list[float], Field(min_length=2, max_length=2)] = Field(
        default_factory=lambda: [-3.0, 3.0]
    )

    @model_validator(mode="after")
    def ranges_ordered(self) -> Self:
        for name, rng in [
            ("rate_range", self.rate_range),
            ("pitch_shift_range", self.pitch_shift_range),
            ("volume_range", self.volume_range),
        ]:
            if rng[0] > rng[1]:
                raise ValueError(f"{name} min {rng[0]} > max {rng[1]}")
        return self


class SheProvesConfig(BaseModel):
    incident_window_start_fraction: float = Field(ge=0.0, le=1.0)
    pre_incident_phases: list[str] = Field(default_factory=list)
    incident_phases: list[str] = Field(default_factory=list)
    post_incident_phases: list[str] = Field(default_factory=list)

    @field_validator("pre_incident_phases", "incident_phases", "post_incident_phases")
    @classmethod
    def valid_phases(cls, v: list[str]) -> list[str]:
        valid = set(scene_phase_values())
        bad = [p for p in v if p not in valid]
        if bad:
            raise ValueError(f"Invalid scene phases: {bad}. Valid: {sorted(valid)}")
        return v


class ElephantConfig(BaseModel):
    scene_type: str
    alert_triggered: bool = False
    alert_at_phase: str | None = None
    alert_onset_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    attack_type: str | None = None


class SceneConfig(BaseModel):
    scene_id: str
    project: str
    language: str = "he"
    violence_typology: str
    tier: Literal["A", "B", "C"]
    random_seed: int = 0
    speakers: list[SpeakerRef] = Field(min_length=1)
    script_template: str
    script_slots: dict = Field(default_factory=dict)
    intensity_arc: list[int] = Field(min_length=1)
    she_proves: SheProvesConfig | None = None
    elephant: ElephantConfig | None = None
    target_duration_minutes: float = Field(gt=0)
    min_pre_incident_seconds: float | None = None
    min_pre_alert_seconds: float | None = None
    acoustic_scene: AcousticSceneConfig | None = None
    prosody: dict[str, ProsodyBounds] = Field(default_factory=dict)
    output_dir: str = "data/he"

    @field_validator("project")
    @classmethod
    def valid_project(cls, v: str) -> str:
        if v not in _VALID_PROJECTS:
            raise ValueError(f"project {v!r} not in {sorted(_VALID_PROJECTS)}")
        return v

    @field_validator("violence_typology")
    @classmethod
    def valid_typology(cls, v: str) -> str:
        valid = violence_typology_codes()
        if v not in valid:
            raise ValueError(f"violence_typology {v!r} not in taxonomy. Valid: {sorted(valid)}")
        return v

    @field_validator("intensity_arc")
    @classmethod
    def valid_intensity_arc(cls, v: list[int]) -> list[int]:
        bad = [x for x in v if x not in range(1, 6)]
        if bad:
            raise ValueError(f"intensity_arc values must be 1–5; got {bad}")
        return v

    @model_validator(mode="after")
    def project_specific_config_present(self) -> Self:
        if self.project == "she_proves" and self.she_proves is None:
            # Allow missing she_proves for minimal/test configs
            pass
        if self.project == "elephant_in_the_room" and self.elephant is None:
            pass
        # Tier B and C must have acoustic_scene
        if self.tier in {"B", "C"} and self.acoustic_scene is None:
            raise ValueError(f"Tier {self.tier} scene requires acoustic_scene config")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        with Path(path).open("r", encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))
