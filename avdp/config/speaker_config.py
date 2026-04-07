"""Pydantic models for speaker persona configuration."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from avdp.config.taxonomy import speaker_role_codes

_SPEAKER_ID_RE = re.compile(r"^[A-Z]{2,4}_[MF]_\d{1,3}-\d{1,3}_\d{3}$")


class ProsodyBaseline(BaseModel):
    rate: float = Field(gt=0, le=3.0, default=1.0)
    pitch_hz: float = Field(gt=0, default=120.0)
    volume_db: float = Field(ge=-20.0, le=20.0, default=0.0)


class StyleEntry(BaseModel):
    style: str
    rate_multiplier: float = Field(gt=0, le=3.0, default=1.0)
    pitch_delta_st: float = Field(ge=-12.0, le=12.0, default=0.0)
    volume_delta_db: float = Field(ge=-20.0, le=20.0, default=0.0)


class DisfluencyProfile(BaseModel):
    filled_pause_prob: float = Field(ge=0.0, le=1.0, default=0.05)
    false_start_prob: float = Field(ge=0.0, le=1.0, default=0.02)
    truncation_prob: float = Field(ge=0.0, le=1.0, default=0.01)


class SpeakerConfig(BaseModel):
    speaker_id: str
    role: str
    gender: Literal["male", "female"]
    age_range: str
    context: Literal["she_proves", "elephant_in_the_room", "both"]
    tts_voice_id: str
    tts_provider: Literal["azure", "google"] = "azure"
    prosody_baseline: ProsodyBaseline = Field(default_factory=ProsodyBaseline)
    # Keys are intensity levels 1–5 as ints (parsed from YAML int keys)
    style_map: dict[int, StyleEntry] = Field(default_factory=dict)
    disfluency: DisfluencyProfile = Field(default_factory=DisfluencyProfile)
    split: Literal["train", "val", "test_synth"] = "train"

    @field_validator("speaker_id")
    @classmethod
    def valid_speaker_id_format(cls, v: str) -> str:
        if not _SPEAKER_ID_RE.match(v):
            raise ValueError(f"speaker_id {v!r} does not match pattern ROLE_G_AGE-AGE_NNN")
        return v

    @field_validator("role")
    @classmethod
    def valid_role(cls, v: str) -> str:
        valid = speaker_role_codes()
        if v not in valid:
            raise ValueError(f"role {v!r} not in taxonomy. Valid: {sorted(valid)}")
        return v

    @field_validator("style_map", mode="before")
    @classmethod
    def coerce_style_map_keys(cls, v: object) -> object:
        # YAML loads integer keys as ints; ensure they're ints
        if isinstance(v, dict):
            return {int(k): val for k, val in v.items()}
        return v

    @model_validator(mode="after")
    def style_map_intensity_range(self) -> SpeakerConfig:
        for k in self.style_map:
            if k not in range(1, 6):
                raise ValueError(f"style_map key {k} out of valid intensity range 1–5")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> SpeakerConfig:
        with Path(path).open("r", encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))

    def style_for_intensity(self, intensity: int) -> StyleEntry:
        """Return the closest style entry for the given intensity level."""
        if intensity in self.style_map:
            return self.style_map[intensity]
        # Fall back to nearest available intensity
        available = sorted(self.style_map.keys())
        if not available:
            return StyleEntry(style="General")
        nearest = min(available, key=lambda k: abs(k - intensity))
        return self.style_map[nearest]


# Convenience alias used in scene configs
SpeakerRef = Annotated[
    dict,
    Field(description="Inline speaker reference within a scene config"),
]
