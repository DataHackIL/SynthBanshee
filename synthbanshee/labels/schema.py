"""Pydantic models for AVDP label schema (spec.md §5).

These models define the structure of per-clip metadata JSON files
and per-event strong label JSONL records.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from synthbanshee.config.taxonomy import (
    emotional_state_values,
    intensity_levels,
    speaker_role_codes,
    tier1_category_codes,
    tier2_parent,
    tier2_subtype_codes,
    violence_typology_codes,
)

_QUALITY_FLAGS = frozenset(
    {
        "low_snr",
        "clipping",
        "short_silence_pad",
        "label_uncertainty",
        "iaa_disagreement",
        "synthetic_artifact",
    }
)

_ASCII_SAFE_RE = re.compile(r"^[\x00-\x7F\xa1-\xff]*$")


def _assert_ascii_safe(value: str, field_name: str) -> str:
    """Reject strings containing UTF-8 above U+00A1 (spec §2.5)."""
    for ch in value:
        if ord(ch) > 0x00A1:
            raise ValueError(
                f"{field_name} contains character U+{ord(ch):04X} above U+00A1; "
                "Hebrew text must not appear in metadata string fields"
            )
    return value


# ---------------------------------------------------------------------------
# Nested sub-models
# ---------------------------------------------------------------------------


class SpeakerInfo(BaseModel):
    speaker_id: str
    role: str
    gender: Literal["male", "female"]
    age_range: str
    tts_voice_id: str

    @field_validator("role")
    @classmethod
    def valid_role(cls, v: str) -> str:
        valid = speaker_role_codes()
        if v not in valid:
            raise ValueError(f"role {v!r} not in taxonomy")
        return v


class ClipAcousticScene(BaseModel):
    room_type: str | None = None
    device: str | None = None
    ir_source: str | None = None
    background_events: list[dict] = Field(default_factory=list)


class WeakLabel(BaseModel):
    has_violence: bool
    violence_categories: list[str] = Field(default_factory=list)
    max_intensity: int = Field(ge=1, le=5)
    violence_typology: str

    @field_validator("violence_typology")
    @classmethod
    def valid_typology(cls, v: str) -> str:
        valid = violence_typology_codes()
        if v not in valid:
            raise ValueError(f"violence_typology {v!r} not in taxonomy")
        return v

    @field_validator("violence_categories")
    @classmethod
    def valid_categories(cls, v: list[str]) -> list[str]:
        valid = tier1_category_codes()
        bad = [c for c in v if c not in valid]
        if bad:
            raise ValueError(f"Unknown violence_categories: {bad}")
        return v

    @field_validator("max_intensity")
    @classmethod
    def valid_intensity(cls, v: int) -> int:
        if v not in intensity_levels():
            raise ValueError(f"max_intensity {v} not in 1–5")
        return v


class PreprocessingApplied(BaseModel):
    resampled_to_16k: bool = False
    downmixed_to_mono: bool = False
    spectral_filtered: bool = False
    denoised: bool = False
    normalized_dbfs: float | None = None
    silence_padded: bool = False


# ---------------------------------------------------------------------------
# Per-clip metadata (one .json per clip)
# ---------------------------------------------------------------------------


class ClipMetadata(BaseModel):
    clip_id: str
    project: str
    language: str
    violence_typology: str
    tier: Literal["A", "B", "C"]
    duration_seconds: float = Field(gt=0)
    sample_rate: int = 16000
    channels: int = 1
    snr_db_estimated: float | None = None
    scene_config: str | None = None
    random_seed: int = 0
    generation_date: str
    generator_version: str
    is_synthetic: Literal[True] = True
    tts_engine: str = "azure_he_IL"
    acoustic_scene: ClipAcousticScene = Field(default_factory=ClipAcousticScene)
    speakers: list[SpeakerInfo] = Field(default_factory=list)
    weak_label: WeakLabel
    preprocessing_applied: PreprocessingApplied = Field(default_factory=PreprocessingApplied)
    dirty_file_path: str | None = None
    transcript_path: str | None = None
    quality_flags: list[str] = Field(default_factory=list)
    annotator_confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    iaa_reviewed: bool = False
    she_proves_meta: dict | None = None
    elephant_meta: dict | None = None

    @field_validator("violence_typology")
    @classmethod
    def valid_typology(cls, v: str) -> str:
        valid = violence_typology_codes()
        if v not in valid:
            raise ValueError(f"violence_typology {v!r} not in taxonomy")
        return v

    @field_validator("quality_flags")
    @classmethod
    def valid_quality_flags(cls, v: list[str]) -> list[str]:
        bad = [f for f in v if f not in _QUALITY_FLAGS]
        if bad:
            raise ValueError(f"Unknown quality_flags: {bad}. Valid: {sorted(_QUALITY_FLAGS)}")
        return v

    @field_validator("clip_id", "project", "tts_engine", "violence_typology", "generator_version")
    @classmethod
    def ascii_safe_string(cls, v: str, info) -> str:
        return _assert_ascii_safe(v, info.field_name)


# ---------------------------------------------------------------------------
# Per-event strong labels (one record per event in JSONL)
# ---------------------------------------------------------------------------


class EventLabel(BaseModel):
    event_id: str
    clip_id: str
    onset: float = Field(ge=0.0)
    offset: float = Field(gt=0.0)
    tier1_category: str
    tier2_subtype: str
    intensity: int = Field(ge=1, le=5)
    speaker_id: str | None = None
    speaker_role: str | None = None
    emotional_state: str | None = None
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    label_source: Literal["auto", "human", "auto_reviewed"] = "auto"
    iaa_reviewed: bool = False
    notes: str | None = None

    @field_validator("tier1_category")
    @classmethod
    def valid_tier1(cls, v: str) -> str:
        valid = tier1_category_codes()
        if v not in valid:
            raise ValueError(f"tier1_category {v!r} not in taxonomy")
        return v

    @field_validator("tier2_subtype")
    @classmethod
    def valid_tier2(cls, v: str) -> str:
        valid = tier2_subtype_codes()
        if v not in valid:
            raise ValueError(f"tier2_subtype {v!r} not in taxonomy")
        return v

    @field_validator("intensity")
    @classmethod
    def valid_intensity(cls, v: int) -> int:
        if v not in intensity_levels():
            raise ValueError(f"intensity {v} not in 1–5")
        return v

    @field_validator("speaker_role")
    @classmethod
    def valid_speaker_role(cls, v: str | None) -> str | None:
        if v is None:
            return v
        valid = speaker_role_codes()
        if v not in valid:
            raise ValueError(f"speaker_role {v!r} not in taxonomy")
        return v

    @field_validator("emotional_state")
    @classmethod
    def valid_emotional_state(cls, v: str | None) -> str | None:
        if v is None:
            return v
        valid = emotional_state_values()
        if v not in valid:
            raise ValueError(f"emotional_state {v!r} not valid. Valid: {valid}")
        return v

    @model_validator(mode="after")
    def tier2_matches_tier1(self) -> EventLabel:
        parent = tier2_parent(self.tier2_subtype)
        if parent is not None and parent != self.tier1_category:
            raise ValueError(
                f"tier2_subtype {self.tier2_subtype!r} belongs to parent {parent!r}, "
                f"but tier1_category is {self.tier1_category!r}"
            )
        return self

    @model_validator(mode="after")
    def offset_after_onset(self) -> EventLabel:
        if self.offset <= self.onset:
            raise ValueError(f"offset {self.offset} must be > onset {self.onset}")
        return self
