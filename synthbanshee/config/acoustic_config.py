"""Pydantic models for acoustic scene configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

_VALID_ROOM_TYPES = {
    "small_bedroom",
    "apartment_kitchen",
    "living_room",
    "clinic_office",
    "welfare_office",
    "open_office_corridor",
}

_VALID_DEVICES = {
    "phone_in_hand",
    "phone_in_pocket",
    "phone_on_table",
    "pi_budget_mic",
}


class RoomDimensionsRange(BaseModel):
    min: Annotated[list[float], Field(min_length=3, max_length=3)]
    max: Annotated[list[float], Field(min_length=3, max_length=3)]

    @model_validator(mode="after")
    def min_lt_max(self) -> RoomDimensionsRange:
        for lo, hi in zip(self.min, self.max, strict=True):
            if lo > hi:
                raise ValueError(f"min dimension {lo} exceeds max {hi}")
        return self


class BackgroundEvent(BaseModel):
    type: str
    onset_seconds: float | None = None
    onset_at_phase: str | None = None
    onset_offset_seconds: float | None = None
    level_db: float | None = None
    loop: bool = False
    asset_path: str | None = None

    @model_validator(mode="after")
    def onset_defined(self) -> BackgroundEvent:
        if self.onset_seconds is None and self.onset_at_phase is None:
            raise ValueError("BackgroundEvent must specify onset_seconds or onset_at_phase")
        return self


class AcousticSceneConfig(BaseModel):
    room_type: str
    device: str
    speaker_distance_meters: float = Field(gt=0)
    victim_distance_meters: float = Field(gt=0)
    ir_source: str = "pyroomacoustics"
    room_dimensions_range: RoomDimensionsRange | None = None
    rt60_range: Annotated[list[float], Field(min_length=2, max_length=2)] | None = None
    background_events: list[BackgroundEvent] = Field(default_factory=list)
    snr_target_db: float = 20.0

    @field_validator("room_type")
    @classmethod
    def valid_room_type(cls, v: str) -> str:
        if v not in _VALID_ROOM_TYPES:
            raise ValueError(f"Unknown room_type {v!r}. Valid: {sorted(_VALID_ROOM_TYPES)}")
        return v

    @field_validator("device")
    @classmethod
    def valid_device(cls, v: str) -> str:
        if v not in _VALID_DEVICES:
            raise ValueError(f"Unknown device {v!r}. Valid: {sorted(_VALID_DEVICES)}")
        return v

    @field_validator("rt60_range")
    @classmethod
    def rt60_valid(cls, v: list[float] | None) -> list[float] | None:
        if v is not None and v[0] > v[1]:
            raise ValueError(f"rt60_range min {v[0]} > max {v[1]}")
        return v

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AcousticSceneConfig:
        return cls.model_validate(data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AcousticSceneConfig:
        with Path(path).open("r", encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))
