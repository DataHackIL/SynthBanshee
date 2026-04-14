"""Pydantic model for a full generation run configuration (milestone 1.4).

A run config specifies everything needed to orchestrate a batch generation:
  - Which project and tier to generate
  - Per-typology clip count targets
  - Where to find scene configs
  - Train/val/test split fractions
  - Retry and error handling behaviour
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from synthbanshee.config.taxonomy import violence_typology_codes

_VALID_PROJECTS = {"she_proves", "elephant_in_the_room"}


class TypologyTarget(BaseModel):
    """Clip count target for one violence typology."""

    violence_typology: str
    count: int = Field(gt=0)

    @field_validator("violence_typology")
    @classmethod
    def valid_typology(cls, v: str) -> str:
        valid = violence_typology_codes()
        if v not in valid:
            raise ValueError(f"violence_typology {v!r} not in taxonomy. Valid: {sorted(valid)}")
        return v


class SplitFractions(BaseModel):
    """Train/val/test fraction targets for speaker-disjoint splits.

    All three fractions must be positive and sum to exactly 1.0.
    """

    train: float = Field(default=0.70, gt=0.0, lt=1.0)
    val: float = Field(default=0.15, gt=0.0, lt=1.0)
    test: float = Field(default=0.15, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def fractions_sum_to_one(self) -> Self:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split fractions must sum to 1.0; got {total:.4f}")
        return self


class RunConfig(BaseModel):
    """Configuration for a full batch generation run."""

    run_id: str
    project: str
    tier: Literal["A", "B", "C"] = "A"
    language: str = "he"
    random_seed: int = 42
    output_dir: str = Field(
        default_factory=lambda: os.environ.get("SYNTHBANSHEE_DATA_DIR") or "data/he"
    )
    scene_configs_dir: str = "configs/scenes"
    targets: list[TypologyTarget] = Field(min_length=1)
    splits: SplitFractions = Field(default_factory=SplitFractions)
    max_retries: int = Field(default=3, ge=1)
    fail_fast: bool = False

    @field_validator("project")
    @classmethod
    def valid_project(cls, v: str) -> str:
        if v not in _VALID_PROJECTS:
            raise ValueError(f"project {v!r} not in {sorted(_VALID_PROJECTS)}")
        return v

    @property
    def total_target(self) -> int:
        """Total clip count across all typology targets."""
        return sum(t.count for t in self.targets)

    def targets_by_typology(self) -> dict[str, int]:
        """Return {violence_typology: count} for all targets."""
        return {t.violence_typology: t.count for t in self.targets}

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        with Path(path).open("r", encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))
