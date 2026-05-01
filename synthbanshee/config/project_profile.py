"""Pydantic model for project-specific default profiles (milestone M13).

A project profile defines default values for gap timing, overlap probabilities,
loudness targets, preprocessing config, and acoustic augmentation defaults.
Profiles are loaded from YAML files in ``configs/run_configs/`` and new profiles
can be added without code changes.

Spec reference: docs/audio_generation_v3_design.md §M13
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from synthbanshee.config.preprocessing_config import PreprocessingConfig

# Default search directory for profile YAML files.
_PROFILE_DIR = Path("configs/run_configs")


class GapRange(BaseModel):
    """Min/max gap duration in seconds for a single context key."""

    lo: float = Field(ge=0.0)
    hi: float = Field(ge=0.0)


class GapTimingDefaults(BaseModel):
    """Project-specific gap timing table.

    Keys match the context keys used by ``TurnGapController``:
    vic_low, vic_i3, vic_i4, vic_i5, agg_low, agg_high, agg_pause.
    """

    vic_low: GapRange = GapRange(lo=0.30, hi=0.60)
    vic_i3: GapRange = GapRange(lo=0.15, hi=0.35)
    vic_i4: GapRange = GapRange(lo=0.05, hi=0.15)
    vic_i5: GapRange = GapRange(lo=0.10, hi=0.30)
    agg_low: GapRange = GapRange(lo=0.20, hi=0.50)
    agg_high: GapRange = GapRange(lo=0.05, hi=0.20)
    agg_pause: GapRange = GapRange(lo=0.60, hi=1.40)

    def to_table(self) -> dict[str, tuple[float, float]]:
        """Convert to the ``{context_key: (lo, hi)}`` dict used by the gap controller."""
        return {
            key: (getattr(self, key).lo, getattr(self, key).hi)
            for key in (
                "vic_low",
                "vic_i3",
                "vic_i4",
                "vic_i5",
                "agg_low",
                "agg_high",
                "agg_pause",
            )
        }


class OverlapDefaults(BaseModel):
    """Project-specific overlap and barge-in probability defaults."""

    agg_pause_prob: float = Field(default=0.30, ge=0.0, le=1.0)


class LoudnessDefaults(BaseModel):
    """Default RMS loudness targets per role (dBFS)."""

    agg_rms_dbfs: float = Field(default=-20.0, ge=-60.0, le=0.0)
    vic_rms_dbfs: float = Field(default=-24.0, ge=-60.0, le=0.0)


class AcousticDefaults(BaseModel):
    """Default acoustic augmentation parameters for Tier B/C scenes."""

    snr_target_db: float = 20.0
    preferred_devices: list[str] = Field(default_factory=list)
    preferred_room_types: list[str] = Field(default_factory=list)


class ProjectProfile(BaseModel):
    """Project-specific default profile.

    Loaded from a YAML file in ``configs/run_configs/``.  Each field provides
    a default value layer that applies when no per-scene override is given.
    """

    name: str
    description: str = ""
    gap_timing: GapTimingDefaults = Field(default_factory=GapTimingDefaults)
    overlap: OverlapDefaults = Field(default_factory=OverlapDefaults)
    loudness: LoudnessDefaults = Field(default_factory=LoudnessDefaults)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    acoustic: AcousticDefaults = Field(default_factory=AcousticDefaults)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ProjectProfile:
        with Path(path).open("r", encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))


# ---- Profile registry (data-driven, no hardcoded profile names) ----

_profile_cache: dict[str, ProjectProfile] = {}


def _discover_profiles(profile_dir: Path) -> dict[str, Path]:
    """Scan *profile_dir* for ``profile_*.yaml`` files.

    Returns ``{profile_name: path}`` where *profile_name* is the filename stem
    with the ``profile_`` prefix stripped (e.g. ``profile_she_proves.yaml`` →
    ``"she_proves"``).
    """
    result: dict[str, Path] = {}
    if not profile_dir.is_dir():
        return result
    for p in sorted(profile_dir.glob("profile_*.yaml")):
        name = p.stem.removeprefix("profile_")
        result[name] = p
    return result


def load_profile(
    name: str,
    *,
    profile_dir: Path = _PROFILE_DIR,
    extra: dict[str, Any] | None = None,
) -> ProjectProfile:
    """Load a project profile by name.

    Args:
        name: Profile name (e.g. ``"she_proves"``, ``"elephant"``).
            ``"generic"`` returns a default ``ProjectProfile`` with framework
            defaults (no YAML file required).
        profile_dir: Directory to search for ``profile_<name>.yaml``.
        extra: Optional dict merged on top of the loaded YAML data (useful
            for RunConfig-level overrides).

    Raises:
        FileNotFoundError: If no matching profile YAML exists and *name* is
            not ``"generic"``.
    """
    if name == "generic":
        return ProjectProfile(name="generic", description="Framework defaults")

    cache_key = f"{profile_dir}::{name}"
    if cache_key in _profile_cache:
        return _profile_cache[cache_key]

    available = _discover_profiles(profile_dir)
    if name not in available:
        raise FileNotFoundError(
            f"No profile YAML for {name!r} in {profile_dir}. "
            f"Available: {sorted(available) or '(none)'}. "
            f"Expected file: profile_{name}.yaml"
        )

    profile = ProjectProfile.from_yaml(available[name])
    _profile_cache[cache_key] = profile
    return profile


def clear_profile_cache() -> None:
    """Clear the in-memory profile cache (useful for testing)."""
    _profile_cache.clear()
