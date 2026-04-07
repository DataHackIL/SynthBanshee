"""Taxonomy loader — single source of truth for all AVDP label codes.

All label codes must be loaded from configs/taxonomy.yaml, never hardcoded.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

_TAXONOMY_PATH = Path(__file__).parent.parent.parent / "configs" / "taxonomy.yaml"


@lru_cache(maxsize=1)
def load_taxonomy() -> dict:
    """Load and cache the taxonomy YAML. Call this instead of hardcoding codes."""
    with _TAXONOMY_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def violence_typology_codes() -> frozenset[str]:
    return frozenset(load_taxonomy()["violence_typology"].keys())


def tier1_category_codes() -> frozenset[str]:
    return frozenset(load_taxonomy()["tier1_category"].keys())


def tier2_subtype_codes() -> frozenset[str]:
    return frozenset(load_taxonomy()["tier2_subtype"].keys())


def speaker_role_codes() -> frozenset[str]:
    return frozenset(load_taxonomy()["speaker_roles"].keys())


def emotional_state_values() -> list[str]:
    return list(load_taxonomy()["emotional_states"])


def intensity_levels() -> frozenset[int]:
    return frozenset(int(k) for k in load_taxonomy()["intensity"])


def scene_phase_values() -> list[str]:
    return list(load_taxonomy()["scene_phases"])


def tier2_parent(subtype_code: str) -> str | None:
    """Return the tier1 parent code for a given tier2 subtype, or None if not found."""
    subtypes = load_taxonomy()["tier2_subtype"]
    entry = subtypes.get(subtype_code)
    return entry["parent"] if entry else None
