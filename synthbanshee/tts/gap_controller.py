"""Psychologically-motivated inter-turn gap controller.

Replaces fixed ``pause_before_s`` values with context-sensitive silence durations
drawn from project-specific tables (spec §4.5).  All draws are reproducible via
a caller-supplied ``random.Random`` instance seeded from the scene's ``random_seed``.

Gap table reference: docs/audio_generation_v3_design.md §4.5
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import NamedTuple

from synthbanshee.script.types import DialogueTurn


class _GapRange(NamedTuple):
    """Inclusive min/max gap in seconds."""

    lo: float
    hi: float


# ---------------------------------------------------------------------------
# Project-specific gap tables
# Each table maps a (role, context_key) string to a _GapRange (seconds).
# context_key values:
#   vic_low        — VIC responds to AGG at I1–I2 (normal conversation)
#   vic_i3         — VIC responds to AGG accusation at I3
#   vic_i4         — VIC responds to AGG command at I4
#   vic_i5         — VIC responds to AGG threat at I5
#   agg_low        — AGG turn at I1–I2 (normal)
#   agg_high       — AGG turn at I3–I5 (cutting in)
#   agg_pause      — AGG deliberate self-pause at I4–I5 (menacing)
# ---------------------------------------------------------------------------

_SHE_PROVES_GAPS: dict[str, _GapRange] = {
    "vic_low": _GapRange(0.300, 0.600),
    "vic_i3": _GapRange(0.150, 0.350),
    "vic_i4": _GapRange(0.050, 0.150),
    "vic_i5": _GapRange(0.100, 0.300),
    "agg_low": _GapRange(0.200, 0.500),
    "agg_high": _GapRange(0.050, 0.200),
    "agg_pause": _GapRange(0.600, 1.400),
}

_ELEPHANT_GAPS: dict[str, _GapRange] = {
    "vic_low": _GapRange(0.250, 0.500),
    "vic_i3": _GapRange(0.100, 0.300),
    "vic_i4": _GapRange(0.050, 0.100),
    "vic_i5": _GapRange(0.080, 0.200),
    "agg_low": _GapRange(0.150, 0.400),
    "agg_high": _GapRange(0.050, 0.150),
    "agg_pause": _GapRange(0.500, 1.200),
}

# Fallback table for projects without a bespoke config (e.g. NEG/NEU confusors
# that don't belong to either primary project).  Ranges sit within the same
# psychological ballpark as the primary projects so confusor gaps are never
# mechanically fixed while violent scenes have timing logic.
_DEFAULT_GAPS: dict[str, _GapRange] = {
    "vic_low": _GapRange(0.300, 0.600),
    "vic_i3": _GapRange(0.150, 0.350),
    "vic_i4": _GapRange(0.050, 0.150),
    "vic_i5": _GapRange(0.100, 0.300),
    "agg_low": _GapRange(0.200, 0.500),
    "agg_high": _GapRange(0.050, 0.200),
    "agg_pause": _GapRange(0.600, 1.400),
}

_PROJECT_TABLES: dict[str, dict[str, _GapRange]] = {
    "she_proves": _SHE_PROVES_GAPS,
    "elephant_in_the_room": _ELEPHANT_GAPS,
}

# Probability that an AGG turn at I4–I5 is a deliberate menacing self-pause
# rather than an immediate cutting-in response.
_AGG_PAUSE_PROB = 0.30


def _role(speaker_id: str) -> str:
    """Extract the role prefix from a speaker_id (e.g. 'AGG_M_30-45_001' → 'AGG')."""
    return speaker_id.split("_")[0]


@dataclass
class TurnGapController:
    """Return psychologically-motivated inter-turn silence durations.

    Args:
        project: Scene project identifier (``"she_proves"`` or
            ``"elephant_in_the_room"``).  Unknown projects fall back to
            ``_DEFAULT_GAPS`` so that NEG/NEU confusor scenes always receive
            timing logic rather than mechanical fixed gaps.
    """

    project: str

    def __post_init__(self) -> None:
        self._table = _PROJECT_TABLES.get(self.project, _DEFAULT_GAPS)

    def gap_seconds(
        self,
        current_turn: DialogueTurn,
        prev_turn: DialogueTurn | None,
        rng: random.Random,
    ) -> float:
        """Return a gap duration (in seconds) to insert before *current_turn*.

        Args:
            current_turn: The turn about to be spoken.
            prev_turn: The immediately preceding turn, or ``None`` for the
                first turn in a scene (returns a short default gap).
            rng: Seeded ``random.Random`` instance for reproducible draws.

        Returns:
            Gap duration in seconds (≥ 0.0).
        """
        if prev_turn is None:
            # First turn: use a brief ambient lead-in from the low range.
            r = self._table["agg_low"]
            return rng.uniform(r.lo, r.hi)

        role = _role(current_turn.speaker_id)
        intensity = current_turn.intensity
        prev_intensity = prev_turn.intensity

        if role == "VIC":
            context_key = self._vic_context(prev_intensity)
        elif role == "AGG":
            context_key = self._agg_context(intensity, rng)
        else:
            # Unknown role (e.g. bystander, WIT, AUT) — use low AGG range.
            context_key = "agg_low"

        r = self._table[context_key]
        return rng.uniform(r.lo, r.hi)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _vic_context(prev_intensity: int) -> str:
        if prev_intensity <= 2:
            return "vic_low"
        if prev_intensity == 3:
            return "vic_i3"
        if prev_intensity == 4:
            return "vic_i4"
        return "vic_i5"  # intensity 5

    @staticmethod
    def _agg_context(current_intensity: int, rng: random.Random) -> str:
        if current_intensity <= 2:
            return "agg_low"
        # I3–I5: default to cutting-in, but at I4–I5 allow menacing self-pause.
        if current_intensity >= 4 and rng.random() < _AGG_PAUSE_PROB:
            return "agg_pause"
        return "agg_high"
