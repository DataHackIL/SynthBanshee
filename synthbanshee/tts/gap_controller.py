"""Psychologically-motivated inter-turn gap controller.

Replaces fixed ``pause_before_s`` values with context-sensitive silence durations
drawn from project-specific tables (spec §4.5).  All draws are reproducible via
a caller-supplied ``random.Random`` instance seeded from the scene's ``random_seed``.

At I3–I5 intensities, the controller also decides whether the current turn
*overlaps* or *barges in* on the previous turn, according to the asymmetric
probability tables in §4.6.  When overlap/barge-in is selected, the returned
amount is drawn from a fixed range representing the depth of the interruption
rather than a silence gap.

Gap table reference: docs/audio_generation_v3_design.md §4.5
Overlap probability table: docs/audio_generation_v3_design.md §4.6
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import NamedTuple

from synthbanshee.script.types import DialogueTurn
from synthbanshee.tts.mixer import MixMode


class _GapRange(NamedTuple):
    """Inclusive min/max gap in seconds."""

    lo: float
    hi: float


# ---------------------------------------------------------------------------
# Project-specific gap tables
# Each table maps a context_key string to a _GapRange (seconds).
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

# Fallback table used when a project identifier is not found in _PROJECT_TABLES.
# In the CLI pipeline, SceneConfig validates project against _VALID_PROJECTS so
# only the two primary projects reach render_scene(); this fallback is a safety
# net for programmatic use and future project additions.  NEG/NEU confusor
# scenes use she_proves or elephant_in_the_room as their project identifier and
# therefore receive a bespoke table — not this fallback.
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

# ---------------------------------------------------------------------------
# Overlap probability tables (§4.6)
# Maps (current_role, current_intensity) → (barge_in_prob, overlap_prob).
# AGG at I3–I5 may cut off VIC; VIC at any I may (rarely) cut off AGG.
# ---------------------------------------------------------------------------

_OVERLAP_PROBS: dict[tuple[str, int], tuple[float, float]] = {
    # (current_role, current_intensity) → (barge_in_prob, overlap_prob)
    ("AGG", 3): (0.10, 0.15),
    ("AGG", 4): (0.25, 0.20),
    ("AGG", 5): (0.40, 0.15),
    ("VIC", 1): (0.05, 0.10),
    ("VIC", 2): (0.05, 0.10),
    ("VIC", 3): (0.05, 0.10),
    ("VIC", 4): (0.05, 0.10),
    ("VIC", 5): (0.05, 0.10),
}

# Depth ranges (seconds) drawn when an overlap/barge-in mode is selected.
_OVERLAP_DEPTH_RANGE = _GapRange(0.10, 0.35)  # how far into prev turn OVERLAP starts
_BARGE_IN_DEPTH_RANGE = _GapRange(0.20, 0.50)  # how far into prev turn BARGE_IN starts


@dataclass
class TurnGapController:
    """Return psychologically-motivated inter-turn silence durations and mix modes.

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
        current_role: str,
    ) -> tuple[float, MixMode]:
        """Return an (amount, MixMode) pair for the transition to *current_turn*.

        For ``MixMode.SEQUENTIAL``, *amount* is a silence gap in seconds inserted
        before the turn.  For ``MixMode.OVERLAP`` and ``MixMode.BARGE_IN``,
        *amount* is the overlap depth in seconds — how far back into the previous
        turn the current turn starts.

        Args:
            current_turn: The turn about to be spoken.
            prev_turn: The immediately preceding turn, or ``None`` for the
                first turn in a scene (returns a short default gap, SEQUENTIAL).
            rng: Seeded ``random.Random`` instance for reproducible draws.
            current_role: Semantic role of the current speaker (``"AGG"`` or
                ``"VIC"``), taken from ``SpeakerConfig.role``.  Must not be
                inferred from ``speaker_id`` because Elephant scenes use persona
                prefixes (``BEN_``, ``SW_``) rather than role prefixes.

        Returns:
            ``(amount_s, MixMode)`` tuple.
        """
        if prev_turn is None:
            # First turn: use a brief ambient lead-in from the low range.
            r = self._table["agg_low"]
            return rng.uniform(r.lo, r.hi), MixMode.SEQUENTIAL

        intensity = current_turn.intensity

        # --- Overlap / barge-in decision (§4.6) ---
        probs = _OVERLAP_PROBS.get((current_role, intensity))
        if probs is not None:
            barge_in_p, overlap_p = probs
            roll = rng.random()
            if roll < barge_in_p:
                depth = rng.uniform(_BARGE_IN_DEPTH_RANGE.lo, _BARGE_IN_DEPTH_RANGE.hi)
                return depth, MixMode.BARGE_IN
            if roll < barge_in_p + overlap_p:
                depth = rng.uniform(_OVERLAP_DEPTH_RANGE.lo, _OVERLAP_DEPTH_RANGE.hi)
                return depth, MixMode.OVERLAP

        # --- SEQUENTIAL path ---
        prev_intensity = prev_turn.intensity
        if current_role == "VIC":
            context_key = self._vic_context(prev_intensity)
        elif current_role == "AGG":
            context_key = self._agg_context(intensity, rng)
        else:
            # Unknown role (e.g. bystander, WIT, AUT) — use low AGG range.
            context_key = "agg_low"

        r = self._table[context_key]
        return rng.uniform(r.lo, r.hi), MixMode.SEQUENTIAL

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
