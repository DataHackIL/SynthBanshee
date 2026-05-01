"""Psychologically-motivated inter-turn gap controller.

Replaces fixed ``pause_before_s`` values with context-sensitive silence durations
drawn from project-specific tables (spec §4.5).  All draws are reproducible via
a caller-supplied ``random.Random`` instance seeded from the scene's ``random_seed``.

At all intensities the controller checks whether the current turn *overlaps* or
*barges in* on the previous turn, according to the asymmetric probability tables
in §4.6.  Low-intensity (I1–I2) transitions carry small non-zero probabilities so
that confusor scenes receive realistic overlap without the model learning
"overlap = violence".  High-intensity (I3–I5) transitions carry higher
probabilities matching the dominant-behaviour pattern in the spec.  When
overlap/barge-in is selected, the returned amount is drawn from a fixed range
representing the depth of the interruption rather than a silence gap.

Gap table reference: docs/audio_generation_v3_design.md §4.5
Overlap probability table: docs/audio_generation_v3_design.md §4.6
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from synthbanshee.script.types import DialogueTurn
from synthbanshee.tts.mix_mode import MixMode

if TYPE_CHECKING:
    from synthbanshee.config.project_profile import ProjectProfile


class _GapRange(NamedTuple):
    """Inclusive min/max gap in seconds."""

    lo: float
    hi: float


# ---------------------------------------------------------------------------
# Legacy project-specific gap tables (fallback when no ProjectProfile is provided)
#
# When a ProjectProfile is active (M13), gap timing comes from the profile YAML
# and these tables are NOT used.  They remain as backward-compatible fallbacks
# for programmatic callers that construct TurnGapController without a profile.
#
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
# Maps (prev_role, current_role, current_intensity) → (barge_in_prob, overlap_prob).
# Keyed on the *transition* (who is cutting off whom) to prevent same-role
# self-interruption, which has no psychological basis in the spec.
# ---------------------------------------------------------------------------

_OVERLAP_PROBS: dict[tuple[str, str, int], tuple[float, float]] = {
    # AGG cuts off VIC at I1–I2 (confusor/low-intensity scenes — spec §4.6 note)
    # Non-zero rates prevent the model from learning "overlap = violence".
    ("VIC", "AGG", 1): (0.02, 0.05),
    ("VIC", "AGG", 2): (0.05, 0.08),
    # AGG cuts off VIC at I3–I5 (spec §4.6 table rows 1–3)
    ("VIC", "AGG", 3): (0.10, 0.15),
    ("VIC", "AGG", 4): (0.25, 0.20),
    ("VIC", "AGG", 5): (0.40, 0.15),
    # VIC cuts off AGG at any intensity (spec §4.6 table row 4)
    ("AGG", "VIC", 1): (0.05, 0.10),
    ("AGG", "VIC", 2): (0.05, 0.10),
    ("AGG", "VIC", 3): (0.05, 0.10),
    ("AGG", "VIC", 4): (0.05, 0.10),
    ("AGG", "VIC", 5): (0.05, 0.10),
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
        gap_table_override: Optional externally-supplied gap table (e.g. from
            a ``ProjectProfile``).  When provided, this table is used instead
            of the hardcoded per-project tables, making gap timing fully
            data-driven.  Keys must match the standard context keys
            (``vic_low``, ``vic_i3``, etc.).
        agg_pause_prob_override: Optional override for the probability that an
            AGG turn at I4-I5 uses a deliberate menacing self-pause.
    """

    project: str
    gap_table_override: dict[str, _GapRange] | None = None
    agg_pause_prob_override: float | None = None

    def __post_init__(self) -> None:
        if self.gap_table_override is not None:
            self._table = self.gap_table_override
        else:
            self._table = _PROJECT_TABLES.get(self.project, _DEFAULT_GAPS)
        self._agg_pause_prob = (
            self.agg_pause_prob_override
            if self.agg_pause_prob_override is not None
            else _AGG_PAUSE_PROB
        )

    @classmethod
    def from_profile(
        cls,
        project: str,
        profile: ProjectProfile,
    ) -> TurnGapController:
        """Create a controller with gap table and overlap prob from a ``ProjectProfile``.

        Args:
            project: Project identifier (passed through for fallback).
            profile: A ``ProjectProfile`` instance providing gap timing and
                overlap probability defaults.
        """
        gt = profile.gap_timing
        gap_table: dict[str, _GapRange] = {
            "vic_low": _GapRange(gt.vic_low.lo, gt.vic_low.hi),
            "vic_i3": _GapRange(gt.vic_i3.lo, gt.vic_i3.hi),
            "vic_i4": _GapRange(gt.vic_i4.lo, gt.vic_i4.hi),
            "vic_i5": _GapRange(gt.vic_i5.lo, gt.vic_i5.hi),
            "agg_low": _GapRange(gt.agg_low.lo, gt.agg_low.hi),
            "agg_high": _GapRange(gt.agg_high.lo, gt.agg_high.hi),
            "agg_pause": _GapRange(gt.agg_pause.lo, gt.agg_pause.hi),
        }

        return cls(
            project=project,
            gap_table_override=gap_table,
            agg_pause_prob_override=profile.overlap.agg_pause_prob,
        )

    def gap_seconds(
        self,
        current_turn: DialogueTurn,
        prev_turn: DialogueTurn | None,
        rng: random.Random,
        current_role: str,
        prev_role: str | None = None,
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
            prev_role: Semantic role of the *previous* speaker.  ``None`` when
                ``prev_turn`` is ``None`` (first turn).  Overlap/barge-in is only
                considered when both roles are known and differ, matching the
                spec §4.6 transition table.

        Returns:
            ``(amount_s, MixMode)`` tuple.
        """
        if prev_turn is None:
            # First turn: use a brief ambient lead-in from the low range.
            r = self._table["agg_low"]
            return rng.uniform(r.lo, r.hi), MixMode.SEQUENTIAL

        intensity = current_turn.intensity

        # --- Overlap / barge-in decision (§4.6) ---
        # Only consider overlap when both roles are known; prevents same-role
        # self-interruption (e.g. AGG→AGG) which has no spec support.
        if prev_role is not None:
            probs = _OVERLAP_PROBS.get((prev_role, current_role, intensity))
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

    def _agg_context(self, current_intensity: int, rng: random.Random) -> str:
        if current_intensity <= 2:
            return "agg_low"
        # I3-I5: default to cutting-in, but at I4-I5 allow menacing self-pause.
        if current_intensity >= 4 and rng.random() < self._agg_pause_prob:
            return "agg_pause"
        return "agg_high"
