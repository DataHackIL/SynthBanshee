"""MixMode enum — how a turn is positioned relative to the previous turn.

Kept in a lightweight module (no audio deps) so that gap_controller and any
other timing-only code can import it without pulling in the full mixer stack.
"""

from __future__ import annotations

from enum import Enum


class MixMode(Enum):
    """Placement strategy for a turn in the output buffer."""

    SEQUENTIAL = "sequential"  # silence gap before this turn (default)
    OVERLAP = "overlap"  # start before prev ends; both turns audible
    BARGE_IN = "barge_in"  # start before prev ends; prev is cut off
