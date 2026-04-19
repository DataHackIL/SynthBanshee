"""Stateful cross-turn emotional controller (M7 — spec §4.3).

One ``SpeakerState`` instance is created per speaker at the start of
``TTSRenderer.render_scene()`` and updated after every turn.  The accumulated
prosody offsets are applied on top of the base ``StyleEntry`` values so that a
speaker who has been escalating for several turns carries that momentum into
subsequent turns rather than resetting to a neutral baseline at each turn
boundary.

Design doc reference: docs/audio_generation_v3_design.md §4.3
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Per-role intensity target tables
#
# Each entry maps an intensity level (1–5) to the target state offsets that
# the speaker converges toward as turns at that intensity accumulate:
#   (rate_offset_multiplier, pitch_offset_semitones, volume_offset_db)
#
# AGG: higher intensity → faster rate, higher pitch, louder.
# VIC: higher AGG pressure → VIC compliance shifts to slower, lower volume,
#      slight pitch rise from tension.
# ---------------------------------------------------------------------------

_AGG_TARGETS: dict[int, tuple[float, float, float]] = {
    1: (1.00, 0.0, 0.0),
    2: (1.03, 0.3, 1.0),
    3: (1.08, 0.8, 2.0),
    4: (1.12, 1.5, 3.0),
    5: (1.15, 2.0, 4.0),
}

_VIC_TARGETS: dict[int, tuple[float, float, float]] = {
    1: (1.00, 0.0, 0.0),
    2: (0.98, 0.2, -0.5),
    3: (0.94, 0.5, -1.5),
    4: (0.90, 0.8, -2.5),
    5: (0.87, 1.0, -3.0),
}

# Unknown roles (bystanders, witnesses, etc.) hold neutral state.
_NEUTRAL_TARGETS: dict[int, tuple[float, float, float]] = {k: (1.0, 0.0, 0.0) for k in range(1, 6)}

# Fraction of the gap between current state and target that is closed per turn
# when intensity is steady or rising (escalation).
_DRIFT_RATE_ESCALATE: float = 0.60

# Fraction closed per turn when intensity drops (decay).
_DRIFT_RATE_DECAY: float = 0.30


def _target_for(role: str, intensity: int) -> tuple[float, float, float]:
    """Return the target offsets for *role* at *intensity* (clamped to 1–5)."""
    table: dict[int, tuple[float, float, float]]
    if role == "AGG":
        table = _AGG_TARGETS
    elif role == "VIC":
        table = _VIC_TARGETS
    else:
        table = _NEUTRAL_TARGETS
    return table[max(1, min(5, intensity))]


@dataclass
class SpeakerState:
    """Accumulated prosody-offset state for a single speaker across turns.

    Starts at neutral (rate_offset=1.0, all deltas 0.0).  Call
    ``update(turn.intensity, speaker.role)`` after each rendered turn to drift
    the state.  Pass the state to ``TTSRenderer.render_utterance()`` via the
    ``speaker_state`` kwarg so the offsets are baked into the SSML.

    Attributes:
        intensity_history: Ordered list of intensities seen for this speaker.
        rate_offset: Multiplicative rate modifier applied on top of
            ``StyleEntry.rate_multiplier``.  Starts at 1.0 (no change).
        pitch_offset_st: Additive semitone shift on top of
            ``StyleEntry.pitch_delta_st``.  Starts at 0.0.
        volume_offset_db: Additive dB shift on top of
            ``StyleEntry.volume_delta_db``.  Starts at 0.0.
        breathiness_level: Reserved for M12 breathiness processing.
            0.0 = modal voice; 1.0 = maximum breathiness.  Not applied by
            M7 — stored here so M12 can wire it without a schema change.
    """

    intensity_history: list[int] = field(default_factory=list)
    rate_offset: float = 1.0
    pitch_offset_st: float = 0.0
    volume_offset_db: float = 0.0
    breathiness_level: float = 0.0  # reserved for M12

    def update(self, new_intensity: int, speaker_role: str) -> None:
        """Drift state toward the target for *new_intensity* and *speaker_role*.

        Escalation (new_intensity ≥ previous) closes 60 % of the gap between
        current state and target.  De-escalation closes only 30 % — state
        decays more slowly than it accumulates.

        Args:
            new_intensity: Intensity of the turn that was just rendered (1–5).
            speaker_role: Semantic role of the speaker (e.g. ``"AGG"``,
                ``"VIC"``), taken from ``SpeakerConfig.role``.
        """
        new_intensity = max(1, min(5, new_intensity))
        prev_intensity = self.intensity_history[-1] if self.intensity_history else new_intensity
        self.intensity_history.append(new_intensity)

        drift = _DRIFT_RATE_ESCALATE if new_intensity >= prev_intensity else _DRIFT_RATE_DECAY
        t_rate, t_pitch, t_vol = _target_for(speaker_role, new_intensity)

        self.rate_offset += drift * (t_rate - self.rate_offset)
        self.pitch_offset_st += drift * (t_pitch - self.pitch_offset_st)
        self.volume_offset_db += drift * (t_vol - self.volume_offset_db)

    def to_metadata_dict(self) -> dict[str, float]:
        """Serialize current state for per-turn generation metadata (prep for M11).

        Returns:
            Dict with keys ``rate_offset``, ``pitch_offset_st``,
            ``volume_offset_db``, ``breathiness_level``.
        """
        return {
            "rate_offset": round(self.rate_offset, 4),
            "pitch_offset_st": round(self.pitch_offset_st, 4),
            "volume_offset_db": round(self.volume_offset_db, 4),
            "breathiness_level": round(self.breathiness_level, 4),
        }
