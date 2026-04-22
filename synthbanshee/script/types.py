"""Shared types for the script generation and TTS mixing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from synthbanshee.tts.ssml_types import PhraseHint


@dataclass
class DialogueTurn:
    """One utterance from a single speaker in a generated scene script.

    Attributes:
        speaker_id: Matches a SpeakerConfig.speaker_id in the scene.
        text: Original Hebrew UTF-8 text as produced by the LLM.  This is
            the authoritative source of record and is never mutated after
            creation.  Also accessible as ``text_original``.
        intensity: 1–5, drives SSML style selection in TTSRenderer.
        pause_before_s: Silence gap (seconds) inserted before this turn in the mix.
        emotional_state: LLM-generated hint; used as a secondary style cue.
        text_spoken: Post-normalization text actually sent to TTS.  Populated
            by the Hebrew gender-disambiguation step (M1); defaults to ``text``
            when no normalization has been applied.
        normalization_rules_triggered: Ordered list of disambiguation rule IDs
            applied to produce ``text_spoken`` (e.g. ``["POSS_SHEL", "PREP_LAKH"]``).
            Empty when ``text_spoken == text``.
        phrase_hints: LLM-annotated prosody hints for emotionally loaded phrases
            within this turn (M2b).  Offsets reference ``text`` (``text_original``).
            Populated by the script generator for I3–I5 turns; empty otherwise.
        speaker_state_snapshot: Serialized ``SpeakerState`` (via
            ``to_metadata_dict()``) captured immediately before this turn was
            rendered — i.e. the accumulated prosody state that was applied.
            Populated by ``TTSRenderer.render_scene()``; empty dict when the
            turn was rendered outside that context.
    """

    speaker_id: str
    text: str
    intensity: int
    pause_before_s: float = 0.3
    emotional_state: str = "neutral"
    text_spoken: str = ""
    normalization_rules_triggered: list[str] = field(default_factory=list)
    # M2b: LLM-annotated phrase-level prosody hints (offsets in text_original).
    phrase_hints: list[PhraseHint] = field(default_factory=list)
    # M7: snapshot of SpeakerState.to_metadata_dict() at render time (pre-update).
    # Populated by TTSRenderer.render_scene(); empty when rendered without state.
    speaker_state_snapshot: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Default text_spoken to the original LLM text when not explicitly set.
        if not self.text_spoken:
            self.text_spoken = self.text

    @property
    def text_original(self) -> str:
        """Alias for ``text``; the unmodified LLM output."""
        return self.text


@dataclass
class MixedScene:
    """Audio result of mixing multiple per-turn TTS segments into one scene.

    Attributes:
        samples: Float32 numpy array, mono, 16 kHz.
        sample_rate: Always 16000.
        turn_onsets_s: Per-turn audible onset in seconds (backward-compat alias
            for ``audible_onsets_s``; safe for all consumers since values are
            always within the final waveform duration).
        turn_offsets_s: Per-turn audible end in seconds (backward-compat alias
            for ``audible_ends_s``; for BARGE_IN-interrupted turns this is the
            truncation point, not the original TTS end).
        duration_s: Total scene duration in seconds.
        speaker_ids: Speaker ID for each turn (parallel with onsets/offsets).
        script_onsets_s: Sequential-world onset — where the turn would start if
            no overlap were applied (§4.6).
        script_offsets_s: Sequential-world offset (script_onset + TTS duration).
        rendered_onsets_s: Actual onset in the output buffer, accounting for
            overlap/barge-in positioning.
        rendered_offsets_s: Actual offset in the output buffer after any
            barge-in truncation has been applied.  For uninterrupted turns this
            is ``rendered_onset + TTS duration``; for turns interrupted by a
            BARGE_IN it is the truncation point in the final mixed output.
        audible_onsets_s: Same as ``rendered_onsets_s`` for all turns.
        audible_ends_s: End of the audible portion.  Matches
            ``rendered_offsets_s`` for all turns, including BARGE_IN-interrupted
            turns where both fields reflect the truncation point.
    """

    samples: np.ndarray
    sample_rate: int
    turn_onsets_s: list[float]
    turn_offsets_s: list[float]
    duration_s: float
    speaker_ids: list[str] = field(default_factory=list)
    # M8a: three-timeline timestamps (§4.6).
    script_onsets_s: list[float] = field(default_factory=list)
    script_offsets_s: list[float] = field(default_factory=list)
    rendered_onsets_s: list[float] = field(default_factory=list)
    rendered_offsets_s: list[float] = field(default_factory=list)
    audible_onsets_s: list[float] = field(default_factory=list)
    audible_ends_s: list[float] = field(default_factory=list)
