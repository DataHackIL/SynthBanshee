"""Shared types for the script generation and TTS mixing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DialogueTurn:
    """One utterance from a single speaker in a generated scene script.

    Attributes:
        speaker_id: Matches a SpeakerConfig.speaker_id in the scene.
        text: Hebrew UTF-8 utterance text. Never placed in filenames.
        intensity: 1–5, drives SSML style selection in TTSRenderer.
        pause_before_s: Silence gap (seconds) inserted before this turn in the mix.
        emotional_state: LLM-generated hint; used as a secondary style cue.
    """

    speaker_id: str
    text: str
    intensity: int
    pause_before_s: float = 0.3
    emotional_state: str = "neutral"


@dataclass
class MixedScene:
    """Audio result of mixing multiple per-turn TTS segments into one scene.

    Attributes:
        samples: Float32 numpy array, mono, 16 kHz.
        sample_rate: Always 16000.
        turn_onsets_s: Per-turn onset time in seconds (after silence pad).
        turn_offsets_s: Per-turn offset time in seconds.
        duration_s: Total scene duration in seconds.
        speaker_ids: Speaker ID for each turn (parallel with onsets/offsets).
    """

    samples: np.ndarray
    sample_rate: int
    turn_onsets_s: list[float]
    turn_offsets_s: list[float]
    duration_s: float
    speaker_ids: list[str] = field(default_factory=list)
