"""Shared types for the acoustic augmentation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AugmentedEvent:
    """Record of one background or SFX event applied during augmentation.

    onset_s and offset_s are raw times relative to the un-padded MixedScene.
    They do not include any preprocessing silence padding offset such as
    PreprocessingResult.silence_pad_applied_s.  Downstream code (for example,
    cli.py when writing labels) is responsible for adding that padding offset
    if final preprocessed-clip coordinates are required.
    """

    type: str  # taxonomy ACOU_* code or ambient label (e.g. "tv_ambient")
    onset_s: float  # seconds from start of clip
    offset_s: float  # seconds from start of clip
    level_db: float  # level relative to 0 dBFS (negative = quieter)


@dataclass
class AugmentationResult:
    """Result of applying the full acoustic augmentation stack to a MixedScene.

    Returned by NoiseMixer.mix(); consumed by Stage 3 → Stage 4 of the pipeline.
    The acoustic_scene dict is suitable for direct inclusion in ClipMetadata.
    """

    samples: np.ndarray  # float32, mono, 16 kHz — post-room, post-device, post-mix
    sample_rate: int  # always 16_000
    room_type: str
    device: str
    ir_source: str
    speaker_distance_meters: float
    snr_db_actual: float  # measured after mixing; logged in ClipMetadata
    events: list[AugmentedEvent] = field(default_factory=list)

    @property
    def acoustic_scene_dict(self) -> dict:
        """Return the acoustic_scene block for ClipMetadata (spec §5)."""
        return {
            "room_type": self.room_type,
            "device": self.device,
            "ir_source": self.ir_source,
            "speaker_distance_meters": self.speaker_distance_meters,
            "snr_db_actual": round(self.snr_db_actual, 2),
            "background_events": [
                {
                    "type": ev.type,
                    "onset_s": round(ev.onset_s, 3),
                    "offset_s": round(ev.offset_s, 3),
                    "level_db": round(ev.level_db, 1),
                }
                for ev in self.events
            ],
        }
