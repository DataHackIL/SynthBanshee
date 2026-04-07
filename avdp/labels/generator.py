"""Stub auto-label generator: produces AVDP-schema labels from script structure.

In Phase 0 this is a stub: it takes a list of pre-defined script events
(with taxonomy codes, onsets, and offsets already known) and converts them
to validated EventLabel objects and a ClipMetadata record.

In later phases this will be driven by the LLM-generated script and the
augmentation log (which provides SFX onset/offset times).
"""

from __future__ import annotations

import datetime
from pathlib import Path

import jsonlines

from avdp import __version__
from avdp.labels.schema import (
    ClipAcousticScene,
    ClipMetadata,
    EventLabel,
    PreprocessingApplied,
    SpeakerInfo,
    WeakLabel,
)


class ScriptEvent:
    """Minimal description of one labeled event derived from the scene script."""

    def __init__(
        self,
        *,
        tier1_category: str,
        tier2_subtype: str,
        onset: float,
        offset: float,
        intensity: int,
        speaker_id: str | None = None,
        speaker_role: str | None = None,
        emotional_state: str | None = None,
        confidence: float = 1.0,
        notes: str | None = None,
    ) -> None:
        self.tier1_category = tier1_category
        self.tier2_subtype = tier2_subtype
        self.onset = onset
        self.offset = offset
        self.intensity = intensity
        self.speaker_id = speaker_id
        self.speaker_role = speaker_role
        self.emotional_state = emotional_state
        self.confidence = confidence
        self.notes = notes


class LabelGenerator:
    """Generate AVDP-schema labels from a structured list of script events."""

    def __init__(self, generator_version: str = __version__) -> None:
        self.generator_version = generator_version

    def generate_event_labels(
        self,
        clip_id: str,
        events: list[ScriptEvent],
    ) -> list[EventLabel]:
        """Convert script events into validated EventLabel objects."""
        labels: list[EventLabel] = []
        for idx, evt in enumerate(events):
            event_id = f"{clip_id}_EVT_{idx:03d}"
            labels.append(
                EventLabel(
                    event_id=event_id,
                    clip_id=clip_id,
                    onset=evt.onset,
                    offset=evt.offset,
                    tier1_category=evt.tier1_category,
                    tier2_subtype=evt.tier2_subtype,
                    intensity=evt.intensity,
                    speaker_id=evt.speaker_id,
                    speaker_role=evt.speaker_role,
                    emotional_state=evt.emotional_state,
                    confidence=evt.confidence,
                    label_source="auto",
                    notes=evt.notes,
                )
            )
        return labels

    def generate_clip_metadata(
        self,
        *,
        clip_id: str,
        project: str,
        violence_typology: str,
        tier: str,
        duration_seconds: float,
        events: list[EventLabel],
        speakers: list[SpeakerInfo] | None = None,
        scene_config_path: str | None = None,
        random_seed: int = 0,
        acoustic_scene: ClipAcousticScene | None = None,
        preprocessing: PreprocessingApplied | None = None,
        dirty_file_path: str | None = None,
        transcript_path: str | None = None,
        snr_db_estimated: float | None = None,
        quality_flags: list[str] | None = None,
    ) -> ClipMetadata:
        """Build a ClipMetadata record from pipeline outputs."""
        violence_categories = sorted(
            {e.tier1_category for e in events if e.tier1_category != "NONE"}
        )
        max_intensity = max((e.intensity for e in events), default=1)
        has_violence = any(e.tier1_category != "NONE" for e in events)

        return ClipMetadata(
            clip_id=clip_id,
            project=project,
            language="he",
            violence_typology=violence_typology,
            tier=tier,
            duration_seconds=duration_seconds,
            sample_rate=16000,
            channels=1,
            snr_db_estimated=snr_db_estimated,
            scene_config=scene_config_path,
            random_seed=random_seed,
            generation_date=datetime.date.today().isoformat(),
            generator_version=self.generator_version,
            is_synthetic=True,
            tts_engine="azure_he_IL",
            acoustic_scene=acoustic_scene or ClipAcousticScene(),
            speakers=speakers or [],
            weak_label=WeakLabel(
                has_violence=has_violence,
                violence_categories=violence_categories,
                max_intensity=max_intensity,
                violence_typology=violence_typology,
            ),
            preprocessing_applied=preprocessing or PreprocessingApplied(),
            dirty_file_path=dirty_file_path,
            transcript_path=transcript_path,
            quality_flags=quality_flags or [],
        )

    def write_strong_labels_jsonl(
        self, labels: list[EventLabel], path: Path
    ) -> None:
        """Serialize event labels to a JSONL file (one JSON object per line)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with jsonlines.open(path, mode="w") as writer:
            for label in labels:
                writer.write(label.model_dump())

    def read_strong_labels_jsonl(self, path: Path) -> list[EventLabel]:
        """Deserialize event labels from a JSONL file."""
        labels: list[EventLabel] = []
        with jsonlines.open(path) as reader:
            for record in reader:
                labels.append(EventLabel.model_validate(record))
        return labels

    def write_clip_metadata_json(self, metadata: ClipMetadata, path: Path) -> None:
        """Serialize clip metadata to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            fh.write(metadata.model_dump_json(indent=2))

    def read_clip_metadata_json(self, path: Path) -> ClipMetadata:
        """Deserialize clip metadata from a JSON file."""
        return ClipMetadata.model_validate_json(path.read_text(encoding="utf-8"))
