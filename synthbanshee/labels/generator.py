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
from typing import TYPE_CHECKING, Literal

import jsonlines

from synthbanshee import __version__
from synthbanshee.labels.schema import (
    ClipAcousticScene,
    ClipMetadata,
    EventLabel,
    PreprocessingApplied,
    SpeakerInfo,
    WeakLabel,
)

if TYPE_CHECKING:
    from synthbanshee.script.types import MixedScene

# One sample at 16 kHz — minimum label duration for BARGE_IN-zeroed turns.
_MIN_LABEL_DURATION_S = 1.0 / 16_000

# Truncation detection threshold (seconds).  Quantization of onset_sample via
# int() can leave audible_end up to 1 sample (1/16000 ≈ 6.25e-5 s) below the
# unquantized script_offset on purely SEQUENTIAL turns.  A threshold of two
# samples (≈1.25e-4 s) comfortably clears that noise while remaining far below
# the minimum real barge-in depth (_BARGE_IN_DEPTH_RANGE.lo = 0.10 s).
_TRUNCATION_THRESHOLD_S = 2.0 / 16_000


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
        truncated: bool = False,
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
        self.truncated = truncated
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
                    truncated=evt.truncated,
                    notes=evt.notes,
                )
            )
        return labels

    def generate_events_from_scene(
        self,
        clip_id: str,
        events: list[ScriptEvent],
        scene: MixedScene,
    ) -> list[EventLabel]:
        """Generate EventLabels using audible timing from a MixedScene.

        Onset/offset for each label comes from ``scene.audible_onsets_s`` /
        ``scene.audible_ends_s`` rather than the ScriptEvent's own onset/offset.
        When ``scene.script_offsets_s`` is populated and a turn's audible
        duration (``audible_end - audible_onset``) is strictly shorter than its
        script duration (``script_offset - script_onset``), ``truncated=True`` is
        set on the resulting label.  This captures BARGE_IN-interrupted turns
        without falsely flagging OVERLAP turns whose absolute audible end may be
        earlier than the script offset for reasons unrelated to truncation.

        For fully-barged-in turns where the audible end equals the onset (zero
        audible duration), the label offset is floored to
        ``onset + _MIN_LABEL_DURATION_S`` (one sample at 16 kHz) so that the
        ``offset > onset`` invariant is maintained; ``truncated`` is still ``True``.

        Args:
            clip_id: Clip identifier used to form event_id strings.
            events: One ScriptEvent per turn providing taxonomy codes and
                speaker metadata.  Must be parallel to scene.audible_onsets_s.
            scene: MixedScene produced by SceneMixer with three-timeline fields.

        Returns:
            One EventLabel per turn, timestamped from the audible timeline.

        Raises:
            ValueError: If len(events) != number of turns in scene.
        """
        n = len(scene.audible_onsets_s)
        if len(events) != n:
            raise ValueError(f"events length {len(events)} does not match scene turns {n}")
        labels: list[EventLabel] = []
        for idx, (evt, onset, end) in enumerate(
            zip(events, scene.audible_onsets_s, scene.audible_ends_s, strict=True)
        ):
            # Detect truncation by comparing the *audible duration* against the
            # *script duration* for this turn.  Using absolute onset vs. script
            # onset would incorrectly flag OVERLAP/BARGE_IN turns that started
            # early but spoke their full audio.
            script_onset = scene.script_onsets_s[idx] if idx < len(scene.script_onsets_s) else None
            script_offset = (
                scene.script_offsets_s[idx] if idx < len(scene.script_offsets_s) else None
            )
            if script_onset is not None and script_offset is not None:
                script_duration = script_offset - script_onset
                audible_duration = end - onset
                scene_truncated = audible_duration < script_duration - _TRUNCATION_THRESHOLD_S
            else:
                scene_truncated = False
            truncated = evt.truncated or scene_truncated
            # Floor zero-duration audible spans (fully-barged-in turns) to one
            # sample so the offset > onset validator is satisfied.
            safe_end = max(end, onset + _MIN_LABEL_DURATION_S)
            event_id = f"{clip_id}_EVT_{idx:03d}"
            labels.append(
                EventLabel(
                    event_id=event_id,
                    clip_id=clip_id,
                    onset=onset,
                    offset=safe_end,
                    tier1_category=evt.tier1_category,
                    tier2_subtype=evt.tier2_subtype,
                    intensity=evt.intensity,
                    speaker_id=evt.speaker_id,
                    speaker_role=evt.speaker_role,
                    emotional_state=evt.emotional_state,
                    confidence=evt.confidence,
                    label_source="auto",
                    truncated=truncated,
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
        tier: Literal["A", "B", "C"],
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

    def write_strong_labels_jsonl(self, labels: list[EventLabel], path: Path) -> None:
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
