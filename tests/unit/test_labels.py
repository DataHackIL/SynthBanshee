"""Unit tests for label schema and auto-generator (Phase 0.5)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from avdp.labels.generator import LabelGenerator, ScriptEvent
from avdp.labels.schema import (
    ClipMetadata,
    EventLabel,
    WeakLabel,
)

# ---------------------------------------------------------------------------
# EventLabel tests
# ---------------------------------------------------------------------------


def _make_event(**overrides) -> dict:
    base = {
        "event_id": "SP_IT_A_0001_00_EVT_000",
        "clip_id": "sp_it_a_0001_00",
        "onset": 1.0,
        "offset": 3.5,
        "tier1_category": "VERB",
        "tier2_subtype": "VERB_SHOUT",
        "intensity": 3,
        "speaker_role": "AGG",
        "emotional_state": "anger",
        "confidence": 0.95,
        "label_source": "auto",
    }
    base.update(overrides)
    return base


class TestEventLabel:
    def test_valid_event(self):
        label = EventLabel(**_make_event())
        assert label.tier1_category == "VERB"
        assert label.tier2_subtype == "VERB_SHOUT"

    def test_invalid_tier1(self):
        with pytest.raises(ValidationError, match="tier1_category"):
            EventLabel(**_make_event(tier1_category="EXPLODE"))

    def test_invalid_tier2(self):
        with pytest.raises(ValidationError, match="tier2_subtype"):
            EventLabel(**_make_event(tier2_subtype="VERB_NONE_EXISTING"))

    def test_tier2_parent_mismatch(self):
        """PHYS_HARD belongs to PHYS, not VERB — should raise."""
        with pytest.raises(ValidationError, match="parent"):
            EventLabel(**_make_event(tier1_category="VERB", tier2_subtype="PHYS_HARD"))

    def test_invalid_intensity(self):
        with pytest.raises(ValidationError, match="intensity"):
            EventLabel(**_make_event(intensity=6))

    def test_offset_before_onset(self):
        with pytest.raises(ValidationError, match="offset"):
            EventLabel(**_make_event(onset=5.0, offset=3.0))

    def test_invalid_emotional_state(self):
        with pytest.raises(ValidationError, match="emotional_state"):
            EventLabel(**_make_event(emotional_state="ecstatic"))

    def test_invalid_speaker_role(self):
        with pytest.raises(ValidationError, match="speaker_role"):
            EventLabel(**_make_event(speaker_role="BOSS"))

    def test_round_trip_json(self):
        label = EventLabel(**_make_event())
        json_str = label.model_dump_json()
        label2 = EventLabel.model_validate_json(json_str)
        assert label == label2

    def test_none_category_with_none_subtype(self):
        label = EventLabel(**_make_event(tier1_category="NONE", tier2_subtype="NONE_AMBIENT"))
        assert label.tier1_category == "NONE"

    def test_phys_subtype_correct_parent(self):
        label = EventLabel(**_make_event(tier1_category="PHYS", tier2_subtype="PHYS_HARD"))
        assert label.tier2_subtype == "PHYS_HARD"


# ---------------------------------------------------------------------------
# WeakLabel tests
# ---------------------------------------------------------------------------


class TestWeakLabel:
    def test_valid_weak_label(self):
        wl = WeakLabel(
            has_violence=True,
            violence_categories=["VERB", "PHYS"],
            max_intensity=4,
            violence_typology="IT",
        )
        assert wl.has_violence is True

    def test_invalid_typology(self):
        with pytest.raises(ValidationError, match="violence_typology"):
            WeakLabel(
                has_violence=False,
                max_intensity=1,
                violence_typology="BADTYPE",
            )

    def test_invalid_category(self):
        with pytest.raises(ValidationError, match="violence_categories"):
            WeakLabel(
                has_violence=True,
                violence_categories=["EXPLODE"],
                max_intensity=3,
                violence_typology="SV",
            )


# ---------------------------------------------------------------------------
# ClipMetadata tests
# ---------------------------------------------------------------------------


def _make_metadata(**overrides) -> ClipMetadata:
    base = dict(
        clip_id="sp_it_a_0001_00",
        project="she_proves",
        language="he",
        violence_typology="IT",
        tier="A",
        duration_seconds=10.5,
        generation_date="2026-04-06",
        generator_version="0.1.0",
        is_synthetic=True,
        weak_label=WeakLabel(
            has_violence=True,
            violence_categories=["VERB"],
            max_intensity=3,
            violence_typology="IT",
        ),
    )
    base.update(overrides)
    return ClipMetadata(**base)


class TestClipMetadata:
    def test_valid_metadata(self):
        m = _make_metadata()
        assert m.is_synthetic is True
        assert m.sample_rate == 16000

    def test_invalid_quality_flag(self):
        with pytest.raises(ValidationError, match="quality_flags"):
            _make_metadata(quality_flags=["not_a_real_flag"])

    def test_round_trip_json(self):
        m = _make_metadata()
        json_str = m.model_dump_json()
        m2 = ClipMetadata.model_validate_json(json_str)
        assert m.clip_id == m2.clip_id
        assert m.weak_label == m2.weak_label

    def test_ascii_safe_clip_id(self):
        # Hebrew characters in clip_id should be rejected
        with pytest.raises(ValidationError):
            _make_metadata(clip_id="\u05e9\u05dc\u05d5\u05dd")


# ---------------------------------------------------------------------------
# LabelGenerator tests
# ---------------------------------------------------------------------------


class TestLabelGenerator:
    def setup_method(self):
        self.gen = LabelGenerator()

    def test_generate_event_labels(self):
        events = [
            ScriptEvent(
                tier1_category="VERB",
                tier2_subtype="VERB_SHOUT",
                onset=1.0,
                offset=3.0,
                intensity=3,
                speaker_id="AGG_M_30-45_001",
                speaker_role="AGG",
                emotional_state="anger",
            ),
            ScriptEvent(
                tier1_category="DIST",
                tier2_subtype="DIST_PLEAD",
                onset=3.5,
                offset=5.0,
                intensity=3,
                speaker_id="VIC_F_25-40_002",
                speaker_role="VIC",
                emotional_state="fear",
            ),
            ScriptEvent(
                tier1_category="ACOU",
                tier2_subtype="ACOU_SLAM",
                onset=5.1,
                offset=5.4,
                intensity=4,
            ),
        ]
        labels = self.gen.generate_event_labels("sp_it_a_0001_00", events)
        assert len(labels) == 3
        assert labels[0].event_id == "sp_it_a_0001_00_EVT_000"
        assert labels[1].tier2_subtype == "DIST_PLEAD"
        assert labels[2].tier1_category == "ACOU"

    def test_generate_clip_metadata(self):
        events = [
            ScriptEvent(
                tier1_category="VERB",
                tier2_subtype="VERB_SHOUT",
                onset=1.0,
                offset=3.0,
                intensity=3,
            ),
        ]
        labels = self.gen.generate_event_labels("test_clip", events)
        metadata = self.gen.generate_clip_metadata(
            clip_id="test_clip",
            project="she_proves",
            violence_typology="SV",
            tier="A",
            duration_seconds=5.0,
            events=labels,
        )
        assert metadata.is_synthetic is True
        assert metadata.weak_label.has_violence is True
        assert "VERB" in metadata.weak_label.violence_categories

    def test_jsonl_round_trip(self, tmp_path):
        events = [
            ScriptEvent(
                tier1_category="PHYS",
                tier2_subtype="PHYS_SOFT",
                onset=2.0,
                offset=2.5,
                intensity=4,
                speaker_role="AGG",
            ),
        ]
        labels = self.gen.generate_event_labels("clip_001", events)
        jsonl_path = tmp_path / "labels.jsonl"
        self.gen.write_strong_labels_jsonl(labels, jsonl_path)
        loaded = self.gen.read_strong_labels_jsonl(jsonl_path)
        assert len(loaded) == 1
        assert loaded[0] == labels[0]

    def test_metadata_json_round_trip(self, tmp_path):
        metadata = _make_metadata()
        json_path = tmp_path / "meta.json"
        self.gen.write_clip_metadata_json(metadata, json_path)
        loaded = self.gen.read_clip_metadata_json(json_path)
        assert loaded.clip_id == metadata.clip_id
        assert loaded.is_synthetic is True
