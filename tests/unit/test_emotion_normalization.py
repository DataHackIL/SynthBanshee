"""Unit tests for M4 — emotional-state normalization in cli.py."""

from __future__ import annotations

import logging

import pytest

from synthbanshee.cli import _EMOTION_ALIASES, _normalize_emotion


class TestNormalizeEmotion:
    def test_known_state_returned_verbatim(self):
        state, alias_used = _normalize_emotion("anger")
        assert state == "anger"
        assert alias_used is False

    def test_case_and_whitespace_stripped(self):
        """LLM outputs like 'Calm', 'ANGER', 'neutral ' must resolve without aliases."""
        assert _normalize_emotion("Calm") == ("calm", False)
        assert _normalize_emotion("ANGER") == ("anger", False)
        assert _normalize_emotion("neutral ") == ("neutral", False)

    def test_alias_maps_to_canonical(self):
        state, alias_used = _normalize_emotion("worried")
        assert state == "distress"
        assert alias_used is True

    def test_alias_covers_all_entries(self):
        """Every key in _EMOTION_ALIASES must resolve to a known taxonomy state."""
        from synthbanshee.config.taxonomy import emotional_state_values

        known = set(emotional_state_values())
        for alias, canonical in _EMOTION_ALIASES.items():
            assert canonical in known, (
                f"_EMOTION_ALIASES[{alias!r}] = {canonical!r} is not in taxonomy"
            )

    def test_alias_emits_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="synthbanshee.cli"):
            _normalize_emotion("relaxed")
        assert any("remapped" in r.message for r in caplog.records)

    def test_unknown_state_raises(self):
        with pytest.raises(ValueError, match="Unknown emotional_state"):
            _normalize_emotion("euphoric")

    def test_unknown_state_error_mentions_taxonomy(self):
        with pytest.raises(ValueError, match="taxonomy.yaml"):
            _normalize_emotion("xyzzy")

    @pytest.mark.parametrize("raw,expected", list(_EMOTION_ALIASES.items()))
    def test_alias_mapping_table(self, raw, expected):
        """Every entry in _EMOTION_ALIASES must round-trip through _normalize_emotion."""
        state, alias_used = _normalize_emotion(raw)
        assert state == expected
        assert alias_used is True


class TestNewTaxonomyStates:
    """Verify newly added canonical states are accepted by EventLabel."""

    def test_shame_is_valid_taxonomy_state(self):
        from synthbanshee.config.taxonomy import emotional_state_values

        assert "shame" in emotional_state_values()

    def test_desperation_is_valid_taxonomy_state(self):
        from synthbanshee.config.taxonomy import emotional_state_values

        assert "desperation" in emotional_state_values()

    def test_shame_normalizes_directly(self):
        state, alias_used = _normalize_emotion("shame")
        assert state == "shame"
        assert alias_used is False

    def test_desperation_normalizes_directly(self):
        state, alias_used = _normalize_emotion("desperation")
        assert state == "desperation"
        assert alias_used is False
