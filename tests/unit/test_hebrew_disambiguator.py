"""Unit tests for synthbanshee.script.hebrew_disambiguator (M1).

Every lexicon entry is tested individually so that a bad niqqud character in
the replacement string is caught immediately rather than silently producing
wrong TTS output.
"""

from __future__ import annotations

import pytest

from synthbanshee.script.hebrew_disambiguator import (
    _LEXICON,
    NormalizationResult,
    disambiguate_for_speaker,
    disambiguate_turns,
)
from synthbanshee.script.types import DialogueTurn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agg_vic(text: str) -> NormalizationResult:
    """Convenience: disambiguate as AGG speaker addressing VIC."""
    return disambiguate_for_speaker(text, speaker_role="AGG", addressee_role="VIC")


# ---------------------------------------------------------------------------
# disambiguate_for_speaker — direction guard
# ---------------------------------------------------------------------------


class TestDirectionGuard:
    """Substitutions must only fire for AGG→VIC; all other pairs are no-ops."""

    def test_agg_vic_may_modify(self):
        result = disambiguate_for_speaker("שלך", "AGG", "VIC")
        assert result.text_spoken == "שֶׁלָּךְ"
        assert result.normalization_rules_triggered == ["POSS_SHEL"]

    def test_vic_agg_no_change(self):
        result = disambiguate_for_speaker("שלך", "VIC", "AGG")
        assert result.text_spoken == "שלך"
        assert result.normalization_rules_triggered == []

    def test_agg_agg_no_change(self):
        result = disambiguate_for_speaker("הלכת", "AGG", "AGG")
        assert result.text_spoken == "הלכת"
        assert result.normalization_rules_triggered == []

    def test_unk_speaker_no_change(self):
        result = disambiguate_for_speaker("שלך", "UNK", "VIC")
        assert result.text_spoken == "שלך"
        assert result.normalization_rules_triggered == []

    def test_vic_vic_no_change(self):
        result = disambiguate_for_speaker("עשית", "VIC", "VIC")
        assert result.text_spoken == "עשית"
        assert result.normalization_rules_triggered == []


# ---------------------------------------------------------------------------
# disambiguate_for_speaker — pass-through
# ---------------------------------------------------------------------------


class TestPassThrough:
    def test_unlisted_word_unchanged(self):
        text = "שלום, מה שלומך?"
        result = _agg_vic(text)
        assert result.text_spoken == text
        assert result.normalization_rules_triggered == []

    def test_empty_string(self):
        result = _agg_vic("")
        assert result.text_spoken == ""
        assert result.normalization_rules_triggered == []

    def test_already_vocalized_text_unchanged(self):
        # Niqqud already present — surface form (unvocalized) won't match
        text = "הָלַכְתְּ"
        result = _agg_vic(text)
        assert result.text_spoken == text
        assert result.normalization_rules_triggered == []

    def test_non_hebrew_text_unchanged(self):
        text = "hello world"
        result = _agg_vic(text)
        assert result.text_spoken == text
        assert result.normalization_rules_triggered == []


# ---------------------------------------------------------------------------
# disambiguate_for_speaker — individual lexicon entries
# ---------------------------------------------------------------------------


class TestLexiconEntries:
    """One test per entry in _LEXICON ensuring the surface→feminine mapping is correct."""

    @pytest.mark.parametrize("entry", _LEXICON, ids=lambda e: e.rule_id)
    def test_surface_replaced_by_feminine(self, entry):
        """Each surface form surrounded by spaces must be replaced by its feminine variant."""
        text = f"אמרתי {entry.surface} לאחרונה"
        result = _agg_vic(text)
        assert entry.surface not in result.text_spoken, (
            f"Rule {entry.rule_id}: surface form still present after disambiguation"
        )
        assert entry.feminine_spoken in result.text_spoken, (
            f"Rule {entry.rule_id}: feminine form not inserted"
        )
        assert entry.rule_id in result.normalization_rules_triggered

    @pytest.mark.parametrize("entry", _LEXICON, ids=lambda e: e.rule_id)
    def test_surface_at_start_of_string(self, entry):
        text = f"{entry.surface} הוא הדבר"
        result = _agg_vic(text)
        assert entry.feminine_spoken in result.text_spoken

    @pytest.mark.parametrize("entry", _LEXICON, ids=lambda e: e.rule_id)
    def test_surface_at_end_of_string(self, entry):
        text = f"רציתי לדבר {entry.surface}"
        result = _agg_vic(text)
        assert entry.feminine_spoken in result.text_spoken

    @pytest.mark.parametrize("entry", _LEXICON, ids=lambda e: e.rule_id)
    def test_surface_not_matched_as_substring(self, entry):
        """A longer Hebrew word that CONTAINS the surface form must not be corrupted."""
        # Prepend and append a Hebrew letter to simulate embedding in a longer word.
        # The lexicon entry should NOT match inside a longer word.
        longer_word = "מ" + entry.surface + "ם"
        result = _agg_vic(longer_word)
        # The longer word is not a standalone match so text should be unchanged.
        assert result.text_spoken == longer_word

    @pytest.mark.parametrize("entry", _LEXICON, ids=lambda e: e.rule_id)
    def test_surface_not_matched_after_vocalized_letter(self, entry):
        """A surface form immediately following a niqqud-bearing letter must not match.

        Regression guard for the U+0591–U+05C7 combining-mark boundary fix: before
        the fix, a niqqud character after the preceding letter was not treated as a
        token character, so the lookbehind saw a non-letter at that position and
        allowed an in-word match.
        """
        # בָּ = bet (U+05D1) + dagesh (U+05BC) + patah (U+05B7): a vocalized letter
        # immediately before the surface form simulates a partially-vocalized longer word.
        adjacent = "בָּ" + entry.surface
        result = _agg_vic(adjacent)
        # Must not match: the surface form is not a standalone token here.
        assert result.text_spoken == adjacent


# ---------------------------------------------------------------------------
# disambiguate_for_speaker — multiple substitutions
# ---------------------------------------------------------------------------


class TestMultipleSubstitutions:
    def test_two_tokens_in_one_turn(self):
        text = "מה עשית עם שלך?"
        result = _agg_vic(text)
        assert "עָשִׂיתְ" in result.text_spoken
        assert "שֶׁלָּךְ" in result.text_spoken
        assert "VERB_PAST_ASIT" in result.normalization_rules_triggered
        assert "POSS_SHEL" in result.normalization_rules_triggered

    def test_same_token_twice(self):
        text = "לך ואחרי זה לך שוב"
        result = _agg_vic(text)
        # Both occurrences of לך should be replaced
        assert result.text_spoken.count("לָךְ") == 2

    def test_rules_triggered_list_contains_each_rule_once(self):
        text = "הלכת לשם ועשית את זה"
        result = _agg_vic(text)
        # Each triggered rule_id should appear at most once in the result list
        seen = result.normalization_rules_triggered
        assert len(seen) == len(set(seen)), "Duplicate rule IDs in triggered list"


# ---------------------------------------------------------------------------
# NormalizationResult
# ---------------------------------------------------------------------------


class TestNormalizationResult:
    def test_no_rules_gives_identical_text(self):
        text = "בוקר טוב"
        result = _agg_vic(text)
        assert result.text_spoken == text
        assert result.normalization_rules_triggered == []

    def test_result_is_dataclass(self):
        result = _agg_vic("שלך")
        assert isinstance(result, NormalizationResult)


# ---------------------------------------------------------------------------
# DialogueTurn integration
# ---------------------------------------------------------------------------


class TestDialogueTurnDefaults:
    def test_text_spoken_defaults_to_text(self):
        turn = DialogueTurn(speaker_id="AGG_001", text="שלום", intensity=1)
        assert turn.text_spoken == "שלום"

    def test_text_original_property(self):
        turn = DialogueTurn(speaker_id="AGG_001", text="שלום", intensity=1)
        assert turn.text_original == "שלום"

    def test_explicit_text_spoken_preserved(self):
        turn = DialogueTurn(
            speaker_id="AGG_001",
            text="שלך",
            intensity=1,
            text_spoken="שֶׁלָּךְ",
            normalization_rules_triggered=["POSS_SHEL"],
        )
        assert turn.text_spoken == "שֶׁלָּךְ"
        assert turn.text == "שלך"

    def test_normalization_rules_default_empty(self):
        turn = DialogueTurn(speaker_id="AGG_001", text="שלום", intensity=1)
        assert turn.normalization_rules_triggered == []


# ---------------------------------------------------------------------------
# disambiguate_turns
# ---------------------------------------------------------------------------


class TestDisambiguateTurns:
    def _make_turns(self) -> list[DialogueTurn]:
        return [
            DialogueTurn(speaker_id="AGG_001", text="מה עשית?", intensity=2),
            DialogueTurn(speaker_id="VIC_001", text="לא עשיתי כלום.", intensity=1),
        ]

    def _roles(self) -> dict[str, str]:
        return {"AGG_001": "AGG", "VIC_001": "VIC"}

    def test_agg_turn_modified(self):
        turns = disambiguate_turns(self._make_turns(), self._roles())
        agg_turn = next(t for t in turns if t.speaker_id == "AGG_001")
        assert "עָשִׂיתְ" in agg_turn.text_spoken
        assert "VERB_PAST_ASIT" in agg_turn.normalization_rules_triggered

    def test_vic_turn_not_modified(self):
        turns = disambiguate_turns(self._make_turns(), self._roles())
        vic_turn = next(t for t in turns if t.speaker_id == "VIC_001")
        # VIC→AGG direction: no feminine substitution applied
        assert vic_turn.text_spoken == vic_turn.text
        assert vic_turn.normalization_rules_triggered == []

    def test_original_turns_not_mutated(self):
        original = self._make_turns()
        original_texts = [t.text for t in original]
        disambiguate_turns(original, self._roles())
        assert [t.text for t in original] == original_texts

    def test_returns_same_length(self):
        turns = self._make_turns()
        result = disambiguate_turns(turns, self._roles())
        assert len(result) == len(turns)

    def test_empty_scene(self):
        assert disambiguate_turns([], {}) == []

    def test_single_speaker_scene_no_crash(self):
        turns = [DialogueTurn(speaker_id="AGG_001", text="שלך", intensity=1)]
        result = disambiguate_turns(turns, {"AGG_001": "AGG"})
        # Only one speaker — addressee role is UNK; no modification
        assert result[0].text_spoken == "שלך"

    def test_three_speaker_agg_prefers_vic_over_bys(self):
        """In a 3-role scene (AGG, VIC, BYS) the AGG turn must be disambiguated
        toward VIC, not BYS, regardless of dict insertion order."""
        turns = [DialogueTurn(speaker_id="AGG_001", text="מה עשית?", intensity=2)]
        # BYS is inserted first — without priority ordering this would be picked
        roles = {"AGG_001": "AGG", "BYS_001": "BYS", "VIC_001": "VIC"}
        result = disambiguate_turns(turns, roles)
        agg_turn = result[0]
        assert "עָשִׂיתְ" in agg_turn.text_spoken, "AGG turn should be disambiguated (addressee is VIC)"

    def test_three_speaker_bys_gets_no_disambiguation(self):
        """A BYS speaker addressing AGG must not receive feminine substitution."""
        turns = [DialogueTurn(speaker_id="BYS_001", text="מה עשית?", intensity=1)]
        roles = {"AGG_001": "AGG", "VIC_001": "VIC", "BYS_001": "BYS"}
        result = disambiguate_turns(turns, roles)
        bys_turn = result[0]
        # BYS→AGG is not AGG→VIC, so no substitution should fire
        assert bys_turn.text_spoken == bys_turn.text
