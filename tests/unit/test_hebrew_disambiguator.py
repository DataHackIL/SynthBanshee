"""Unit tests for synthbanshee.script.hebrew_disambiguator (M1).

Every lexicon entry is tested individually so that a bad niqqud character in
the replacement string is caught immediately rather than silently producing
wrong TTS output.
"""

from __future__ import annotations

import pytest

from synthbanshee.script.hebrew_disambiguator import (
    _LEXICON,
    _LEXICON_2P_FEMININE,
    _LEXICON_2P_MASCULINE,
    NormalizationResult,
    check_gender_ambiguity,
    disambiguate_for_speaker,
    disambiguate_turns,
)
from synthbanshee.script.types import DialogueTurn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _addr_female(text: str) -> NormalizationResult:
    """Disambiguate with a male speaker addressing a female."""
    return disambiguate_for_speaker(text, speaker_gender="male", addressee_gender="female")


def _addr_male(text: str) -> NormalizationResult:
    """Disambiguate with a female speaker addressing a male."""
    return disambiguate_for_speaker(text, speaker_gender="female", addressee_gender="male")


# ---------------------------------------------------------------------------
# disambiguate_for_speaker — gender guard
# ---------------------------------------------------------------------------


class TestGenderGuard:
    """Substitutions must fire based on addressee gender, not role."""

    def test_addressee_female_applies_feminine_rules(self):
        result = disambiguate_for_speaker("\u05e9\u05dc\u05da", "male", "female")
        assert result.text_spoken == "\u05e9\u05b6\u05c1\u05dc\u05b8\u05bc\u05da\u05b0"
        assert result.normalization_rules_triggered == ["F2_POSS_SHEL"]

    def test_addressee_male_applies_masculine_rules(self):
        result = disambiguate_for_speaker("\u05e9\u05dc\u05da", "female", "male")
        assert result.text_spoken == "\u05e9\u05b6\u05c1\u05dc\u05b0\u05bc\u05da\u05b8"
        assert result.normalization_rules_triggered == ["M2_POSS_SHEL"]

    def test_unknown_addressee_gender_no_change(self):
        result = disambiguate_for_speaker("\u05e9\u05dc\u05da", "male", "")
        assert result.text_spoken == "\u05e9\u05dc\u05da"
        assert result.normalization_rules_triggered == []

    def test_unknown_addressee_gender_unk_no_change(self):
        result = disambiguate_for_speaker("\u05e9\u05dc\u05da", "male", "unknown")
        assert result.text_spoken == "\u05e9\u05dc\u05da"
        assert result.normalization_rules_triggered == []


# ---------------------------------------------------------------------------
# disambiguate_for_speaker — pass-through
# ---------------------------------------------------------------------------


class TestPassThrough:
    def test_unlisted_word_unchanged(self):
        text = "\u05e9\u05dc\u05d5\u05dd, \u05de\u05d4 \u05e9\u05dc\u05d5\u05de\u05da?"
        result = _addr_female(text)
        assert result.text_spoken == text
        assert result.normalization_rules_triggered == []

    def test_empty_string(self):
        result = _addr_female("")
        assert result.text_spoken == ""
        assert result.normalization_rules_triggered == []

    def test_already_vocalized_text_unchanged(self):
        text = "\u05d4\u05b8\u05dc\u05b7\u05db\u05b0\u05ea\u05b0\u05bc"
        result = _addr_female(text)
        assert result.text_spoken == text
        assert result.normalization_rules_triggered == []

    def test_non_hebrew_text_unchanged(self):
        text = "hello world"
        result = _addr_female(text)
        assert result.text_spoken == text
        assert result.normalization_rules_triggered == []


# ---------------------------------------------------------------------------
# disambiguate_for_speaker — individual feminine lexicon entries
# ---------------------------------------------------------------------------


class TestFeminineLexiconEntries:
    """One test per entry in _LEXICON_2P_FEMININE."""

    @pytest.mark.parametrize("entry", _LEXICON_2P_FEMININE, ids=lambda e: e.rule_id)
    def test_surface_replaced_by_feminine(self, entry):
        text = (
            f"\u05d4\u05d9\u05d5\u05dd {entry.surface} \u05dc\u05d0\u05d7\u05e8\u05d5\u05e0\u05d4"
        )
        result = _addr_female(text)
        assert entry.spoken_form in result.text_spoken, (
            f"Rule {entry.rule_id}: spoken form not inserted"
        )
        assert entry.rule_id in result.normalization_rules_triggered

    @pytest.mark.parametrize("entry", _LEXICON_2P_FEMININE, ids=lambda e: e.rule_id)
    def test_surface_at_start_of_string(self, entry):
        text = f"{entry.surface} \u05d4\u05d5\u05d0 \u05d4\u05d3\u05d1\u05e8"
        result = _addr_female(text)
        assert entry.spoken_form in result.text_spoken

    @pytest.mark.parametrize("entry", _LEXICON_2P_FEMININE, ids=lambda e: e.rule_id)
    def test_surface_at_end_of_string(self, entry):
        text = f"\u05e8\u05e6\u05d9\u05ea\u05d9 \u05dc\u05d3\u05d1\u05e8 {entry.surface}"
        result = _addr_female(text)
        assert entry.spoken_form in result.text_spoken

    @pytest.mark.parametrize("entry", _LEXICON_2P_FEMININE, ids=lambda e: e.rule_id)
    def test_surface_not_matched_as_substring(self, entry):
        longer_word = "\u05de" + entry.surface + "\u05dd"
        result = _addr_female(longer_word)
        assert result.text_spoken == longer_word

    @pytest.mark.parametrize("entry", _LEXICON_2P_FEMININE, ids=lambda e: e.rule_id)
    def test_surface_not_matched_after_vocalized_letter(self, entry):
        adjacent = "\u05d1\u05b8\u05bc" + entry.surface
        result = _addr_female(adjacent)
        assert result.text_spoken == adjacent

    @pytest.mark.parametrize("entry", _LEXICON_2P_FEMININE, ids=lambda e: e.rule_id)
    def test_masculine_lexicon_does_not_fire_for_female_addressee(self, entry):
        """When addressee is female, masculine rules must not fire."""
        text = (
            f"\u05d4\u05d9\u05d5\u05dd {entry.surface} \u05dc\u05d0\u05d7\u05e8\u05d5\u05e0\u05d4"
        )
        result = _addr_female(text)
        assert not any(rid.startswith("M2_") for rid in result.normalization_rules_triggered)


# ---------------------------------------------------------------------------
# disambiguate_for_speaker — individual masculine lexicon entries
# ---------------------------------------------------------------------------


class TestMasculineLexiconEntries:
    """One test per entry in _LEXICON_2P_MASCULINE."""

    @pytest.mark.parametrize("entry", _LEXICON_2P_MASCULINE, ids=lambda e: e.rule_id)
    def test_surface_replaced_by_masculine(self, entry):
        text = (
            f"\u05d4\u05d9\u05d5\u05dd {entry.surface} \u05dc\u05d0\u05d7\u05e8\u05d5\u05e0\u05d4"
        )
        result = _addr_male(text)
        assert entry.spoken_form in result.text_spoken, (
            f"Rule {entry.rule_id}: spoken form not inserted"
        )
        assert entry.rule_id in result.normalization_rules_triggered

    @pytest.mark.parametrize("entry", _LEXICON_2P_MASCULINE, ids=lambda e: e.rule_id)
    def test_surface_at_start_of_string(self, entry):
        text = f"{entry.surface} \u05d4\u05d5\u05d0 \u05d4\u05d3\u05d1\u05e8"
        result = _addr_male(text)
        assert entry.spoken_form in result.text_spoken

    @pytest.mark.parametrize("entry", _LEXICON_2P_MASCULINE, ids=lambda e: e.rule_id)
    def test_surface_at_end_of_string(self, entry):
        text = f"\u05e8\u05e6\u05d9\u05ea\u05d9 \u05dc\u05d3\u05d1\u05e8 {entry.surface}"
        result = _addr_male(text)
        assert entry.spoken_form in result.text_spoken

    @pytest.mark.parametrize("entry", _LEXICON_2P_MASCULINE, ids=lambda e: e.rule_id)
    def test_surface_not_matched_as_substring(self, entry):
        longer_word = "\u05de" + entry.surface + "\u05dd"
        result = _addr_male(longer_word)
        assert result.text_spoken == longer_word

    @pytest.mark.parametrize("entry", _LEXICON_2P_MASCULINE, ids=lambda e: e.rule_id)
    def test_feminine_lexicon_does_not_fire_for_male_addressee(self, entry):
        """When addressee is male, feminine rules must not fire."""
        text = (
            f"\u05d4\u05d9\u05d5\u05dd {entry.surface} \u05dc\u05d0\u05d7\u05e8\u05d5\u05e0\u05d4"
        )
        result = _addr_male(text)
        assert not any(rid.startswith("F2_") for rid in result.normalization_rules_triggered)


# ---------------------------------------------------------------------------
# disambiguate_for_speaker — multiple substitutions
# ---------------------------------------------------------------------------


class TestMultipleSubstitutions:
    def test_two_tokens_in_one_turn_female(self):
        text = "\u05de\u05d4 \u05e2\u05e9\u05d9\u05ea \u05e2\u05dd \u05e9\u05dc\u05da?"
        result = _addr_female(text)
        assert "F2_VERB_ASIT" in result.normalization_rules_triggered
        assert "F2_POSS_SHEL" in result.normalization_rules_triggered

    def test_two_tokens_in_one_turn_male(self):
        text = "\u05de\u05d4 \u05e2\u05e9\u05d9\u05ea \u05e2\u05dd \u05e9\u05dc\u05da?"
        result = _addr_male(text)
        assert "M2_VERB_ASITA" in result.normalization_rules_triggered
        assert "M2_POSS_SHEL" in result.normalization_rules_triggered

    def test_same_token_twice(self):
        text = "\u05dc\u05da \u05d5\u05d0\u05d7\u05e8\u05d9 \u05d6\u05d4 \u05dc\u05da \u05e9\u05d5\u05d1"
        result = _addr_female(text)
        assert result.text_spoken.count("\u05dc\u05b8\u05da\u05b0") == 2

    def test_rules_triggered_list_contains_each_rule_once(self):
        text = "\u05d4\u05dc\u05db\u05ea \u05dc\u05e9\u05dd \u05d5\u05e2\u05e9\u05d9\u05ea \u05d0\u05ea \u05d6\u05d4"
        result = _addr_female(text)
        seen = result.normalization_rules_triggered
        assert len(seen) == len(set(seen)), "Duplicate rule IDs in triggered list"


# ---------------------------------------------------------------------------
# Bidirectional disambiguation — same surface, different output
# ---------------------------------------------------------------------------


class TestBidirectional:
    """The same unvocalized token must produce different niqqud for M vs F addressees."""

    def test_lekha_vs_lakh(self):
        """'\u05dc\u05da' must become '\u05dc\u05b0\u05da\u05b8' (lekha) for male and '\u05dc\u05b8\u05da\u05b0' (lakh) for female."""
        fem = _addr_female("\u05dc\u05da")
        masc = _addr_male("\u05dc\u05da")
        assert fem.text_spoken != masc.text_spoken
        assert "F2_PREP_LAKH" in fem.normalization_rules_triggered
        assert "M2_PREP_LEKHA" in masc.normalization_rules_triggered

    def test_shelakh_vs_shelkha(self):
        fem = _addr_female("\u05e9\u05dc\u05da")
        masc = _addr_male("\u05e9\u05dc\u05da")
        assert fem.text_spoken != masc.text_spoken

    def test_verb_past_asit_vs_asita(self):
        fem = _addr_female("\u05e2\u05e9\u05d9\u05ea")
        masc = _addr_male("\u05e2\u05e9\u05d9\u05ea")
        assert fem.text_spoken != masc.text_spoken
        assert "F2_VERB_ASIT" in fem.normalization_rules_triggered
        assert "M2_VERB_ASITA" in masc.normalization_rules_triggered


# ---------------------------------------------------------------------------
# NormalizationResult
# ---------------------------------------------------------------------------


class TestNormalizationResult:
    def test_no_rules_gives_identical_text(self):
        text = "\u05d1\u05d5\u05e7\u05e8 \u05d8\u05d5\u05d1"
        result = _addr_female(text)
        assert result.text_spoken == text
        assert result.normalization_rules_triggered == []

    def test_result_is_dataclass(self):
        result = _addr_female("\u05e9\u05dc\u05da")
        assert isinstance(result, NormalizationResult)


# ---------------------------------------------------------------------------
# check_gender_ambiguity — QA gate
# ---------------------------------------------------------------------------


class TestCheckGenderAmbiguity:
    def test_clean_text_no_warnings(self):
        assert check_gender_ambiguity("\u05d1\u05d5\u05e7\u05e8 \u05d8\u05d5\u05d1") == []

    def test_vocalized_text_no_warnings(self):
        # Already vocalized — the scanner should not match niqqud-bearing text.
        assert check_gender_ambiguity("\u05e9\u05b6\u05c1\u05dc\u05b8\u05bc\u05da\u05b0") == []

    def test_ambiguous_surface_flagged(self):
        warnings = check_gender_ambiguity("\u05d0\u05de\u05e8\u05ea\u05d9 \u05e9\u05dc\u05da")
        assert len(warnings) == 1
        assert "gender_ambiguity" in warnings[0]
        assert "\u05e9\u05dc\u05da" in warnings[0]

    def test_multiple_ambiguous_surfaces(self):
        warnings = check_gender_ambiguity(
            "\u05e9\u05dc\u05da \u05d5\u05d0\u05d7\u05e8\u05d9 \u05dc\u05da"
        )
        assert len(warnings) == 2

    def test_embedded_surface_not_flagged(self):
        # Surface form inside a longer word — should NOT be flagged.
        longer = "\u05de\u05e9\u05dc\u05da\u05dd"
        assert check_gender_ambiguity(longer) == []


# ---------------------------------------------------------------------------
# DialogueTurn integration
# ---------------------------------------------------------------------------


class TestDialogueTurnDefaults:
    def test_text_spoken_defaults_to_text(self):
        turn = DialogueTurn(speaker_id="AGG_001", text="\u05e9\u05dc\u05d5\u05dd", intensity=1)
        assert turn.text_spoken == "\u05e9\u05dc\u05d5\u05dd"

    def test_text_original_property(self):
        turn = DialogueTurn(speaker_id="AGG_001", text="\u05e9\u05dc\u05d5\u05dd", intensity=1)
        assert turn.text_original == "\u05e9\u05dc\u05d5\u05dd"

    def test_explicit_text_spoken_preserved(self):
        turn = DialogueTurn(
            speaker_id="AGG_001",
            text="\u05e9\u05dc\u05da",
            intensity=1,
            text_spoken="\u05e9\u05b6\u05c1\u05dc\u05b8\u05bc\u05da\u05b0",
            normalization_rules_triggered=["F2_POSS_SHEL"],
        )
        assert turn.text_spoken == "\u05e9\u05b6\u05c1\u05dc\u05b8\u05bc\u05da\u05b0"
        assert turn.text == "\u05e9\u05dc\u05da"

    def test_normalization_rules_default_empty(self):
        turn = DialogueTurn(speaker_id="AGG_001", text="\u05e9\u05dc\u05d5\u05dd", intensity=1)
        assert turn.normalization_rules_triggered == []


# ---------------------------------------------------------------------------
# disambiguate_turns — with explicit genders
# ---------------------------------------------------------------------------


class TestDisambiguateTurns:
    def _make_turns(self) -> list[DialogueTurn]:
        return [
            DialogueTurn(
                speaker_id="AGG_001", text="\u05de\u05d4 \u05e2\u05e9\u05d9\u05ea?", intensity=2
            ),
            DialogueTurn(
                speaker_id="VIC_001",
                text="\u05dc\u05da \u05d6\u05d4 \u05dc\u05d0 \u05de\u05e9\u05e0\u05d4.",
                intensity=1,
            ),
        ]

    def _roles(self) -> dict[str, str]:
        return {"AGG_001": "AGG", "VIC_001": "VIC"}

    def _genders(self) -> dict[str, str]:
        return {"AGG_001": "male", "VIC_001": "female"}

    def test_agg_addressing_female_vic(self):
        turns = disambiguate_turns(self._make_turns(), self._roles(), self._genders())
        agg_turn = next(t for t in turns if t.speaker_id == "AGG_001")
        # AGG (male) addressing VIC (female) — feminine 2P rules fire
        assert "F2_VERB_ASIT" in agg_turn.normalization_rules_triggered

    def test_vic_addressing_male_agg(self):
        turns = disambiguate_turns(self._make_turns(), self._roles(), self._genders())
        vic_turn = next(t for t in turns if t.speaker_id == "VIC_001")
        # VIC (female) addressing AGG (male) — masculine 2P rules fire
        assert "M2_PREP_LEKHA" in vic_turn.normalization_rules_triggered

    def test_original_turns_not_mutated(self):
        original = self._make_turns()
        original_texts = [t.text for t in original]
        disambiguate_turns(original, self._roles(), self._genders())
        assert [t.text for t in original] == original_texts

    def test_returns_same_length(self):
        turns = self._make_turns()
        result = disambiguate_turns(turns, self._roles(), self._genders())
        assert len(result) == len(turns)

    def test_empty_scene(self):
        assert disambiguate_turns([], {}) == []

    def test_single_speaker_scene_no_crash(self):
        turns = [DialogueTurn(speaker_id="AGG_001", text="\u05e9\u05dc\u05da", intensity=1)]
        result = disambiguate_turns(turns, {"AGG_001": "AGG"}, {"AGG_001": "male"})
        # Only one speaker — no addressee; no modification but QA gate flags it
        assert result[0].text_spoken == "\u05e9\u05dc\u05da"

    def test_three_speaker_agg_prefers_vic(self):
        turns = [
            DialogueTurn(
                speaker_id="AGG_001", text="\u05de\u05d4 \u05e2\u05e9\u05d9\u05ea?", intensity=2
            )
        ]
        roles = {"AGG_001": "AGG", "BYS_001": "BYS", "VIC_001": "VIC"}
        genders = {"AGG_001": "male", "BYS_001": "male", "VIC_001": "female"}
        result = disambiguate_turns(turns, roles, genders)
        agg_turn = result[0]
        assert "F2_VERB_ASIT" in agg_turn.normalization_rules_triggered

    def test_three_speaker_bys_gets_no_disambiguation_toward_vic(self):
        """BYS addressing AGG (male) should get masculine rules, not feminine."""
        turns = [
            DialogueTurn(
                speaker_id="BYS_001", text="\u05de\u05d4 \u05e2\u05e9\u05d9\u05ea?", intensity=1
            )
        ]
        roles = {"AGG_001": "AGG", "VIC_001": "VIC", "BYS_001": "BYS"}
        genders = {"AGG_001": "male", "VIC_001": "female", "BYS_001": "male"}
        result = disambiguate_turns(turns, roles, genders)
        # BYS→VIC (first non-self in priority order is VIC)
        bys_turn = result[0]
        # BYS addresses VIC (female) per priority — feminine rules fire
        assert "F2_VERB_ASIT" in bys_turn.normalization_rules_triggered

    def test_unknown_role_fallback(self):
        turns = [DialogueTurn(speaker_id="A", text="\u05e9\u05dc\u05da", intensity=1)]
        roles = {"A": "CALLER", "B": "WITNESS"}
        genders = {"A": "male", "B": "female"}
        result = disambiguate_turns(turns, roles, genders)
        # Addressee is B (female) — feminine rules fire
        assert "F2_POSS_SHEL" in result[0].normalization_rules_triggered


# ---------------------------------------------------------------------------
# disambiguate_turns — legacy fallback (no speaker_genders)
# ---------------------------------------------------------------------------


class TestLegacyFallback:
    """When speaker_genders is None, genders are inferred from roles."""

    def test_agg_assumed_male_vic_assumed_female(self):
        turns = [
            DialogueTurn(
                speaker_id="AGG_001", text="\u05de\u05d4 \u05e2\u05e9\u05d9\u05ea?", intensity=2
            ),
        ]
        roles = {"AGG_001": "AGG", "VIC_001": "VIC"}
        result = disambiguate_turns(turns, roles)  # no genders
        assert "F2_VERB_ASIT" in result[0].normalization_rules_triggered

    def test_vic_assumed_female_addressing_male_agg(self):
        turns = [
            DialogueTurn(
                speaker_id="VIC_001",
                text="\u05dc\u05da \u05d6\u05d4 \u05dc\u05d0 \u05de\u05e9\u05e0\u05d4.",
                intensity=1,
            ),
        ]
        roles = {"AGG_001": "AGG", "VIC_001": "VIC"}
        result = disambiguate_turns(turns, roles)  # no genders
        # VIC→AGG: AGG is assumed male, so masculine rules should fire
        assert "M2_PREP_LEKHA" in result[0].normalization_rules_triggered


# ---------------------------------------------------------------------------
# QA gate integration in disambiguate_turns
# ---------------------------------------------------------------------------


class TestQAGateInTurns:
    def test_ambiguous_forms_flagged_in_quality_gate_failures(self):
        """Turns with unknown addressee gender should have QA warnings."""
        turns = [DialogueTurn(speaker_id="A", text="\u05e9\u05dc\u05da", intensity=1)]
        # Only one speaker — no addressee gender can be determined.
        result = disambiguate_turns(turns, {"A": "AGG"}, {"A": "male"})
        # The unresolved \u05e9\u05dc\u05da should be flagged.
        assert any("gender_ambiguity" in w for w in result[0].quality_gate_failures)

    def test_resolved_forms_not_flagged(self):
        """After successful disambiguation, no QA warnings should appear."""
        turns = [DialogueTurn(speaker_id="AGG_001", text="\u05e9\u05dc\u05da", intensity=1)]
        roles = {"AGG_001": "AGG", "VIC_001": "VIC"}
        genders = {"AGG_001": "male", "VIC_001": "female"}
        result = disambiguate_turns(turns, roles, genders)
        assert not any("gender_ambiguity" in w for w in result[0].quality_gate_failures)

    def test_existing_gate_failures_preserved(self):
        """Pre-existing quality_gate_failures must not be overwritten."""
        turns = [
            DialogueTurn(
                speaker_id="AGG_001",
                text="\u05e9\u05dc\u05d5\u05dd",
                intensity=1,
                quality_gate_failures=["f0_check: too high"],
            )
        ]
        roles = {"AGG_001": "AGG", "VIC_001": "VIC"}
        genders = {"AGG_001": "male", "VIC_001": "female"}
        result = disambiguate_turns(turns, roles, genders)
        assert "f0_check: too high" in result[0].quality_gate_failures


# ---------------------------------------------------------------------------
# Lexicon consistency checks
# ---------------------------------------------------------------------------


class TestLexiconConsistency:
    def test_no_duplicate_rule_ids(self):
        ids = [e.rule_id for e in _LEXICON]
        assert len(ids) == len(set(ids)), (
            f"Duplicate rule IDs: {[x for x in ids if ids.count(x) > 1]}"
        )

    def test_feminine_and_masculine_cover_same_surfaces(self):
        """Every surface form in the feminine lexicon should have a masculine counterpart."""
        fem_surfaces = {e.surface for e in _LEXICON_2P_FEMININE}
        masc_surfaces = {e.surface for e in _LEXICON_2P_MASCULINE}
        assert fem_surfaces == masc_surfaces, (
            f"Mismatched surfaces: fem-only={fem_surfaces - masc_surfaces}, "
            f"masc-only={masc_surfaces - fem_surfaces}"
        )

    def test_feminine_entries_target_female(self):
        for entry in _LEXICON_2P_FEMININE:
            assert entry.target_gender == "female", f"{entry.rule_id} has wrong target_gender"

    def test_masculine_entries_target_male(self):
        for entry in _LEXICON_2P_MASCULINE:
            assert entry.target_gender == "male", f"{entry.rule_id} has wrong target_gender"

    def test_spoken_form_differs_from_surface(self):
        for entry in _LEXICON:
            assert entry.spoken_form != entry.surface, (
                f"{entry.rule_id}: spoken_form is identical to surface — no disambiguation"
            )

    def test_feminine_and_masculine_produce_different_spoken_forms(self):
        """For each surface, the fem and masc spoken forms must differ."""
        fem_by_surface = {e.surface: e.spoken_form for e in _LEXICON_2P_FEMININE}
        masc_by_surface = {e.surface: e.spoken_form for e in _LEXICON_2P_MASCULINE}
        for surface in fem_by_surface:
            assert fem_by_surface[surface] != masc_by_surface[surface], (
                f"Surface '{surface}': fem and masc spoken forms are identical"
            )
