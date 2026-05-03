"""Hebrew gender disambiguation for TTS text normalization (pipeline M1).

Azure TTS produces inconsistent gender readings for morphologically ambiguous
unvocalized Hebrew second-person forms.  When the addressee gender is known
from the scene config, we insert niqqud (Hebrew vowel diacritics) that force
the TTS engine to produce the correct gendered pronunciation.

Strategy: rule-based lexicon substitution keyed on **addressee gender** for
2nd-person forms (pronouns, prepositions, past-tense verbs).  Each lexicon
entry targets a surface form where the unvocalized consonant string is
identical for masculine and feminine, and replaces it with a fully vocalized
form that is phonetically unambiguous.

Only ``DialogueTurn.text_spoken`` is modified; the original LLM output
(``DialogueTurn.text``) is preserved unchanged for auditing and caching.

Reference: docs/audio_generation_v3_design.md §4.1, M1

.. note:: Rule IDs changed in this version.

   Feminine entries were renamed from ``POSS_SHEL`` to ``F2_POSS_SHEL`` (etc.)
   to distinguish them from the new masculine ``M2_*`` entries.  Update any
   log-analysis or dashboard filters that match on the old ID format.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from synthbanshee.script.types import DialogueTurn


class LexiconEntry(NamedTuple):
    """One disambiguation rule mapping an ambiguous surface form to a vocalized replacement."""

    rule_id: str
    surface: str  # unvocalized form to match in source text
    spoken_form: str  # niqqud-vocalized form for TTS; forces gendered reading
    target_gender: str  # "female" or "male" — the gender this form is correct for
    description: str  # human-readable note


class _AmbiguousForm(NamedTuple):
    """Single source-of-truth entry for a gender-ambiguous Hebrew surface form.

    Both feminine and masculine ``LexiconEntry`` objects are generated from
    each ``_AmbiguousForm`` — eliminating duplication between the two
    lexicons and guaranteeing that every surface form has both gendered
    variants.
    """

    base_id: str
    surface: str
    female_spoken: str
    male_spoken: str
    description: str  # short gloss, e.g. "'yours'"


# ---------------------------------------------------------------------------
# Lexicon data — single source of truth
# ---------------------------------------------------------------------------
#
# Sources: Shay's direct listening feedback on debug_run_1, listening test
# 2026-05-03 (issue #63), Hebrew morphological reference.
#
# Each entry targets a form where:
#   (a) the written unvocalized text is identical for masc and fem, and
#   (b) Azure TTS may read the wrong gendered form without niqqud.

_AMBIGUOUS_FORMS: tuple[_AmbiguousForm, ...] = (
    # --- Possessive / object pronouns ---
    #              base_id         surface   female_spoken            male_spoken              description
    _AmbiguousForm(
        "POSS_SHEL",
        "\u05e9\u05dc\u05da",
        "\u05e9\u05b6\u05c1\u05dc\u05b8\u05bc\u05da\u05b0",
        "\u05e9\u05b6\u05c1\u05dc\u05b0\u05bc\u05da\u05b8",
        "'yours' — shelakh (f) / shelkha (m)",
    ),
    _AmbiguousForm(
        "PRON_OTKH",
        "\u05d0\u05d5\u05ea\u05da",
        "\u05d0\u05d5\u05b9\u05ea\u05b8\u05da\u05b0",
        "\u05d0\u05d5\u05b9\u05ea\u05b0\u05da\u05b8",
        "'you' (obj) — otakh (f) / otkha (m)",
    ),
    _AmbiguousForm(
        "PRON_ITKH",
        "\u05d0\u05ea\u05da",
        "\u05d0\u05b4\u05ea\u05b8\u05bc\u05da\u05b0",
        "\u05d0\u05b4\u05ea\u05b0\u05bc\u05da\u05b8",
        "'with you' — itakh (f) / itkha (m)",
    ),
    # --- Prepositional clitics ---
    _AmbiguousForm(
        "PREP_LAKH",
        "\u05dc\u05da",
        "\u05dc\u05b8\u05da\u05b0",
        "\u05dc\u05b0\u05da\u05b8",
        "'to/for you' — lakh (f) / lekha (m)",
    ),
    _AmbiguousForm(
        "PREP_BAKH",
        "\u05d1\u05da",
        "\u05d1\u05b8\u05bc\u05da\u05b0",
        "\u05d1\u05b0\u05bc\u05da\u05b8",
        "'in/at you' — bakh (f) / bekha (m)",
    ),
    _AmbiguousForm(
        "PREP_MIMEKH",
        "\u05de\u05de\u05da",
        "\u05de\u05b4\u05de\u05b5\u05bc\u05da\u05b0",
        "\u05de\u05b4\u05de\u05b0\u05bc\u05da\u05b8",
        "'from you' — mimekh (f) / mimkha (m)",
    ),
    _AmbiguousForm(
        "PREP_ALAIKH",
        "\u05e2\u05dc\u05d9\u05da",
        "\u05e2\u05b8\u05dc\u05b7\u05d9\u05b4\u05da\u05b0",
        "\u05e2\u05b8\u05dc\u05b6\u05d9\u05da\u05b8",
        "'on/about you' — alaikh (f) / alekha (m)",
    ),
    _AmbiguousForm(
        "PREP_ELAIKH",
        "\u05d0\u05dc\u05d9\u05da",
        "\u05d0\u05b5\u05dc\u05b7\u05d9\u05b4\u05da\u05b0",
        "\u05d0\u05b5\u05dc\u05b6\u05d9\u05da\u05b8",
        "'toward you' — elaikh (f) / elekha (m)",
    ),
    # --- Past-tense 2sg verbs ---
    # Fem ends /-t/ (shva on tav), masc /-ta/ (kamatz on tav)
    _AmbiguousForm(
        "VERB_HALAKHT",
        "\u05d4\u05dc\u05db\u05ea",
        "\u05d4\u05b8\u05dc\u05b7\u05db\u05b0\u05ea\u05b0\u05bc",
        "\u05d4\u05b8\u05dc\u05b7\u05db\u05b0\u05ea\u05b8\u05bc",
        "'you went' — halakht (f) / halakhta (m)",
    ),
    _AmbiguousForm(
        "VERB_ASIT",
        "\u05e2\u05e9\u05d9\u05ea",
        "\u05e2\u05b8\u05e9\u05c2\u05b4\u05d9\u05ea\u05b0",
        "\u05e2\u05b8\u05e9\u05c2\u05b4\u05d9\u05ea\u05b8",
        "'you did/made' — asit (f) / asita (m)",
    ),
    _AmbiguousForm(
        "VERB_KHASHAVT",
        "\u05d7\u05e9\u05d1\u05ea",
        "\u05d7\u05b8\u05e9\u05c1\u05b7\u05d1\u05b0\u05ea\u05b0\u05bc",
        "\u05d7\u05b8\u05e9\u05c1\u05b7\u05d1\u05b0\u05ea\u05b8\u05bc",
        "'you thought' — khashavt (f) / khashavta (m)",
    ),
    _AmbiguousForm(
        "VERB_DIBART",
        "\u05d3\u05d9\u05d1\u05e8\u05ea",
        "\u05d3\u05b4\u05bc\u05d9\u05d1\u05b7\u05bc\u05e8\u05b0\u05ea\u05b0\u05bc",
        "\u05d3\u05b4\u05bc\u05d9\u05d1\u05b7\u05bc\u05e8\u05b0\u05ea\u05b8\u05bc",
        "'you spoke' — dibart (f) / dibarta (m)",
    ),
    _AmbiguousForm(
        "VERB_YADAT",
        "\u05d9\u05d3\u05e2\u05ea",
        "\u05d9\u05b8\u05d3\u05b7\u05e2\u05b0\u05ea\u05b0\u05bc",
        "\u05d9\u05b8\u05d3\u05b7\u05e2\u05b0\u05ea\u05b8\u05bc",
        "'you knew' — yadat (f) / yadata (m)",
    ),
    _AmbiguousForm(
        "VERB_HAYIT",
        "\u05d4\u05d9\u05d9\u05ea",
        "\u05d4\u05b8\u05d9\u05b4\u05d9\u05ea\u05b0",
        "\u05d4\u05b8\u05d9\u05b4\u05d9\u05ea\u05b8",
        "'you were' — hayit (f) / hayita (m)",
    ),
    _AmbiguousForm(
        "VERB_RATSIT",
        "\u05e8\u05e6\u05d9\u05ea",
        "\u05e8\u05b8\u05e6\u05b4\u05d9\u05ea\u05b0",
        "\u05e8\u05b8\u05e6\u05b4\u05d9\u05ea\u05b8",
        "'you wanted' — ratsit (f) / ratsita (m)",
    ),
    _AmbiguousForm(
        "VERB_ANIT",
        "\u05e2\u05e0\u05d9\u05ea",
        "\u05e2\u05b8\u05e0\u05b4\u05d9\u05ea\u05b0",
        "\u05e2\u05b8\u05e0\u05b4\u05d9\u05ea\u05b8",
        "'you answered' — anit (f) / anita (m)",
    ),
    _AmbiguousForm(
        "VERB_AMART",
        "\u05d0\u05de\u05e8\u05ea",
        "\u05d0\u05b8\u05de\u05b7\u05e8\u05b0\u05ea\u05b0\u05bc",
        "\u05d0\u05b8\u05de\u05b7\u05e8\u05b0\u05ea\u05b8\u05bc",
        "'you said' — amart (f) / amarta (m)",
    ),
    _AmbiguousForm(
        "VERB_SHAMAT",
        "\u05e9\u05de\u05e2\u05ea",
        "\u05e9\u05c1\u05b8\u05de\u05b7\u05e2\u05b0\u05ea\u05b0\u05bc",
        "\u05e9\u05c1\u05b8\u05de\u05b7\u05e2\u05b0\u05ea\u05b8\u05bc",
        "'you heard' — shamat (f) / shamata (m)",
    ),
    _AmbiguousForm(
        "VERB_RAIT",
        "\u05e8\u05d0\u05d9\u05ea",
        "\u05e8\u05b8\u05d0\u05b4\u05d9\u05ea\u05b0",
        "\u05e8\u05b8\u05d0\u05b4\u05d9\u05ea\u05b8",
        "'you saw' — ra'it (f) / ra'ita (m)",
    ),
    _AmbiguousForm(
        "VERB_LAKAKHT",
        "\u05dc\u05e7\u05d7\u05ea",
        "\u05dc\u05b8\u05e7\u05b7\u05d7\u05b0\u05ea\u05b0\u05bc",
        "\u05dc\u05b8\u05e7\u05b7\u05d7\u05b0\u05ea\u05b8\u05bc",
        "'you took' — lakakht (f) / lakakhta (m)",
    ),
    _AmbiguousForm(
        "VERB_NATATT",
        "\u05e0\u05ea\u05ea",
        "\u05e0\u05b8\u05ea\u05b7\u05ea\u05b0\u05bc",
        "\u05e0\u05b8\u05ea\u05b7\u05ea\u05b8\u05bc",
        "'you gave' — natatt (f) / natata (m)",
    ),
    _AmbiguousForm(
        "VERB_BAT",
        "\u05d1\u05d0\u05ea",
        "\u05d1\u05b8\u05d0\u05ea\u05b0",
        "\u05d1\u05b8\u05d0\u05ea\u05b8",
        "'you came' — bat (f) / bata (m)",
    ),
    _AmbiguousForm(
        "VERB_YATSAT",
        "\u05d9\u05e6\u05d0\u05ea",
        "\u05d9\u05b8\u05e6\u05b8\u05d0\u05ea\u05b0",
        "\u05d9\u05b8\u05e6\u05b8\u05d0\u05ea\u05b8",
        "'you went out' — yatsat (f) / yatsata (m)",
    ),
    _AmbiguousForm(
        "VERB_SHAKHAKHT",
        "\u05e9\u05db\u05d7\u05ea",
        "\u05e9\u05c1\u05b8\u05db\u05b7\u05d7\u05b0\u05ea\u05b0\u05bc",
        "\u05e9\u05c1\u05b8\u05db\u05b7\u05d7\u05b0\u05ea\u05b8\u05bc",
        "'you forgot' — shakhakht (f) / shakhakhta (m)",
    ),
    _AmbiguousForm(
        "VERB_HITKHALAT",
        "\u05d4\u05ea\u05d7\u05dc\u05ea",
        "\u05d4\u05b4\u05ea\u05b0\u05d7\u05b7\u05dc\u05b0\u05ea\u05b0\u05bc",
        "\u05d4\u05b4\u05ea\u05b0\u05d7\u05b7\u05dc\u05b0\u05ea\u05b8\u05bc",
        "'you started' — hitkhalat (f) / hitkhalta (m)",
    ),
    _AmbiguousForm(
        "VERB_HIFSAKT",
        "\u05d4\u05e4\u05e1\u05e7\u05ea",
        "\u05d4\u05b4\u05e4\u05b0\u05e1\u05b7\u05e7\u05b0\u05ea\u05b0\u05bc",
        "\u05d4\u05b4\u05e4\u05b0\u05e1\u05b7\u05e7\u05b0\u05ea\u05b8\u05bc",
        "'you stopped' — hifsakt (f) / hifsakta (m)",
    ),
)

# ---------------------------------------------------------------------------
# Generated per-gender lexicons (from _AMBIGUOUS_FORMS)
# ---------------------------------------------------------------------------

_LEXICON_2P_FEMININE: tuple[LexiconEntry, ...] = tuple(
    LexiconEntry(
        rule_id=f"F2_{f.base_id}",
        surface=f.surface,
        spoken_form=f.female_spoken,
        target_gender="female",
        description=f.description,
    )
    for f in _AMBIGUOUS_FORMS
)

_LEXICON_2P_MASCULINE: tuple[LexiconEntry, ...] = tuple(
    LexiconEntry(
        rule_id=f"M2_{f.base_id}",
        surface=f.surface,
        spoken_form=f.male_spoken,
        target_gender="male",
        description=f.description,
    )
    for f in _AMBIGUOUS_FORMS
)

# Combined lexicon for parametrized tests.
_LEXICON: tuple[LexiconEntry, ...] = _LEXICON_2P_FEMININE + _LEXICON_2P_MASCULINE

# ---------------------------------------------------------------------------
# Surface forms that remain gender-ambiguous after disambiguation
# ---------------------------------------------------------------------------
# Used by the QA gate to flag turns that may still contain wrong-gender forms.
# These are common 2nd-person surface forms whose unvocalized spelling is
# shared between masculine and feminine.  If any of these survive in
# text_spoken without niqqud, they are suspicious.
#
# KNOWN LIMITATION: some surface forms overlap with unrelated Hebrew words
# (e.g. "לך" can also be the imperative "go!", "באת" overlaps with a noun
# form).  The gate is therefore a *heuristic* that flags candidates for
# review, not a definitive error detector.  False positives are expected.
_AMBIGUOUS_SURFACES: frozenset[str] = frozenset(f.surface for f in _AMBIGUOUS_FORMS)

# ---------------------------------------------------------------------------
# Regex compilation
# ---------------------------------------------------------------------------

_HEB_TOKEN_CHAR = r"[\u0591-\u05C7\u05D0-\u05EA]"


def _compile_entries(
    entries: tuple[LexiconEntry, ...],
) -> tuple[tuple[LexiconEntry, re.Pattern[str]], ...]:
    return tuple(
        (
            entry,
            re.compile(
                r"(?<!"
                + _HEB_TOKEN_CHAR
                + r")"
                + re.escape(entry.surface)
                + r"(?!"
                + _HEB_TOKEN_CHAR
                + r")"
            ),
        )
        for entry in entries
    )


_COMPILED_2P_FEMININE = _compile_entries(_LEXICON_2P_FEMININE)
_COMPILED_2P_MASCULINE = _compile_entries(_LEXICON_2P_MASCULINE)

# Pre-compile a single scanner regex for the QA gate: matches any unvocalized
# ambiguous surface form that was NOT replaced by a niqqud-bearing form.
_AMBIGUOUS_SCANNER = re.compile(
    r"(?<!"
    + _HEB_TOKEN_CHAR
    + r")(?:"
    + "|".join(re.escape(s) for s in sorted(_AMBIGUOUS_SURFACES, key=len, reverse=True))
    + r")(?!"
    + _HEB_TOKEN_CHAR
    + r")"
)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class NormalizationResult:
    """Outcome of gender disambiguation for one utterance.

    Attributes:
        text_spoken: Modified text ready to send to TTS.  Identical to the
            input when no rules fire.
        normalization_rules_triggered: Ordered list of ``LexiconEntry.rule_id``
            values for every substitution that was applied.  Empty when the
            input required no changes.
    """

    text_spoken: str
    normalization_rules_triggered: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def disambiguate_for_speaker(
    text: str,
    addressee_gender: str,
) -> NormalizationResult:
    """Apply gender corrections based on addressee gender.

    Second-person forms (pronouns, prepositions, verbs) are selected based
    on the *addressee* gender.  The rules insert niqqud to force the TTS
    engine to read the correct gendered form.

    Args:
        text: UTF-8 Hebrew source text (``DialogueTurn.text``).
        addressee_gender: Gender of the primary addressee (``"male"`` or
            ``"female"``).

    Returns:
        :class:`NormalizationResult` carrying the (possibly modified)
        ``text_spoken`` and the list of rule IDs that fired.
    """
    # Select the correct compiled lexicon based on addressee gender.
    if addressee_gender == "female":
        compiled = _COMPILED_2P_FEMININE
    elif addressee_gender == "male":
        compiled = _COMPILED_2P_MASCULINE
    else:
        return NormalizationResult(text_spoken=text)

    result = text
    triggered: list[str] = []
    for entry, pattern in compiled:
        new_text, n = pattern.subn(entry.spoken_form, result)
        if n:
            result = new_text
            triggered.append(entry.rule_id)

    return NormalizationResult(
        text_spoken=result,
        normalization_rules_triggered=triggered,
    )


def check_gender_ambiguity(text_spoken: str) -> list[str]:
    """Flag remaining gender-ambiguous surface forms after disambiguation.

    Scans *text_spoken* for unvocalized tokens that appear in the lexicon
    but were not replaced (e.g. because the addressee gender was unknown).

    .. note::

       This is a **heuristic** gate.  Some surface forms overlap with
       unrelated Hebrew words (e.g. "\u05dc\u05da" = imperative "go!", "\u05d1\u05d0\u05ea" can be a
       noun).  False positives are expected — treat warnings as candidates
       for review, not confirmed errors.

    Returns:
        List of warning strings, one per ambiguous token found.  Empty if
        no suspicious forms remain.
    """
    warnings: list[str] = []
    for match in _AMBIGUOUS_SCANNER.finditer(text_spoken):
        token = match.group()
        warnings.append(f"gender_ambiguity: unresolved '{token}' at position {match.start()}")
    return warnings


def disambiguate_turns(
    turns: list[DialogueTurn],
    speaker_roles: dict[str, str],
    speaker_genders: Mapping[str, str] | None = None,
) -> list[DialogueTurn]:
    """Apply gender disambiguation to every turn in a scene.

    For each turn the addressee is inferred heuristically from the other
    speakers in *speaker_roles*.  Gender information is looked up from
    *speaker_genders* (mapping ``{speaker_id: "male"|"female"}``).

    When *speaker_genders* is ``None``, falls back to legacy behaviour:
    AGG is assumed male, VIC is assumed female.

    The original ``DialogueTurn`` objects are never mutated.  New instances
    are returned with ``text_spoken`` and ``normalization_rules_triggered``
    populated.

    Args:
        turns: Ordered list of turns from :class:`ScriptGenerator`.
        speaker_roles: Mapping ``{speaker_id: role}`` for all speakers in
            the scene (built from ``SceneConfig.speakers``).
        speaker_genders: Mapping ``{speaker_id: gender}`` for all speakers.
            When ``None``, gender is inferred from role (legacy fallback).

    Returns:
        New list of :class:`DialogueTurn` with disambiguation applied.
    """

    _ROLE_PRIORITY = ["VIC", "AGG", "BYS"]

    # Legacy gender inference from role.
    _ROLE_GENDER_FALLBACK: dict[str, str] = {
        "AGG": "male",
        "VIC": "female",
    }

    def _gender(speaker_id: str) -> str:
        if speaker_genders and speaker_id in speaker_genders:
            return speaker_genders[speaker_id]
        role = speaker_roles.get(speaker_id, "UNK")
        return _ROLE_GENDER_FALLBACK.get(role, "")

    def _addressee_id(speaker_id: str) -> str:
        """Return the speaker_id of the most likely addressee."""
        my_role = speaker_roles.get(speaker_id, "UNK")
        # Build candidates: other speakers ordered by role priority.
        others = [(sid, role) for sid, role in speaker_roles.items() if sid != speaker_id]
        for candidate_role in _ROLE_PRIORITY:
            if candidate_role == my_role:
                continue
            for sid, role in others:
                if role == candidate_role:
                    return sid
        # Fall back to first other speaker.
        for sid, _role in others:
            return sid
        return ""

    result: list[DialogueTurn] = []
    for turn in turns:
        addr_id = _addressee_id(turn.speaker_id)
        addr_gender = _gender(addr_id) if addr_id else ""

        norm = disambiguate_for_speaker(
            turn.text_spoken or turn.text,
            addr_gender,
        )

        # QA gate: flag remaining ambiguous forms.
        qa_warnings = check_gender_ambiguity(norm.text_spoken)
        gate_failures = list(turn.quality_gate_failures) + qa_warnings

        result.append(
            dataclasses.replace(
                turn,
                text_spoken=norm.text_spoken,
                normalization_rules_triggered=norm.normalization_rules_triggered,
                quality_gate_failures=gate_failures,
            )
        )
    return result
