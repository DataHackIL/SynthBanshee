"""Hebrew gender disambiguation for TTS text normalization (pipeline M1).

Azure TTS produces inconsistent gender readings for morphologically ambiguous
unvocalized Hebrew second-person forms.  When the addressee gender is known
from the scene config, we insert niqqud (Hebrew vowel diacritics) that force
the TTS engine to produce the correct gendered pronunciation.

Strategy: rule-based lexicon substitution keyed on **addressee gender** (for
2nd-person forms) and **speaker gender** (for 1st-person forms).  Each lexicon
entry targets a surface form where the unvocalized consonant string is
identical for masculine and feminine, and replaces it with a fully vocalized
form that is phonetically unambiguous.

Only ``DialogueTurn.text_spoken`` is modified; the original LLM output
(``DialogueTurn.text``) is preserved unchanged for auditing and caching.

Reference: docs/audio_generation_v3_design.md §4.1, M1
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
    person: str  # "2" = addressee-keyed, "1" = speaker-keyed
    description: str  # human-readable note


# ---------------------------------------------------------------------------
# Lexicon — second-person forms keyed on ADDRESSEE gender
# ---------------------------------------------------------------------------
#
# Sources: Shay's direct listening feedback on debug_run_1, listening test
# 2026-05-03 (issue #63), Hebrew morphological reference.
#
# Each entry targets a form where:
#   (a) the written unvocalized text is identical for masc and fem, and
#   (b) Azure TTS may read the wrong gendered form without niqqud.

_LEXICON_2P_FEMININE: tuple[LexiconEntry, ...] = (
    # --- Possessive / object pronouns (addressee is female) ---
    LexiconEntry(
        "F2_POSS_SHEL",
        "\u05e9\u05dc\u05da",
        "\u05e9\u05b6\u05c1\u05dc\u05b8\u05bc\u05da\u05b0",
        "female",
        "2",
        "shelakh (f) \u2190 shelkha (m default); 'yours'",
    ),
    LexiconEntry(
        "F2_PRON_OTKH",
        "\u05d0\u05d5\u05ea\u05da",
        "\u05d0\u05d5\u05b9\u05ea\u05b8\u05da\u05b0",
        "female",
        "2",
        "otakh (f) \u2190 otkha (m default); 'you' (direct object)",
    ),
    LexiconEntry(
        "F2_PRON_ITKH",
        "\u05d0\u05ea\u05da",
        "\u05d0\u05b4\u05ea\u05b8\u05bc\u05da\u05b0",
        "female",
        "2",
        "itakh (f) \u2190 itkha (m default); 'with you'",
    ),
    # --- Prepositional clitics (addressee is female) ---
    LexiconEntry(
        "F2_PREP_LAKH",
        "\u05dc\u05da",
        "\u05dc\u05b8\u05da\u05b0",
        "female",
        "2",
        "lakh (f) \u2190 lekha (m default); 'to/for you'",
    ),
    LexiconEntry(
        "F2_PREP_BAKH",
        "\u05d1\u05da",
        "\u05d1\u05b8\u05bc\u05da\u05b0",
        "female",
        "2",
        "bakh (f) \u2190 bekha (m default); 'in/at you'",
    ),
    LexiconEntry(
        "F2_PREP_MIMEKH",
        "\u05de\u05de\u05da",
        "\u05de\u05b4\u05de\u05b5\u05bc\u05da\u05b0",
        "female",
        "2",
        "mimekh (f) \u2190 mimkha (m default); 'from you'",
    ),
    LexiconEntry(
        "F2_PREP_ALAIKH",
        "\u05e2\u05dc\u05d9\u05da",
        "\u05e2\u05b8\u05dc\u05b7\u05d9\u05b4\u05da\u05b0",
        "female",
        "2",
        "alaikh (f) \u2190 alekha (m default); 'on/about you'",
    ),
    LexiconEntry(
        "F2_PREP_ELAIKH",
        "\u05d0\u05dc\u05d9\u05da",
        "\u05d0\u05b5\u05dc\u05b7\u05d9\u05b4\u05da\u05b0",
        "female",
        "2",
        "elaikh (f) \u2190 elekha (m default); 'toward you'",
    ),
    # --- Past-tense 2sg verbs (addressee is female) ---
    # Fem ends /-t/ (shva on tav), masc /-ta/ (kamatz on tav)
    LexiconEntry(
        "F2_VERB_HALAKHT",
        "\u05d4\u05dc\u05db\u05ea",
        "\u05d4\u05b8\u05dc\u05b7\u05db\u05b0\u05ea\u05b0\u05bc",
        "female",
        "2",
        "halakht (f) \u2190 halakhta (m default); 'you went'",
    ),
    LexiconEntry(
        "F2_VERB_ASIT",
        "\u05e2\u05e9\u05d9\u05ea",
        "\u05e2\u05b8\u05e9\u05c2\u05b4\u05d9\u05ea\u05b0",
        "female",
        "2",
        "asit (f) \u2190 asita (m default); 'you did/made'",
    ),
    LexiconEntry(
        "F2_VERB_KHASHAVT",
        "\u05d7\u05e9\u05d1\u05ea",
        "\u05d7\u05b8\u05e9\u05c1\u05b7\u05d1\u05b0\u05ea\u05b0\u05bc",
        "female",
        "2",
        "khashavt (f) \u2190 khashavta (m default); 'you thought'",
    ),
    LexiconEntry(
        "F2_VERB_DIBART",
        "\u05d3\u05d9\u05d1\u05e8\u05ea",
        "\u05d3\u05b4\u05bc\u05d9\u05d1\u05b7\u05bc\u05e8\u05b0\u05ea\u05b0\u05bc",
        "female",
        "2",
        "dibart (f) \u2190 dibarta (m default); 'you spoke'",
    ),
    LexiconEntry(
        "F2_VERB_YADAT",
        "\u05d9\u05d3\u05e2\u05ea",
        "\u05d9\u05b8\u05d3\u05b7\u05e2\u05b0\u05ea\u05b0\u05bc",
        "female",
        "2",
        "yadat (f) \u2190 yadata (m default); 'you knew'",
    ),
    LexiconEntry(
        "F2_VERB_HAYIT",
        "\u05d4\u05d9\u05d9\u05ea",
        "\u05d4\u05b8\u05d9\u05b4\u05d9\u05ea\u05b0",
        "female",
        "2",
        "hayit (f) \u2190 hayita (m default); 'you were'",
    ),
    LexiconEntry(
        "F2_VERB_RATSIT",
        "\u05e8\u05e6\u05d9\u05ea",
        "\u05e8\u05b8\u05e6\u05b4\u05d9\u05ea\u05b0",
        "female",
        "2",
        "ratsit (f) \u2190 ratsita (m default); 'you wanted'",
    ),
    # --- Additional past-tense 2sg verbs (addressee is female) ---
    LexiconEntry(
        "F2_VERB_ANIT",
        "\u05e2\u05e0\u05d9\u05ea",
        "\u05e2\u05b8\u05e0\u05b4\u05d9\u05ea\u05b0",
        "female",
        "2",
        "anit (f) \u2190 anita (m default); 'you answered'",
    ),
    LexiconEntry(
        "F2_VERB_AMART",
        "\u05d0\u05de\u05e8\u05ea",
        "\u05d0\u05b8\u05de\u05b7\u05e8\u05b0\u05ea\u05b0\u05bc",
        "female",
        "2",
        "amart (f) \u2190 amarta (m default); 'you said'",
    ),
    LexiconEntry(
        "F2_VERB_SHAMAT",
        "\u05e9\u05de\u05e2\u05ea",
        "\u05e9\u05c1\u05b8\u05de\u05b7\u05e2\u05b0\u05ea\u05b0\u05bc",
        "female",
        "2",
        "shamat (f) \u2190 shamata (m default); 'you heard'",
    ),
    LexiconEntry(
        "F2_VERB_RAIT",
        "\u05e8\u05d0\u05d9\u05ea",
        "\u05e8\u05b8\u05d0\u05b4\u05d9\u05ea\u05b0",
        "female",
        "2",
        "ra'it (f) \u2190 ra'ita (m default); 'you saw'",
    ),
    LexiconEntry(
        "F2_VERB_LAKAKHT",
        "\u05dc\u05e7\u05d7\u05ea",
        "\u05dc\u05b8\u05e7\u05b7\u05d7\u05b0\u05ea\u05b0\u05bc",
        "female",
        "2",
        "lakakht (f) \u2190 lakakhta (m default); 'you took'",
    ),
    LexiconEntry(
        "F2_VERB_NATATT",
        "\u05e0\u05ea\u05ea",
        "\u05e0\u05b8\u05ea\u05b7\u05ea\u05b0\u05bc",
        "female",
        "2",
        "natatt (f) \u2190 natata (m default); 'you gave'",
    ),
    LexiconEntry(
        "F2_VERB_BAT",
        "\u05d1\u05d0\u05ea",
        "\u05d1\u05b8\u05d0\u05ea\u05b0",
        "female",
        "2",
        "bat (f) \u2190 bata (m default); 'you came'",
    ),
    LexiconEntry(
        "F2_VERB_YATSAT",
        "\u05d9\u05e6\u05d0\u05ea",
        "\u05d9\u05b8\u05e6\u05b8\u05d0\u05ea\u05b0",
        "female",
        "2",
        "yatsat (f) \u2190 yatsata (m default); 'you went out'",
    ),
    LexiconEntry(
        "F2_VERB_SHAKHAKHT",
        "\u05e9\u05db\u05d7\u05ea",
        "\u05e9\u05c1\u05b8\u05db\u05b7\u05d7\u05b0\u05ea\u05b0\u05bc",
        "female",
        "2",
        "shakhakht (f) \u2190 shakhakhta (m default); 'you forgot'",
    ),
    LexiconEntry(
        "F2_VERB_HITKHALAT",
        "\u05d4\u05ea\u05d7\u05dc\u05ea",
        "\u05d4\u05b4\u05ea\u05b0\u05d7\u05b7\u05dc\u05b0\u05ea\u05b0\u05bc",
        "female",
        "2",
        "hitkhalat (f) \u2190 hitkhalta (m default); 'you started'",
    ),
    LexiconEntry(
        "F2_VERB_HIFSAKT",
        "\u05d4\u05e4\u05e1\u05e7\u05ea",
        "\u05d4\u05b4\u05e4\u05b0\u05e1\u05b7\u05e7\u05b0\u05ea\u05b0\u05bc",
        "female",
        "2",
        "hifsakt (f) \u2190 hifsakta (m default); 'you stopped'",
    ),
)

_LEXICON_2P_MASCULINE: tuple[LexiconEntry, ...] = (
    # --- Possessive / object pronouns (addressee is male) ---
    LexiconEntry(
        "M2_POSS_SHEL",
        "\u05e9\u05dc\u05da",
        "\u05e9\u05b6\u05c1\u05dc\u05b0\u05bc\u05da\u05b8",
        "male",
        "2",
        "shelkha (m) \u2190 shelakh (f); 'yours'",
    ),
    LexiconEntry(
        "M2_PRON_OTKH",
        "\u05d0\u05d5\u05ea\u05da",
        "\u05d0\u05d5\u05b9\u05ea\u05b0\u05da\u05b8",
        "male",
        "2",
        "otkha (m) \u2190 otakh (f); 'you' (direct object)",
    ),
    LexiconEntry(
        "M2_PRON_ITKH",
        "\u05d0\u05ea\u05da",
        "\u05d0\u05b4\u05ea\u05b0\u05bc\u05da\u05b8",
        "male",
        "2",
        "itkha (m) \u2190 itakh (f); 'with you'",
    ),
    # --- Prepositional clitics (addressee is male) ---
    LexiconEntry(
        "M2_PREP_LEKHA",
        "\u05dc\u05da",
        "\u05dc\u05b0\u05da\u05b8",
        "male",
        "2",
        "lekha (m) \u2190 lakh (f); 'to/for you'",
    ),
    LexiconEntry(
        "M2_PREP_BEKHA",
        "\u05d1\u05da",
        "\u05d1\u05b0\u05bc\u05da\u05b8",
        "male",
        "2",
        "bekha (m) \u2190 bakh (f); 'in/at you'",
    ),
    LexiconEntry(
        "M2_PREP_MIMKHA",
        "\u05de\u05de\u05da",
        "\u05de\u05b4\u05de\u05b0\u05bc\u05da\u05b8",
        "male",
        "2",
        "mimkha (m) \u2190 mimekh (f); 'from you'",
    ),
    LexiconEntry(
        "M2_PREP_ALEKHA",
        "\u05e2\u05dc\u05d9\u05da",
        "\u05e2\u05b8\u05dc\u05b6\u05d9\u05da\u05b8",
        "male",
        "2",
        "alekha (m) \u2190 alaikh (f); 'on/about you'",
    ),
    LexiconEntry(
        "M2_PREP_ELEKHA",
        "\u05d0\u05dc\u05d9\u05da",
        "\u05d0\u05b5\u05dc\u05b6\u05d9\u05da\u05b8",
        "male",
        "2",
        "elekha (m) \u2190 elaikh (f); 'toward you'",
    ),
    # --- Past-tense 2sg verbs (addressee is male) ---
    # Masc ends /-ta/ (kamatz+dagesh on tav), fem /-t/ (shva on tav)
    LexiconEntry(
        "M2_VERB_HALAKHTA",
        "\u05d4\u05dc\u05db\u05ea",
        "\u05d4\u05b8\u05dc\u05b7\u05db\u05b0\u05ea\u05b8\u05bc",
        "male",
        "2",
        "halakhta (m) \u2190 halakht (f); 'you went'",
    ),
    LexiconEntry(
        "M2_VERB_ASITA",
        "\u05e2\u05e9\u05d9\u05ea",
        "\u05e2\u05b8\u05e9\u05c2\u05b4\u05d9\u05ea\u05b8",
        "male",
        "2",
        "asita (m) \u2190 asit (f); 'you did/made'",
    ),
    LexiconEntry(
        "M2_VERB_KHASHAVTA",
        "\u05d7\u05e9\u05d1\u05ea",
        "\u05d7\u05b8\u05e9\u05c1\u05b7\u05d1\u05b0\u05ea\u05b8\u05bc",
        "male",
        "2",
        "khashavta (m) \u2190 khashavt (f); 'you thought'",
    ),
    LexiconEntry(
        "M2_VERB_DIBARTA",
        "\u05d3\u05d9\u05d1\u05e8\u05ea",
        "\u05d3\u05b4\u05bc\u05d9\u05d1\u05b7\u05bc\u05e8\u05b0\u05ea\u05b8\u05bc",
        "male",
        "2",
        "dibarta (m) \u2190 dibart (f); 'you spoke'",
    ),
    LexiconEntry(
        "M2_VERB_YADATA",
        "\u05d9\u05d3\u05e2\u05ea",
        "\u05d9\u05b8\u05d3\u05b7\u05e2\u05b0\u05ea\u05b8\u05bc",
        "male",
        "2",
        "yadata (m) \u2190 yadat (f); 'you knew'",
    ),
    LexiconEntry(
        "M2_VERB_HAYITA",
        "\u05d4\u05d9\u05d9\u05ea",
        "\u05d4\u05b8\u05d9\u05b4\u05d9\u05ea\u05b8",
        "male",
        "2",
        "hayita (m) \u2190 hayit (f); 'you were'",
    ),
    LexiconEntry(
        "M2_VERB_RATSITA",
        "\u05e8\u05e6\u05d9\u05ea",
        "\u05e8\u05b8\u05e6\u05b4\u05d9\u05ea\u05b8",
        "male",
        "2",
        "ratsita (m) \u2190 ratsit (f); 'you wanted'",
    ),
    # --- Additional past-tense 2sg verbs (addressee is male) ---
    LexiconEntry(
        "M2_VERB_ANITA",
        "\u05e2\u05e0\u05d9\u05ea",
        "\u05e2\u05b8\u05e0\u05b4\u05d9\u05ea\u05b8",
        "male",
        "2",
        "anita (m) \u2190 anit (f); 'you answered'",
    ),
    LexiconEntry(
        "M2_VERB_AMARTA",
        "\u05d0\u05de\u05e8\u05ea",
        "\u05d0\u05b8\u05de\u05b7\u05e8\u05b0\u05ea\u05b8\u05bc",
        "male",
        "2",
        "amarta (m) \u2190 amart (f); 'you said'",
    ),
    LexiconEntry(
        "M2_VERB_SHAMATA",
        "\u05e9\u05de\u05e2\u05ea",
        "\u05e9\u05c1\u05b8\u05de\u05b7\u05e2\u05b0\u05ea\u05b8\u05bc",
        "male",
        "2",
        "shamata (m) \u2190 shamat (f); 'you heard'",
    ),
    LexiconEntry(
        "M2_VERB_RAITA",
        "\u05e8\u05d0\u05d9\u05ea",
        "\u05e8\u05b8\u05d0\u05b4\u05d9\u05ea\u05b8",
        "male",
        "2",
        "ra'ita (m) \u2190 ra'it (f); 'you saw'",
    ),
    LexiconEntry(
        "M2_VERB_LAKAKHTA",
        "\u05dc\u05e7\u05d7\u05ea",
        "\u05dc\u05b8\u05e7\u05b7\u05d7\u05b0\u05ea\u05b8\u05bc",
        "male",
        "2",
        "lakakhta (m) \u2190 lakakht (f); 'you took'",
    ),
    LexiconEntry(
        "M2_VERB_NATATA",
        "\u05e0\u05ea\u05ea",
        "\u05e0\u05b8\u05ea\u05b7\u05ea\u05b8\u05bc",
        "male",
        "2",
        "natata (m) \u2190 natatt (f); 'you gave'",
    ),
    LexiconEntry(
        "M2_VERB_BATA",
        "\u05d1\u05d0\u05ea",
        "\u05d1\u05b8\u05d0\u05ea\u05b8",
        "male",
        "2",
        "bata (m) \u2190 bat (f); 'you came'",
    ),
    LexiconEntry(
        "M2_VERB_YATSATA",
        "\u05d9\u05e6\u05d0\u05ea",
        "\u05d9\u05b8\u05e6\u05b8\u05d0\u05ea\u05b8",
        "male",
        "2",
        "yatsata (m) \u2190 yatsat (f); 'you went out'",
    ),
    LexiconEntry(
        "M2_VERB_SHAKHAKHTA",
        "\u05e9\u05db\u05d7\u05ea",
        "\u05e9\u05c1\u05b8\u05db\u05b7\u05d7\u05b0\u05ea\u05b8\u05bc",
        "male",
        "2",
        "shakhakhta (m) \u2190 shakhakht (f); 'you forgot'",
    ),
    LexiconEntry(
        "M2_VERB_HITKHALTA",
        "\u05d4\u05ea\u05d7\u05dc\u05ea",
        "\u05d4\u05b4\u05ea\u05b0\u05d7\u05b7\u05dc\u05b0\u05ea\u05b8\u05bc",
        "male",
        "2",
        "hitkhalta (m) \u2190 hitkhalat (f); 'you started'",
    ),
    LexiconEntry(
        "M2_VERB_HIFSAKTA",
        "\u05d4\u05e4\u05e1\u05e7\u05ea",
        "\u05d4\u05b4\u05e4\u05b0\u05e1\u05b7\u05e7\u05b0\u05ea\u05b8\u05bc",
        "male",
        "2",
        "hifsakta (m) \u2190 hifsakt (f); 'you stopped'",
    ),
)

# Combined lexicon for backward-compatible parametrized tests.
_LEXICON: tuple[LexiconEntry, ...] = _LEXICON_2P_FEMININE + _LEXICON_2P_MASCULINE

# ---------------------------------------------------------------------------
# Surface forms that remain gender-ambiguous after disambiguation
# ---------------------------------------------------------------------------
# Used by the QA gate to flag turns that may still contain wrong-gender forms.
# These are common 2nd-person surface forms whose unvocalized spelling is
# shared between masculine and feminine.  If any of these survive in
# text_spoken without niqqud, they are suspicious.
_AMBIGUOUS_SURFACES: frozenset[str] = frozenset(entry.surface for entry in _LEXICON)

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
    speaker_gender: str,
    addressee_gender: str,
) -> NormalizationResult:
    """Apply gender corrections based on speaker and addressee genders.

    Second-person forms (pronouns, prepositions, verbs) are selected based
    on the *addressee* gender.  The rules insert niqqud to force the TTS
    engine to read the correct gendered form.

    Args:
        text: UTF-8 Hebrew source text (``DialogueTurn.text``).
        speaker_gender: Gender of the turn's speaker (``"male"`` or
            ``"female"``).
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
        spk_gender = _gender(turn.speaker_id)
        addr_gender = _gender(addr_id) if addr_id else ""

        norm = disambiguate_for_speaker(
            turn.text_spoken or turn.text,
            spk_gender,
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
