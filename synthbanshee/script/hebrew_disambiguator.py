"""Hebrew gender disambiguation for TTS text normalization (pipeline M1).

Azure TTS defaults to masculine inflection for morphologically ambiguous
unvocalized Hebrew second-person forms.  When the aggressor (AGG) addresses
the victim (VIC) — the dominant address direction in the current scenes —
this produces systematic gender errors that are audible to native speakers
and confirmed by direct listening (Shay, debug_run_1).

Strategy: rule-based lexicon substitution that inserts niqqud (Hebrew vowel
diacritics) for the highest-risk tokens so that Azure TTS receives an
unambiguous phonetic target and reads the correct feminine form.

Only ``DialogueTurn.text_spoken`` is modified; the original LLM output
(``DialogueTurn.text``) is preserved unchanged for auditing and caching.

Reference: docs/audio_generation_v3_design.md §4.1, M1
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from synthbanshee.script.types import DialogueTurn


class LexiconEntry(NamedTuple):
    """One disambiguation rule mapping an ambiguous surface form to its vocalized feminine replacement."""

    rule_id: str
    surface: str  # unvocalized form to match in source text
    feminine_spoken: str  # niqqud-vocalized form for TTS; forces feminine reading
    description: str  # human-readable note (masc default → correct fem reading)


# ---------------------------------------------------------------------------
# Lexicon — high-risk second-person forms
# ---------------------------------------------------------------------------
#
# Sources: Shay's direct listening feedback on debug_run_1 ("שלך always
# pronounced 'shelkha' instead of 'shelakh', הלכת pronounced 'halakhta'
# instead of 'halakht'"), Gemini review, Hebrew morphological reference.
#
# Each entry targets a form where:
#   (a) the written unvocalized text is identical for masc and fem, and
#   (b) Azure TTS (he-IL-AvriNeural) defaults to the masculine reading.
#
# The niqqud replacement is the *only* change — the surrounding text is
# untouched.  IPA equivalents are in the description for verification.

_LEXICON: tuple[LexiconEntry, ...] = (
    # --- Possessive / object pronouns ---
    LexiconEntry(
        "POSS_SHEL",
        "שלך",
        "שֶׁלָּךְ",
        "shelakh (f) ← shelkha (m default); 'yours'",
    ),
    LexiconEntry(
        "PRON_OTKH",
        "אותך",
        "אוֹתָךְ",
        "otakh (f) ← otkha (m default); 'you' (direct object)",
    ),
    LexiconEntry(
        "PRON_ITZAKH",
        "אתך",
        "אִתָּךְ",
        "itakh (f) ← itkha (m default); 'with you'",
    ),
    # --- Prepositional clitics ---
    LexiconEntry(
        "PREP_LAKH",
        "לך",
        "לָךְ",
        "lakh (f) ← lekha (m default); 'to/for you'",
    ),
    LexiconEntry(
        "PREP_BAKH",
        "בך",
        "בָּךְ",
        "bakh (f) ← bekha (m default); 'in/at you'",
    ),
    LexiconEntry(
        "PREP_MIMEKH",
        "ממך",
        "מִמֵּךְ",
        "mimekh (f) ← mimkha (m default); 'from you'",
    ),
    LexiconEntry(
        "PREP_ALAIKH",
        "עליך",
        "עָלַיִךְ",
        "alaikh (f) ← alekha (m default); 'on/about you'",
    ),
    LexiconEntry(
        "PREP_ELAIKH",
        "אליך",
        "אֵלַיִךְ",
        "elaikh (f) ← elekha (m default); 'toward you'",
    ),
    # --- Past-tense 2sg verbs — fem ends /-t/ (תְּ), masc /-ta/ (תָּ) ---
    LexiconEntry(
        "VERB_PAST_HALAKHT",
        "הלכת",
        "הָלַכְתְּ",
        "halakht (f) ← halakhta (m default); 'you went'",
    ),
    LexiconEntry(
        "VERB_PAST_ASIT",
        "עשית",
        "עָשִׂיתְ",
        "asit (f) ← asita (m default); 'you did/made'",
    ),
    LexiconEntry(
        "VERB_PAST_KHASHAVT",
        "חשבת",
        "חָשַׁבְתְּ",
        "khashavt (f) ← khashavta (m default); 'you thought'",
    ),
    LexiconEntry(
        "VERB_PAST_DIBBART",
        "דיברת",
        "דִּיבַּרְתְּ",
        "dibart (f) ← dibarta (m default); 'you spoke'",
    ),
    LexiconEntry(
        "VERB_PAST_YADAT",
        "ידעת",
        "יָדַעְתְּ",
        "yadat (f) ← yadata (m default); 'you knew'",
    ),
    LexiconEntry(
        "VERB_PAST_HAYIT",
        "היית",
        "הָיִיתְ",
        "hayit (f) ← hayita (m default); 'you were'",
    ),
    LexiconEntry(
        "VERB_PAST_RATSIT",
        "רצית",
        "רָצִיתְ",
        "ratsit (f) ← ratsita (m default); 'you wanted'",
    ),
)

# Pre-compile one regex per lexicon entry.
# Lookahead/lookbehind on Hebrew token characters ensure we match whole
# tokens, not substrings embedded inside longer words.  The guard covers
# both Hebrew base letters (U+05D0–U+05EA) and Hebrew combining marks /
# related marks (U+0591–U+05C7), so a partially-vocalized letter immediately
# adjacent to the surface form does not create a false word boundary that
# would allow an in-word match.
_HEB_TOKEN_CHAR = r"[\u0591-\u05C7\u05D0-\u05EA]"

_COMPILED: tuple[tuple[LexiconEntry, re.Pattern[str]], ...] = tuple(
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
    for entry in _LEXICON
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
    speaker_role: str,
    addressee_role: str,
) -> NormalizationResult:
    """Apply feminine gender corrections when AGG addresses VIC.

    Substitutions are only applied when *speaker_role* is ``"AGG"`` and
    *addressee_role* is ``"VIC"`` — the dominant AGG→VIC direction in all
    current scene configurations.  All other role pairs are returned
    unchanged, preserving VIC self-references and VIC→AGG address forms.

    Args:
        text: UTF-8 Hebrew source text (``DialogueTurn.text``).
        speaker_role: Role of the turn's speaker, e.g. ``"AGG"``.
        addressee_role: Role of the primary addressee, e.g. ``"VIC"``.

    Returns:
        :class:`NormalizationResult` carrying the (possibly modified)
        ``text_spoken`` and the list of rule IDs that fired.
    """
    if not (speaker_role == "AGG" and addressee_role == "VIC"):
        return NormalizationResult(text_spoken=text)

    result = text
    triggered: list[str] = []
    for entry, pattern in _COMPILED:
        new_text, n = pattern.subn(entry.feminine_spoken, result)
        if n:
            result = new_text
            triggered.append(entry.rule_id)

    return NormalizationResult(
        text_spoken=result,
        normalization_rules_triggered=triggered,
    )


def disambiguate_turns(
    turns: list[DialogueTurn],
    speaker_roles: dict[str, str],
) -> list[DialogueTurn]:
    """Apply gender disambiguation to every turn in a scene.

    For each turn the addressee role is inferred as the role of the first
    other speaker in *speaker_roles* — correct for the standard two-speaker
    AGG/VIC scene.  For scenes with more than two speaker roles the heuristic
    may be inaccurate; it is still better than applying no disambiguation.

    The original ``DialogueTurn`` objects are never mutated.  New instances
    are returned with ``text_spoken`` and ``normalization_rules_triggered``
    populated.

    Args:
        turns: Ordered list of turns from :class:`ScriptGenerator`.
        speaker_roles: Mapping ``{speaker_id: role}`` for all speakers in
            the scene (built from ``SceneConfig.speakers``).

    Returns:
        New list of :class:`DialogueTurn` with disambiguation applied.
    """

    # Priority order for addressee selection in multi-speaker scenes.
    # AGG prefers VIC; VIC prefers AGG; BYS is always last resort.
    _ROLE_PRIORITY = ["VIC", "AGG", "BYS"]

    def _addressee_role(speaker_id: str) -> str:
        my_role = speaker_roles.get(speaker_id, "UNK")
        other_roles = {
            role for sid, role in speaker_roles.items() if sid != speaker_id and role != my_role
        }
        for candidate in _ROLE_PRIORITY:
            if candidate in other_roles:
                return candidate
        # Fall back to any other role not in the priority list.
        for sid, role in speaker_roles.items():
            if sid != speaker_id and role != my_role:
                return role
        return "UNK"

    result: list[DialogueTurn] = []
    for turn in turns:
        addr_role = _addressee_role(turn.speaker_id)
        norm = disambiguate_for_speaker(
            turn.text_spoken or turn.text,
            speaker_roles.get(turn.speaker_id, "UNK"),
            addr_role,
        )
        result.append(
            dataclasses.replace(
                turn,
                text_spoken=norm.text_spoken,
                normalization_rules_triggered=norm.normalization_rules_triggered,
            )
        )
    return result
