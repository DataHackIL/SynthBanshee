"""Phrase-level prosody types and resolution utilities (M2b — spec §4.2b).

Two sources produce ``PhraseProsody`` objects:

1. **LLM-generated hints** — the script generator asks the LLM to annotate
   1–2 emotionally loaded phrases per I3–I5 turn by their character offsets
   in ``text_original``.  ``resolve_phrase_hints()`` maps those offsets to
   ``text_spoken`` after all text normalisation has been applied.

2. **Deterministic heuristics** — ``detect_imperative_phrases()`` adds a
   break + slow hint before sentence-final Hebrew imperatives without any LLM
   annotation.

Both sources produce ``PhraseProsody`` objects; the ``SSMLBuilder`` does not
distinguish their origin.

Design doc reference: docs/audio_generation_v3_design.md §4.2b
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from typing import Literal

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class PhraseHint:
    """LLM-provided annotation for one emotionally loaded phrase in a turn.

    Offsets are character positions within ``DialogueTurn.text_original``
    (the raw LLM output, never mutated).  The resolver maps them to
    ``text_spoken`` after all normalisation steps have been applied.

    Attributes:
        phrase_id: Unique within the turn, e.g. ``"t3_p1"``.
        hint: Prosody directive applied to this phrase.
        char_start_original: Inclusive start offset in ``text_original``.
        char_end_original: Exclusive end offset in ``text_original``.
    """

    phrase_id: str
    hint: Literal["stress", "slow", "break_before", "break_after", "menace"]
    char_start_original: int
    char_end_original: int


@dataclass
class PhraseProsody:
    """Resolved phrase span with SSML prosody directives.

    Offsets reference the *final* text string that will be passed to
    ``SSMLBuilder`` (either ``text_spoken`` or the disfluency-injected
    variant, after rebasing via ``rebase_phrase_prosody()``).

    Attributes:
        phrase_id: Matches the originating ``PhraseHint.phrase_id``, or a
            heuristic-generated ID such as ``"heuristic_p0"``.
        char_start: Inclusive start offset in the target text.
        char_end: Exclusive end offset.
        rate: SSML rate value (e.g. ``"+15%"``, ``"slow"``).
            ``None`` means no rate change.
        pitch: SSML pitch value (e.g. ``"+2st"``). ``None`` means no change.
        volume: SSML volume value (e.g. ``"+3dB"``). ``None`` means no change.
        break_before_ms: Milliseconds of silence inserted before the span.
        break_after_ms: Milliseconds of silence inserted after the span.
    """

    phrase_id: str
    char_start: int
    char_end: int
    rate: str | None = None
    pitch: str | None = None
    volume: str | None = None
    break_before_ms: int = 0
    break_after_ms: int = 0


# ---------------------------------------------------------------------------
# Hint → default PhraseProsody values
# ---------------------------------------------------------------------------

_HINT_DEFAULTS: dict[str, dict[str, str | int]] = {
    # Stressed accusatory phrase — faster, louder, higher
    "stress": {"rate": "+15%", "volume": "+3dB", "break_before_ms": 0},
    # Deliberate command slowing — measured menace
    "slow": {"rate": "-20%", "break_before_ms": 150},
    # Pause before the phrase for dramatic weight
    "break_before": {"break_before_ms": 250},
    # Pause after the phrase to let it land
    "break_after": {"break_after_ms": 250},
    # Cold-coercion menacing tone — slower + very slight pitch drop
    "menace": {"rate": "-25%", "break_before_ms": 300},
}


# ---------------------------------------------------------------------------
# Offset alignment utilities
# ---------------------------------------------------------------------------


def _build_offset_map(from_text: str, to_text: str) -> list[int]:
    """Return a forward mapping: ``from_text`` position → ``to_text`` position.

    Uses ``difflib.SequenceMatcher`` to align the two strings.  Positions
    that are deleted in ``to_text`` are mapped to the nearest surviving
    position.  The returned list has length ``len(from_text) + 1`` so that
    the exclusive-end position ``len(from_text)`` maps to ``len(to_text)``.
    """
    mapping: list[int] = [-1] * (len(from_text) + 1)
    matcher = SequenceMatcher(None, from_text, to_text, autojunk=False)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for delta in range(i2 - i1):
                mapping[i1 + delta] = j1 + delta
            mapping[i2] = j2
        elif tag == "replace":
            span_from = i2 - i1
            span_to = j2 - j1
            for delta in range(span_from):
                mapping[i1 + delta] = j1 + delta * span_to // max(span_from, 1)
            mapping[i2] = j2
        elif tag == "delete":
            for delta in range(i2 - i1):
                mapping[i1 + delta] = j1
            mapping[i2] = j1

    # Fill any remaining -1 entries by propagating the last known value.
    last = 0
    for i in range(len(mapping)):
        if mapping[i] >= 0:
            last = mapping[i]
        else:
            mapping[i] = last

    return mapping


# ---------------------------------------------------------------------------
# Resolver: PhraseHint (text_original offsets) → PhraseProsody (text_spoken)
# ---------------------------------------------------------------------------


def resolve_phrase_hints(
    hints: list[PhraseHint],
    text_original: str,
    text_spoken: str,
) -> list[PhraseProsody]:
    """Map ``PhraseHint`` char offsets from ``text_original`` to ``text_spoken``.

    Handles niqqud insertion and any other normalisation changes that shift
    character positions between the two strings.  Spans that collapse to zero
    length after remapping are silently dropped.

    Args:
        hints: LLM-generated phrase annotations (offsets in ``text_original``).
        text_original: The unmodified LLM output.
        text_spoken: Post-normalisation text (after M1 disambiguation).

    Returns:
        List of ``PhraseProsody`` objects with offsets into ``text_spoken``.
    """
    if not hints:
        return []

    offset_map = _build_offset_map(text_original, text_spoken)
    n = len(offset_map)
    result: list[PhraseProsody] = []

    for hint in hints:
        start = offset_map[min(hint.char_start_original, n - 1)]
        end = offset_map[min(hint.char_end_original, n - 1)]
        if start >= end:
            continue

        defaults = _HINT_DEFAULTS.get(hint.hint, {})
        result.append(
            PhraseProsody(
                phrase_id=hint.phrase_id,
                char_start=start,
                char_end=end,
                rate=str(defaults["rate"]) if "rate" in defaults else None,
                pitch=str(defaults["pitch"]) if "pitch" in defaults else None,
                volume=str(defaults["volume"]) if "volume" in defaults else None,
                break_before_ms=int(defaults.get("break_before_ms", 0)),
                break_after_ms=int(defaults.get("break_after_ms", 0)),
            )
        )

    return result


# ---------------------------------------------------------------------------
# Rebase: remap PhraseProsody offsets after disfluency injection
# ---------------------------------------------------------------------------


def rebase_phrase_prosody(
    phrases: list[PhraseProsody],
    from_text: str,
    to_text: str,
) -> list[PhraseProsody]:
    """Remap ``PhraseProsody`` offsets from ``from_text`` to ``to_text``.

    Called when disfluency injection shifts character positions between
    ``text_spoken`` (where phrase hints were resolved) and the final
    disfluency-injected text passed to the SSML builder.  Spans that
    collapse to zero length are dropped.

    Args:
        phrases: Phrases with offsets into ``from_text``.
        from_text: Text the current offsets reference.
        to_text: Text the returned offsets should reference.

    Returns:
        New list of ``PhraseProsody`` with updated offsets.
    """
    if not phrases:
        return []

    offset_map = _build_offset_map(from_text, to_text)
    n = len(offset_map)
    result: list[PhraseProsody] = []

    for phrase in phrases:
        start = offset_map[min(phrase.char_start, n - 1)]
        end = offset_map[min(phrase.char_end, n - 1)]
        if start >= end:
            continue
        result.append(replace(phrase, char_start=start, char_end=end))

    return result


# ---------------------------------------------------------------------------
# Deterministic heuristics: sentence-final imperatives
# ---------------------------------------------------------------------------

# Common Hebrew sentence-final imperatives that always get a pre-break + slow.
# Checked after stripping terminal punctuation from the last word.
_IMPERATIVE_WORDS: frozenset[str] = frozenset(
    {
        "עכשיו",  # now
        "ברור",  # clear / understood
        "לך",  # go
        "צא",  # get out
        "שב",  # sit (down)
        "עמוד",  # stop / stand
        "תשתוק",  # shut up
        "תפסיק",  # stop (it)
        "בצע",  # execute / do it
        "תעשה",  # do it
        "תקשיב",  # listen
        "תסתכל",  # look
        "תראה",  # see / look
        "בוא",  # come (here)
        "תגיד",  # say (it)
        "תחתום",  # sign (it)
    }
)

_STRIP_PUNCT = str.maketrans("", "", "?!.,;:\"'״׃")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def detect_imperative_phrases(text_spoken: str) -> list[PhraseProsody]:
    """Return ``PhraseProsody`` entries for sentence-final Hebrew imperatives.

    Each matched word gets a 200 ms break before it and a ``-15%`` rate
    reduction.  This runs deterministically — no LLM annotation needed.

    Args:
        text_spoken: The normalised turn text.

    Returns:
        List of ``PhraseProsody`` for any matched imperative words.
    """
    result: list[PhraseProsody] = []
    phrase_idx = 0
    cursor = 0

    sentences = _SENTENCE_SPLIT_RE.split(text_spoken)

    for sentence in sentences:
        # Find where this sentence starts in the full text (from cursor onward).
        try:
            sentence_start = text_spoken.index(sentence, cursor)
        except ValueError:
            continue

        words = sentence.split()
        if not words:
            cursor = sentence_start + len(sentence)
            continue

        last_word_raw = words[-1]
        last_word_norm = last_word_raw.translate(_STRIP_PUNCT)

        if last_word_norm in _IMPERATIVE_WORDS:
            # Locate the last word within the sentence string.
            word_offset_in_sentence = sentence.rfind(last_word_raw)
            if word_offset_in_sentence >= 0:
                char_start = sentence_start + word_offset_in_sentence
                char_end = char_start + len(last_word_raw)
                result.append(
                    PhraseProsody(
                        phrase_id=f"heuristic_p{phrase_idx}",
                        char_start=char_start,
                        char_end=char_end,
                        rate="-15%",
                        break_before_ms=200,
                    )
                )
                phrase_idx += 1

        cursor = sentence_start + len(sentence)

    return result


# ---------------------------------------------------------------------------
# Combined resolver: LLM hints + heuristics → PhraseProsody list
# ---------------------------------------------------------------------------


def collect_phrase_prosody(
    phrase_hints: list[PhraseHint],
    text_original: str,
    text_spoken: str,
) -> list[PhraseProsody]:
    """Resolve LLM phrase hints and add deterministic imperative annotations.

    This is the single entry point called by ``TTSRenderer.render_scene()``.

    Args:
        phrase_hints: ``DialogueTurn.phrase_hints`` (may be empty).
        text_original: ``DialogueTurn.text_original`` (unmodified LLM output).
        text_spoken: Post-normalisation text (M1 output).

    Returns:
        Merged, start-sorted list of ``PhraseProsody`` ready for the SSML
        builder.  Overlapping spans are not deduplicated — the SSML builder
        processes them in order.
    """
    from_llm = resolve_phrase_hints(phrase_hints, text_original, text_spoken)
    from_heuristics = detect_imperative_phrases(text_spoken)

    # Merge and sort by start position.
    merged = sorted(from_llm + from_heuristics, key=lambda p: p.char_start)
    return merged
