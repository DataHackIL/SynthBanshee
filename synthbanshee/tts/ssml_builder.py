"""SSML document builder for he-IL TTS voices (Azure and Google).

Generates SSML 1.1 documents with:
  - <voice> elements for each speaker
  - <mstts:express-as> style tags (Azure only — omitted when
    ``supports_style_tags=False``)
  - <prosody> elements for rate, pitch, and volume control
  - Nested per-phrase <prosody> + <break> elements (M2b)
  - Inter-word ``<break time="50ms"/>`` elements to prevent Hebrew word
    merging (#62)

Azure SSML reference:
https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-voice
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from synthbanshee.tts.ssml_types import PhraseProsody

_AZURE_XMLNS = "http://www.w3.org/2001/10/synthesis"
_MSTTS_XMLNS = "http://www.w3.org/2001/mstts"
_SPEAK_LANG = "he-IL"

# Inter-word break duration in milliseconds.  50 ms is the initial estimate
# for signalling a word boundary to Azure / Google he-IL without introducing
# an audible pause.  This value needs empirical validation with real TTS
# output — it may need per-provider tuning if engines respond differently.
_WORD_BREAK_MS = 50


def _inject_word_breaks(
    parent: ET.Element,
    text: str,
    after: ET.Element | None = None,
) -> ET.Element | None:
    """Insert *text* into *parent* with ``<break>`` tags between words.

    Hebrew TTS engines (Azure he-IL, Google Chirp) merge adjacent words into
    unintelligible speech when no explicit boundary cue exists.  This function
    splits *text* on whitespace and inserts a short ``<break>`` element between
    every pair of consecutive words.

    Args:
        parent: The XML element to add content to.
        text: The text to inject (may contain multi-word Hebrew).
        after: If provided, the first text chunk is appended to
            ``after.tail`` instead of ``parent.text``.

    Returns:
        The last child element added to *parent*, or *after* if no ``<break>``
        elements were created (single word or empty text).
    """
    if not text or not text.strip():
        # Pure whitespace or empty — append as-is without inserting breaks.
        if text:
            if after is None:
                parent.text = (parent.text or "") + text
            else:
                after.tail = (after.tail or "") + text
        return after

    words = text.split()
    last = after

    # Preserve any leading whitespace (e.g. space before a text fragment
    # that follows a <break> or <prosody> element).
    leading = text[: len(text) - len(text.lstrip())]
    if leading:
        if last is None:
            parent.text = (parent.text or "") + leading
        else:
            last.tail = (last.tail or "") + leading

    for i, word in enumerate(words):
        if i == 0:
            # First word: append directly (no break needed before it).
            if last is None:
                parent.text = (parent.text or "") + word
            else:
                last.tail = (last.tail or "") + word
        else:
            # Subsequent words: insert <break/> then the word with a
            # leading space in the tail to preserve normal spacing.
            brk = ET.SubElement(parent, "break", attrib={"time": f"{_WORD_BREAK_MS}ms"})
            brk.tail = " " + word
            last = brk

    # Preserve any trailing whitespace.
    trailing = text[len(text.rstrip()) :]
    if trailing:
        if last is None:
            parent.text = (parent.text or "") + trailing
        else:
            last.tail = (last.tail or "") + trailing

    return last


@dataclass
class UtteranceSpec:
    """Specifies one TTS utterance to be included in an SSML document."""

    text: str
    voice_id: str  # e.g. "he-IL-AvriNeural"
    style: str = "General"  # Azure mstts:express-as style
    rate_multiplier: float = 1.0  # speaking rate relative to voice default
    pitch_delta_st: float = 0.0  # pitch offset in semitones
    volume_delta_db: float = 0.0  # volume offset in dB
    phrase_prosody: list[PhraseProsody] = field(default_factory=list)


def _semitones_to_percent(st: float) -> str:
    """Convert a semitone shift to the Azure pitch % format (e.g. '+5%' / '-10%')."""
    # Approximation: 1 semitone ≈ 5.946% pitch change
    pct = round(st * 5.946)
    return f"+{pct}%" if pct >= 0 else f"{pct}%"


def _rate_to_string(rate: float) -> str:
    """Format a rate multiplier as a percentage string for <prosody rate=...>."""
    pct = round((rate - 1.0) * 100)
    return f"+{pct}%" if pct >= 0 else f"{pct}%"


def _volume_to_string(db: float) -> str:
    """Format a dB volume offset as a percentage for <prosody volume=...>."""
    # Azure volume is 0–100; default is 100. Map dB linearly (rough).
    pct = round(db * 1.0)
    return f"+{pct}%" if pct >= 0 else f"{pct}%"


def _apply_phrase_prosody(
    parent: ET.Element,
    text: str,
    phrases: list[PhraseProsody],
) -> None:
    """Inject per-phrase ``<prosody>`` and ``<break>`` elements into *parent*.

    Splits *text* around phrase spans and wraps each phrase in a nested
    ``<prosody>`` element with optional ``<break time="…"/>`` elements
    inserted before and/or after.  Inter-word ``<break>`` elements are
    inserted within each text fragment to prevent Hebrew word merging (#62).
    Overlapping spans are skipped (the span whose ``char_start`` falls
    before the previous span's ``char_end`` is silently dropped).

    Args:
        parent: The XML element that will receive the mixed text/element content.
        text: The utterance text (plain UTF-8).
        phrases: Resolved phrase spans with prosody directives.
    """
    cursor = 0
    prev: ET.Element | None = None

    def _append_text(s: str) -> None:
        """Append *s* with inter-word ``<break>`` elements."""
        nonlocal prev
        if not s:
            return
        result = _inject_word_breaks(parent, s, after=prev)
        if result is not None:
            prev = result

    def _append_break(ms: int) -> ET.Element:
        nonlocal prev
        el = ET.SubElement(parent, "break", attrib={"time": f"{ms}ms"})
        prev = el
        return el

    for phrase in sorted(phrases, key=lambda p: p.char_start):
        if phrase.char_start >= phrase.char_end:
            continue
        if phrase.char_start < cursor:
            # Overlapping span — skip to avoid malformed nesting.
            continue

        # Plain text before this phrase.
        if phrase.char_start > cursor:
            _append_text(text[cursor : phrase.char_start])

        # Optional break before the phrase.
        if phrase.break_before_ms > 0:
            _append_break(phrase.break_before_ms)

        # Phrase-level prosody wrapper (omitted when only breaks are requested).
        phrase_attrs: dict[str, str] = {}
        if phrase.rate is not None:
            phrase_attrs["rate"] = phrase.rate
        if phrase.pitch is not None:
            phrase_attrs["pitch"] = phrase.pitch
        if phrase.volume is not None:
            phrase_attrs["volume"] = phrase.volume

        phrase_text = text[phrase.char_start : phrase.char_end]
        if phrase_attrs:
            pe = ET.SubElement(parent, "prosody", attrib=phrase_attrs)
            _inject_word_breaks(pe, phrase_text)
            prev = pe
        else:
            _append_text(phrase_text)

        # Optional break after the phrase.
        if phrase.break_after_ms > 0:
            _append_break(phrase.break_after_ms)

        cursor = phrase.char_end

    # Remaining text after all phrases (empty-string guard in _append_text
    # handles the case where cursor == len(text)).
    _append_text(text[cursor:])


class SSMLBuilder:
    """Build SSML documents for he-IL TTS rendering (Azure and Google)."""

    def build_single(self, utt: UtteranceSpec, *, supports_style_tags: bool = True) -> str:
        """Build an SSML document for a single utterance.

        Args:
            utt: Utterance specification.
            supports_style_tags: When False, ``<mstts:express-as>`` elements
                and the ``xmlns:mstts`` namespace declaration are omitted.
                Set to False for Google TTS and any non-Azure backend.
        """
        return self.build_multi([utt], supports_style_tags=supports_style_tags)

    def build_multi(
        self,
        utterances: list[UtteranceSpec],
        *,
        supports_style_tags: bool = True,
    ) -> str:
        """Build an SSML document containing multiple utterances.

        Each utterance is wrapped in its own <voice> block, so different
        voices can appear in the same request.

        Args:
            utterances: List of utterance specifications.
            supports_style_tags: When False, ``<mstts:express-as>`` elements
                are omitted.  Use False for Google TTS.
        """
        ET.register_namespace("", _AZURE_XMLNS)

        speak_attribs: dict[str, str] = {
            "version": "1.0",
            "xmlns": _AZURE_XMLNS,
            "xml:lang": _SPEAK_LANG,
        }
        if supports_style_tags:
            ET.register_namespace("mstts", _MSTTS_XMLNS)
            speak_attribs["xmlns:mstts"] = _MSTTS_XMLNS

        speak = ET.Element("speak", attrib=speak_attribs)

        for utt in utterances:
            voice = ET.SubElement(speak, "voice", attrib={"name": utt.voice_id})

            # Add express-as only when a non-default style is requested AND
            # the backend supports Azure-style tags.
            if supports_style_tags and utt.style and utt.style.lower() not in {"general", ""}:
                express = ET.SubElement(
                    voice,
                    "mstts:express-as",
                    attrib={"style": utt.style},
                )
                parent = express
            else:
                parent = voice

            # Build prosody attributes
            prosody_attrs: dict[str, str] = {}
            if abs(utt.rate_multiplier - 1.0) > 0.01:
                prosody_attrs["rate"] = _rate_to_string(utt.rate_multiplier)
            if abs(utt.pitch_delta_st) > 0.1:
                prosody_attrs["pitch"] = _semitones_to_percent(utt.pitch_delta_st)
            if abs(utt.volume_delta_db) > 0.1:
                prosody_attrs["volume"] = _volume_to_string(utt.volume_delta_db)

            if prosody_attrs:
                inner = ET.SubElement(parent, "prosody", attrib=prosody_attrs)
            else:
                inner = parent

            if utt.phrase_prosody:
                _apply_phrase_prosody(inner, utt.text, utt.phrase_prosody)
            else:
                _inject_word_breaks(inner, utt.text)

        raw = ET.tostring(speak, encoding="unicode", xml_declaration=False)
        return '<?xml version="1.0" encoding="UTF-8"?>\n' + raw

    def build_from_speaker_config(
        self,
        text: str,
        voice_id: str,
        style: str,
        rate_multiplier: float = 1.0,
        pitch_delta_st: float = 0.0,
        volume_delta_db: float = 0.0,
        phrase_prosody: list[PhraseProsody] | None = None,
        *,
        supports_style_tags: bool = True,
    ) -> str:
        """Convenience wrapper used by the render pipeline.

        Args:
            supports_style_tags: Forwarded to ``build_single``.  Pass
                ``provider.capabilities.supports_style_tags`` here.
        """
        utt = UtteranceSpec(
            text=text,
            voice_id=voice_id,
            style=style,
            rate_multiplier=rate_multiplier,
            pitch_delta_st=pitch_delta_st,
            volume_delta_db=volume_delta_db,
            phrase_prosody=phrase_prosody or [],
        )
        return self.build_single(utt, supports_style_tags=supports_style_tags)
