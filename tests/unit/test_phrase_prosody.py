"""Unit tests for M2b — per-phrase SSML prosody injection.

Covers:
  - _build_offset_map: SequenceMatcher-based char-offset alignment
  - resolve_phrase_hints: PhraseHint (text_original) → PhraseProsody (text_spoken)
  - rebase_phrase_prosody: remapping after disfluency injection
  - detect_imperative_phrases: deterministic Hebrew imperative heuristics
  - collect_phrase_prosody: combined entry point
  - _apply_phrase_prosody (via SSMLBuilder): nested SSML element injection
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from synthbanshee.tts.ssml_builder import SSMLBuilder, UtteranceSpec, _apply_phrase_prosody
from synthbanshee.tts.ssml_types import (
    PhraseHint,
    PhraseProsody,
    _build_offset_map,
    collect_phrase_prosody,
    detect_imperative_phrases,
    rebase_phrase_prosody,
    resolve_phrase_hints,
)

# ---------------------------------------------------------------------------
# _build_offset_map
# ---------------------------------------------------------------------------


class TestBuildOffsetMap:
    def test_identity_mapping(self) -> None:
        text = "hello"
        m = _build_offset_map(text, text)
        for i in range(len(text) + 1):
            assert m[i] == i

    def test_map_length_is_from_len_plus_one(self) -> None:
        m = _build_offset_map("abc", "abcd")
        assert len(m) == 4  # len("abc") + 1

    def test_insertion_shifts_end(self) -> None:
        # from_text: "ac", to_text: "abc" — 'b' inserted between a and c
        m = _build_offset_map("ac", "abc")
        assert m[0] == 0  # 'a' → position 0
        assert m[2] == 3  # exclusive end of "ac" → exclusive end of "abc"

    def test_deletion_collapses_deleted_chars(self) -> None:
        # from_text: "abc", to_text: "ac" — 'b' deleted
        m = _build_offset_map("abc", "ac")
        assert m[0] == 0  # 'a' stays at 0
        assert m[3] == 2  # exclusive end maps to end of "ac"

    def test_no_negative_entries(self) -> None:
        m = _build_offset_map("hello world", "hi world")
        assert all(v >= 0 for v in m)

    def test_empty_from_text(self) -> None:
        m = _build_offset_map("", "abc")
        assert m == [0]

    def test_empty_to_text(self) -> None:
        m = _build_offset_map("abc", "")
        # All positions should map to 0 (end of empty string)
        assert all(v == 0 for v in m)


# ---------------------------------------------------------------------------
# resolve_phrase_hints
# ---------------------------------------------------------------------------


class TestResolvePhraseHints:
    def test_empty_hints_returns_empty(self) -> None:
        assert resolve_phrase_hints([], "text", "text") == []

    def test_identity_text_preserves_offsets(self) -> None:
        text = "hello world"
        hint = PhraseHint(
            phrase_id="t1_p0",
            hint="stress",
            char_start_original=6,
            char_end_original=11,
        )
        result = resolve_phrase_hints([hint], text, text)
        assert len(result) == 1
        assert result[0].char_start == 6
        assert result[0].char_end == 11
        assert result[0].phrase_id == "t1_p0"

    def test_hint_defaults_applied_stress(self) -> None:
        text = "hello world"
        hint = PhraseHint(
            phrase_id="t1_p0",
            hint="stress",
            char_start_original=0,
            char_end_original=5,
        )
        result = resolve_phrase_hints([hint], text, text)
        assert result[0].rate == "+15%"
        assert result[0].volume == "+3dB"

    def test_hint_defaults_applied_menace(self) -> None:
        text = "hello world"
        hint = PhraseHint(
            phrase_id="t1_p0",
            hint="menace",
            char_start_original=0,
            char_end_original=5,
        )
        result = resolve_phrase_hints([hint], text, text)
        assert result[0].rate == "-25%"
        assert result[0].break_before_ms == 300

    def test_hint_defaults_applied_slow(self) -> None:
        text = "hello world"
        hint = PhraseHint("t1_p0", "slow", 0, 5)
        result = resolve_phrase_hints([hint], text, text)
        assert result[0].rate == "-20%"
        assert result[0].break_before_ms == 150

    def test_hint_defaults_applied_break_before(self) -> None:
        hint = PhraseHint("t1_p0", "break_before", 0, 5)
        result = resolve_phrase_hints([hint], "hello", "hello")
        assert result[0].break_before_ms == 250
        assert result[0].rate is None

    def test_hint_defaults_applied_break_after(self) -> None:
        hint = PhraseHint("t1_p0", "break_after", 0, 5)
        result = resolve_phrase_hints([hint], "hello", "hello")
        assert result[0].break_after_ms == 250
        assert result[0].rate is None

    def test_zero_length_span_dropped(self) -> None:
        hint = PhraseHint("t1_p0", "stress", 3, 3)
        result = resolve_phrase_hints([hint], "hello", "hello")
        assert result == []

    def test_oob_hint_clamped_not_raised(self) -> None:
        text = "abc"
        hint = PhraseHint("t1_p0", "stress", 0, 999)
        # Should not raise; end clamped to len(text)
        result = resolve_phrase_hints([hint], text, text)
        assert len(result) == 1

    def test_text_with_insertion_remaps_correctly(self) -> None:
        # text_original: "עברית" (5 chars), text_spoken: "עִבְרִית" (niqqud added, 8 chars)
        # Offsets [0,5] in original → should span full text in spoken
        text_orig = "עברית"
        text_spoken = "עִבְרִית"
        hint = PhraseHint("t1_p0", "stress", 0, len(text_orig))
        result = resolve_phrase_hints([hint], text_orig, text_spoken)
        assert len(result) == 1
        assert result[0].char_start == 0
        assert result[0].char_end > 0

    def test_multiple_hints_preserved(self) -> None:
        text = "one two three"
        hints = [
            PhraseHint("t1_p0", "stress", 0, 3),
            PhraseHint("t1_p1", "slow", 4, 7),
        ]
        result = resolve_phrase_hints(hints, text, text)
        assert len(result) == 2
        assert result[0].phrase_id == "t1_p0"
        assert result[1].phrase_id == "t1_p1"


# ---------------------------------------------------------------------------
# rebase_phrase_prosody
# ---------------------------------------------------------------------------


class TestRebasePhrasePrody:
    def test_empty_list_returns_empty(self) -> None:
        assert rebase_phrase_prosody([], "abc", "abc") == []

    def test_identity_rebase_preserves_offsets(self) -> None:
        phrase = PhraseProsody("p0", 2, 5, rate="+15%")
        result = rebase_phrase_prosody([phrase], "hello", "hello")
        assert result[0].char_start == 2
        assert result[0].char_end == 5

    def test_insertion_shifts_offsets_forward(self) -> None:
        # "world" → "dear world" — 5 chars inserted at start
        phrase = PhraseProsody("p0", 0, 5, rate="+15%")
        result = rebase_phrase_prosody([phrase], "world", "dear world")
        assert result[0].char_start >= 0
        assert result[0].char_end > result[0].char_start

    def test_prosody_values_preserved(self) -> None:
        phrase = PhraseProsody("p0", 0, 3, rate="-20%", break_before_ms=150)
        result = rebase_phrase_prosody([phrase], "abc", "abc")
        assert result[0].rate == "-20%"
        assert result[0].break_before_ms == 150

    def test_collapsed_span_dropped(self) -> None:
        # Zero-length span is dropped regardless of rebasing.
        phrase2 = PhraseProsody("p0", 2, 2, rate="+15%")  # already zero-length
        result = rebase_phrase_prosody([phrase2], "abc", "ac")
        assert result == []


# ---------------------------------------------------------------------------
# detect_imperative_phrases
# ---------------------------------------------------------------------------


class TestDetectImperativePhrases:
    def test_sentence_ending_imperative_detected(self) -> None:
        text = "אתה צריך לשמוע אותי תקשיב"
        result = detect_imperative_phrases(text)
        assert len(result) == 1
        assert result[0].rate == "-15%"
        assert result[0].break_before_ms == 200

    def test_phrase_id_uses_heuristic_prefix(self) -> None:
        result = detect_imperative_phrases("אתה תסתכל")
        assert result[0].phrase_id.startswith("heuristic_p")

    def test_non_imperative_sentence_skipped(self) -> None:
        result = detect_imperative_phrases("אני הולך הביתה")
        assert result == []

    def test_multiple_sentences_multiple_imperatives(self) -> None:
        text = "תסתכל. ועכשיו תקשיב"
        result = detect_imperative_phrases(text)
        # "תסתכל" ends first sentence, "תקשיב" ends second
        assert len(result) == 2
        assert result[0].phrase_id == "heuristic_p0"
        assert result[1].phrase_id == "heuristic_p1"

    def test_imperative_with_punctuation_stripped(self) -> None:
        # "תשתוק!" — '!' stripped before lookup
        result = detect_imperative_phrases("אתה תשתוק!")
        assert len(result) == 1

    def test_char_offsets_within_text_bounds(self) -> None:
        text = "אתה תפסיק"
        result = detect_imperative_phrases(text)
        assert len(result) == 1
        p = result[0]
        assert 0 <= p.char_start < p.char_end <= len(text)
        assert text[p.char_start : p.char_end] == "תפסיק"

    def test_imperative_not_at_sentence_end_not_detected(self) -> None:
        # "תקשיב אליי" — imperative is NOT the last word
        result = detect_imperative_phrases("תקשיב אליי")
        assert result == []

    def test_empty_text_returns_empty(self) -> None:
        assert detect_imperative_phrases("") == []


# ---------------------------------------------------------------------------
# collect_phrase_prosody
# ---------------------------------------------------------------------------


class TestCollectPhraseProsody:
    def test_empty_hints_and_no_imperatives(self) -> None:
        result = collect_phrase_prosody([], "שלום עולם", "שלום עולם")
        assert result == []

    def test_merged_result_sorted_by_start(self) -> None:
        # LLM hint at offset 10, imperative heuristic earlier in text
        text = "בוא לכאן תפסיק"
        hints = [
            PhraseHint("t1_p0", "stress", 9, 14),  # "תפסיק" area (approx)
        ]
        result = collect_phrase_prosody(hints, text, text)
        starts = [p.char_start for p in result]
        assert starts == sorted(starts)

    def test_only_heuristics_when_no_hints(self) -> None:
        text = "אתה תקשיב"
        result = collect_phrase_prosody([], text, text)
        assert any(p.phrase_id.startswith("heuristic") for p in result)

    def test_only_llm_hints_when_no_imperatives(self) -> None:
        text = "הוא אמר לי משהו"
        hints = [PhraseHint("t1_p0", "slow", 7, 11)]
        result = collect_phrase_prosody(hints, text, text)
        assert any(p.phrase_id == "t1_p0" for p in result)


# ---------------------------------------------------------------------------
# _apply_phrase_prosody — SSML element injection
# ---------------------------------------------------------------------------


def _make_parent() -> ET.Element:
    return ET.Element("prosody")


class TestApplyPhraseProsody:
    def test_no_phrases_sets_text(self) -> None:
        parent = _make_parent()
        _apply_phrase_prosody(parent, "hello world", [])
        assert parent.text == "hello world"
        assert len(list(parent)) == 0  # no children

    def test_single_phrase_mid_text(self) -> None:
        parent = _make_parent()
        phrase = PhraseProsody("p0", 6, 11, rate="+15%")
        _apply_phrase_prosody(parent, "hello world", [phrase])
        children = list(parent)
        assert parent.text == "hello "
        assert len(children) == 1
        assert children[0].tag == "prosody"
        assert children[0].text == "world"
        assert children[0].get("rate") == "+15%"

    def test_phrase_at_start_of_text(self) -> None:
        parent = _make_parent()
        phrase = PhraseProsody("p0", 0, 5, rate="-20%")
        _apply_phrase_prosody(parent, "hello world", [phrase])
        children = list(parent)
        assert parent.text is None or parent.text == ""
        assert children[0].text == "hello"
        assert children[0].tail == " world"

    def test_phrase_at_end_of_text(self) -> None:
        parent = _make_parent()
        phrase = PhraseProsody("p0", 6, 11, rate="+15%")
        _apply_phrase_prosody(parent, "hello world", [phrase])
        children = list(parent)
        assert children[0].text == "world"
        assert children[0].tail is None or children[0].tail == ""

    def test_break_before_inserted(self) -> None:
        parent = _make_parent()
        phrase = PhraseProsody("p0", 0, 5, break_before_ms=250)
        _apply_phrase_prosody(parent, "hello", [phrase])
        children = list(parent)
        # First child should be <break>, second should be plain text (no prosody wrapper
        # since no rate/pitch/volume)
        assert children[0].tag == "break"
        assert children[0].get("time") == "250ms"

    def test_break_after_inserted(self) -> None:
        parent = _make_parent()
        phrase = PhraseProsody("p0", 0, 5, break_after_ms=300)
        _apply_phrase_prosody(parent, "hello", [phrase])
        children = list(parent)
        assert any(c.tag == "break" and c.get("time") == "300ms" for c in children)

    def test_overlapping_span_skipped(self) -> None:
        parent = _make_parent()
        p1 = PhraseProsody("p0", 0, 8, rate="+15%")
        p2 = PhraseProsody("p1", 3, 8, rate="-20%")  # overlaps p1
        _apply_phrase_prosody(parent, "hello world", [p1, p2])
        # Only one prosody child expected (p2 skipped)
        children = [c for c in list(parent) if c.tag == "prosody"]
        assert len(children) == 1

    def test_two_non_overlapping_phrases(self) -> None:
        parent = _make_parent()
        p1 = PhraseProsody("p0", 0, 3, rate="+15%")
        p2 = PhraseProsody("p1", 4, 7, rate="-20%")
        _apply_phrase_prosody(parent, "one two three", [p1, p2])
        children = [c for c in list(parent) if c.tag == "prosody"]
        assert len(children) == 2
        assert children[0].text == "one"
        assert children[1].text == "two"

    def test_phrase_without_prosody_attrs_no_wrapper(self) -> None:
        parent = _make_parent()
        # break_only phrase — no rate/pitch/volume
        phrase = PhraseProsody("p0", 0, 5, break_before_ms=200)
        _apply_phrase_prosody(parent, "hello world", [phrase])
        prosody_children = [c for c in list(parent) if c.tag == "prosody"]
        assert len(prosody_children) == 0  # no nested prosody, only break


# ---------------------------------------------------------------------------
# SSMLBuilder integration — phrase prosody in full SSML output
# ---------------------------------------------------------------------------


class TestSSMLBuilderWithPhrases:
    def _parse(self, ssml: str) -> ET.Element:
        # Strip the XML declaration for easier parsing
        body = ssml.split("\n", 1)[1] if ssml.startswith("<?xml") else ssml
        return ET.fromstring(body)

    def test_no_phrases_plain_text(self) -> None:
        builder = SSMLBuilder()
        utt = UtteranceSpec(text="שלום", voice_id="he-IL-AvriNeural")
        ssml = builder.build_single(utt)
        root = self._parse(ssml)
        # Find the voice element and check text
        voice = root.find("{http://www.w3.org/2001/10/synthesis}voice")
        assert voice is not None
        assert "שלום" in ET.tostring(voice, encoding="unicode")

    def test_phrase_produces_nested_prosody(self) -> None:
        builder = SSMLBuilder()
        phrase = PhraseProsody("p0", 0, 5, rate="+15%")
        utt = UtteranceSpec(
            text="hello world",
            voice_id="he-IL-AvriNeural",
            phrase_prosody=[phrase],
        )
        ssml = builder.build_single(utt)
        assert 'rate="+15%"' in ssml

    def test_build_from_speaker_config_with_phrases(self) -> None:
        builder = SSMLBuilder()
        phrase = PhraseProsody("p0", 6, 11, rate="-20%", break_before_ms=150)
        ssml = builder.build_from_speaker_config(
            text="hello world",
            voice_id="he-IL-AvriNeural",
            style="General",
            phrase_prosody=[phrase],
        )
        assert "break" in ssml
        assert "-20%" in ssml

    def test_build_from_speaker_config_no_phrases(self) -> None:
        builder = SSMLBuilder()
        ssml = builder.build_from_speaker_config(
            text="hello",
            voice_id="he-IL-AvriNeural",
            style="General",
        )
        assert "hello" in ssml
        assert "<break" not in ssml
