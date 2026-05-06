"""Tests for ``synthbanshee.package.asr_sanity`` (#87 Whisper sanity check).

The heavyweight Whisper inference itself is exercised by the local Tier-3
manual run (see CLAUDE.md "ASR sanity check policy") — these tests cover
the lightweight surface that runs without ``torch``/``transformers``:

  - ``normalize_for_wer`` — punctuation stripping + whitespace collapse so
    the spike script and ``qa-report --asr`` agree on what counts as a word.
  - ``AsrMetrics.is_silence_collapse`` — the threshold predicate that decides
    whether a clip is flagged.
  - ``_read_reference_text`` — bracket-line stripping (the ``[SPEAKER: …]``
    headers in the .txt files must not enter the reference passed to jiwer).
"""

from __future__ import annotations

from synthbanshee.package.asr_sanity import (
    AsrMetrics,
    _read_reference_text,
    normalize_for_wer,
)


class TestNormalizeForWer:
    def test_strips_ascii_punctuation(self):
        assert normalize_for_wer("hello, world!") == "hello world"

    def test_collapses_repeated_whitespace(self):
        assert normalize_for_wer("hello   \n\tworld") == "hello world"

    def test_preserves_hebrew_characters(self):
        text = "שלום, עולם!"
        assert "שלום" in normalize_for_wer(text)
        assert "עולם" in normalize_for_wer(text)

    def test_preserves_digits(self):
        assert normalize_for_wer("abc 123 def") == "abc 123 def"

    def test_empty_input(self):
        assert normalize_for_wer("") == ""

    def test_only_punctuation_collapses_to_empty(self):
        assert normalize_for_wer("...!?,") == ""


class TestAsrMetricsThreshold:
    def test_clip_below_threshold_is_flagged(self):
        m = AsrMetrics(wer=0.30, length_ratio=0.71, hyp_words=143, ref_words=190, hyp_text="…")
        assert m.is_silence_collapse(min_length_ratio=0.85) is True

    def test_clip_at_threshold_is_not_flagged(self):
        m = AsrMetrics(wer=0.10, length_ratio=0.85, hyp_words=170, ref_words=200, hyp_text="…")
        # is_silence_collapse uses strict less-than: equal-to-threshold passes.
        assert m.is_silence_collapse(min_length_ratio=0.85) is False

    def test_clip_at_baseline_is_not_flagged(self):
        m = AsrMetrics(wer=0.05, length_ratio=1.00, hyp_words=234, ref_words=234, hyp_text="…")
        assert m.is_silence_collapse(min_length_ratio=0.85) is False


class TestReadReferenceText:
    def test_strips_bracket_lines(self, tmp_path):
        # Mirrors the .txt format produced by cli.py:
        #   [CLIP_ID: ...]
        #   [SPEAKER: ... | ROLE: ... | ONSET: ... | OFFSET: ...]
        #   <hebrew text>
        #   [ACTION: ... | INTENSITY: ...]
        path = tmp_path / "clip.txt"
        path.write_text(
            "[CLIP_ID: clip_001_00]\n"
            "[SPEAKER: AGG_M_30-45_001 | ROLE: AGG | ONSET: 0.50 | OFFSET: 4.20]\n"
            "שלום עולם\n"
            "[ACTION: NONE_AMBIENT | INTENSITY: 1]\n",
            encoding="utf-8",
        )
        ref = _read_reference_text(path)
        assert ref == "שלום עולם"

    def test_preserves_multiple_speech_lines(self, tmp_path):
        path = tmp_path / "clip.txt"
        path.write_text(
            "[CLIP_ID: clip_001_00]\n"
            "[SPEAKER: A | ROLE: AGG | ONSET: 0 | OFFSET: 1]\n"
            "first\n"
            "[ACTION: X | INTENSITY: 1]\n"
            "[SPEAKER: B | ROLE: VIC | ONSET: 1 | OFFSET: 2]\n"
            "second\n"
            "[ACTION: Y | INTENSITY: 1]\n",
            encoding="utf-8",
        )
        ref = _read_reference_text(path)
        assert ref == "first\nsecond"
