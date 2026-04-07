"""Unit tests for synthbanshee.script.generator and related utilities."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from synthbanshee.script.generator import (
    ScriptGenerator,
    inject_disfluency,
    validate_script,
)
from synthbanshee.script.types import DialogueTurn

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SPEAKER_IDS = {"AGG_M_30-45_001", "VIC_F_25-40_002"}

_VALID_TURNS = [
    DialogueTurn(
        speaker_id="AGG_M_30-45_001",
        text="שלום, מה שלומך היום?",
        intensity=1,
        pause_before_s=0.3,
        emotional_state="neutral",
    ),
    DialogueTurn(
        speaker_id="VIC_F_25-40_002",
        text="בסדר, תודה. איך הולך לך?",
        intensity=1,
        pause_before_s=0.5,
        emotional_state="neutral",
    ),
    DialogueTurn(
        speaker_id="AGG_M_30-45_001",
        text="אני לא מרוצה. צריך לדבר על משהו חשוב.",
        intensity=3,
        pause_before_s=0.3,
        emotional_state="angry",
    ),
]

_VALID_TURNS_JSON = {
    "turns": [
        {
            "speaker_id": t.speaker_id,
            "text": t.text,
            "intensity": t.intensity,
            "pause_before_s": t.pause_before_s,
            "emotional_state": t.emotional_state,
        }
        for t in _VALID_TURNS
    ]
}


# ---------------------------------------------------------------------------
# inject_disfluency
# ---------------------------------------------------------------------------


class TestInjectDisfluency:
    def test_no_op_on_single_sentence(self):
        text = "שלום מה שלומך"
        result = inject_disfluency(text, prob=1.0, rng_seed=0)
        assert result == text

    def test_inserts_pause_between_sentences(self):
        text = "שלום. מה שלומך?"
        result = inject_disfluency(text, prob=1.0, rng_seed=0)
        # A Hebrew pause token should appear between the two sentences
        assert any(p in result for p in ["אממ", "אה", "אנ"])
        # Original sentences are still present
        assert "שלום" in result
        assert "מה שלומך" in result

    def test_zero_prob_no_change(self):
        text = "שלום. מה שלומך? טוב מאוד."
        result = inject_disfluency(text, prob=0.0, rng_seed=42)
        assert result == text

    def test_reproducible_with_seed(self):
        text = "טוב. מה נשמע? הכל בסדר."
        r1 = inject_disfluency(text, prob=0.8, rng_seed=7)
        r2 = inject_disfluency(text, prob=0.8, rng_seed=7)
        assert r1 == r2

    def test_different_seeds_may_differ(self):
        text = "טוב. מה נשמע? הכל בסדר. ומה איתך?"
        results = {inject_disfluency(text, prob=0.5, rng_seed=s) for s in range(20)}
        # With prob 0.5 over multiple seeds we expect at least two distinct outputs
        assert len(results) >= 2


# ---------------------------------------------------------------------------
# validate_script
# ---------------------------------------------------------------------------


class TestValidateScript:
    def test_valid_turns_no_errors(self):
        errors = validate_script(_VALID_TURNS, _SPEAKER_IDS)
        assert errors == []

    def test_empty_text_flagged(self):
        bad = [DialogueTurn(speaker_id="AGG_M_30-45_001", text="  ", intensity=1)]
        errors = validate_script(bad, _SPEAKER_IDS)
        assert any("empty text" in e for e in errors)

    def test_unknown_speaker_flagged(self):
        bad = [DialogueTurn(speaker_id="UNKNOWN_001", text="שלום", intensity=1)]
        errors = validate_script(bad, _SPEAKER_IDS)
        assert any("speaker_id" in e for e in errors)

    def test_invalid_intensity_flagged(self):
        bad = [DialogueTurn(speaker_id="AGG_M_30-45_001", text="שלום", intensity=0)]
        errors = validate_script(bad, _SPEAKER_IDS)
        assert any("intensity" in e for e in errors)

    def test_repetition_flagged(self):
        repeated = "שלום " * 5
        bad = [DialogueTurn(speaker_id="AGG_M_30-45_001", text=repeated, intensity=1)]
        errors = validate_script(bad, _SPEAKER_IDS)
        assert any("consecutive" in e for e in errors)

    def test_non_hebrew_text_flagged(self):
        bad = [DialogueTurn(speaker_id="AGG_M_30-45_001", text="hello world", intensity=1)]
        errors = validate_script(bad, _SPEAKER_IDS)
        assert any("Hebrew" in e for e in errors)


# ---------------------------------------------------------------------------
# ScriptGenerator — cache
# ---------------------------------------------------------------------------


class TestScriptGeneratorCache:
    def _make_generator(self, tmp_path: Path) -> ScriptGenerator:
        return ScriptGenerator(provider="anthropic", cache_dir=tmp_path / "scripts")

    def _scene_kwargs(self, scene_id: str = "TEST_001") -> dict:
        return dict(
            scene_id=scene_id,
            project="she_proves",
            violence_typology="IT",
            script_template="synthbanshee/script/templates/she_proves/intimate_terror_coercive_control.j2",
            script_slots={"relationship": "spouse", "setting": "kitchen"},
            intensity_arc=[1, 2, 3],
            target_duration_minutes=1.0,
            speakers=[
                {"speaker_id": "AGG_M_30-45_001", "role": "AGG", "gender": "male"},
                {"speaker_id": "VIC_F_25-40_002", "role": "VIC", "gender": "female"},
            ],
            random_seed=0,
        )

    def test_cache_hit_skips_llm(self, tmp_path: Path):
        gen = self._make_generator(tmp_path)
        # Pre-populate cache
        key = gen._cache_key(
            "TEST_001",
            "synthbanshee/script/templates/she_proves/intimate_terror_coercive_control.j2",
            {"relationship": "spouse", "setting": "kitchen"},
            [1, 2, 3],
            0,
            ["AGG_M_30-45_001", "VIC_F_25-40_002"],
        )
        gen._save_to_cache(key, _VALID_TURNS)

        with patch.object(gen, "_call_llm") as mock_llm:
            turns = gen.generate(**self._scene_kwargs())

        mock_llm.assert_not_called()
        assert len(turns) == len(_VALID_TURNS)
        assert turns[0].speaker_id == _VALID_TURNS[0].speaker_id

    def test_cache_miss_calls_llm_and_saves(self, tmp_path: Path):
        gen = self._make_generator(tmp_path)

        with patch.object(gen, "_call_llm", return_value=json.dumps(_VALID_TURNS_JSON)):
            turns = gen.generate(**self._scene_kwargs("SCENE_NEW"))

        assert len(turns) == len(_VALID_TURNS)
        # Cache file should now exist
        key = gen._cache_key(
            "SCENE_NEW",
            "synthbanshee/script/templates/she_proves/intimate_terror_coercive_control.j2",
            {"relationship": "spouse", "setting": "kitchen"},
            [1, 2, 3],
            0,
            ["AGG_M_30-45_001", "VIC_F_25-40_002"],
        )
        assert gen._cache_path(key).exists()

    def test_validation_error_raises(self, tmp_path: Path):
        gen = self._make_generator(tmp_path)
        bad_response = json.dumps(
            {
                "turns": [
                    {
                        "speaker_id": "UNKNOWN_SPEAKER",
                        "text": "hello world",  # no Hebrew, unknown speaker
                        "intensity": 1,
                        "pause_before_s": 0.3,
                        "emotional_state": "neutral",
                    }
                ]
            }
        )

        with (
            patch.object(gen, "_call_llm", return_value=bad_response),
            pytest.raises(ValueError, match="Script validation failed"),
        ):
            gen.generate(**self._scene_kwargs("BAD_SCENE"))


# ---------------------------------------------------------------------------
# ScriptGenerator — _parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_plain_json(self):
        turns = ScriptGenerator._parse_response(json.dumps(_VALID_TURNS_JSON))
        assert len(turns) == len(_VALID_TURNS)
        assert turns[0].speaker_id == _VALID_TURNS[0].speaker_id

    def test_json_in_markdown_fence(self):
        raw = f"```json\n{json.dumps(_VALID_TURNS_JSON)}\n```"
        turns = ScriptGenerator._parse_response(raw)
        assert len(turns) == len(_VALID_TURNS)

    def test_json_in_plain_fence(self):
        raw = f"```\n{json.dumps(_VALID_TURNS_JSON)}\n```"
        turns = ScriptGenerator._parse_response(raw)
        assert len(turns) == len(_VALID_TURNS)

    def test_invalid_json_raises(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            ScriptGenerator._parse_response("not json")


# ---------------------------------------------------------------------------
# ScriptGenerator — provider validation
# ---------------------------------------------------------------------------


class TestScriptGeneratorInit:
    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError, match="provider"):
            ScriptGenerator(provider="gemini")

    def test_default_models(self):
        gen_a = ScriptGenerator(provider="anthropic")
        assert "claude" in gen_a._model

        gen_o = ScriptGenerator(provider="openai")
        assert "gpt" in gen_o._model

    def test_custom_model(self):
        gen = ScriptGenerator(provider="anthropic", model="claude-haiku-4-5-20251001")
        assert gen._model == "claude-haiku-4-5-20251001"
