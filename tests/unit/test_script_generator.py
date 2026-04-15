"""Unit tests for synthbanshee.script.generator and related utilities."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

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

    def test_negative_pause_flagged(self):
        bad = [
            DialogueTurn(
                speaker_id="AGG_M_30-45_001", text="שלום", intensity=1, pause_before_s=-0.1
            )
        ]
        errors = validate_script(bad, _SPEAKER_IDS)
        assert any("pause_before_s" in e for e in errors)

    def test_excessive_pause_flagged(self):
        bad = [
            DialogueTurn(speaker_id="AGG_M_30-45_001", text="שלום", intensity=1, pause_before_s=2.0)
        ]
        errors = validate_script(bad, _SPEAKER_IDS)
        assert any("pause_before_s" in e for e in errors)

    def test_valid_pause_no_error(self):
        ok = [
            DialogueTurn(speaker_id="AGG_M_30-45_001", text="שלום", intensity=1, pause_before_s=1.0)
        ]
        errors = validate_script(ok, _SPEAKER_IDS)
        assert not any("pause_before_s" in e for e in errors)


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

    def test_verbose_log_called_on_cache_hit(self, tmp_path: Path):
        """verbose_log receives a cache-hit message when the script is already cached."""
        gen = self._make_generator(tmp_path)
        key = gen._cache_key(
            "TEST_001",
            "synthbanshee/script/templates/she_proves/intimate_terror_coercive_control.j2",
            {"relationship": "spouse", "setting": "kitchen"},
            [1, 2, 3],
            0,
            ["AGG_M_30-45_001", "VIC_F_25-40_002"],
        )
        gen._save_to_cache(key, _VALID_TURNS)

        log_messages: list[str] = []
        with patch.object(gen, "_call_llm"):
            gen.generate(**self._scene_kwargs(), verbose_log=log_messages.append)

        assert any("cache hit" in m for m in log_messages)

    def test_verbose_log_called_on_cache_miss(self, tmp_path: Path):
        """verbose_log receives a cache-miss message when the LLM is called."""
        gen = self._make_generator(tmp_path)

        log_messages: list[str] = []
        with patch.object(gen, "_call_llm", return_value=json.dumps(_VALID_TURNS_JSON)):
            gen.generate(
                **self._scene_kwargs("SCENE_VERBOSE_MISS"), verbose_log=log_messages.append
            )

        assert any("cache miss" in m for m in log_messages)

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


# ---------------------------------------------------------------------------
# ScriptGenerator — LLM provider dispatch (_call_llm, _call_anthropic, _call_openai)
# ---------------------------------------------------------------------------


class TestLLMDispatch:
    """Cover _call_anthropic, _call_openai, and _call_llm routing."""

    def test_call_llm_routes_to_anthropic(self, tmp_path: Path):
        gen = ScriptGenerator(provider="anthropic", cache_dir=tmp_path)
        with patch.object(gen, "_call_anthropic", return_value="resp") as mock_a:
            result = gen._call_llm("prompt")
        mock_a.assert_called_once_with("prompt")
        assert result == "resp"

    def test_call_llm_routes_to_openai(self, tmp_path: Path):
        gen = ScriptGenerator(provider="openai", cache_dir=tmp_path)
        with patch.object(gen, "_call_openai", return_value="resp") as mock_o:
            result = gen._call_llm("prompt")
        mock_o.assert_called_once_with("prompt")
        assert result == "resp"

    def test_call_anthropic_uses_sdk(self, tmp_path: Path):
        gen = ScriptGenerator(provider="anthropic", model="claude-test", cache_dir=tmp_path)
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="שלום")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        mock_anthropic_mod = ModuleType("anthropic")
        mock_anthropic_mod.Anthropic = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"anthropic": mock_anthropic_mod}):
            result = gen._call_anthropic("test prompt")

        mock_client.messages.create.assert_called_once_with(
            model="claude-test",
            max_tokens=4096,
            messages=[{"role": "user", "content": "test prompt"}],
        )
        assert result == "שלום"

    def test_call_openai_uses_sdk(self, tmp_path: Path):
        gen = ScriptGenerator(provider="openai", model="gpt-test", cache_dir=tmp_path)
        mock_choice = MagicMock()
        mock_choice.message.content = "שלום מהאי"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        mock_openai_mod = ModuleType("openai")
        mock_openai_mod.OpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"openai": mock_openai_mod}):
            result = gen._call_openai("test prompt")

        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-test",
            messages=[{"role": "user", "content": "test prompt"}],
            max_tokens=4096,
        )
        assert result == "שלום מהאי"

    def test_call_openai_empty_content_returns_empty_string(self, tmp_path: Path):
        gen = ScriptGenerator(provider="openai", model="gpt-test", cache_dir=tmp_path)
        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        mock_openai_mod = ModuleType("openai")
        mock_openai_mod.OpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"openai": mock_openai_mod}):
            result = gen._call_openai("prompt")

        assert result == ""
