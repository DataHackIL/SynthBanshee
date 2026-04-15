"""LLM-based script generator for AVDP scenes.

Renders a Jinja2 prompt template (from the scene config's script_template field)
and calls an LLM (Anthropic Claude or OpenAI GPT-4) to produce a structured Hebrew
dialogue.  Results are cached on disk keyed by a SHA-256 of all generation inputs
so identical scene configs never hit the API twice.

Cache key components: scene_id, script_template path, script_slots JSON,
intensity_arc, random_seed, provider, model, speaker IDs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from collections.abc import Callable
from pathlib import Path

from synthbanshee.script.types import DialogueTurn

_DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-6"
_DEFAULT_OPENAI_MODEL = "gpt-4o"

# Hebrew filled-pause tokens inserted by inject_disfluency
_HE_FILLED_PAUSES = ["אממ", "אה", "אנ"]


def inject_disfluency(
    text: str,
    prob: float = 0.10,
    rng_seed: int | None = None,
) -> str:
    """Insert Hebrew filled-pause tokens between sentences with probability *prob*.

    Operates on sentence boundaries (splits on '.', '!', '?' followed by space).
    The original text is never truncated — pauses are inserted, not substituted.

    Args:
        text: Hebrew UTF-8 text.
        prob: Probability of inserting a filled pause between any two sentences.
        rng_seed: Optional seed for reproducibility.

    Returns:
        Modified text with occasional Hebrew filled pauses.
    """
    import random

    rng = random.Random(rng_seed)
    # Split into sentences keeping the delimiters
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(parts) <= 1:
        return text

    result_parts: list[str] = [parts[0]]
    for part in parts[1:]:
        if rng.random() < prob:
            pause_token = rng.choice(_HE_FILLED_PAUSES)
            result_parts.append(pause_token)
        result_parts.append(part)
    return " ".join(result_parts)


def validate_script(
    turns: list[DialogueTurn],
    known_speaker_ids: set[str],
) -> list[str]:
    """Validate a generated script for spec compliance.

    Checks:
    - All turns have non-empty Hebrew text
    - All speaker_ids appear in known_speaker_ids
    - Intensity values are 1–5
    - No 4+ consecutive identical tokens (LLM repetition artifact)

    Returns:
        List of error message strings (empty → valid).
    """
    errors: list[str] = []
    for i, turn in enumerate(turns):
        prefix = f"turn[{i}]"

        if not turn.text.strip():
            errors.append(f"{prefix}: empty text")
            continue

        if turn.speaker_id not in known_speaker_ids:
            errors.append(f"{prefix}: speaker_id {turn.speaker_id!r} not in known speakers")

        if turn.intensity not in range(1, 6):
            errors.append(f"{prefix}: intensity {turn.intensity} out of range 1–5")

        # Validate pause_before_s: must be finite and within [0.0, 1.5] s
        if not math.isfinite(turn.pause_before_s) or not (0.0 <= turn.pause_before_s <= 1.5):
            errors.append(
                f"{prefix}: pause_before_s {turn.pause_before_s} must be finite and in [0.0, 1.5]"
            )

        # Detect repetition: 4+ consecutive identical whitespace-split tokens
        tokens = turn.text.split()
        run = 1
        for j in range(1, len(tokens)):
            if tokens[j] == tokens[j - 1]:
                run += 1
                if run >= 4:
                    errors.append(f"{prefix}: 4+ consecutive identical tokens ({tokens[j]!r})")
                    break
            else:
                run = 1

        # Check text contains at least some Unicode Hebrew characters
        has_hebrew = any(
            unicodedata.name(ch, "").startswith("HEBREW") for ch in turn.text if ch.strip()
        )
        if not has_hebrew:
            errors.append(f"{prefix}: text contains no Hebrew characters")

    return errors


class ScriptGenerator:
    """Generate a structured Hebrew dialogue from a scene config using an LLM.

    Supports Anthropic (Claude) and OpenAI (GPT-4o) providers.
    Results are cached to ``cache_dir`` so identical scenes never re-call the API.
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str | None = None,
        cache_dir: Path | str | None = None,
    ) -> None:
        if provider not in {"anthropic", "openai"}:
            raise ValueError(f"provider must be 'anthropic' or 'openai', got {provider!r}")
        self._provider = provider
        self._model = model or (
            _DEFAULT_ANTHROPIC_MODEL if provider == "anthropic" else _DEFAULT_OPENAI_MODEL
        )
        self._cache_dir = Path(
            cache_dir
            if cache_dir is not None
            else os.environ.get("SYNTHBANSHEE_SCRIPT_CACHE_DIR") or "assets/scripts"
        )

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _cache_key(
        self,
        scene_id: str,
        script_template: str,
        script_slots: dict,
        intensity_arc: list[int],
        random_seed: int,
        speaker_ids: list[str],
    ) -> str:
        payload = json.dumps(
            {
                "scene_id": scene_id,
                "script_template": script_template,
                "script_slots": script_slots,
                "intensity_arc": intensity_arc,
                "random_seed": random_seed,
                "provider": self._provider,
                "model": self._model,
                "speaker_ids": sorted(speaker_ids),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _load_from_cache(self, key: str) -> list[DialogueTurn] | None:
        p = self._cache_path(key)
        if not p.exists():
            return None
        raw = json.loads(p.read_text(encoding="utf-8"))
        return [DialogueTurn(**t) for t in raw["turns"]]

    def _save_to_cache(self, key: str, turns: list[DialogueTurn]) -> None:
        p = self._cache_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "turns": [
                {
                    "speaker_id": t.speaker_id,
                    "text": t.text,
                    "intensity": t.intensity,
                    "pause_before_s": t.pause_before_s,
                    "emotional_state": t.emotional_state,
                }
                for t in turns
            ]
        }
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Template rendering
    # ------------------------------------------------------------------

    def _render_prompt(
        self,
        template_path: str,
        scene_id: str,
        project: str,
        violence_typology: str,
        script_slots: dict,
        intensity_arc: list[int],
        target_duration_minutes: float,
        speakers: list[dict],
    ) -> str:
        """Render the Jinja2 prompt template to a string."""
        from jinja2 import Environment, FileSystemLoader, StrictUndefined

        tpl_path = Path(template_path)
        env = Environment(
            loader=FileSystemLoader(str(tpl_path.parent)),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
        )
        template = env.get_template(tpl_path.name)
        return template.render(
            scene_id=scene_id,
            project=project,
            violence_typology=violence_typology,
            script_slots=script_slots,
            intensity_arc=intensity_arc,
            target_duration_minutes=target_duration_minutes,
            speakers=speakers,
        )

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _call_anthropic(self, prompt: str) -> str:
        import anthropic

        client = anthropic.Anthropic()
        message = client.messages.create(
            model=self._model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def _call_openai(self, prompt: str) -> str:
        import openai

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
        )
        return response.choices[0].message.content or ""

    def _call_llm(self, prompt: str) -> str:
        if self._provider == "anthropic":
            return self._call_anthropic(prompt)
        return self._call_openai(prompt)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(raw: str) -> list[DialogueTurn]:
        """Extract dialogue turns from the LLM's JSON response.

        Accepts a raw response that may include markdown code fences.
        """
        # Strip markdown code fences if present
        stripped = raw.strip()
        fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)```", stripped)
        if fence_match:
            stripped = fence_match.group(1).strip()

        data = json.loads(stripped)
        turns_raw = data.get("turns", [])
        turns: list[DialogueTurn] = []
        for t in turns_raw:
            turns.append(
                DialogueTurn(
                    speaker_id=t["speaker_id"],
                    text=t["text"],
                    intensity=int(t.get("intensity", 1)),
                    pause_before_s=float(t.get("pause_before_s", 0.3)),
                    emotional_state=str(t.get("emotional_state", "neutral")),
                )
            )
        return turns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        scene_id: str,
        project: str,
        violence_typology: str,
        script_template: str,
        script_slots: dict,
        intensity_arc: list[int],
        target_duration_minutes: float,
        speakers: list[dict],
        random_seed: int = 0,
        verbose_log: Callable[[str], None] | None = None,
    ) -> list[DialogueTurn]:
        """Generate a dialogue script for the given scene parameters.

        Args:
            scene_id: Unique scene identifier.
            project: 'she_proves' or 'elephant_in_the_room'.
            violence_typology: e.g. 'IT', 'SV', 'NEU'.
            script_template: Path to the Jinja2 prompt template file.
            script_slots: Template slot values (e.g. relationship, setting).
            intensity_arc: Sequence of 1–5 intensity levels for the scene.
            target_duration_minutes: Desired total clip duration.
            speakers: List of speaker dicts with speaker_id, role, gender, etc.
            random_seed: Seed used in the cache key for reproducibility.

        Returns:
            List of DialogueTurn objects forming the full scene script.

        Raises:
            ValueError: If the LLM response cannot be parsed or fails validation.
        """
        speaker_ids = [s["speaker_id"] for s in speakers]
        key = self._cache_key(
            scene_id, script_template, script_slots, intensity_arc, random_seed, speaker_ids
        )

        cached = self._load_from_cache(key)
        if cached is not None:
            if verbose_log is not None:
                verbose_log(f"  [dim]script: cache hit ({len(cached)} turns)[/dim]")
            return cached

        if verbose_log is not None:
            verbose_log(f"  [dim]script: cache miss — calling {self._provider}/{self._model}[/dim]")
        prompt = self._render_prompt(
            template_path=script_template,
            scene_id=scene_id,
            project=project,
            violence_typology=violence_typology,
            script_slots=script_slots,
            intensity_arc=intensity_arc,
            target_duration_minutes=target_duration_minutes,
            speakers=speakers,
        )

        raw_response = self._call_llm(prompt)
        turns = self._parse_response(raw_response)

        errors = validate_script(turns, known_speaker_ids=set(speaker_ids))
        if errors:
            raise ValueError(
                f"Script validation failed for scene {scene_id}:\n" + "\n".join(errors)
            )

        self._save_to_cache(key, turns)
        return turns
