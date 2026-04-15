"""TTS rendering orchestrator with asset cache.

Cache key: SHA-256 of the rendered SSML string, which captures voice, text,
style, prosody parameters, and any randomization — so two calls with different
intensity or prosody will never collide.
Cache format: raw WAV bytes stored as {cache_key}.wav in the cache directory.

Spec reference: docs/implementation_plan.md §0.3
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from pathlib import Path

from synthbanshee.config.speaker_config import SpeakerConfig
from synthbanshee.script.types import DialogueTurn, MixedScene
from synthbanshee.tts.azure_provider import AzureProvider
from synthbanshee.tts.ssml_builder import SSMLBuilder


class TTSRenderer:
    """Render TTS utterances with a file-system render cache.

    Caches WAV bytes keyed on the full SSML string so that calls with
    different voice, text, style, or prosody always produce distinct entries.
    """

    def __init__(
        self,
        provider: AzureProvider | None = None,
        cache_dir: Path | str | None = None,
    ) -> None:
        self._provider = provider or AzureProvider()
        self._cache_dir = Path(
            cache_dir
            if cache_dir is not None
            else os.environ.get("SYNTHBANSHEE_CACHE_DIR") or "assets/speech"
        )
        self._ssml_builder = SSMLBuilder()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, ssml: str) -> str:
        return hashlib.sha256(ssml.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.wav"

    def _load_from_cache(self, key: str) -> bytes | None:
        p = self._cache_path(key)
        if p.exists():
            return p.read_bytes()
        return None

    def _save_to_cache(self, key: str, wav_bytes: bytes) -> Path:
        p = self._cache_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(wav_bytes)
        return p

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_utterance(
        self,
        text: str,
        speaker: SpeakerConfig,
        intensity: int = 1,
        *,
        randomize: bool = False,
        rng_seed: int | None = None,
    ) -> tuple[bytes, str, bool]:
        """Render a single utterance and return (wav_bytes, cache_key, cache_hit).

        Args:
            text: Hebrew utterance text (UTF-8). Must not contain Hebrew in
                  filenames — the text goes to TTS, not into any filename.
            speaker: Speaker persona (SpeakerConfig).
            intensity: Intensity level 1–5, used to look up style_map.
            randomize: If True, apply small random prosody variation.
            rng_seed: Random seed for reproducible prosody variation.

        Returns:
            Tuple of (raw WAV bytes, cache key string, cache_hit bool).
            cache_hit is True when the result was served from disk cache.
        """
        style_entry = speaker.style_for_intensity(intensity)

        rate = style_entry.rate_multiplier
        pitch = style_entry.pitch_delta_st
        volume = style_entry.volume_delta_db

        # Apply baseline offsets
        rate *= speaker.prosody_baseline.rate
        volume += speaker.prosody_baseline.volume_db

        if randomize:
            import random

            rng = random.Random(rng_seed)
            rate *= rng.uniform(0.97, 1.03)
            pitch += rng.uniform(-0.5, 0.5)
            volume += rng.uniform(-1.0, 1.0)

        ssml = self._ssml_builder.build_from_speaker_config(
            text=text,
            voice_id=speaker.tts_voice_id,
            style=style_entry.style,
            rate_multiplier=rate,
            pitch_delta_st=pitch,
            volume_delta_db=volume,
        )

        cache_key = self._cache_key(ssml)
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached, cache_key, True

        wav_bytes = self._provider.synthesize(ssml)
        self._save_to_cache(cache_key, wav_bytes)
        return wav_bytes, cache_key, False

    def render_utterance_to_file(
        self,
        text: str,
        speaker: SpeakerConfig,
        output_path: Path | str,
        intensity: int = 1,
    ) -> Path:
        """Render an utterance and write the WAV file to output_path.

        Returns the output path.
        """
        wav_bytes, _, _hit = self.render_utterance(text, speaker, intensity)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(wav_bytes)
        return output_path

    def render_scene(
        self,
        turns: list[DialogueTurn],
        speakers: dict[str, SpeakerConfig],
        *,
        randomize: bool = False,
        rng_seed: int | None = None,
        disfluency: bool = False,
        verbose_log: Callable[[str], None] | None = None,
    ) -> MixedScene:
        """Render a multi-speaker dialogue script to a MixedScene.

        Each turn is rendered individually (with caching) then mixed into a
        single audio stream by SceneMixer.  Turn onset/offset times are derived
        from the mix, not from TTS durations, so the label generator should
        always use MixedScene.turn_onsets_s / turn_offsets_s.

        Args:
            turns: Ordered list of DialogueTurn objects from ScriptGenerator.
            speakers: Mapping from speaker_id to SpeakerConfig.
            randomize: Apply small random prosody variation per turn.
            rng_seed: Seed for reproducible prosody variation.
            disfluency: If True, inject Hebrew filled pauses into each turn's
                        text using the speaker's disfluency profile.

        Returns:
            MixedScene with concatenated audio and per-turn timing metadata.

        Raises:
            KeyError: If a turn references a speaker_id not in *speakers*.
        """
        import random

        from synthbanshee.script.generator import inject_disfluency
        from synthbanshee.tts.mixer import SceneMixer

        rng = random.Random(rng_seed)
        mixer = SceneMixer()

        segments: list[tuple[bytes, float, str]] = []
        for i, turn in enumerate(turns):
            speaker = speakers[turn.speaker_id]
            text = turn.text
            if disfluency:
                text = inject_disfluency(
                    text,
                    prob=speaker.disfluency.filled_pause_prob,
                    rng_seed=rng.randint(0, 2**31),
                )
            wav_bytes, _, hit = self.render_utterance(
                text,
                speaker,
                turn.intensity,
                randomize=randomize,
                rng_seed=rng.randint(0, 2**31) if randomize else None,
            )
            if verbose_log is not None:
                status = "cache hit" if hit else "rendered"
                verbose_log(
                    f"  [dim]turn {i + 1:02d}/{len(turns):02d}"
                    f" [{turn.speaker_id}] intensity={turn.intensity}"
                    f" → {status}[/dim]"
                )
            segments.append((wav_bytes, turn.pause_before_s, turn.speaker_id))

        return mixer.mix_sequential(segments)
