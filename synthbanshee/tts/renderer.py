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
from synthbanshee.tts.mix_mode import MixMode
from synthbanshee.tts.provider import TTSProvider
from synthbanshee.tts.speaker_state import SpeakerState
from synthbanshee.tts.ssml_builder import SSMLBuilder
from synthbanshee.tts.ssml_types import PhraseProsody, collect_phrase_prosody, rebase_phrase_prosody

# Verbose-log callback type.  Strings passed to this callback may contain
# Rich markup tags (e.g. ``[dim]…[/dim]``).  Callers that do not use Rich
# (e.g. standard loggers) should strip or ignore the markup.
_VerboseLog = Callable[[str], None]


class TTSRenderer:
    """Render TTS utterances with a file-system render cache.

    Caches WAV bytes keyed on the full SSML string so that calls with
    different voice, text, style, or prosody always produce distinct entries.

    Supports multiple backends.  The backend for each utterance is chosen
    from ``providers`` based on ``SpeakerConfig.tts_provider``.  When only
    one provider is supplied (legacy usage), it is used for all speakers
    regardless of their ``tts_provider`` field.

    Args:
        provider: Single provider for backward compatibility.  Mutually
            exclusive with ``providers``.
        providers: Mapping of provider name (``"azure"``, ``"google"``) to
            provider instance.  When present, the renderer dispatches based
            on ``speaker.tts_provider``.
        cache_dir: Directory for the WAV render cache.
    """

    def __init__(
        self,
        provider: AzureProvider | None = None,
        cache_dir: Path | str | None = None,
        *,
        providers: dict[str, TTSProvider] | None = None,
    ) -> None:
        if providers is not None and provider is not None:
            raise ValueError("Specify either 'provider' or 'providers', not both.")
        if providers is not None:
            self._providers: dict[str, TTSProvider] = providers
            self._legacy_mode = False
        else:
            self._providers = {"azure": provider or AzureProvider()}
            self._legacy_mode = True
        self._cache_dir = Path(
            cache_dir
            if cache_dir is not None
            else os.environ.get("SYNTHBANSHEE_CACHE_DIR") or "assets/speech"
        )
        self._ssml_builder = SSMLBuilder()

    def _get_provider(self, speaker: SpeakerConfig) -> TTSProvider:
        """Return the provider for *speaker*.

        In legacy single-provider mode (``provider=`` kwarg at construction),
        the registered provider is always used regardless of the speaker's
        ``tts_provider`` field — backward-compatible behaviour.

        In explicit multi-provider mode (``providers=`` kwarg), dispatch is
        strict: a ``KeyError`` is raised if the speaker's backend is not
        registered.
        """
        key = speaker.tts_provider
        if key in self._providers:
            return self._providers[key]
        if self._legacy_mode:
            return next(iter(self._providers.values()))
        raise KeyError(
            f"No provider registered for tts_provider={key!r}. Registered: {list(self._providers)}"
        )

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
        speaker_state: SpeakerState | None = None,
        phrase_prosody: list[PhraseProsody] | None = None,
    ) -> tuple[bytes, str, bool]:
        """Render a single utterance and return (wav_bytes, cache_key, cache_hit).

        Args:
            text: Hebrew utterance text (UTF-8). Must not contain Hebrew in
                  filenames — the text goes to TTS, not into any filename.
            speaker: Speaker persona (SpeakerConfig).
            intensity: Intensity level 1–5, used to look up style_map.
            randomize: If True, apply small random prosody variation.
            rng_seed: Random seed for reproducible prosody variation.
            speaker_state: Optional accumulated cross-turn state (M7).  When
                supplied, its offsets are multiplied/added on top of the base
                style values before SSML is built.
            phrase_prosody: Optional resolved per-phrase prosody spans (M2b).
                Offsets must reference *text* (the final string passed here).

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

        # Apply accumulated cross-turn state offsets (M7).
        if speaker_state is not None:
            rate *= speaker_state.rate_offset
            pitch += speaker_state.pitch_offset_st
            volume += speaker_state.volume_offset_db

        if randomize:
            import random

            rng = random.Random(rng_seed)
            rate *= rng.uniform(0.97, 1.03)
            pitch += rng.uniform(-0.5, 0.5)
            volume += rng.uniform(-1.0, 1.0)

        provider = self._get_provider(speaker)
        ssml = self._ssml_builder.build_from_speaker_config(
            text=text,
            voice_id=speaker.tts_voice_id,
            style=style_entry.style,
            rate_multiplier=rate,
            pitch_delta_st=pitch,
            volume_delta_db=volume,
            phrase_prosody=phrase_prosody,
            supports_style_tags=provider.capabilities.supports_style_tags,
        )

        cache_key = self._cache_key(ssml)
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached, cache_key, True

        wav_bytes = provider.synthesize(ssml)
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
        verbose_log: _VerboseLog | None = None,
        project: str = "she_proves",
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
            project: Project identifier passed to ``TurnGapController`` to
                     select psychologically-motivated gap ranges (§4.5).

        Returns:
            MixedScene with concatenated audio and per-turn timing metadata.

        Raises:
            KeyError: If a turn references a speaker_id not in *speakers*.
        """
        import random

        from synthbanshee.script.generator import inject_disfluency
        from synthbanshee.tts.gap_controller import TurnGapController
        from synthbanshee.tts.mixer import SceneMixer

        rng = random.Random(rng_seed)
        # Separate gap RNG (same seed, isolated stream) so that enabling/
        # disabling disfluency or prosody randomization never shifts gap draws,
        # and vice versa.
        gap_rng = random.Random(rng_seed)
        mixer = SceneMixer()
        gap_ctrl = TurnGapController(project=project)

        # M7: one SpeakerState per speaker; starts neutral, updated after each turn.
        states: dict[str, SpeakerState] = {sid: SpeakerState() for sid in speakers}

        segments: list[tuple[bytes, float, str, float | None, MixMode]] = []
        prev_turn: DialogueTurn | None = None
        for i, turn in enumerate(turns):
            speaker = speakers[turn.speaker_id]
            state = states[turn.speaker_id]
            # Use text_spoken (post-gender-disambiguation) rather than the
            # raw LLM text so that niqqud corrections reach the TTS engine.
            text = turn.text_spoken
            # M2b: resolve phrase hints (LLM annotations + imperative heuristics).
            phrases = collect_phrase_prosody(
                turn.phrase_hints,
                turn.text_original,
                turn.text_spoken,
            )
            if disfluency:
                new_text = inject_disfluency(
                    text,
                    prob=speaker.disfluency.filled_pause_prob,
                    rng_seed=rng.randint(0, 2**31),
                )
                if phrases and new_text != text:
                    phrases = rebase_phrase_prosody(phrases, text, new_text)
                text = new_text
            # Snapshot pre-render state for reproducibility metadata (prep for M11).
            turn.speaker_state_snapshot = state.to_metadata_dict()
            wav_bytes, _, hit = self.render_utterance(
                text,
                speaker,
                turn.intensity,
                randomize=randomize,
                rng_seed=rng.randint(0, 2**31) if randomize else None,
                speaker_state=state,
                phrase_prosody=phrases if phrases else None,
            )
            # Update state after rendering so the first turn always uses neutral
            # state and drift accumulates from the second turn onward.
            state.update(turn.intensity, speaker.role)
            if verbose_log is not None:
                status = "cache hit" if hit else "rendered"
                verbose_log(
                    f"  [dim]turn {i + 1:02d}/{len(turns):02d}"
                    f" [{turn.speaker_id}] intensity={turn.intensity}"
                    f" → {status}[/dim]"
                )
            prev_role = speakers[prev_turn.speaker_id].role if prev_turn is not None else None
            gap_s, mix_mode = gap_ctrl.gap_seconds(
                turn, prev_turn, gap_rng, speaker.role, prev_role
            )
            segments.append(
                (
                    wav_bytes,
                    gap_s,
                    turn.speaker_id,
                    speaker.style_for_intensity(turn.intensity).rms_target_dbfs,
                    mix_mode,
                )
            )
            prev_turn = turn

        return mixer.mix_sequential(segments)
