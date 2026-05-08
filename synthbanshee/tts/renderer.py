"""TTS rendering orchestrator with asset cache.

Cache key: SHA-256 of the rendered SSML string, which captures voice, text,
style, prosody parameters, and any randomization — so two calls with different
intensity or prosody will never collide.
Cache format: raw WAV bytes stored as {cache_key}.wav in the cache directory.

Spec reference: docs/implementation_plan.md §0.3
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from synthbanshee.config.speaker_config import SpeakerConfig

if TYPE_CHECKING:
    from synthbanshee.config.project_profile import ProjectProfile
from synthbanshee.script.types import DialogueTurn, MixedScene
from synthbanshee.tts.provider import TTSProvider
from synthbanshee.tts.quality_gates import run_quality_gates
from synthbanshee.tts.speaker_state import SpeakerState
from synthbanshee.tts.ssml_builder import SSMLBuilder
from synthbanshee.tts.ssml_types import PhraseProsody, collect_phrase_prosody, rebase_phrase_prosody

_log = logging.getLogger(__name__)

# Verbose-log callback type.  Strings passed to this callback may contain
# Rich markup tags (e.g. ``[dim]…[/dim]``).  Callers that do not use Rich
# (e.g. standard loggers) should strip or ignore the markup.
_VerboseLog = Callable[[str], None]


# ---------------------------------------------------------------------------
# Effective-prosody cap (#87)
# ---------------------------------------------------------------------------
#
# Bound the post-state, post-randomization prosody before SSML emission.  This
# defends against two correlated failure modes that surfaced in #87:
#
#   1. M7 ``SpeakerState`` drift compounds with #51's M15 style_map values to
#      push effective pitch past +2.5 st (~+15%) and effective rate past 1.30×
#      at I4/I5 turns.  Listening test (2026-05-03) flagged that range as
#      cartoonish ("helium / oompa-loompa").
#   2. The same effective-prosody range trips Whisper-large-v3's silence-
#      detection heuristic — pitch/rate combinations outside the natural-speech
#      training distribution cause the encoder to mis-segment chunks as silence,
#      producing 6× WER regression with length-ratio collapse to ~0.7
#      (the #87 fingerprint, also seen on `sp_it_a_0001`).
#
# Volume is intentionally NOT capped here — Whisper's log-mel feature extractor
# internally normalizes loudness (per #82's lever probe, 7 of 8 peak/RMS variants
# produced byte-identical Whisper hypotheses), so loudness is not a Whisper-trip
# dimension.  Existing ±50% Azure clamp in ``ssml_builder._volume_to_string``
# remains the only volume bound.
#
# Caps are anchored to the pre-#51 effective prosody envelope, which produced
# the 04-15 reference clips with WER 0.04–0.08.  Tighter caps would diverge
# further from M15 listening-test calibration; looser caps would re-trip
# Whisper.  Revisit with a paired listening test + Whisper sanity check
# (`qa-report --asr`) before changing.
_EFFECTIVE_PITCH_MAX_ST = 2.0  # ≈ +12% — pre-#51 AGG never exceeded this with drift
_EFFECTIVE_PITCH_MIN_ST = -3.0  # pre-#51 VIC went down to -4 baseline + drift
_EFFECTIVE_RATE_MAX = 1.20
# #91 R experiment: rate floor lifted 0.85 → 0.95.  PR #90's symmetric pitch cap
# closed most of the #87 WER gap on `sp_it_a_0001` (0.322 → 0.129) but left a
# residual to the 04-15 baseline (0.056).  #89 falsified pitch as the residual
# driver; rate is the next single-knob lever.  Hypothesis: VIC's I4/I5 slowdown
# (style_map 0.90 × baseline 1.0 × drift → floors at 0.85) is what trips Whisper's
# silence-detection heuristic.  Lifting the floor risks flattening VIC's
# deliberate distress cue — listening test is the merge gate.
_EFFECTIVE_RATE_MIN = 0.95


# ---------------------------------------------------------------------------
# VIC high-intensity express-as override (originally #97 spike, kept generic)
# ---------------------------------------------------------------------------
#
# Per-call A/B knob for Azure ``<mstts:express-as>`` styles on VIC at high
# intensity, without touching any speaker YAML or the provider's global
# ``supports_style_tags`` capability.  When the env var is set:
#   - VIC role at intensity ≥ 4
#   - The renderer overrides ``style_entry.style`` with the env value AND
#     forces ``supports_style_tags=True`` for this single SSML build (so the
#     Azure default ``False`` is preserved for every other turn).
#
# Origin: the 2026-05-07 listening test on PR #95 said VIC at I3–I5 sounds
# robotic; rate/pitch alone don't carry distress on Hebrew Azure voices.
# #97 used this knob to A/B-test ``fearful`` and found Azure he-IL returns
# byte-identical audio regardless of the express-as wrapper — see
# ``tests/fixtures/issue_97_fearful_ssml.json``.  The mechanism is left in
# place so the next style spike (e.g. ``terrified``) can run with no code
# changes; if all styles are confirmed no-op, delete this block and the
# unit test in ``tests/unit/test_vic_style_override.py``.
_VIC_STYLE_OVERRIDE_ENV_VAR = "SYNTHBANSHEE_VIC_STYLE_OVERRIDE"
_VIC_STYLE_OVERRIDE_MIN_INTENSITY = 4
_VIC_STYLE_OVERRIDE_ROLE = "VIC"


def _apply_effective_prosody_cap(
    rate: float,
    pitch: float,
    volume: float,
    *,
    out_cap_events: list[dict] | None = None,
) -> tuple[float, float, float]:
    """Clamp ``rate`` and ``pitch`` to the #87 effective-prosody envelope.

    Returns the (possibly clamped) ``(rate, pitch, volume)`` tuple.  When
    ``out_cap_events`` is provided, one event dict is appended per dimension
    that was actually clamped, with keys ``dim``, ``pre_cap``, ``post_cap``.
    Volume is returned unchanged (see module-level rationale).
    """
    new_pitch = max(_EFFECTIVE_PITCH_MIN_ST, min(_EFFECTIVE_PITCH_MAX_ST, pitch))
    if new_pitch != pitch:
        if out_cap_events is not None:
            out_cap_events.append(
                {"dim": "pitch", "pre_cap": round(pitch, 4), "post_cap": round(new_pitch, 4)}
            )
        _log.warning("Effective pitch %+.2f st clamped to %+.2f st (#87 cap)", pitch, new_pitch)
    new_rate = max(_EFFECTIVE_RATE_MIN, min(_EFFECTIVE_RATE_MAX, rate))
    if new_rate != rate:
        if out_cap_events is not None:
            out_cap_events.append(
                {"dim": "rate", "pre_cap": round(rate, 4), "post_cap": round(new_rate, 4)}
            )
        _log.warning("Effective rate %.3f clamped to %.3f (#87 cap)", rate, new_rate)
    return new_rate, new_pitch, volume


class TTSRenderer:
    """Render TTS utterances with a file-system render cache.

    Caches WAV bytes keyed on the full SSML string so that calls with
    different voice, text, style, or prosody always produce distinct entries.

    Supports multiple backends.  The backend for each utterance is chosen
    based on ``SpeakerConfig.tts_provider``.  Dispatch is always strict:
    a ``KeyError`` is raised if the speaker's backend is not registered.

    Args:
        provider: Convenience shorthand that registers a single provider
            under the ``"azure"`` key.  Mutually exclusive with
            ``providers``.
        providers: Mapping of provider name (``"azure"``, ``"google"``) to
            provider instance.  When present, the renderer dispatches based
            on ``speaker.tts_provider``.
        cache_dir: Directory for the WAV render cache.
    """

    def __init__(
        self,
        provider: TTSProvider | None = None,
        cache_dir: Path | str | None = None,
        *,
        providers: dict[str, TTSProvider] | None = None,
    ) -> None:
        if providers is not None and provider is not None:
            raise ValueError("Specify either 'provider' or 'providers', not both.")
        if providers is not None:
            self._providers: dict[str, TTSProvider] = providers
        else:
            if provider is None:
                from synthbanshee.tts.azure_provider import AzureProvider

                provider = AzureProvider()
            self._providers = {"azure": provider}
        self._cache_dir = Path(
            cache_dir
            if cache_dir is not None
            else os.environ.get("SYNTHBANSHEE_CACHE_DIR") or "assets/speech"
        )
        self._ssml_builder = SSMLBuilder()

    def _get_provider(self, speaker: SpeakerConfig) -> TTSProvider:
        """Return the provider for *speaker*.

        Dispatch is always strict: a ``KeyError`` is raised if the
        speaker's ``tts_provider`` is not registered.  Since
        ``SpeakerConfig.tts_provider`` defaults to ``"azure"``, the
        single-provider convenience (``provider=``) works transparently
        for Azure speakers.
        """
        key = speaker.tts_provider
        if key in self._providers:
            return self._providers[key]
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
        out_cap_events: list[dict] | None = None,
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
            out_cap_events: Optional list that the renderer appends to when
                the #87 effective-prosody cap fires.  One dict per clamped
                dimension; see ``_apply_effective_prosody_cap``.  Pass an empty
                list to capture; pass ``None`` to ignore.

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

        # #64: clamp total pitch to StyleEntry bounds (±12 st) to prevent
        # unbounded drift when speaker_state + randomization stack up.
        pitch = max(-12.0, min(12.0, pitch))

        if randomize:
            import random

            rng = random.Random(rng_seed)
            rate *= rng.uniform(0.97, 1.03)
            pitch += rng.uniform(-0.5, 0.5)
            volume += rng.uniform(-1.0, 1.0)

        # #87: cap effective post-state, post-randomization prosody to keep
        # the SSML inside the envelope that Whisper can read AND that the
        # listener finds natural at high intensities.
        rate, pitch, volume = _apply_effective_prosody_cap(
            rate, pitch, volume, out_cap_events=out_cap_events
        )

        provider = self._get_provider(speaker)

        # Optional VIC@I≥4 express-as style override (see module docstring).
        # Forces supports_style_tags=True for this single call so the override
        # actually takes effect — without flipping the provider's global flag.
        style_for_ssml = style_entry.style
        supports_style_tags = provider.capabilities.supports_style_tags
        override_style = os.environ.get(_VIC_STYLE_OVERRIDE_ENV_VAR)
        if (
            override_style
            and speaker.role == _VIC_STYLE_OVERRIDE_ROLE
            and intensity >= _VIC_STYLE_OVERRIDE_MIN_INTENSITY
        ):
            style_for_ssml = override_style
            supports_style_tags = True

        ssml = self._ssml_builder.build_from_speaker_config(
            text=text,
            voice_id=speaker.tts_voice_id,
            style=style_for_ssml,
            rate_multiplier=rate,
            pitch_delta_st=pitch,
            volume_delta_db=volume,
            phrase_prosody=phrase_prosody,
            supports_style_tags=supports_style_tags,
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
        project_profile: ProjectProfile | None = None,
        quality_gates: bool = True,
        quality_gate_retries: int = 2,
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
            project_profile: Optional ``ProjectProfile`` instance (M13).
                     When provided, gap timing and overlap probabilities are
                     loaded from the profile instead of the hardcoded tables.
            quality_gates: If True (default), run M15 turn-level quality gates
                     after each render.  Set False to skip for faster batch runs.
            quality_gate_retries: Number of re-render attempts on gate failure
                     (default 2).  Each retry uses a different random seed.

        Returns:
            MixedScene with concatenated audio and per-turn timing metadata.

        Raises:
            KeyError: If a turn references a speaker_id not in *speakers*.
        """
        import random

        from synthbanshee.script.generator import inject_disfluency
        from synthbanshee.tts.gap_controller import TurnGapController
        from synthbanshee.tts.mixer import SceneMixer, Segment

        rng = random.Random(rng_seed)
        # Separate gap RNG (same seed, isolated stream) so that enabling/
        # disabling disfluency or prosody randomization never shifts gap draws,
        # and vice versa.
        gap_rng = random.Random(rng_seed)
        mixer = SceneMixer()
        if project_profile is not None:
            gap_ctrl = TurnGapController.from_profile(project, project_profile)
        else:
            gap_ctrl = TurnGapController(project=project)

        # M7: one SpeakerState per speaker; starts neutral, updated after each turn.
        states: dict[str, SpeakerState] = {sid: SpeakerState() for sid in speakers}

        segments: list[Segment] = []
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
            render_seed = rng.randint(0, 2**31) if randomize else None
            cap_events: list[dict] = []
            wav_bytes, _, hit = self.render_utterance(
                text,
                speaker,
                turn.intensity,
                randomize=randomize,
                rng_seed=render_seed,
                speaker_state=state,
                phrase_prosody=phrases if phrases else None,
                out_cap_events=cap_events,
            )
            # M15: run turn-level quality gates with retry on failure.
            if quality_gates:
                from synthbanshee.tts.speaker_state import MAX_F0_DRIFT_ST

                gate_result = run_quality_gates(wav_bytes, speaker.gender)
                max_retries = quality_gate_retries if randomize else 0
                retries_attempted = 0
                while not gate_result.passed and retries_attempted < max_retries:
                    retries_attempted += 1
                    render_seed = rng.randint(0, 2**31)
                    cap_events = []
                    wav_bytes, _, hit = self.render_utterance(
                        text,
                        speaker,
                        turn.intensity,
                        randomize=True,
                        rng_seed=render_seed,
                        speaker_state=state,
                        phrase_prosody=phrases if phrases else None,
                        out_cap_events=cap_events,
                    )
                    gate_result = run_quality_gates(wav_bytes, speaker.gender)
                if not gate_result.passed:
                    failure_str = f"{gate_result.gate_name}: {gate_result.detail}"
                    turn.quality_gate_failures.append(failure_str)
                    if verbose_log is not None:
                        verbose_log(
                            f"  [yellow]turn {i + 1:02d}/{len(turns):02d}"
                            f" [{turn.speaker_id}] quality gate FAILED"
                            f" (after {retries_attempted} retries):"
                            f" {failure_str}[/yellow]"
                        )
            # #87: persist effective-prosody cap activations on the turn so cli.py
            # can roll them up into ClipMetadata.generation_metadata.
            turn.effective_prosody_caps = cap_events
            # Update state after rendering so the first turn always uses neutral
            # state and drift accumulates from the second turn onward.
            state.update(turn.intensity, speaker.role)
            # M15: warn if accumulated F0 drift exceeds the 2.0 st bound.
            if quality_gates and state.f0_drift_exceeded and verbose_log is not None:
                verbose_log(
                    f"  [yellow]turn {i + 1:02d}/{len(turns):02d}"
                    f" [{turn.speaker_id}] F0 drift {state.pitch_offset_st:.2f} st"
                    f" exceeds ±{MAX_F0_DRIFT_ST} st bound[/yellow]"
                )
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
            rms_target = speaker.style_for_intensity(turn.intensity).rms_target_dbfs
            # M13: if the speaker config doesn't set an RMS target for this
            # intensity, fall back to the project profile's role-level default.
            if rms_target is None and project_profile is not None:
                if speaker.role == "AGG":
                    rms_target = project_profile.loudness.agg_rms_dbfs
                elif speaker.role == "VIC":
                    rms_target = project_profile.loudness.vic_rms_dbfs
            segments.append(
                Segment(
                    wav_bytes=wav_bytes,
                    amount_s=gap_s,
                    speaker_id=turn.speaker_id,
                    rms_target_dbfs=rms_target,
                    mix_mode=mix_mode,
                    intensity=turn.intensity,
                )
            )
            prev_turn = turn

        return mixer.mix_sequential(segments)
