"""TTS rendering orchestrator with asset cache.

Cache key: SHA-256 of (voice_id, normalized_text).
Cache format: raw WAV bytes stored as {cache_key}.wav in the cache directory.

Spec reference: docs/implementation_plan.md §0.3
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from avdp.config.speaker_config import SpeakerConfig
from avdp.tts.azure_provider import AzureProvider
from avdp.tts.ssml_builder import SSMLBuilder

_DEFAULT_CACHE_DIR = Path("assets/speech")


class TTSRenderer:
    """Render TTS utterances with a file-system render cache.

    Caches WAV bytes keyed on (voice_id, text) so identical utterances are
    only synthesized once, even across pipeline runs.
    """

    def __init__(
        self,
        provider: AzureProvider | None = None,
        cache_dir: Path | str = _DEFAULT_CACHE_DIR,
    ) -> None:
        self._provider = provider or AzureProvider()
        self._cache_dir = Path(cache_dir)
        self._ssml_builder = SSMLBuilder()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, voice_id: str, text: str) -> str:
        h = hashlib.sha256(f"{voice_id}\x00{text}".encode()).hexdigest()
        return h

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
    ) -> tuple[bytes, str]:
        """Render a single utterance and return (wav_bytes, cache_key).

        Args:
            text: Hebrew utterance text (UTF-8). Must not contain Hebrew in
                  filenames — the text goes to TTS, not into any filename.
            speaker: Speaker persona (SpeakerConfig).
            intensity: Intensity level 1–5, used to look up style_map.
            randomize: If True, apply small random prosody variation.
            rng_seed: Random seed for reproducible prosody variation.

        Returns:
            Tuple of (raw WAV bytes, cache key string).
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

        cache_key = self._cache_key(speaker.tts_voice_id, text)
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached, cache_key

        wav_bytes = self._provider.synthesize(ssml)
        self._save_to_cache(cache_key, wav_bytes)
        return wav_bytes, cache_key

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
        wav_bytes, _ = self.render_utterance(text, speaker, intensity)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(wav_bytes)
        return output_path
