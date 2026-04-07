"""SceneMixer: concatenate per-speaker TTS WAV segments into a single audio scene.

Each segment is a (wav_bytes, pause_before_s, speaker_id) triple.  The mixer
decodes WAV bytes using soundfile, resamples to 16 kHz if needed, prepends the
requested silence gap, and concatenates all segments into a single float32 mono
array while preserving speaker IDs in the mix metadata.

The output MixedScene carries per-turn onset/offset times so the label generator
can derive event timing from the mix log rather than re-estimating it from the
final waveform.

Spec reference: docs/spec.md §3.1
"""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from synthbanshee.augment.preprocessing import _resample
from synthbanshee.script.types import MixedScene

_TARGET_SR = 16_000


class SceneMixer:
    """Mix a sequence of TTS segments into a single-track 16 kHz scene."""

    def mix_sequential(
        self,
        segments: list[tuple[bytes, float, str]],
    ) -> MixedScene:
        """Concatenate segments in order, separated by silence gaps.

        Args:
            segments: List of (wav_bytes, pause_before_s, speaker_id) triples.
                      wav_bytes must be valid WAV data (any SR / channels).
                      pause_before_s is inserted *before* each segment.
                      speaker_id is stored in the MixedScene for labelling.

        Returns:
            MixedScene with all segments concatenated at 16 kHz mono.
        """
        all_samples: list[np.ndarray] = []
        turn_onsets: list[float] = []
        turn_offsets: list[float] = []
        speaker_ids: list[str] = []
        current_pos_s: float = 0.0

        for wav_bytes, pause_s, speaker_id in segments:
            # --- Decode WAV ---
            with io.BytesIO(wav_bytes) as buf:
                data, src_sr = sf.read(buf, dtype="float32", always_2d=True)

            # Downmix to mono
            mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]

            # Resample to 16 kHz if needed
            if src_sr != _TARGET_SR:
                mono = _resample(mono, src_sr, _TARGET_SR)

            # Prepend silence gap
            if pause_s > 0.0:
                silence = np.zeros(int(pause_s * _TARGET_SR), dtype=np.float32)
                all_samples.append(silence)
                current_pos_s += pause_s

            onset_s = current_pos_s
            turn_onsets.append(onset_s)

            all_samples.append(mono.astype(np.float32))
            seg_duration_s = len(mono) / _TARGET_SR
            current_pos_s += seg_duration_s

            turn_offsets.append(current_pos_s)
            speaker_ids.append(speaker_id)

        combined = np.concatenate(all_samples) if all_samples else np.zeros(0, dtype=np.float32)

        return MixedScene(
            samples=combined,
            sample_rate=_TARGET_SR,
            turn_onsets_s=turn_onsets,
            turn_offsets_s=turn_offsets,
            duration_s=float(len(combined)) / _TARGET_SR,
            speaker_ids=speaker_ids,
        )
