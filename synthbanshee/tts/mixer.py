"""SceneMixer: concatenate per-speaker TTS WAV segments into a single audio scene.

Each segment is a (wav_bytes, pause_before_s, speaker_id, rms_target_dbfs) 4-tuple.
The mixer decodes WAV bytes using soundfile, resamples to 16 kHz if needed, applies
optional per-turn RMS gain (M3), prepends the requested silence gap, and concatenates
all segments into a single float32 mono array while preserving speaker IDs in the mix
metadata.

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


def _apply_rms_gain(mono: np.ndarray, rms_target_dbfs: float) -> np.ndarray:
    """Scale *mono* so its RMS equals *rms_target_dbfs* dBFS.

    Near-silent frames (RMS < −80 dBFS) are returned unchanged to avoid
    amplifying noise floors to absurd levels.  The result is not clipped;
    clipping is handled downstream by the preprocessing peak-normalizer.

    Args:
        mono: Float32 mono audio array (values in [−1, 1] range expected).
        rms_target_dbfs: Desired RMS level in dBFS (negative float, e.g. −20.0).

    Returns:
        Gain-adjusted float32 array.
    """
    rms = float(np.sqrt(np.mean(mono**2)))
    if rms < 1e-4:  # ~−80 dBFS — treat as silence, skip gain
        return mono
    current_dbfs = 20.0 * np.log10(rms)
    gain_db = rms_target_dbfs - current_dbfs
    gain_linear = 10.0 ** (gain_db / 20.0)
    return (mono * gain_linear).astype(np.float32)


class SceneMixer:
    """Mix a sequence of TTS segments into a single-track 16 kHz scene."""

    def mix_sequential(
        self,
        segments: list[tuple[bytes, float, str, float | None]],
    ) -> MixedScene:
        """Concatenate segments in order, separated by silence gaps.

        Args:
            segments: List of (wav_bytes, pause_before_s, speaker_id,
                      rms_target_dbfs) 4-tuples.
                      wav_bytes must be valid WAV data (any SR / channels).
                      pause_before_s is inserted *before* each segment.
                      speaker_id is stored in the MixedScene for labelling.
                      rms_target_dbfs: if not None, the segment is gain-adjusted
                      so its RMS matches this target (dBFS) after decoding and
                      resampling.

        Returns:
            MixedScene with all segments concatenated at 16 kHz mono.
        """
        all_samples: list[np.ndarray] = []
        turn_onsets: list[float] = []
        turn_offsets: list[float] = []
        speaker_ids: list[str] = []
        current_pos_s: float = 0.0

        for wav_bytes, pause_s, speaker_id, rms_target_dbfs in segments:
            # --- Decode WAV ---
            with io.BytesIO(wav_bytes) as buf:
                data, src_sr = sf.read(buf, dtype="float32", always_2d=True)

            # Downmix to mono
            mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]

            # Resample to 16 kHz if needed
            if src_sr != _TARGET_SR:
                mono = _resample(mono, src_sr, _TARGET_SR)

            # M3: apply per-turn RMS gain before mixing
            if rms_target_dbfs is not None:
                mono = _apply_rms_gain(mono, rms_target_dbfs)

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
