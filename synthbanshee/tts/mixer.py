"""SceneMixer: concatenate per-speaker TTS WAV segments into a single audio scene.

Each segment is a (wav_bytes, amount_s, speaker_id, rms_target_dbfs, mix_mode) 5-tuple.
The mixer decodes WAV bytes using soundfile, resamples to 16 kHz if needed, applies
optional per-turn RMS gain (M3), then places each turn in the output buffer according
to its MixMode:

  SEQUENTIAL  — insert a silence gap of *amount_s* before this turn (existing behaviour).
  OVERLAP     — start *amount_s* seconds before the previous turn ends; both turns are
                audible in the overlap region.
  BARGE_IN    — same positioning as OVERLAP but the previous turn's audio is truncated
                at the point where the current turn begins.

The output MixedScene carries per-turn timing metadata on three timelines (§4.6):

  script_onsets_s / script_offsets_s    — sequential-world positions (no overlap applied);
                                          script_offsets_s retains the original TTS duration
                                          even for BARGE_IN-truncated turns.
  rendered_onsets_s / rendered_offsets_s — actual onset/offset in the output buffer;
                                          for BARGE_IN-interrupted turns, rendered_offsets_s
                                          is updated to the truncation point.
  audible_onsets_s / audible_ends_s     — what is audible (same as rendered; both are
                                          at the truncation point for interrupted turns).

For backward compatibility, turn_onsets_s and turn_offsets_s mirror the *audible* timeline
(audible_onsets_s and audible_ends_s respectively) so that all offsets stay within the
final waveform duration.

Spec reference: docs/audio_generation_v3_design.md §4.6
"""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from synthbanshee.augment.preprocessing import _resample
from synthbanshee.script.types import MixedScene
from synthbanshee.tts.mix_mode import MixMode

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
        segments: list[tuple[bytes, float, str, float | None, MixMode]],
    ) -> MixedScene:
        """Place segments in the output buffer according to their MixMode.

        Args:
            segments: List of (wav_bytes, amount_s, speaker_id,
                      rms_target_dbfs, mix_mode) 5-tuples.
                      wav_bytes — valid WAV data (any SR / channels).
                      amount_s — silence gap for SEQUENTIAL; overlap depth for
                        OVERLAP / BARGE_IN (how far back into the previous turn
                        the current turn starts).
                      speaker_id — stored in the MixedScene for labelling.
                      rms_target_dbfs — if not None, the segment is gain-adjusted
                        so its RMS matches this target (dBFS).
                      mix_mode — placement strategy (MixMode enum).

        Returns:
            MixedScene with all segments mixed at 16 kHz mono, carrying three
            sets of per-turn timing timestamps (script / rendered / audible).
        """
        # Decoded and gain-adjusted mono arrays paired with their onset sample.
        placed: list[tuple[np.ndarray, int]] = []  # (mono, onset_sample_idx)

        # Three-timeline metadata (§4.6).
        script_onsets: list[float] = []
        script_offsets: list[float] = []
        rendered_onsets: list[float] = []
        rendered_offsets: list[float] = []
        audible_onsets: list[float] = []
        audible_ends: list[float] = []
        speaker_ids: list[str] = []

        # render_cursor_s tracks the end of the last placed segment in buffer time.
        # script_cursor_s advances sequentially (overlap is not subtracted).
        render_cursor_s: float = 0.0
        script_cursor_s: float = 0.0

        for wav_bytes, amount_s, speaker_id, rms_target_dbfs, mix_mode in segments:
            # Clamp amount_s: negative values are nonsensical for all mix modes
            # (gap for SEQUENTIAL; overlap depth for OVERLAP / BARGE_IN).
            amount_s = max(0.0, amount_s)

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

            mono = mono.astype(np.float32)
            seg_duration_s = len(mono) / _TARGET_SR

            # --- Determine onset position ---
            if mix_mode == MixMode.SEQUENTIAL:
                gap_s = amount_s
                onset_s = render_cursor_s + gap_s
                script_onset_s = script_cursor_s + gap_s
                script_cursor_s = script_onset_s + seg_duration_s
            else:
                # OVERLAP / BARGE_IN: go back *amount_s* into the previous turn,
                # but never before that turn's actual rendered onset (COPILOT-5).
                if placed:
                    prev_mono, prev_onset_sample = placed[-1]
                    prev_onset_s = prev_onset_sample / _TARGET_SR
                    prev_duration_s = len(prev_mono) / _TARGET_SR
                    prev_offset_s = prev_onset_s + prev_duration_s
                    overlap_s = min(amount_s, prev_duration_s)
                    onset_s = max(prev_onset_s, prev_offset_s - overlap_s)
                else:
                    onset_s = max(0.0, render_cursor_s - amount_s)
                # In script space the turn still follows the previous one (no gap).
                script_onset_s = script_cursor_s
                script_cursor_s = script_onset_s + seg_duration_s

            onset_sample = int(onset_s * _TARGET_SR)
            # Quantise onset/offset to the actual sample boundaries used for buffer
            # placement so all timeline metadata stays exactly sample-aligned.
            onset_s = float(onset_sample) / _TARGET_SR
            offset_s = float(onset_sample + len(mono)) / _TARGET_SR

            # --- BARGE_IN: truncate the previous segment at the barge-in point ---
            if mix_mode == MixMode.BARGE_IN and placed:
                prev_mono, prev_onset_sample = placed[-1]
                max_samples = onset_sample - prev_onset_sample
                if max_samples <= 0:
                    # Full-depth barge-in: new turn starts at or before the previous
                    # turn's onset — replace it with an empty array.
                    placed[-1] = (np.zeros(0, dtype=np.float32), prev_onset_sample)
                    prev_onset_s_f = float(prev_onset_sample) / _TARGET_SR
                    rendered_offsets[-1] = prev_onset_s_f
                    audible_ends[-1] = prev_onset_s_f
                elif max_samples < len(prev_mono):
                    placed[-1] = (prev_mono[:max_samples], prev_onset_sample)
                    # onset_s is already quantised to the sample boundary above.
                    rendered_offsets[-1] = onset_s
                    audible_ends[-1] = onset_s

            placed.append((mono, onset_sample))

            # Recompute cursor from the actual max end-sample across all placed
            # segments so BARGE_IN truncation can move it earlier and OVERLAP
            # never overstates the true audio extent.
            render_cursor_s = max(onset + len(m) for m, onset in placed) / _TARGET_SR

            script_onsets.append(script_onset_s)
            script_offsets.append(script_onset_s + seg_duration_s)
            rendered_onsets.append(onset_s)
            rendered_offsets.append(offset_s)
            audible_onsets.append(onset_s)
            audible_ends.append(offset_s)
            speaker_ids.append(speaker_id)

        # --- Build output buffer ---
        if placed:
            total_samples = max(onset + len(mono) for mono, onset in placed)
            combined = np.zeros(total_samples, dtype=np.float32)
            for mono, onset in placed:
                combined[onset : onset + len(mono)] += mono
        else:
            combined = np.zeros(0, dtype=np.float32)

        # turn_onsets_s / turn_offsets_s mirror the audible timeline so that all
        # offset values stay within the waveform duration (backward-compat, COPILOT-2).
        return MixedScene(
            samples=combined,
            sample_rate=_TARGET_SR,
            turn_onsets_s=audible_onsets,
            turn_offsets_s=audible_ends,
            duration_s=float(len(combined)) / _TARGET_SR,
            speaker_ids=speaker_ids,
            script_onsets_s=script_onsets,
            script_offsets_s=script_offsets,
            rendered_onsets_s=rendered_onsets,
            rendered_offsets_s=rendered_offsets,
            audible_onsets_s=audible_onsets,
            audible_ends_s=audible_ends,
        )
