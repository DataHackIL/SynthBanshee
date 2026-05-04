"""SceneMixer: concatenate per-speaker TTS WAV segments into a single audio scene.

Each segment is a ``Segment`` dataclass (wav_bytes, amount_s, speaker_id,
rms_target_dbfs, mix_mode, intensity).  The mixer decodes WAV bytes using
soundfile, resamples to 16 kHz if needed, applies optional per-turn RMS
gain (M3) and a Lombard spectral tilt at I4–I5 (#65), then places each turn
in the output buffer according to its MixMode:

  SEQUENTIAL  — insert a silence gap of *amount_s* before this turn (existing behaviour).
  OVERLAP     — start *amount_s* seconds before the previous turn ends; both turns are
                audible in the overlap region.
  BARGE_IN    — the new turn onsets inside the previous turn (anchored against the
                previous turn's *speech* end, not file end — #66); the previous turn
                continues through its natural speech end with a linear fade-out spanning
                the overlap region, producing an audible crossfade between the two voices.

The output MixedScene carries per-turn timing metadata on three timelines (§4.6):

  script_onsets_s / script_offsets_s    — sequential-world positions (no overlap applied);
                                          script_offsets_s retains the original TTS duration
                                          even for BARGE_IN-truncated turns.
  rendered_onsets_s / rendered_offsets_s — actual onset/offset in the output buffer;
                                          for BARGE_IN-interrupted turns, rendered_offsets_s
                                          is updated to the truncation point (the previous
                                          turn's speech end), which sits *after* the new
                                          turn's rendered onset by the overlap-region length.
  audible_onsets_s / audible_ends_s     — what is audible (same as rendered; both at the
                                          truncation point for interrupted turns).

For backward compatibility, turn_onsets_s and turn_offsets_s mirror the *audible* timeline
(audible_onsets_s and audible_ends_s respectively) so that all offsets stay within the
final waveform duration.

Spec reference: docs/audio_generation_v3_design.md §4.6
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

from synthbanshee.augment.preprocessing import _resample
from synthbanshee.script.types import MixedScene
from synthbanshee.tts.mix_mode import MixMode

_TARGET_SR = 16_000
_EDGE_FADE_SAMPLES = 160  # 10ms at 16 kHz — eliminates DC-offset clicks at turn boundaries

# #66 BARGE_IN tail-trim: TTS voices pad the end of an utterance with 100–300 ms
# of near-silence (Azure ~150 ms, Google ~200 ms).  Anchoring BARGE_IN onset to
# file end therefore lands the overlap inside that padding — listener hears the
# previous speaker stop, then a gap, then the new speaker start.  We measure
# speech end via a windowed RMS scan and anchor the overlap to that instead.
_SPEECH_END_WINDOW_MS = 30.0
_SPEECH_END_THRESHOLD_DBFS = -40.0

# #65 Lombard effect: post-TTS spectral tilt at high intensities.
# Real raised voices boost high-frequency energy; pure amplitude scaling sounds
# like mic proximity instead of shouting.  Mapping is intentionally narrow —
# only I4 and I5 get a high-shelf boost; all other intensities are untouched.
# Values are listening-test-derived placeholders (#65); revisit on re-listening.
_LOMBARD_GAIN_DB: dict[int, float] = {4: 2.0, 5: 3.5}
_LOMBARD_SHELF_HZ = 2500.0


@dataclass
class Segment:
    """A single TTS turn ready to be placed on the scene timeline.

    Carries everything ``SceneMixer.mix_sequential`` needs to decode, condition,
    and position one turn.  Replaces the historical 5/6-element tuple — fields
    are named so call sites and reviewers can't transpose them.

    Attributes:
        wav_bytes: Raw WAV (any SR / channels — the mixer downmixes and resamples).
        amount_s: Silence gap (SEQUENTIAL) or overlap depth (OVERLAP / BARGE_IN).
        speaker_id: Stored on the resulting MixedScene for labelling.
        rms_target_dbfs: If not None, the segment is gain-adjusted to this dBFS.
        mix_mode: Placement strategy.
        intensity: 1–5 turn intensity.  Drives the Lombard high-shelf at I4/I5
            (#65); any other value is a no-op.  Defaults to 1 so test fixtures
            that don't care about Lombard get the natural "no boost" sentinel.
    """

    wav_bytes: bytes
    amount_s: float
    speaker_id: str
    rms_target_dbfs: float | None
    mix_mode: MixMode
    intensity: int = 1


def _precompute_lombard_coeffs() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Pre-bake biquad coefficients for every intensity that gets a shelf.

    The coefficients depend only on (gain_db, f0, sample_rate), and we know all
    three at import time.  Computing once avoids redundant trig in the per-turn
    hot loop.
    """
    return {
        intensity: _highshelf_biquad(gain_db, _LOMBARD_SHELF_HZ, _TARGET_SR)
        for intensity, gain_db in _LOMBARD_GAIN_DB.items()
    }


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


def _highshelf_biquad(gain_db: float, f0: float, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (b, a) biquad coefficients for an RBJ high-shelf filter.

    Uses slope S=1, which is the canonical "natural" shelf shape.
    """
    a_amp = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / sample_rate
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    # alpha for slope S=1 simplifies to sin(w0)/sqrt(2)
    alpha = sin_w0 / math.sqrt(2.0)
    sqrt_a = math.sqrt(a_amp)
    two_sqrt_a_alpha = 2.0 * sqrt_a * alpha

    b0 = a_amp * ((a_amp + 1.0) + (a_amp - 1.0) * cos_w0 + two_sqrt_a_alpha)
    b1 = -2.0 * a_amp * ((a_amp - 1.0) + (a_amp + 1.0) * cos_w0)
    b2 = a_amp * ((a_amp + 1.0) + (a_amp - 1.0) * cos_w0 - two_sqrt_a_alpha)
    a0 = (a_amp + 1.0) - (a_amp - 1.0) * cos_w0 + two_sqrt_a_alpha
    a1 = 2.0 * ((a_amp - 1.0) - (a_amp + 1.0) * cos_w0)
    a2 = (a_amp + 1.0) - (a_amp - 1.0) * cos_w0 - two_sqrt_a_alpha

    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return b, a


def _apply_lombard_tilt(mono: np.ndarray, intensity: int) -> np.ndarray:
    """Apply a Lombard-effect spectral tilt for I4/I5 turns (#65).

    A high-shelf biquad at 2.5 kHz boosts upper formants so the turn sounds
    like a raised voice rather than a microphone moved closer.  Asymptotic
    gain is +2.0 dB at I4 and +3.5 dB at I5; at the 2.5 kHz knee the gain is
    half that.  Any other intensity is a no-op.

    The input *mono* is assumed to be at ``_TARGET_SR`` (16 kHz) — the mixer
    resamples before this step, so the precomputed biquad coefficients match.

    Args:
        mono: Float mono audio array at 16 kHz.
        intensity: 1–5 intensity level.

    Returns:
        float32 array with the shelf applied, or *mono* unchanged when no
        shelf is defined for this intensity.  No clipping is performed;
        downstream peak limiting handles amplitude headroom.
    """
    coeffs = _LOMBARD_COEFFS.get(intensity)
    if coeffs is None or len(mono) == 0:
        return mono
    b, a = coeffs
    return lfilter(b, a, mono).astype(np.float32)


_LOMBARD_COEFFS: dict[int, tuple[np.ndarray, np.ndarray]] = _precompute_lombard_coeffs()


def _speech_end_sample(
    mono: np.ndarray,
    sample_rate: int = _TARGET_SR,
    window_ms: float = _SPEECH_END_WINDOW_MS,
    threshold_dbfs: float = _SPEECH_END_THRESHOLD_DBFS,
) -> int:
    """Return the sample index just after the last window of speech-level energy.

    Walks the tail backwards in non-overlapping windows; returns the end index
    of the first window (from the tail) whose RMS exceeds *threshold_dbfs*.
    Returns ``len(mono)`` when the input is empty (defensive: caller already
    handles the empty case but the contract stays safe under refactors), and
    ``0`` when no window crosses the threshold (silent input).

    Used by BARGE_IN onset placement (#66) to skip TTS trailing-silence padding.
    """
    if len(mono) == 0:
        return 0
    win = max(1, int(window_ms * sample_rate / 1000.0))
    threshold = 10.0 ** (threshold_dbfs / 20.0)
    for end in range(len(mono), 0, -win):
        start = max(0, end - win)
        rms = float(np.sqrt(np.mean(mono[start:end] ** 2)))
        if rms >= threshold:
            return end
    return 0


def _apply_edge_fades(mono: np.ndarray, n: int = _EDGE_FADE_SAMPLES) -> np.ndarray:
    """Apply linear fade-in and fade-out of *n* samples to eliminate boundary clicks.

    Short segments (< 2*n samples) get proportionally shorter fades.
    """
    if len(mono) == 0 or n <= 0:
        return mono
    fade_len = min(n, len(mono) // 2)
    if fade_len <= 0:
        return mono
    out = mono.copy()
    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    out[:fade_len] *= fade_in
    out[-fade_len:] *= fade_out
    return out


class SceneMixer:
    """Mix a sequence of TTS segments into a single-track 16 kHz scene."""

    def mix_sequential(self, segments: list[Segment]) -> MixedScene:
        """Place segments in the output buffer according to their MixMode.

        See ``Segment`` for the per-turn input fields.

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
        mix_modes: list[str] = []

        # render_cursor_s tracks the end of the last placed segment in buffer time.
        # script_cursor_s advances sequentially (overlap is not subtracted).
        render_cursor_s: float = 0.0
        script_cursor_s: float = 0.0

        for seg in segments:
            # Clamp amount_s: negative values are nonsensical for all mix modes
            # (gap for SEQUENTIAL; overlap depth for OVERLAP / BARGE_IN).
            amount_s = max(0.0, seg.amount_s)

            # --- Decode WAV ---
            with io.BytesIO(seg.wav_bytes) as buf:
                data, src_sr = sf.read(buf, dtype="float32", always_2d=True)

            # Downmix to mono
            mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]

            # Resample to 16 kHz if needed
            if src_sr != _TARGET_SR:
                mono = _resample(mono, src_sr, _TARGET_SR)

            # M3: apply per-turn RMS gain before mixing
            if seg.rms_target_dbfs is not None:
                mono = _apply_rms_gain(mono, seg.rms_target_dbfs)

            # #65: apply Lombard high-shelf tilt at I4/I5 (no-op below I4).
            # Runs after RMS gain so the boost shapes the already-loud signal.
            mono = _apply_lombard_tilt(mono, seg.intensity)

            # M14: apply fade-in/fade-out to eliminate DC-offset clicks.
            # The fade-in also masks the lfilter transient introduced by Lombard.
            mono = _apply_edge_fades(mono.astype(np.float32))
            seg_duration_s = len(mono) / _TARGET_SR

            # --- Determine onset position ---
            if seg.mix_mode == MixMode.SEQUENTIAL:
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
                    if seg.mix_mode == MixMode.BARGE_IN:
                        # #66: anchor against speech end, not file end — TTS
                        # trailing-silence padding would otherwise eat the
                        # entire overlap and produce an audible turn-taking gap.
                        anchor_samples = _speech_end_sample(prev_mono, _TARGET_SR)
                    else:
                        anchor_samples = len(prev_mono)
                    anchor_duration_s = float(anchor_samples) / _TARGET_SR
                    overlap_s = min(amount_s, anchor_duration_s)
                    onset_s = max(prev_onset_s, prev_onset_s + anchor_duration_s - overlap_s)
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

            # --- BARGE_IN: continue prev through its speech end, fade out across overlap ---
            # Old behaviour cut prev *at* the new onset, leaving zero audible overlap and
            # (worse) a perceptual gap when the cut landed inside TTS trailing silence (#66).
            # New behaviour: prev plays through its natural speech end (or the file end —
            # whichever is reached first), with a linear fade-out applied across the overlap
            # region so the two voices crossfade naturally.
            if seg.mix_mode == MixMode.BARGE_IN and placed:
                prev_mono, prev_onset_sample = placed[-1]
                onset_offset_in_prev = onset_sample - prev_onset_sample
                # Truncate at speech end so trailing TTS silence doesn't leak into the buffer
                # past the new turn's onset.
                speech_end_in_prev = _speech_end_sample(prev_mono, _TARGET_SR)
                truncate_at = min(speech_end_in_prev, len(prev_mono))
                if onset_offset_in_prev <= 0:
                    # Full-depth barge-in: new turn starts at or before prev's onset —
                    # replace it with an empty array.
                    placed[-1] = (np.zeros(0, dtype=np.float32), prev_onset_sample)
                    prev_onset_s_f = float(prev_onset_sample) / _TARGET_SR
                    rendered_offsets[-1] = prev_onset_s_f
                    audible_ends[-1] = prev_onset_s_f
                elif truncate_at > onset_offset_in_prev:
                    # Standard overlap path: prev extends past the new onset by the
                    # overlap region; fade out across that region.
                    truncated = prev_mono[:truncate_at].copy()
                    fade_n = truncate_at - onset_offset_in_prev
                    truncated[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
                    placed[-1] = (truncated, prev_onset_sample)
                    new_offset_s = float(prev_onset_sample + truncate_at) / _TARGET_SR
                    rendered_offsets[-1] = new_offset_s
                    audible_ends[-1] = new_offset_s
                # Else (truncate_at <= onset_offset_in_prev): prev's speech already ended
                # before the new onset.  Leave prev untouched — its trailing silence (if any)
                # will simply sum with the new turn at low energy, producing no audible gap.

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
            speaker_ids.append(seg.speaker_id)
            mix_modes.append(seg.mix_mode.value)

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
            mix_modes=mix_modes,
        )
