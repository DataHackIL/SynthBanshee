"""Background noise and SFX event mixer for AVDP Tier B clips.

Responsibilities
----------------
* Load or synthesise audio for each BackgroundEvent in AcousticSceneConfig.
* Scale all looping ambient tracks to collectively achieve config.snr_target_db
  relative to the speech signal.
* Insert foreground SFX events (ACOU_* taxonomy codes) at their script-specified
  onsets and log their actual onset/offset times for label generation.
* Return the augmented signal, a structured event log, and the measured SNR.

Asset loading order (per event)
--------------------------------
1. BackgroundEvent.asset_path — explicit path if given and file exists.
2. assets_dir/sfx/<type>*.wav — for ACOU_* types.
3. assets_dir/ambient/<type>*.wav — for ambient types.
4. Synthetic fallback — always available; logs a quality note.

Spec reference: docs/spec.md §3.1 (Stage 3 acoustic augmentation)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt

from synthbanshee.augment.types import AugmentedEvent
from synthbanshee.config.acoustic_config import AcousticSceneConfig, BackgroundEvent

_TARGET_SR = 16_000

# Taxonomy codes that represent foreground sound events (not part of SNR floor).
_ACOU_TYPES = frozenset(
    [
        "ACOU_BREAK",
        "ACOU_SLAM",
        "ACOU_THROW",
        "ACOU_FOOT",
        "ACOU_FALL",
    ]
)

# Default fractional positions for phase-relative onset resolution when
# no phase_boundaries map is provided.
_PHASE_FRACTIONS: dict[str, float] = {
    "baseline": 0.05,
    "tension": 0.25,
    "escalation": 0.45,
    "peak": 0.65,
    "aftermath": 0.80,
    "de-escalation": 0.90,
}


# ---------------------------------------------------------------------------
# Internal audio helpers
# ---------------------------------------------------------------------------


def _rms(samples: np.ndarray) -> float:
    """Return RMS of float32 array."""
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def _to_dbfs(samples: np.ndarray) -> float:
    """Return peak dBFS (−inf if silent)."""
    peak = float(np.max(np.abs(samples)))
    if peak == 0.0:
        return -math.inf
    return 20.0 * math.log10(peak)


def _speech_rms(samples: np.ndarray, frame_ms: int = 20, sr: int = _TARGET_SR) -> float:
    """Estimate RMS over the loudest 50% of frames (excludes silence padding)."""
    frame_n = int(sr * frame_ms / 1000)
    if frame_n == 0 or len(samples) < frame_n:
        return _rms(samples)
    n_frames = len(samples) // frame_n
    frames = samples[: n_frames * frame_n].reshape(n_frames, frame_n)
    frame_rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    threshold = np.median(frame_rms)
    active = frame_rms[frame_rms >= threshold]
    return (
        float(np.sqrt(np.mean(active**2)))
        if len(active) > 0
        else float(np.sqrt(np.mean(frame_rms**2)))
    )


def _snr_db(speech: np.ndarray, noise: np.ndarray) -> float:
    """Return SNR in dB (signal = speech, noise = added background)."""
    s_rms = _speech_rms(speech)
    n_rms = _rms(noise)
    if s_rms == 0.0:
        return -math.inf
    if n_rms == 0.0:
        return math.inf
    return 20.0 * math.log10(s_rms / n_rms)


def _pad_or_trim(audio: np.ndarray, target_n: int) -> np.ndarray:
    """Return float32 array of exactly target_n samples (tiled or truncated)."""
    if len(audio) == 0:
        return np.zeros(target_n, dtype=np.float32)
    if len(audio) >= target_n:
        return audio[:target_n].astype(np.float32)
    # Tile to fill
    reps = math.ceil(target_n / len(audio))
    return np.tile(audio, reps)[:target_n].astype(np.float32)


def _load_audio(path: Path, sr: int = _TARGET_SR) -> np.ndarray:
    """Load a WAV file as float32 mono at *sr* Hz."""
    data, src_sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if src_sr != sr:
        from math import gcd

        g = gcd(sr, src_sr)
        mono = resample_poly(mono, sr // g, src_sr // g).astype(np.float32)
    return mono.astype(np.float32)


# ---------------------------------------------------------------------------
# Synthetic audio generators (fallbacks when no asset files are present)
# ---------------------------------------------------------------------------


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate pink (1/f) noise via FFT shaping."""
    white = rng.standard_normal(n).astype(np.float64)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1.0  # avoid divide-by-zero at DC
    fft /= np.sqrt(freqs)
    pink = np.fft.irfft(fft, n=n).astype(np.float32)
    return pink


def _synthesise_ambient(
    event_type: str,
    duration_s: float,
    sr: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate synthetic ambient audio for common looping event types."""
    n = int(duration_s * sr)
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    if event_type == "hvac_hum":
        # Narrowband noise centred on 120 Hz (2nd harmonic of 60 Hz) + 100 Hz
        t = np.arange(n, dtype=np.float64) / sr
        hum = (
            0.5 * np.sin(2 * np.pi * 100 * t)
            + 0.3 * np.sin(2 * np.pi * 120 * t)
            + 0.2 * np.sin(2 * np.pi * 200 * t)
        )
        hum += 0.05 * rng.standard_normal(n)
        return hum.astype(np.float32)

    if event_type == "distant_phone_ring":
        # NANP dual-tone ring: 480 Hz + 620 Hz, 2 s on / 4 s off pattern
        t = np.arange(n, dtype=np.float64) / sr
        tone = np.sin(2 * np.pi * 480 * t) + np.sin(2 * np.pi * 620 * t)
        cycle = int(6 * sr)  # 6-second ring cycle
        pattern = np.zeros(n, dtype=np.float32)
        for start in range(0, n, cycle):
            end = min(start + int(2 * sr), n)
            pattern[start:end] = tone[start:end]
        return (0.3 * pattern).astype(np.float32)

    # Default (tv_ambient and unknown types): bandpassed pink noise
    pink = _pink_noise(n, rng)
    nyq = sr / 2.0
    sos = butter(4, [300 / nyq, 4000 / nyq], btype="band", output="sos")
    filtered = sosfilt(sos, pink).astype(np.float32)
    return filtered


def _synthesise_sfx(event_type: str, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a synthetic placeholder SFX burst.

    For ACOU_FOOT, returns a sequence of short clicks.
    For all other ACOU_* types, returns white noise with an exponential decay.
    """
    if event_type == "ACOU_FOOT":
        # 3-5 footstep clicks, each ~30 ms
        n_steps = int(rng.integers(3, 6))
        step_n = int(0.03 * sr)
        gap_n = int(rng.uniform(0.08, 0.15) * sr)
        total_n = n_steps * (step_n + gap_n)
        out = np.zeros(total_n, dtype=np.float32)
        for i in range(n_steps):
            start = i * (step_n + gap_n)
            noise = rng.standard_normal(step_n).astype(np.float32)
            env = np.exp(-np.linspace(0, 8, step_n)).astype(np.float32)
            out[start : start + step_n] = noise * env
        return out

    # Default burst: white noise with exponential decay
    durations = {
        "ACOU_BREAK": (0.4, 0.8),
        "ACOU_SLAM": (0.15, 0.35),
        "ACOU_THROW": (0.2, 0.5),
        "ACOU_FALL": (0.5, 1.2),
    }
    dur_min, dur_max = durations.get(event_type, (0.3, 0.6))
    dur_s = float(rng.uniform(dur_min, dur_max))
    n = int(dur_s * sr)
    noise = rng.standard_normal(n).astype(np.float32)
    decay_rate = float(rng.uniform(5.0, 12.0))
    env = np.exp(-decay_rate * np.linspace(0, 1, n)).astype(np.float32)
    # Sharp transient onset
    onset_n = min(int(0.005 * sr), n)
    burst = noise * env
    burst[:onset_n] *= np.linspace(0, 1, onset_n)
    return burst


# ---------------------------------------------------------------------------
# Onset resolution
# ---------------------------------------------------------------------------


def _resolve_onset_s(
    ev: BackgroundEvent,
    duration_s: float,
    phase_boundaries: dict[str, float] | None,
    rng: np.random.Generator,
) -> float:
    """Return the resolved onset in seconds for a non-looping BackgroundEvent."""
    if ev.onset_seconds is not None:
        base = float(ev.onset_seconds)
    elif ev.onset_at_phase is not None:
        if phase_boundaries is not None and ev.onset_at_phase in phase_boundaries:
            base = float(phase_boundaries[ev.onset_at_phase])
        else:
            frac = _PHASE_FRACTIONS.get(ev.onset_at_phase, 0.5)
            base = frac * duration_s
    else:
        base = 0.0

    offset = float(ev.onset_offset_seconds) if ev.onset_offset_seconds is not None else 0.0
    return max(0.0, min(base + offset, duration_s - 0.1))


# ---------------------------------------------------------------------------
# NoiseMixer
# ---------------------------------------------------------------------------


class NoiseMixer:
    """Mix looping ambient audio and SFX events into a speech signal.

    Parameters
    ----------
    assets_dir:
        Root of the assets tree.  The mixer looks for SFX in
        ``assets_dir/sfx/`` and ambient loops in ``assets_dir/ambient/``.
    """

    def __init__(self, assets_dir: Path = Path("assets")) -> None:
        self._sfx_dir = assets_dir / "sfx"
        self._ambient_dir = assets_dir / "ambient"

    def mix(
        self,
        samples: np.ndarray,
        sr: int,
        config: AcousticSceneConfig,
        rng_seed: int = 0,
        phase_boundaries: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, list[AugmentedEvent], float]:
        """Mix background events from config into speech samples.

        Args:
            samples: Float32 mono speech signal.
            sr: Sample rate in Hz (should be 16 000).
            config: AcousticSceneConfig with background_events and snr_target_db.
            rng_seed: Reproducibility seed.
            phase_boundaries: Optional mapping of phase name → onset_s used to
                resolve BackgroundEvent.onset_at_phase fields.  When absent,
                default fractional positions (_PHASE_FRACTIONS) are used.

        Returns:
            Tuple of:
            * augmented float32 mono samples (same length as input)
            * list of AugmentedEvent records (raw clip-relative times,
              before silence-pad offset)
            * snr_db_actual — measured SNR between speech and added noise
        """
        rng = np.random.default_rng(rng_seed)
        duration_s = len(samples) / sr
        mixed = samples.astype(np.float32).copy()
        events: list[AugmentedEvent] = []

        ambient_events = [e for e in config.background_events if e.loop]
        sfx_events = [e for e in config.background_events if not e.loop]

        # --- Looping ambient tracks → scale to snr_target_db ---
        ambient_sum: np.ndarray | None = None
        if ambient_events:
            parts = [
                _synthesise_ambient(ev.type, duration_s, sr, rng)
                if not self._has_asset(ev)
                else _load_audio(self._resolve_asset_path(ev), sr)
                for ev in ambient_events
            ]
            # Mix all ambient tracks at equal weight before SNR scaling
            combined = np.zeros(len(mixed), dtype=np.float32)
            for part in parts:
                combined += _pad_or_trim(part, len(mixed))
            ambient_sum = combined

            speech_rms = _speech_rms(samples, sr=sr)
            ambient_rms = _rms(ambient_sum)
            if speech_rms > 0.0 and ambient_rms > 0.0:
                target_noise_rms = speech_rms / (10.0 ** (config.snr_target_db / 20.0))
                gain = target_noise_rms / ambient_rms
                ambient_sum = (ambient_sum * gain).astype(np.float32)

            mixed = (mixed + ambient_sum).astype(np.float32)
            actual_level = _to_dbfs(ambient_sum)
            for ev in ambient_events:
                events.append(
                    AugmentedEvent(
                        type=ev.type,
                        onset_s=0.0,
                        offset_s=duration_s,
                        level_db=actual_level,
                    )
                )

        # --- Non-looping SFX events ---
        for ev in sfx_events:
            onset_s = _resolve_onset_s(ev, duration_s, phase_boundaries, rng)
            sfx_audio = self._load_or_synthesise_sfx(ev, sr, rng)
            onset_n = int(onset_s * sr)
            copy_n = min(len(sfx_audio), len(mixed) - onset_n)
            if copy_n <= 0 or onset_n >= len(mixed):
                continue

            chunk = sfx_audio[:copy_n].astype(np.float32)
            if ev.level_db is not None:
                target_amp = float(10.0 ** (ev.level_db / 20.0))
                peak = float(np.max(np.abs(chunk))) + 1e-9
                chunk = (chunk * (target_amp / peak)).astype(np.float32)

            mixed[onset_n : onset_n + copy_n] += chunk
            offset_s = onset_s + copy_n / sr
            events.append(
                AugmentedEvent(
                    type=ev.type,
                    onset_s=onset_s,
                    offset_s=offset_s,
                    level_db=ev.level_db if ev.level_db is not None else _to_dbfs(chunk),
                )
            )

        # --- Measure actual SNR ---
        added_noise = mixed - samples.astype(np.float32)
        snr_actual = _snr_db(samples.astype(np.float32), added_noise)

        return mixed, events, snr_actual

    # ------------------------------------------------------------------
    # Asset resolution helpers
    # ------------------------------------------------------------------

    def _has_asset(self, ev: BackgroundEvent) -> bool:
        """Return True if a loadable asset exists for this event."""
        if ev.asset_path is not None:
            return Path(ev.asset_path).exists()
        search_dir = self._sfx_dir if ev.type in _ACOU_TYPES else self._ambient_dir
        return search_dir.exists() and any(search_dir.glob(f"{ev.type}*.wav"))

    def _resolve_asset_path(self, ev: BackgroundEvent) -> Path:
        """Return the first matching asset path for an event."""
        if ev.asset_path is not None:
            return Path(ev.asset_path)
        search_dir = self._sfx_dir if ev.type in _ACOU_TYPES else self._ambient_dir
        return next(search_dir.glob(f"{ev.type}*.wav"))

    def _load_or_synthesise_sfx(
        self, ev: BackgroundEvent, sr: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Return float32 mono audio for a non-looping SFX event."""
        if ev.asset_path is not None:
            p = Path(ev.asset_path)
            if p.exists():
                return _load_audio(p, sr)

        if self._sfx_dir.exists():
            candidates = sorted(self._sfx_dir.glob(f"{ev.type}*.wav"))
            if candidates:
                idx = int(rng.integers(len(candidates)))
                return _load_audio(candidates[idx], sr)

        return _synthesise_sfx(ev.type, sr, rng)
