"""Experimental voice texture augmentation — VIC breathiness (M12).

Applies bandpass-filtered noise (3–7.5 kHz) to VIC speaker turns at
intensity I3–I5, reducing HNR (harmonic-to-noise ratio) to simulate
vocal strain / distress phonation.

This feature is **experimental** and disabled by default.  It must pass
a listening-test gate (§8.3) before being enabled for dataset generation.

Spec reference: docs/audio_generation_v3_design.md §4.4, §M12
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

# Breathiness noise band: 3–7.5 kHz (aspiration noise band for breathy voice).
# Spec recommends 3–8 kHz; upper edge reduced to 7.5 kHz to maintain margin
# below Nyquist at 16 kHz sample rate (Nyquist = 8 kHz).
_BAND_LO_HZ: float = 3000.0
_BAND_HI_HZ: float = 7500.0

# Butterworth filter order for the bandpass
_FILTER_ORDER: int = 4

# Maximum noise gain (at level=1.0) relative to signal RMS.
# Targets ~4 dB HNR reduction (12×log₁₀(1.25) ≈ 4 dB).
# Tuning expected during listening-test gate (§8.3).
_MAX_NOISE_GAIN: float = 0.25


def add_breathiness(
    samples: np.ndarray,
    sample_rate: int,
    level: float,
    *,
    rng_seed: int = 0,
) -> np.ndarray:
    """Add bandpass-filtered noise to simulate breathy phonation.

    Mixes white noise filtered to the 3–7.5 kHz aspiration band into the
    input signal.  The noise amplitude scales linearly with *level* up to
    ``_MAX_NOISE_GAIN`` × signal RMS.

    Args:
        samples: Float32 mono audio array (any sample rate).
        sample_rate: Sample rate in Hz (must be > 2 × ``_BAND_HI_HZ``).
        level: Breathiness level in [0.0, 1.0].  0.0 = no effect (modal
            voice); 1.0 = maximum breathiness.
        rng_seed: Seed for reproducible noise generation.

    Returns:
        New float32 array with breathiness noise added.  Same length as
        input.  Not peak-limited — caller is responsible for downstream
        limiting.

    Raises:
        ValueError: If *level* is outside [0.0, 1.0] or *sample_rate* is
            too low for the filter band.
    """
    if not 0.0 <= level <= 1.0:
        raise ValueError(f"level must be in [0.0, 1.0], got {level}")

    if level == 0.0:
        return samples.copy()

    nyquist = sample_rate / 2.0
    if nyquist <= _BAND_HI_HZ:
        raise ValueError(
            f"sample_rate {sample_rate} Hz is too low for "
            f"bandpass upper edge {_BAND_HI_HZ} Hz (Nyquist={nyquist} Hz)"
        )

    # Compute signal RMS (avoid division by zero on silence)
    sig_rms = float(np.sqrt(np.mean(samples**2)))
    if sig_rms < 1e-10:
        return samples.copy()

    # Generate white noise
    rng = np.random.default_rng(rng_seed)
    noise = rng.standard_normal(len(samples)).astype(np.float32)

    # Bandpass filter the noise to aspiration band
    sos = butter(
        _FILTER_ORDER,
        [_BAND_LO_HZ / nyquist, _BAND_HI_HZ / nyquist],
        btype="band",
        output="sos",
    )
    filtered_noise = sosfilt(sos, noise).astype(np.float32)

    # Normalize filtered noise to unit RMS then scale
    noise_rms = float(np.sqrt(np.mean(filtered_noise**2)))
    if noise_rms < 1e-10:
        return samples.copy()

    target_noise_rms = sig_rms * _MAX_NOISE_GAIN * level
    filtered_noise *= target_noise_rms / noise_rms

    return (samples + filtered_noise).astype(np.float32)
