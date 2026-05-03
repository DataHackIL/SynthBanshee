"""Device-specific audio coloration profiles for AVDP Tier B clips.

Each profile models the frequency response and level characteristics of the
device used to capture a scene.  Profiles are applied as a chain of
scipy Butterworth filters, optional presence boost / extra low-pass shaping,
optional hum injection, and level adjustment.

Spec reference: docs/spec.md §3.1 (device placement, Stage 3 augmentation)
M16 spec: presence boost +2–4 dB @ 2.5–3.5 kHz, gentle rolloff above 6.5 kHz
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

_FILTER_ORDER = 4


def _peaking_eq_coeffs(
    centre_hz: float, gain_db: float, q: float, sr: int
) -> tuple[np.ndarray, np.ndarray]:
    """Design a biquad peaking EQ filter (Audio EQ Cookbook).

    Returns (b, a) transfer function coefficients for use with lfilter.
    """
    a_lin = 10.0 ** (gain_db / 40.0)  # sqrt of linear gain
    w0 = 2.0 * np.pi * centre_hz / sr
    alpha = np.sin(w0) / (2.0 * q)

    b0 = 1.0 + alpha * a_lin
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * a_lin
    a0 = 1.0 + alpha / a_lin
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / a_lin

    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


# Per-device parameters:
#   highpass_hz    — roll-off below this frequency (models mic sensitivity floor)
#   lowpass_hz     — roll-off above this frequency (models bandwidth limit / muffling)
#   presence_boost — optional peaking EQ: (centre_hz, gain_db, Q) or None
#   extra_lowpass_hz — additional 2nd-order Butterworth LP above this freq; None = skip
#   hum_hz         — mains hum frequency to inject; None = no hum
#   hum_dbfs       — amplitude of injected hum in dBFS (ignored when hum_hz is None)
#   level_db       — broadband gain applied after filtering (models pickup / insertion loss)
_PROFILES: dict[str, dict] = {
    "phone_in_hand": {
        "highpass_hz": 200,
        "lowpass_hz": 8_000,
        "presence_boost": (3_000, 3.0, 1.5),  # +3 dB @ 3 kHz, Q=1.5
        "extra_lowpass_hz": 6_500,
        "hum_hz": None,
        "hum_dbfs": None,
        "level_db": 0.0,
    },
    "phone_in_pocket": {
        # Cloth occlusion: strong high-frequency muffling + insertion loss
        "highpass_hz": 300,
        "lowpass_hz": 2_500,
        "presence_boost": None,  # pocket muffling already cuts presence band
        "extra_lowpass_hz": None,
        "hum_hz": None,
        "hum_dbfs": None,
        "level_db": -6.0,
    },
    "phone_on_table": {
        # Surface coupling: slight low-end coupling loss, mild high-frequency rolloff
        "highpass_hz": 100,
        "lowpass_hz": 7_000,
        "presence_boost": (2_800, 2.0, 1.5),  # +2 dB @ 2.8 kHz, Q=1.5
        "extra_lowpass_hz": 6_500,
        "hum_hz": None,
        "hum_dbfs": None,
        "level_db": -3.0,
    },
    "pi_budget_mic": {
        # Budget electret mic: limited bandwidth, 50 Hz mains hum (EU/IL grid)
        "highpass_hz": 80,
        "lowpass_hz": 7_000,
        "presence_boost": None,  # budget mic has no presence lift
        "extra_lowpass_hz": None,
        "hum_hz": 50.0,
        "hum_dbfs": -50.0,
        "level_db": 0.0,
    },
}


class DeviceProfiler:
    """Apply device-specific frequency coloring to a float32 mono signal."""

    def apply(
        self,
        samples: np.ndarray,
        sr: int,
        device: str,
        rng_seed: int = 0,
    ) -> np.ndarray:
        """Return a colored float32 mono array of the same length as input.

        Args:
            samples: Float32 mono audio at ``sr`` Hz.
            sr: Sample rate in Hz.
            device: One of the known device profile keys (e.g. "phone_in_pocket").
            rng_seed: Reserved for future per-device randomisation (unused).

        Returns:
            Float32 mono array — filtered, hum-injected, and level-adjusted.

        Raises:
            KeyError: if ``device`` is not a recognised profile.
        """
        profile = _PROFILES[device]
        out = samples.astype(np.float32).copy()
        nyq = sr / 2.0

        # High-pass filter
        hp_hz = profile["highpass_hz"]
        if hp_hz > 0:
            sos = butter(_FILTER_ORDER, hp_hz / nyq, btype="high", output="sos")
            out = sosfilt(sos, out).astype(np.float32)

        # Low-pass filter
        lp_hz = profile["lowpass_hz"]
        if lp_hz < nyq:
            sos = butter(_FILTER_ORDER, lp_hz / nyq, btype="low", output="sos")
            out = sosfilt(sos, out).astype(np.float32)

        # Presence boost — peaking EQ (M16)
        presence = profile.get("presence_boost")
        if presence is not None:
            centre_hz, gain_db, q = presence
            if centre_hz < nyq:
                b, a = _peaking_eq_coeffs(centre_hz, gain_db, q, sr)
                out = lfilter(b, a, out).astype(np.float32)

        # Additional gentle rolloff above extra_lowpass_hz (M16)
        extra_lowpass_hz = profile.get("extra_lowpass_hz")
        if extra_lowpass_hz is not None and extra_lowpass_hz < nyq:
            sos = butter(2, extra_lowpass_hz / nyq, btype="low", output="sos")
            out = sosfilt(sos, out).astype(np.float32)

        # Mains hum injection (pi_budget_mic)
        if profile["hum_hz"] is not None:
            hum_amp = float(10.0 ** (profile["hum_dbfs"] / 20.0))
            t = np.arange(len(out), dtype=np.float64) / sr
            out = (out + hum_amp * np.sin(2.0 * np.pi * profile["hum_hz"] * t)).astype(np.float32)

        # Broadband level adjustment
        level_db = profile["level_db"]
        if level_db != 0.0:
            out = (out * float(10.0 ** (level_db / 20.0))).astype(np.float32)

        return out
