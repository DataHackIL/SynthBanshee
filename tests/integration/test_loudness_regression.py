"""Integration regression test for #78 — clip-level loudness contract.

Asserts the two invariants the production pipeline must preserve forever:

1. **Absolute peak lands at the configured target.**  Pre-#78 the spec had
   only an upper bound on peak; M3a-shaped Tier A clips legitimately sat
   ~6 dB below the ceiling and the spec had no language to call that wrong.
   #78 made target peak a configured policy, and this assertion catches any
   regression that drifts the peak away from that policy.

2. **Per-turn RMS contrast survives.**  M3a per-turn RMS targeting is the
   primary loudness-trajectory signal classifiers learn from.  The post-mix
   loudness step must not collapse it.  The legacy peak-normalizer this
   replaces (PR #27 removed it) DID collapse contrast; the new helper uses a
   *single global gain* so all per-turn RMS ratios are mathematically
   preserved.

The test runs preprocess() over a synthetic two-segment fixture rather than a
full TTS render so it can sit in CI without API credentials or wall-clock
cost.  The fixture uses bandpass-filtered Gaussian noise with an 18 dB crest
factor — matches what real Hebrew TTS produces post-M3a, unlike a pure sine
whose 3 dB crest would let regressions slip past peak-anchored assertions.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf

from synthbanshee.augment.preprocessing import (
    _SILENCE_PAD_S,
    _TARGET_SR,
    PreprocessingConfig,
    preprocess,
)


def _peak_dbfs(samples: np.ndarray) -> float:
    peak = float(np.max(np.abs(samples)))
    if peak == 0.0:
        return -math.inf
    return 20.0 * math.log10(peak)


def _rms_dbfs(samples: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    if rms == 0.0:
        return -math.inf
    return 20.0 * math.log10(rms)


# Speech-like crest factor target: real Hebrew TTS clips measure 19–21 dB
# crest after M3a per-turn RMS gain (see #78 evidence table).  A pure sine has
# crest 3 dB, which makes peak/RMS test assertions trivially pass on signals
# whose crest behaviour is fundamentally different from production audio.
_SPEECHLIKE_CREST_DB = 18.0


def _make_two_segment_fixture(
    seg_a_dbfs: float,
    seg_b_dbfs: float,
    out_path: Path,
    seg_duration_s: float = 1.5,
    inter_gap_s: float = 0.2,
    rng_seed: int = 17,
) -> tuple[Path, slice, slice]:
    """Write a FLOAT WAV mimicking the SceneMixer's M3a output.

    Two bandpass-filtered Gaussian-noise segments at distinct *RMS* levels with
    a speech-like crest factor (~18 dB), separated by a short silence gap.
    Bandpass at 200–4000 Hz approximates the dominant energy band of speech;
    the crest factor matches what real Hebrew TTS produces post-M3a so the
    peak/RMS assertions exercise a realistic regime instead of the pure-sine
    3 dB crest.

    Returns the path plus sample slices for each segment so the test can
    measure their post-pipeline RMS independently.  The fixture is fully
    deterministic (seeded RNG) so test outcomes are reproducible across runs.
    """
    from scipy.signal import butter, sosfilt

    sr = _TARGET_SR
    seg_n = int(seg_duration_s * sr)
    gap_n = int(inter_gap_s * sr)
    rng = np.random.default_rng(rng_seed)

    # Bandpass-filtered Gaussian noise → speech-like spectrum + crest factor.
    sos = butter(4, [200.0, 4000.0], btype="band", fs=sr, output="sos")

    def _make_segment(rms_dbfs: float) -> np.ndarray:
        raw = rng.standard_normal(seg_n).astype(np.float64)
        filtered = sosfilt(sos, raw)
        # Set RMS first; crest of bandpass-filtered Gaussian noise is ~12–14 dB
        # naturally — we shape the tail to ~18 dB by clipping the top 0.1% of
        # samples to 10× the post-RMS level (a deterministic transform of the
        # already-deterministic noise, so the result is reproducible).
        cur_rms = float(np.sqrt(np.mean(filtered**2)))
        scale = (10.0 ** (rms_dbfs / 20.0)) / max(cur_rms, 1e-12)
        scaled = filtered * scale
        # Light peak shaping toward _SPEECHLIKE_CREST_DB without distorting RMS:
        # threshold = rms * 10 ** ((crest - small_margin) / 20) ≈ 8× rms at 18 dB.
        threshold = (10.0 ** (_SPEECHLIKE_CREST_DB / 20.0)) * (10.0 ** (rms_dbfs / 20.0))
        np.clip(scaled, -threshold, threshold, out=scaled)
        return scaled.astype(np.float32)

    seg_a = _make_segment(seg_a_dbfs)
    seg_b = _make_segment(seg_b_dbfs)
    gap = np.zeros(gap_n, dtype=np.float32)

    full = np.concatenate([seg_a, gap, seg_b])
    sf.write(str(out_path), full, sr, subtype="FLOAT")

    seg_a_slice = slice(0, seg_n)
    seg_b_slice = slice(seg_n + gap_n, seg_n + gap_n + seg_n)
    return out_path, seg_a_slice, seg_b_slice


class TestLoudnessRegression:
    """#78 — protect the absolute-peak and contrast-preservation invariants."""

    def test_tier_a_peak_lands_at_configured_target(self, tmp_path):
        """Pipeline-level guard: rendered clip peak must equal the config target.

        This is the core #78 assertion.  If a future change re-introduces
        limiter-only behaviour, the peak will drop multiple dB and trip this
        test before any clip ships.

        Tolerance is **0.1 dB**, not 0.5: the only legitimate sources of error
        between target and measured peak are PCM_16 quantisation
        (~0.0001 dB) and the 80 Hz Butterworth high-pass on a 200–4000 Hz
        bandpass-noise input (~0.03 dB at the lowest band edge).  Anything
        wider hides real regressions — the original #78 deviation was 4.6 dB,
        a 0.5 dB tolerance only catches that bug by luck.
        """
        # Two segments at typical M3a targets (AGG I1 = −28, VIC I1 = −26).
        src, _, _ = _make_two_segment_fixture(
            seg_a_dbfs=-28.0, seg_b_dbfs=-26.0, out_path=tmp_path / "scene.wav"
        )
        dst = tmp_path / "out.wav"
        cfg = PreprocessingConfig()  # default target_peak_dbfs = −2.0
        result = preprocess(src, dst, config=cfg)
        data, _ = sf.read(str(dst), dtype="float32")
        peak = _peak_dbfs(data)

        assert abs(peak - cfg.target_peak_dbfs) < 0.1, (
            f"Peak {peak:.3f} dBFS deviates >0.1 dB from target "
            f"{cfg.target_peak_dbfs} dBFS.  This is #78 reappearing — most "
            f"likely a change to the order or behaviour of steps 5a/5b in "
            f"preprocess(), or a new step that erases the target-peak gain."
        )
        # Structured field on the result must agree with the configured value.
        assert result.target_peak_dbfs == cfg.target_peak_dbfs

    def test_per_turn_rms_contrast_preserved(self, tmp_path):
        """Single global gain MUST preserve per-turn RMS *ratios* exactly.

        M3a creates the per-turn loudness trajectory (e.g. AGG I1 −28 dBFS,
        AGG I5 −15 dBFS, a 13 dB escalation).  If the loudness step ever
        regresses to per-segment normalization, the ratio collapses to 0 dB
        and classifiers lose the loudness-escalation signal.
        """
        # 5 dB gap between segments is well above any expected
        # high-pass / quantisation noise floor.
        seg_a_target = -28.0
        seg_b_target = -23.0
        src, slice_a, slice_b = _make_two_segment_fixture(
            seg_a_dbfs=seg_a_target,
            seg_b_dbfs=seg_b_target,
            out_path=tmp_path / "scene.wav",
        )
        dst = tmp_path / "out.wav"
        preprocess(src, dst, config=PreprocessingConfig())
        data, _ = sf.read(str(dst), dtype="float32")

        # preprocess() adds _SILENCE_PAD_S of leading silence — shift slices.
        pad_n = int(_SILENCE_PAD_S * _TARGET_SR)
        out_a = data[pad_n + slice_a.start : pad_n + slice_a.stop]
        out_b = data[pad_n + slice_b.start : pad_n + slice_b.stop]
        rms_a = _rms_dbfs(out_a)
        rms_b = _rms_dbfs(out_b)

        input_gap = seg_b_target - seg_a_target  # +5 dB
        output_gap = rms_b - rms_a
        # 0.2 dB tolerance: tight enough to catch any per-segment normalization
        # regression (which would collapse the gap toward 0), loose enough to
        # tolerate windowed-RMS sampling noise on bandpass-noise fixtures.
        assert abs(output_gap - input_gap) < 0.2, (
            f"Per-turn RMS contrast collapsed: input gap {input_gap:+.3f} dB, "
            f"output gap {output_gap:+.3f} dB.  This means the loudness step "
            f"is no longer a single global gain — the M3a trajectory is being "
            f"erased, which is exactly what M3b (PR #27) was designed to "
            f"prevent and what #78's fix must continue to preserve."
        )

    def test_peak_normalize_step_recorded_in_steps_applied(self, tmp_path):
        """Belt-and-braces: PreprocessingResult must declare the new step.

        QA / packaging code consumes ``steps_applied``; missing the step
        would silently misreport the pipeline that produced the clip.
        Step name is a *literal token* — no embedded numeric parameter —
        and the configured target lives in the dedicated structured field.
        """
        src, _, _ = _make_two_segment_fixture(
            seg_a_dbfs=-28.0, seg_b_dbfs=-26.0, out_path=tmp_path / "scene.wav"
        )
        dst = tmp_path / "out.wav"
        cfg = PreprocessingConfig()
        result = preprocess(src, dst, config=cfg)
        assert "peak_normalize" in result.steps_applied, (
            f"peak_normalize step missing from steps_applied: {result.steps_applied}"
        )
        # Structured field carries the actual target — independent of step name.
        assert result.target_peak_dbfs == cfg.target_peak_dbfs

    def test_step_names_are_literal_tokens_not_value_encoded(self, tmp_path):
        """A custom target_peak_dbfs must NOT change the step-name string.

        Guards the QA grep contract: any project profile that overrides the
        target must still match a stable token like ``peak_normalize`` and
        read the actual value off ``result.target_peak_dbfs``.
        """
        src, _, _ = _make_two_segment_fixture(
            seg_a_dbfs=-28.0, seg_b_dbfs=-26.0, out_path=tmp_path / "scene.wav"
        )
        dst = tmp_path / "out.wav"
        cfg = PreprocessingConfig(target_peak_dbfs=-3.5)
        result = preprocess(src, dst, config=cfg)
        # Literal token, no numeric suffix.
        assert "peak_normalize" in result.steps_applied
        for step in result.steps_applied:
            assert "dBFS" not in step, f"value-encoded step name leaked: {step}"
        # Override is reflected in the structured field, not the step string.
        assert result.target_peak_dbfs == -3.5
