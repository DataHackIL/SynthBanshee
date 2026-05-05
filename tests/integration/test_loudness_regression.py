"""Integration regression test for #78 (M3c) — clip-level loudness contract.

Asserts the two invariants the production pipeline must preserve forever:

1. **Absolute peak lands at the configured target.**  Without this, M3a-shaped
   Tier A clips peak ~6 dB below the −1 dBFS ceiling (regression #78); Whisper
   under-transcribes and UTMOS is dominated by limiter characteristics.

2. **Per-turn RMS contrast survives.**  M3a per-turn RMS targeting is the
   primary loudness-trajectory signal classifiers learn from.  The post-mix
   loudness step must not collapse it.  The legacy peak-normalizer this
   replaces (PR #27 removed it) DID collapse contrast; the new helper uses a
   *single global gain* so all per-turn RMS ratios are mathematically
   preserved.

The test runs preprocess() over a synthetic two-segment fixture rather than a
full TTS render so it can sit in CI without API credentials or wall-clock cost.
The synthetic fixture mirrors the structure of a real M3a-targeted scene
(disparate per-turn RMS, head/tail silence) closely enough to catch any
regression in the contract.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf

from synthbanshee.augment.preprocessing import (
    _PEAK_DBFS,
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


def _make_two_segment_fixture(
    seg_a_dbfs: float,
    seg_b_dbfs: float,
    out_path: Path,
    seg_duration_s: float = 1.5,
    inter_gap_s: float = 0.2,
) -> tuple[Path, slice, slice]:
    """Write a FLOAT WAV mimicking the SceneMixer's M3a output.

    Two 440 Hz tone segments at distinct RMS levels separated by a short
    silence gap, with no head/tail padding (preprocess() adds that).
    Returns the path plus the sample slices for each segment so the test
    can measure their post-pipeline RMS independently.
    """
    sr = _TARGET_SR
    seg_n = int(seg_duration_s * sr)
    gap_n = int(inter_gap_s * sr)
    t = np.linspace(0, seg_duration_s, seg_n, endpoint=False)
    base = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    amp_a = 10.0 ** (seg_a_dbfs / 20.0) * math.sqrt(2.0)  # peak from RMS for sine
    amp_b = 10.0 ** (seg_b_dbfs / 20.0) * math.sqrt(2.0)

    seg_a = (base * amp_a).astype(np.float32)
    seg_b = (base * amp_b).astype(np.float32)
    gap = np.zeros(gap_n, dtype=np.float32)

    full = np.concatenate([seg_a, gap, seg_b])
    sf.write(str(out_path), full, sr, subtype="FLOAT")

    seg_a_slice = slice(0, seg_n)
    seg_b_slice = slice(seg_n + gap_n, seg_n + gap_n + seg_n)
    return out_path, seg_a_slice, seg_b_slice


class TestLoudnessRegression:
    """#78 — protect the absolute-peak and contrast-preservation invariants."""

    def test_tier_a_peak_lands_in_useful_range(self, tmp_path):
        """Pipeline-level guard: rendered clip peak must end up close to target.

        This is the core #78 assertion.  If a future change re-introduces
        limiter-only behaviour, the peak will drop ~5 dB and trip this test
        before it reaches the M17 evaluation pipeline.
        """
        # Two segments at typical M3a targets (AGG I1 = −28, VIC I1 = −26).
        src, _, _ = _make_two_segment_fixture(
            seg_a_dbfs=-28.0, seg_b_dbfs=-26.0, out_path=tmp_path / "scene.wav"
        )
        dst = tmp_path / "out.wav"
        cfg = PreprocessingConfig()  # default target_peak_dbfs = −2.0
        preprocess(src, dst, config=cfg)
        data, _ = sf.read(str(dst), dtype="float32")
        peak = _peak_dbfs(data)
        # Acceptance band: tighter than [−3, −1] would flake on PCM_16
        # quantisation + high-pass tone-peak shaving (~0.1 dB combined).
        assert -3.0 <= peak <= _PEAK_DBFS, (
            f"Peak {peak:.2f} dBFS outside [-3.0, {_PEAK_DBFS}]. "
            f"Target was {cfg.target_peak_dbfs} dBFS.  This regression is #78 "
            f"reappearing — most likely a change to the order or behaviour of "
            f"steps 5a/5b in preprocess(), or a new step that erases the "
            f"target-peak gain."
        )

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
        # Allow 0.5 dB slack for high-pass / PCM_16 / windowed-RMS effects.
        assert abs(output_gap - input_gap) < 0.5, (
            f"Per-turn RMS contrast collapsed: input gap {input_gap:+.2f} dB, "
            f"output gap {output_gap:+.2f} dB.  This means the loudness step "
            f"is no longer a single global gain — the M3a trajectory is being "
            f"erased, which is exactly what M3b (PR #27) was designed to "
            f"prevent and what #78's fix (M3c) must continue to preserve."
        )

    def test_peak_normalize_step_recorded_in_steps_applied(self, tmp_path):
        """Belt-and-braces: PreprocessingResult must declare the new step.

        QA / packaging code consumes ``steps_applied``; missing the step
        would silently misreport the pipeline that produced the clip.
        """
        src, _, _ = _make_two_segment_fixture(
            seg_a_dbfs=-28.0, seg_b_dbfs=-26.0, out_path=tmp_path / "scene.wav"
        )
        dst = tmp_path / "out.wav"
        result = preprocess(src, dst, config=PreprocessingConfig())
        steps = " ".join(result.steps_applied)
        assert "peak_normalize_-2.0dBFS" in steps, (
            f"peak_normalize step missing from steps_applied: {result.steps_applied}"
        )
