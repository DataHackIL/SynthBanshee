"""M17 Phase A validation spike — Whisper + UTMOS gates on Hebrew clips.

Implements the prototyping experiments listed in
``docs/automated_eval_design.md`` (Acceptance Criteria table) plus the
follow-up checks raised in PR #77 review:

  - Whisper large-v3 + ivrit-ai/whisper-large-v3 on the clip set; gate < 0.40 median WER.
  - faster-whisper int8 (CT2) on the same clips for CI viability.
  - UTMOS on a multi-severity degradation grid (white-noise SNR sweep + 2 kHz lowpass)
    so a "gate FAIL" verdict has direction-of-degradation evidence behind it.
  - Hallucination detectors (length ratio, repeated-trigram, RTL embed marker count)
    run on every clip and reported empirically.
  - SHA-256 + duration manifest of the clip set for run-to-run audit.

This is a one-shot spike. No production module is created; outputs go to
``state/spikes/m17_phase_a/`` (gitignored). The canonical hand-written report
at ``docs/m17_phase_a_validation_report.md`` is **not** touched by this script
— see ``write_auto_report`` below.

Setup:

    uv pip install --python .venv/bin/python \\
        torch transformers accelerate faster-whisper \\
        jiwer numpy scipy soundfile

(``torchaudio`` and the ``speechmos`` PyPI package are NOT required: the repo
forbids torchaudio in preprocessing, and UTMOS22 strong is loaded via
``torch.hub`` from the pinned ``tarepan/SpeechMOS`` commit below — the
``speechmos`` PyPI package only ships DNSMOS/AECMOS/PLCMOS, not UTMOS.)

Run:

    .venv/bin/python scripts/m17_phase_a_validation.py
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from jiwer import cer, wer
from scipy.signal import butter, filtfilt
from transformers import pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
CLIP_DIR = REPO_ROOT / "data" / "m2a_wettest" / "agg_m_30-45_001"
SPIKE_DIR = REPO_ROOT / "state" / "spikes" / "m17_phase_a"
RESULTS_PATH = SPIKE_DIR / "results.json"
AUTO_REPORT_PATH = SPIKE_DIR / "report_auto.md"

WHISPER_HF_MODELS = [
    "openai/whisper-large-v3",
    "ivrit-ai/whisper-large-v3",
]
FASTER_WHISPER_MODEL = "ivrit-ai/whisper-large-v3-ct2"

WER_GATE = 0.40
UTMOS_SEPARATION_GATE = 0.5

# Degradation grid: known-perceptually-stronger -> known-subtle, plus a
# spectrally-different mechanism (lowpass) to catch UTMOS sensitivity that's
# orthogonal to noise.
DEGRADATIONS: list[dict] = [
    {"id": "wn_snr_-20db", "kind": "white_noise", "snr_db": -20.0},
    {"id": "wn_snr_-10db", "kind": "white_noise", "snr_db": -10.0},
    {"id": "wn_snr_0db", "kind": "white_noise", "snr_db": 0.0},
    {"id": "wn_snr_+10db", "kind": "white_noise", "snr_db": 10.0},
    {"id": "lp_2khz", "kind": "lowpass", "cutoff_hz": 2000.0},
]
# We sample the first N clips for the degradation sweep — saves UTMOS time,
# keeps every degradation paired with the same source for fair comparison.
DEGRADATION_SAMPLE_N = 5
RNG_SEED = 42


@dataclass
class Clip:
    clip_id: str
    wav: np.ndarray
    sr: int
    reference: str
    duration_s: float
    sha256: str
    ref_words: int


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_clips() -> list[Clip]:
    clips: list[Clip] = []
    for wav_path in sorted(CLIP_DIR.glob("*.wav")):
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            raise FileNotFoundError(
                f"Clip {wav_path.name} is missing its transcript sidecar at {txt_path}. "
                "Every .wav must be paired with a .txt reference for WER scoring."
            )
        ref = "\n".join(
            line
            for line in txt_path.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("[")
        ).strip()
        wav, sr = sf.read(wav_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        # Repo invariant (CLAUDE.md): all WAVs are 16 kHz / mono / 16-bit PCM.
        # Whisper's feature extractor + faster-whisper's np.ndarray path both
        # silently treat input as already at 16 kHz — fail loudly if it isn't.
        if sr != 16000:
            raise ValueError(
                f"Clip {wav_path.name} is at {sr} Hz; the spike requires 16 kHz "
                "(repo invariant). Resample or rerender before re-running."
            )
        clips.append(
            Clip(
                clip_id=wav_path.stem,
                wav=wav,
                sr=sr,
                reference=ref,
                duration_s=len(wav) / sr,
                sha256=_sha256_file(wav_path),
                ref_words=len(normalize_for_wer(ref).split()),
            )
        )
    return clips


# Hebrew niqqud range — Whisper output is unpointed; strip diacritics from refs.
NIQQUD_RE = re.compile(r"[֑-ֽֿׁ-ׂׄ-ׇ]")
PUNCT_RE = re.compile(r"[\.,!?;:\"'\(\)\[\]\-–—…״׳]")
WS_RE = re.compile(r"\s+")
# RTL embedding / override marks that surface in some hallucination loops.
RTL_MARKS_RE = re.compile(r"[‫‮‎‏‪‬‭]")
# Hebrew final-form letters — design doc spec line 66: normalize before WER.
# Final and non-final forms are orthographic variants of the same letter; LLM
# output and Whisper output may disagree on form choice without semantic loss.
FINAL_FORMS_MAP = str.maketrans({"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"})


def normalize_for_wer(text: str) -> str:
    text = NIQQUD_RE.sub("", text)
    text = RTL_MARKS_RE.sub("", text)
    text = text.translate(FINAL_FORMS_MAP)
    text = PUNCT_RE.sub(" ", text)
    text = WS_RE.sub(" ", text)
    return text.strip()


# ---------- Hallucination detectors --------------------------------------


def detect_hallucination_signals(reference: str, hypothesis: str) -> dict:
    # length + trigram run on normalized text (fair to both ref and hyp).
    # rtl_mark_count runs on RAW hypothesis because normalize_for_wer strips
    # RTL marks — we still want to know if Whisper emitted them.
    ref_n = normalize_for_wer(reference)
    hyp_n = normalize_for_wer(hypothesis)
    ref_words = ref_n.split()
    hyp_words = hyp_n.split()
    length_ratio = len(hyp_words) / max(1, len(ref_words))

    def max_trigram_repeat(words: list[str]) -> int:
        if len(words) < 3:
            return 0
        counts = Counter(tuple(words[i : i + 3]) for i in range(len(words) - 2))
        return counts.most_common(1)[0][1]

    hyp_repeat = max_trigram_repeat(hyp_words)
    ref_repeat = max_trigram_repeat(ref_words)
    repeat_ratio = hyp_repeat / max(1, ref_repeat)
    rtl_marks = len(RTL_MARKS_RE.findall(hypothesis))
    return {
        "length_ratio": length_ratio,
        "hyp_max_trigram_repeat": hyp_repeat,
        "ref_max_trigram_repeat": ref_repeat,
        "trigram_repeat_ratio": repeat_ratio,
        "rtl_mark_count": rtl_marks,
    }


# ---------- ASR ----------------------------------------------------------


def asr_hf(model_id: str, clips: list[Clip], device: str) -> dict[str, str]:
    asr = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=torch.device(device),  # torch.device works across all transformers versions
        torch_dtype=torch.float32,
        chunk_length_s=30,
        return_timestamps=False,
    )
    out: dict[str, str] = {}
    for c in clips:
        # Pass dict form so the pipeline resamples if input rate ever drifts
        # from the model's expected 16 kHz — bare np.ndarray is silently assumed
        # to already match. (load_clips also asserts 16 kHz; this is belt+braces.)
        result = asr(
            {"raw": c.wav.copy(), "sampling_rate": c.sr},
            generate_kwargs={
                "language": "he",
                "task": "transcribe",
                "num_beams": 1,  # greedy — deterministic given identical input
                "do_sample": False,
            },
        )
        out[c.clip_id] = result["text"]
    del asr
    if device == "mps":
        torch.mps.empty_cache()
    return out


def asr_faster_whisper(model_id: str, clips: list[Clip], compute_type: str) -> dict[str, str]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_id, device="cpu", compute_type=compute_type)
    out: dict[str, str] = {}
    for c in clips:
        segments, _info = model.transcribe(
            c.wav.copy(),
            language="he",
            task="transcribe",
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
        )
        out[c.clip_id] = " ".join(s.text.strip() for s in segments).strip()
    return out


def score_asr_run(model_label: str, hyps: dict[str, str], clips: list[Clip]) -> dict:
    per_clip = []
    wers, cers = [], []
    for c in clips:
        ref_n = normalize_for_wer(c.reference)
        hyp_n = normalize_for_wer(hyps[c.clip_id])
        w, ce = wer(ref_n, hyp_n), cer(ref_n, hyp_n)
        signals = detect_hallucination_signals(c.reference, hyps[c.clip_id])
        per_clip.append(
            {
                "clip_id": c.clip_id,
                "wer": w,
                "cer": ce,
                "ref_words": len(ref_n.split()),
                "hyp_words": len(hyp_n.split()),
                "hallucination_signals": signals,
                "hypothesis": hyps[c.clip_id],
            }
        )
        wers.append(w)
        cers.append(ce)
    return {
        "model_label": model_label,
        "median_wer": float(np.median(wers)),
        "mean_wer": float(np.mean(wers)),
        "median_cer": float(np.median(cers)),
        "mean_cer": float(np.mean(cers)),
        "per_clip": per_clip,
    }


# ---------- Degradations -------------------------------------------------


def apply_white_noise(wav: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    sig_power = float((wav**2).mean()) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.standard_normal(len(wav)).astype(np.float32) * np.sqrt(noise_power)
    return (wav + noise).astype(np.float32)


def apply_lowpass(wav: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    # 8th-order Butterworth via filtfilt (zero-phase, effective 16th order) is
    # a brick-wall — deliberate "spectral surgery" stress test for UTMOS, not
    # a realistic codec/telephony envelope. If E2 ever uses lowpass-style
    # degradation in production, swap for a gentler / linear-phase design.
    nyq = sr / 2.0
    b, a = butter(N=8, Wn=cutoff_hz / nyq, btype="low")
    return filtfilt(b, a, wav).astype(np.float32)


def apply_degradation(wav: np.ndarray, sr: int, spec: dict, rng: np.random.Generator) -> np.ndarray:
    if spec["kind"] == "white_noise":
        return apply_white_noise(wav, spec["snr_db"], rng)
    if spec["kind"] == "lowpass":
        return apply_lowpass(wav, sr, spec["cutoff_hz"])
    raise ValueError(f"unknown degradation kind: {spec['kind']}")


def rms_normalize_to_match(degraded: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, dict]:
    """Scale ``degraded`` so its RMS matches ``clean``'s RMS.

    Isolates SNR / spectral effects from loudness when scoring perceptual MOS:
    UTMOS is loudness-sensitive, so a peak-clipped −20 dB SNR mix would otherwise
    score "quieter, therefore less obvious noise" rather than "more noise."
    Falls back to a uniform peak guard if the RMS-matched signal would clip;
    metrics record both outcomes so the report can surface any compromise.
    """
    clean_rms = float(np.sqrt(np.mean(clean**2)))
    deg_rms_pre = float(np.sqrt(np.mean(degraded**2))) + 1e-12
    scale = clean_rms / deg_rms_pre
    out = degraded * scale
    peak = float(np.max(np.abs(out)))
    peak_clipped = peak > 0.99
    if peak_clipped:
        out = out * (0.99 / peak)
    out = out.astype(np.float32)
    deg_rms_post = float(np.sqrt(np.mean(out**2)))
    return out, {
        "clean_rms": clean_rms,
        "deg_rms_pre_norm": deg_rms_pre,
        "deg_rms_post_norm": deg_rms_post,
        "peak_clipped": peak_clipped,
    }


def is_monotonic_with_severity(white_noise_results: list[dict], slack: float = 0.01) -> bool:
    """True iff UTMOS rises monotonically as SNR rises (less noise → higher score).

    Sorts white-noise variants by SNR ascending, then checks each consecutive
    pair: lower-SNR (more noisy) UTMOS must be ≤ higher-SNR (less noisy) UTMOS,
    plus a small slack to forgive numerical noise.
    """
    by_snr = sorted(white_noise_results, key=lambda d: d["snr_db"])
    return all(
        prev["mean_utmos"] <= nxt["mean_utmos"] + slack
        for prev, nxt in zip(by_snr, by_snr[1:], strict=False)
    )


# ---------- UTMOS --------------------------------------------------------


def utmos_score(model, wav: np.ndarray, sr: int) -> float:
    with torch.no_grad():
        x = torch.from_numpy(wav).unsqueeze(0)
        score = model(x, sr=sr)
    return float(score.squeeze().item())


# ---------- Auto-template report ----------------------------------------


def write_auto_report(results: dict) -> None:
    """Write a data-only template to state/spikes/. The canonical narrative
    report at docs/m17_phase_a_validation_report.md is hand-written; this
    function never touches it."""
    asr_runs = results["asr_runs"]
    md: list[str] = []
    md.append("# M17 Phase A — auto-generated data tables")
    md.append("")
    md.append(f"Run date: {results['metadata']['run_date']}")
    md.append(f"Device: `{results['metadata']['device']}`")
    md.append(f"Clips: {results['metadata']['n_clips']}")
    md.append("")
    md.append("## ASR runs")
    md.append("")
    md.append("| Run | Backend | Model | median WER | mean WER | median CER | mean CER |")
    md.append("|---|---|---|---|---|---|---|")
    for r in asr_runs:
        md.append(
            f"| {r['model_label']} | {r['backend']} | `{r['model_id']}` | "
            f"{r['median_wer']:.3f} | {r['mean_wer']:.3f} | "
            f"{r['median_cer']:.3f} | {r['mean_cer']:.3f} |"
        )
    md.append("")
    md.append("## Per-clip ASR (WER)")
    md.append("")
    md.append("| Clip | " + " | ".join(r["model_label"] for r in asr_runs) + " |")
    md.append("|---" + "|---" * len(asr_runs) + "|")
    clip_ids = [c["clip_id"] for c in asr_runs[0]["per_clip"]]
    for cid in clip_ids:
        cells = [cid]
        for r in asr_runs:
            row = next(p for p in r["per_clip"] if p["clip_id"] == cid)
            cells.append(f"{row['wer']:.3f}")
        md.append("| " + " | ".join(cells) + " |")
    md.append("")
    md.append("## Hallucination detector signals (primary HF run)")
    md.append("")
    md.append("| Clip | length_ratio | hyp 3-gram repeat | trigram ratio | RTL marks | WER |")
    md.append("|---|---|---|---|---|---|")
    primary = next(r for r in asr_runs if r["model_label"] == "ivrit_ai_whisper_large_v3_hf")
    for row in primary["per_clip"]:
        s = row["hallucination_signals"]
        md.append(
            f"| {row['clip_id']} | {s['length_ratio']:.3f} | {s['hyp_max_trigram_repeat']} | "
            f"{s['trigram_repeat_ratio']:.2f} | {s['rtl_mark_count']} | {row['wer']:.3f} |"
        )
    md.append("")
    md.append("## UTMOS — clean baseline")
    md.append("")
    md.append("| Clip | UTMOS |")
    md.append("|---|---|")
    for cid, score in results["utmos"]["clean"].items():
        md.append(f"| {cid} | {score:.3f} |")
    md.append("")
    md.append(
        f"clean mean (all): **{results['utmos']['clean_mean']:.3f}** "
        f"(n={len(results['utmos']['clean'])})  ·  "
        f"clean mean (degradation sample, n={len(results['utmos']['degradation_sample_clip_ids'])}): "
        f"**{results['utmos']['sample_clean_mean']:.3f}**"
    )
    md.append("")
    md.append("## UTMOS — degradation sweep")
    md.append("")
    md.append("| Degradation | mean UTMOS | clean − deg | direction |")
    md.append("|---|---|---|---|")
    # Paired comparison — degraded means cover only the degradation-sample clips,
    # so we subtract from the sample-matched clean mean, not the all-clip mean.
    sample_clean_mean = results["utmos"]["sample_clean_mean"]
    for d in results["utmos"]["degradations"]:
        diff = sample_clean_mean - d["mean_utmos"]
        direction = "↓ as expected" if diff > 0 else "↑ INVERTED"
        md.append(
            f"| `{d['id']}` ({d['kind']}) | {d['mean_utmos']:.3f} | {diff:+.3f} | {direction} |"
        )
    md.append("")
    md.append("Per-clip degraded scores in `results.json` under `utmos.degradations[].per_clip`.")
    md.append("")
    AUTO_REPORT_PATH.write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    np.random.seed(RNG_SEED)
    torch.manual_seed(RNG_SEED)
    SPIKE_DIR.mkdir(parents=True, exist_ok=True)

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"device={device}", flush=True)

    clips = load_clips()
    if not clips:
        raise SystemExit(f"No .wav clips in {CLIP_DIR}")
    print(f"loaded {len(clips)} clips: {[c.clip_id for c in clips]}", flush=True)

    manifest = [
        {
            "clip_id": c.clip_id,
            "duration_s": round(c.duration_s, 3),
            "sample_rate": c.sr,
            "ref_words": c.ref_words,
            "sha256": c.sha256,
        }
        for c in clips
    ]

    asr_runs: list[dict] = []
    for model_id in WHISPER_HF_MODELS:
        label = model_id.replace("/", "_").replace("-", "_") + "_hf"
        print(f"\n=== ASR (HF transformers): {model_id} ===", flush=True)
        try:
            hyps = asr_hf(model_id, clips, device)
        except Exception as e:
            if device == "cpu":
                # Already on CPU — no fallback to attempt; fail loudly.
                raise
            print(f"  HF/{device} failed ({e!r}); retrying on CPU", flush=True)
            hyps = asr_hf(model_id, clips, "cpu")
        run = score_asr_run(label, hyps, clips)
        run["backend"] = "transformers"
        run["model_id"] = model_id
        for row in run["per_clip"]:
            print(
                f"  {row['clip_id']}: WER={row['wer']:.3f} "
                f"len_ratio={row['hallucination_signals']['length_ratio']:.3f} "
                f"trigram_repeat={row['hallucination_signals']['hyp_max_trigram_repeat']}",
                flush=True,
            )
        print(f"  median WER={run['median_wer']:.3f} mean WER={run['mean_wer']:.3f}", flush=True)
        asr_runs.append(run)

    print(f"\n=== ASR (faster-whisper int8 / CPU): {FASTER_WHISPER_MODEL} ===", flush=True)
    try:
        fw_hyps = asr_faster_whisper(FASTER_WHISPER_MODEL, clips, "int8")
        fw_run = score_asr_run("ivrit_ai_whisper_large_v3_ct2_int8_fw", fw_hyps, clips)
        fw_run["backend"] = "faster-whisper"
        fw_run["model_id"] = FASTER_WHISPER_MODEL
        fw_run["compute_type"] = "int8"
        for row in fw_run["per_clip"]:
            print(f"  {row['clip_id']}: WER={row['wer']:.3f}", flush=True)
        print(f"  median WER={fw_run['median_wer']:.3f}", flush=True)
        asr_runs.append(fw_run)
    except Exception as e:
        print(f"  faster-whisper run failed: {e!r}", flush=True)
        fw_run = {
            "model_label": "ivrit_ai_whisper_large_v3_ct2_int8_fw",
            "backend": "faster-whisper",
            "model_id": FASTER_WHISPER_MODEL,
            "compute_type": "int8",
            "error": repr(e),
            "per_clip": [],
            "median_wer": None,
            "mean_wer": None,
            "median_cer": None,
            "mean_cer": None,
        }
        asr_runs.append(fw_run)

    print("\n=== UTMOS ===", flush=True)
    # Pinned to immutable commit (= tarepan/SpeechMOS v1.2.0) to avoid the supply-chain
    # risk of a moving tag. trust_repo=True is required by torch.hub when loading from
    # GitHub; for spike scope we accept this — a production E2 implementation should
    # vendor or hash-lock the model artifacts. See report Limitations §8.
    utmos_model = torch.hub.load(
        "tarepan/SpeechMOS:ed25eacbfa42b99156c36ebec67a733b5dbb9b79",
        "utmos22_strong",
        trust_repo=True,
    )
    rng = np.random.default_rng(RNG_SEED)

    clean_scores = {c.clip_id: utmos_score(utmos_model, c.wav, c.sr) for c in clips}
    for cid, s in clean_scores.items():
        print(f"  clean    {cid}: {s:.3f}", flush=True)
    clean_mean = float(np.mean(list(clean_scores.values())))

    sample_clips = clips[:DEGRADATION_SAMPLE_N]
    degradation_results = []
    for spec in DEGRADATIONS:
        per_clip_scores: dict[str, float] = {}
        per_clip_loudness: dict[str, dict] = {}
        for c in sample_clips:
            raw_degraded = apply_degradation(c.wav, c.sr, spec, rng)
            degraded, loudness_meta = rms_normalize_to_match(raw_degraded, c.wav)
            sample_path = SPIKE_DIR / f"{c.clip_id}__{spec['id']}.wav"
            sf.write(sample_path, degraded, c.sr, subtype="PCM_16")
            per_clip_scores[c.clip_id] = utmos_score(utmos_model, degraded, c.sr)
            per_clip_loudness[c.clip_id] = loudness_meta
        mean_u = float(np.mean(list(per_clip_scores.values())))
        any_clipped = any(m["peak_clipped"] for m in per_clip_loudness.values())
        print(
            f"  degradation {spec['id']}: mean UTMOS = {mean_u:.3f}"
            f"{' (peak-clipped on at least one clip)' if any_clipped else ''}",
            flush=True,
        )
        degradation_results.append(
            {
                **spec,
                "per_clip": per_clip_scores,
                "per_clip_loudness": per_clip_loudness,
                "mean_utmos": mean_u,
            }
        )

    # Gate: paired comparison between the SAME 5 clips clean and degraded.
    # `clean_mean` (n=10) is kept in the JSON as a population baseline, but the
    # gate must use `sample_clean_mean` to avoid mixing populations — degraded
    # means are over `sample_clips` only.
    sample_clean_mean = float(np.mean([clean_scores[c.clip_id] for c in sample_clips]))
    primary_deg = next(d for d in degradation_results if d["id"] == "wn_snr_-10db")
    primary_separation = sample_clean_mean - primary_deg["mean_utmos"]
    primary_gate = "PASS" if primary_separation >= UTMOS_SEPARATION_GATE else "FAIL"
    any_passes = any(
        (sample_clean_mean - d["mean_utmos"]) >= UTMOS_SEPARATION_GATE for d in degradation_results
    )
    monotonic_in_severity = is_monotonic_with_severity(
        [d for d in degradation_results if d["kind"] == "white_noise"]
    )

    asr_gates = {
        r["model_label"]: (
            "PASS"
            if (r.get("median_wer") is not None and r["median_wer"] < WER_GATE)
            else ("FAIL" if r.get("median_wer") is not None else "ERROR")
        )
        for r in asr_runs
    }
    asr_overall = "PASS" if any(g == "PASS" for g in asr_gates.values()) else "FAIL"

    valid_runs = [r for r in asr_runs if r.get("median_wer") is not None]
    preferred = (
        min(valid_runs, key=lambda r: r["median_wer"])["model_label"] if valid_runs else None
    )

    results = {
        "metadata": {
            "run_date": time.strftime("%Y-%m-%d"),
            "n_clips": len(clips),
            "device": device,
            "wer_gate": WER_GATE,
            "utmos_separation_gate": UTMOS_SEPARATION_GATE,
            "rng_seed": RNG_SEED,
        },
        "manifest": manifest,
        "asr_runs": asr_runs,
        "utmos": {
            "clean": clean_scores,
            "clean_mean": clean_mean,
            "sample_clean_mean": sample_clean_mean,
            "degradation_sample_clip_ids": [c.clip_id for c in sample_clips],
            "degradations": degradation_results,
            "primary_separation_db_-10_white_noise": primary_separation,
            "primary_gate": primary_gate,
            "any_separation_passes": any_passes,
            "white_noise_monotonic_in_severity": monotonic_in_severity,
        },
        "gates": {
            "asr_per_run": asr_gates,
            "asr_overall": asr_overall,
            "asr_preferred": preferred,
            "utmos_primary": primary_gate,
        },
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH.relative_to(REPO_ROOT)}", flush=True)
    write_auto_report(results)
    print(f"Wrote {AUTO_REPORT_PATH.relative_to(REPO_ROOT)}", flush=True)

    print("\n=== GATE SUMMARY ===", flush=True)
    print(f"  ASR overall: {asr_overall}", flush=True)
    for label, gate in asr_gates.items():
        run = next(r for r in asr_runs if r["model_label"] == label)
        median = run.get("median_wer")
        median_s = f"{median:.3f}" if median is not None else "ERR"
        print(f"    {label}: median WER {median_s} -> {gate}", flush=True)
    print(
        f"  UTMOS (primary -10 dB white noise): {primary_gate} "
        f"(separation {primary_separation:+.3f})",
        flush=True,
    )
    print(f"  UTMOS any-separation-passes: {any_passes}", flush=True)
    print(f"  UTMOS white-noise monotonic in severity: {monotonic_in_severity}", flush=True)
    print(f"  Preferred ASR run: {preferred}", flush=True)


if __name__ == "__main__":
    main()
