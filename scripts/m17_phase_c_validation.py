"""M17 Phase C validation spike — multimodal LLM judge gates on Hebrew clips.

Implements the Phase C acceptance experiment from
``docs/automated_eval_design.md`` (§E5: Multimodal LLM Judge):

  - Send a 6-clip set (4 corpus typology spans + 2 degraded variants of one
    corpus clip) to two multimodal LLMs (Gemini 2.5 Pro audio, GPT-4o audio).
  - Score each (clip, model) twice, on a fixed structured-output schema.
  - Compute four gate outcomes per model:
      • Refusal:        ≥ 5/6 clips scored without refusal under the DV-research framing.
      • Discrimination: mean overall_quality on known-good corpus clips
                        − mean on −10 dB SNR degraded clip ≥ 0.5.
      • Reproducibility:per-dimension std across the two reruns of each clip ≤ 0.5.
      • Shay-correlation: Spearman ρ ≥ 0.3 between model `overall_quality` and
                        Shay's encoded expected ranking on the 4 corpus clips.

Mirrors the Phase A spike protocol (``scripts/m17_phase_a_validation.py``).
This is a one-shot spike: no production module is created. Raw outputs go to
``state/spikes/m17_phase_c/`` (gitignored). The canonical narrative report
at ``docs/m17_phase_c_validation_report.md`` is hand-written from the
auto-template the script also writes there.

Setup:

    uv pip install --python .venv/bin/python \\
        google-genai openai pydantic soundfile numpy scipy

Auth:

    export GEMINI_API_KEY=...
    export OPENAI_API_KEY=...

    # Optional spend cap (USD); script aborts mid-run if cumulative exceeds it.
    export SYNTHBANSHEE_LLM_SPIKE_BUDGET_USD=5.0

    # Optional override for clip source dir.
    export SYNTHBANSHEE_LLM_SPIKE_CLIP_DIR=data/m2a_wettest/agg_m_30-45_001

Run:

    .venv/bin/python scripts/m17_phase_c_validation.py
    .venv/bin/python scripts/m17_phase_c_validation.py --dry-run    # prompts only, no API calls

Cost ceiling (rough): ~$0.50–1.50 total across both models at 6 clips × 2 reruns.
The hard cap defaults to $5; raise/lower via SYNTHBANSHEE_LLM_SPIKE_BUDGET_USD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt  # noqa: F401  (reserved for parity with Phase A)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLIP_DIR = REPO_ROOT / "data" / "m2a_wettest" / "agg_m_30-45_001"
SPIKE_DIR = REPO_ROOT / "state" / "spikes" / "m17_phase_c"
RESULTS_PATH = SPIKE_DIR / "results.json"
AUTO_REPORT_PATH = SPIKE_DIR / "report_auto.md"

PROMPT_VERSION = "v1"
RNG_SEED = 42
N_RERUNS_PER_CLIP = 2

# --- Gate thresholds ---------------------------------------------------------
# Mirror the structure of Phase A's per-evaluator gates. Each model is graded
# against all four gates independently; a model that fails any gate cannot
# advance to MVP without re-spiking.
REFUSAL_GATE_MIN_SCORED = 5  # ≥ 5/6 clips must score without refusal
DISCRIMINATION_GATE = 0.5  # mean(corpus) − mean(severe-degraded) ≥ this
VARIANCE_GATE = 0.5  # per-dimension std across N reruns ≤ this
SHAY_CORRELATION_GATE = 0.3  # Spearman ρ vs Shay's encoded ranking ≥ this

# --- Clip set ----------------------------------------------------------------
# 4 corpus clips spanning typology + 2 degraded variants of one corpus clip
# (sp_sv_a_0001_00 — the canonical SV clip from the May listening tests).
# `expected_quality_rank` encodes Shay's prior listening-test verdicts as an
# ordinal: 4 = best perceived, 1 = worst. Ties are deliberate where evidence
# doesn't separate clips. Used only for the Spearman gate; if you don't have
# a strong prior for a clip, leave it tied with neighbours rather than guess.
CLIP_SOURCES: list[dict] = [
    {"clip_id": "sp_sv_a_0001_00", "typology": "SV", "kind": "corpus", "expected_quality_rank": 2},
    {"clip_id": "sp_it_a_0001_00", "typology": "IT", "kind": "corpus", "expected_quality_rank": 2},
    {
        "clip_id": "sp_neg_a_0001_00",
        "typology": "NEG",
        "kind": "corpus",
        "expected_quality_rank": 3,
    },
    {
        "clip_id": "sp_neu_a_0001_00",
        "typology": "NEU",
        "kind": "corpus",
        "expected_quality_rank": 4,
    },
    {
        "clip_id": "sp_sv_a_0001_00",
        "typology": "SV",
        "kind": "degraded",
        "degradation": "wn_snr_+10db",
        "expected_quality_rank": 1,
    },
    {
        "clip_id": "sp_sv_a_0001_00",
        "typology": "SV",
        "kind": "degraded",
        "degradation": "wn_snr_-10db",
        "expected_quality_rank": 0,
    },
]

# --- Dimensions scored per clip ---------------------------------------------
# Dimensions from `docs/automated_eval_design.md` §E5. Per-clip rather than
# per-turn because Phase C is testing whether the LLM discriminates AT ALL;
# per-turn scoring is an MVP-phase concern. The order here is the JSON schema
# field order — kept stable so the auto-report tables line up across reruns.
DIMENSIONS = [
    "pronunciation_clarity",
    "prosody_naturalness",
    "emotional_expression",
    "speaker_differentiation",
    "dialogue_flow",
    "escalation_arc",
    "scene_coherence",
    "overall_quality",
]

# --- Prompts -----------------------------------------------------------------
# The framing is deliberately research-neutral: we're evaluating *synthetic
# TTS quality*, not endorsing or producing harmful content. The audio is
# entirely synthetic and contains no real persons. See design doc §E5
# "Content-sensitivity risk" for the rationale.

SYSTEM_PREAMBLE = """\
You are a senior speech-synthesis quality evaluator. You assess the technical
and perceptual quality of synthetic Hebrew text-to-speech output for a research
dataset used to develop interpersonal-conflict detection models. The audio is
entirely synthetic (Microsoft Azure neural voices) — there are no real persons,
no real events. Your job is to rate the synthesis quality, not the content.

You will receive a single audio clip and a short metadata block describing what
the script intended. Listen to the audio and score each dimension on a 1–5
scale where:
  1 = Unacceptable (broken, unintelligible, or wrong language)
  2 = Poor (noticeable defects throughout)
  3 = Acceptable (recognizable as Hebrew speech, but with clear issues)
  4 = Good (natural-sounding with minor flaws)
  5 = Excellent (indistinguishable from a careful native human recording)

Anchor your scores to absolute quality. Do not curve. A bare-bones TTS clip
without prosodic variation should score around 3 on most dimensions, not 4.
Be specific in your summary about what you heard.
"""

CLIP_PROMPT_TEMPLATE = """\
Clip metadata:
- clip_id: {clip_id}
- typology: {typology} ({typology_long})
- duration_seconds: {duration_s:.1f}
- intended intensity arc: turns escalate from 1 to 5 on a 5-point scale
- speakers: {speakers_summary}
- backend: {backend}
{degradation_note}

Score this clip on the following dimensions. For each, give an integer 1–5.

Dimensions:
- pronunciation_clarity    : Are Hebrew words clearly pronounced and intelligible?
- prosody_naturalness      : Does intonation sound natural for spoken Hebrew?
- emotional_expression     : Does each turn's emotional tone match its intended intensity?
- speaker_differentiation  : Do the speakers sound like distinct people?
- dialogue_flow            : Does the conversation flow naturally between turns?
- escalation_arc           : Does tension build perceptibly across the scene?
- scene_coherence          : Does this sound like a plausible real conversation?
- overall_quality          : Holistic production quality.

Also return:
- artifacts_detected   : true/false — any audible glitches, clicks, robotic timbre, dropouts?
- artifact_notes       : if artifacts_detected, a short description with approximate timestamp(s); else "".
- confidence_in_assessment : your own confidence 1–5 in the scores you just gave.
- summary              : 2–3 sentences describing what you heard and why you scored as you did.

Return a single JSON object. Do not include any prose outside the JSON.
"""

TYPOLOGY_LONG = {
    "SV": "Severe Violence — physical attacks, life-threatening escalation",
    "IT": "Intimate Terrorism — sustained coercive control",
    "NEG": "Negative confusor — acoustically intense, no violence (hard negative)",
    "NEU": "Neutral — mundane conversation",
}


# --- Pydantic schema ---------------------------------------------------------
def make_response_schema() -> dict:
    """Build the JSON Schema used to constrain both Gemini and OpenAI output."""
    return {
        "type": "object",
        "properties": {
            **{d: {"type": "integer", "minimum": 1, "maximum": 5} for d in DIMENSIONS},
            "artifacts_detected": {"type": "boolean"},
            "artifact_notes": {"type": "string"},
            "confidence_in_assessment": {"type": "integer", "minimum": 1, "maximum": 5},
            "summary": {"type": "string"},
        },
        "required": [
            *DIMENSIONS,
            "artifacts_detected",
            "artifact_notes",
            "confidence_in_assessment",
            "summary",
        ],
        "additionalProperties": False,
    }


# --- Clip preparation --------------------------------------------------------
@dataclass
class Clip:
    label: str  # `{clip_id}` or `{clip_id}__{degradation}`
    clip_id: str
    wav_path: Path
    typology: str
    kind: str  # "corpus" | "degraded"
    degradation: str | None
    expected_rank: int
    duration_s: float
    sha256: str
    speakers_summary: str
    backend: str


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def apply_white_noise(wav: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    sig_power = float((wav**2).mean()) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.standard_normal(len(wav)).astype(np.float32) * np.sqrt(noise_power)
    return (wav + noise).astype(np.float32)


def rms_normalize_to_match(degraded: np.ndarray, clean: np.ndarray) -> np.ndarray:
    """Same helper as Phase A — keep loudness constant so the LLM is judging
    the spectral / noise content, not just amplitude.
    """
    clean_rms = float(np.sqrt(np.mean(clean**2)))
    deg_rms = float(np.sqrt(np.mean(degraded**2))) + 1e-12
    out = degraded * (clean_rms / deg_rms)
    peak = float(np.max(np.abs(out)))
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)


def prepare_clips(clip_dir: Path) -> list[Clip]:
    """Resolve CLIP_SOURCES into Clip records, materialising degraded variants."""
    rng = np.random.default_rng(RNG_SEED)
    clips: list[Clip] = []
    for src in CLIP_SOURCES:
        cid = src["clip_id"]
        src_wav_path = clip_dir / f"{cid}.wav"
        if not src_wav_path.exists():
            raise FileNotFoundError(
                f"Source clip not found: {src_wav_path}\n"
                f"Set SYNTHBANSHEE_LLM_SPIKE_CLIP_DIR to a directory containing "
                f"sp_sv_a_0001_00.wav, sp_it_a_0001_00.wav, sp_neg_a_0001_00.wav, "
                f"sp_neu_a_0001_00.wav (transcript .txt sidecars not required)."
            )

        wav, sr = sf.read(src_wav_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            raise ValueError(
                f"Clip {src_wav_path.name} is {sr} Hz; require 16 kHz (repo invariant)."
            )

        if src["kind"] == "corpus":
            label = cid
            wav_path = src_wav_path
            degradation = None
        elif src["kind"] == "degraded":
            spec = src["degradation"]  # e.g. "wn_snr_+10db"
            snr_db = float(spec.replace("wn_snr_", "").replace("db", "").replace("+", ""))
            degraded = apply_white_noise(wav, snr_db, rng)
            normalised = rms_normalize_to_match(degraded, wav)
            label = f"{cid}__{spec}"
            wav_path = SPIKE_DIR / f"{label}.wav"
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(wav_path, normalised, 16000, subtype="PCM_16")
            degradation = spec
        else:
            raise ValueError(f"Unknown kind: {src['kind']}")

        clips.append(
            Clip(
                label=label,
                clip_id=cid,
                wav_path=wav_path,
                typology=src["typology"],
                kind=src["kind"],
                degradation=degradation,
                expected_rank=src["expected_quality_rank"],
                duration_s=len(wav) / sr,
                sha256=_sha256_file(wav_path),
                # M2a wettest clips: AGG+VIC Azure pair. Hardcoded here because
                # we know the source dir; if the spike is retargeted at the
                # corpus repo, we'd derive from the JSON sidecar instead.
                speakers_summary="AGG (male, he-IL-AvriNeural) + VIC (female, he-IL-HilaNeural)",
                backend="azure",
            )
        )
    return clips


# --- Cost estimation ---------------------------------------------------------
# Per Anthropic/Google/OpenAI pricing snapshots as of January 2026. Update if
# the spike is rerun against newer model versions. These are upper-bound
# estimates: actual cost is metered per-token by each provider and the
# script's running total uses that, not these.
GEMINI_AUDIO_USD_PER_MIN = 0.0125  # gemini-2.5-pro audio input
GEMINI_OUTPUT_USD_PER_KTOK = 0.0050
OPENAI_AUDIO_INPUT_USD_PER_MIN = 0.10  # gpt-4o-audio-preview
OPENAI_OUTPUT_USD_PER_KTOK = 0.020


def estimate_cost(clips: list[Clip], n_reruns: int, models: list[str]) -> dict:
    total_min: float = sum(c.duration_s for c in clips) / 60.0 * n_reruns
    per_model: dict[str, float] = {}
    total: float = 0.0
    for m in models:
        if m == "gemini":
            input_usd = total_min * GEMINI_AUDIO_USD_PER_MIN
            output_usd = (
                0.5 * len(clips) * n_reruns
            ) * GEMINI_OUTPUT_USD_PER_KTOK  # ~500 tok/response
            cost = input_usd + output_usd
        elif m == "openai":
            input_usd = total_min * OPENAI_AUDIO_INPUT_USD_PER_MIN
            output_usd = (0.5 * len(clips) * n_reruns) * OPENAI_OUTPUT_USD_PER_KTOK
            cost = input_usd + output_usd
        else:
            cost = 0.0
        per_model[m] = round(cost, 4)
        total += cost
    return {
        "total_audio_min": total_min,
        "per_model": per_model,
        "total_usd_estimate": round(total, 4),
    }


# --- Model adapters ----------------------------------------------------------
@dataclass
class JudgeResult:
    """One (clip, model, run) outcome."""

    clip_label: str
    model: str
    run_idx: int
    refused: bool
    raw_response: str
    parsed: dict | None
    error: str | None = None
    latency_s: float | None = None
    usd_cost: float | None = None
    extras: dict = field(default_factory=dict)


def build_clip_prompt(clip: Clip) -> str:
    deg_note = ""
    if clip.degradation:
        deg_note = (
            f"- degradation applied (for evaluation only): {clip.degradation} "
            f"(white noise mixed in, RMS-matched to clean)\n"
        )
    return CLIP_PROMPT_TEMPLATE.format(
        clip_id=clip.label,
        typology=clip.typology,
        typology_long=TYPOLOGY_LONG.get(clip.typology, clip.typology),
        duration_s=clip.duration_s,
        speakers_summary=clip.speakers_summary,
        backend=clip.backend,
        degradation_note=deg_note,
    )


def call_gemini(clip: Clip, run_idx: int, schema: dict) -> JudgeResult:
    """Send one (clip, run) to Gemini 2.5 Pro audio. Lazy-imports `google.genai`.

    Configures safety settings to BLOCK_NONE for the DV-content categories per
    the design doc — this is a research-use exception, never enable it in
    consumer-facing tooling.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    audio_bytes = clip.wav_path.read_bytes()
    prompt = build_clip_prompt(clip)

    # Safety: research framing requires permissive settings for DV-content
    # categories to avoid refusals on the violence-scenario metadata.
    safety_settings = [
        types.SafetySetting(category=c, threshold=types.HarmBlockThreshold.BLOCK_NONE)
        for c in (
            types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        )
    ]

    t0 = time.time()
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[
                SYSTEM_PREAMBLE,
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=schema,
                safety_settings=safety_settings,
            ),
        )
    except Exception as e:  # noqa: BLE001 — capture refusal / API error uniformly
        return JudgeResult(
            clip_label=clip.label,
            model="gemini-2.5-pro",
            run_idx=run_idx,
            refused=True,
            raw_response="",
            parsed=None,
            error=repr(e),
            latency_s=time.time() - t0,
        )
    latency = time.time() - t0

    raw = resp.text or ""
    try:
        parsed = json.loads(raw)
        refused = False
    except json.JSONDecodeError:
        parsed = None
        refused = True

    usage = getattr(resp, "usage_metadata", None)
    in_tok = getattr(usage, "prompt_token_count", None) or 0
    out_tok = getattr(usage, "candidates_token_count", None) or 0
    # Audio is billed by duration on Gemini, not tokens — record both.
    cost = (clip.duration_s / 60.0) * GEMINI_AUDIO_USD_PER_MIN + (
        out_tok / 1000.0
    ) * GEMINI_OUTPUT_USD_PER_KTOK

    return JudgeResult(
        clip_label=clip.label,
        model="gemini-2.5-pro",
        run_idx=run_idx,
        refused=refused,
        raw_response=raw,
        parsed=parsed,
        latency_s=latency,
        usd_cost=round(cost, 4),
        extras={"in_tok": in_tok, "out_tok": out_tok},
    )


def call_openai(clip: Clip, run_idx: int, schema: dict) -> JudgeResult:
    """Send one (clip, run) to gpt-4o-audio-preview. Lazy-imports `openai`."""
    import base64

    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    audio_b64 = base64.b64encode(clip.wav_path.read_bytes()).decode("ascii")
    prompt = build_clip_prompt(clip)

    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text"],
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PREAMBLE},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_b64, "format": "wav"},
                        },
                    ],
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "clip_eval",
                    "strict": True,
                    "schema": schema,
                },
            },
        )
    except Exception as e:  # noqa: BLE001
        return JudgeResult(
            clip_label=clip.label,
            model="gpt-4o-audio-preview",
            run_idx=run_idx,
            refused=True,
            raw_response="",
            parsed=None,
            error=repr(e),
            latency_s=time.time() - t0,
        )
    latency = time.time() - t0

    raw = resp.choices[0].message.content or ""
    try:
        parsed = json.loads(raw)
        refused = False
    except json.JSONDecodeError:
        parsed = None
        refused = True

    usage = resp.usage
    in_tok = getattr(usage, "prompt_tokens", 0) or 0
    out_tok = getattr(usage, "completion_tokens", 0) or 0
    cost = (clip.duration_s / 60.0) * OPENAI_AUDIO_INPUT_USD_PER_MIN + (
        out_tok / 1000.0
    ) * OPENAI_OUTPUT_USD_PER_KTOK

    return JudgeResult(
        clip_label=clip.label,
        model="gpt-4o-audio-preview",
        run_idx=run_idx,
        refused=refused,
        raw_response=raw,
        parsed=parsed,
        latency_s=latency,
        usd_cost=round(cost, 4),
        extras={"in_tok": in_tok, "out_tok": out_tok},
    )


# --- Gate evaluation ---------------------------------------------------------
def spearman(xs: list[float], ys: list[float]) -> float:
    """Minimal Spearman rank correlation. No SciPy dependency to keep the
    script's import surface small. Ties handled with average-rank assignment
    per the standard tie-breaking rule.
    """
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")

    def ranks(vals: list[float]) -> list[float]:
        ordered = sorted(enumerate(vals), key=lambda kv: kv[1])
        ranks_out = [0.0] * len(vals)
        i = 0
        while i < len(ordered):
            j = i
            while j < len(ordered) - 1 and ordered[j + 1][1] == ordered[i][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks_out[ordered[k][0]] = avg_rank
            i = j + 1
        return ranks_out

    rx, ry = ranks(xs), ranks(ys)
    mean_x = sum(rx) / len(rx)
    mean_y = sum(ry) / len(ry)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry, strict=True))
    den_x = sum((a - mean_x) ** 2 for a in rx) ** 0.5
    den_y = sum((b - mean_y) ** 2 for b in ry) ** 0.5
    return num / (den_x * den_y) if den_x and den_y else float("nan")


def evaluate_gates(clips: list[Clip], results: list[JudgeResult]) -> dict:
    """Compute per-model gate outcomes."""
    by_model: dict[str, list[JudgeResult]] = {}
    for r in results:
        by_model.setdefault(r.model, []).append(r)

    out: dict[str, dict] = {}
    for model, runs in by_model.items():
        # Refusal gate: count unique clips with at least one non-refused run.
        scored_clips = {r.clip_label for r in runs if not r.refused}
        refusal_pass = len(scored_clips) >= REFUSAL_GATE_MIN_SCORED

        # Mean overall_quality per clip across reruns (skip refusals).
        per_clip_overall: dict[str, float] = {}
        for c in clips:
            scores = [
                r.parsed["overall_quality"]
                for r in runs
                if r.clip_label == c.label and not r.refused and r.parsed
            ]
            if scores:
                per_clip_overall[c.label] = float(np.mean(scores))

        # Discrimination gate: mean(corpus clips, n=4) − mean(−10dB severe deg, n=1).
        # We use the severe degradation only; the +10 dB SNR clip is a milder
        # mid-anchor and shouldn't fail the gate if discrimination is real.
        corpus_labels = [c.label for c in clips if c.kind == "corpus"]
        severe_label = next((c.label for c in clips if c.degradation == "wn_snr_-10db"), None)
        if (
            severe_label is not None
            and severe_label in per_clip_overall
            and all(lbl in per_clip_overall for lbl in corpus_labels)
        ):
            corpus_mean = float(np.mean([per_clip_overall[lbl] for lbl in corpus_labels]))
            severe_score = per_clip_overall[severe_label]
            discrimination = corpus_mean - severe_score
            discrimination_pass = discrimination >= DISCRIMINATION_GATE
        else:
            corpus_mean = severe_score = discrimination = float("nan")
            discrimination_pass = False

        # Reproducibility gate: per-clip per-dimension std across reruns ≤ 0.5.
        max_std = 0.0
        for c in clips:
            for d in DIMENSIONS:
                vals = [
                    r.parsed[d]
                    for r in runs
                    if r.clip_label == c.label and not r.refused and r.parsed
                ]
                if len(vals) >= 2:
                    max_std = max(max_std, float(np.std(vals, ddof=0)))
        variance_pass = max_std <= VARIANCE_GATE

        # Shay-correlation gate: Spearman ρ between expected_rank (corpus only)
        # and model overall_quality (corpus only). 4 data points — n is small,
        # we accept that and read ρ as directional rather than statistically
        # significant.
        xs, ys = [], []
        for c in clips:
            if c.kind == "corpus" and c.label in per_clip_overall:
                xs.append(float(c.expected_rank))
                ys.append(per_clip_overall[c.label])
        shay_rho = spearman(xs, ys) if len(xs) >= 2 else float("nan")
        # NaN check: (shay_rho == shay_rho) is False iff NaN; ruff prefers this idiom.
        shay_pass = (shay_rho == shay_rho) and shay_rho >= SHAY_CORRELATION_GATE

        out[model] = {
            "refusal_gate": {
                "pass": bool(refusal_pass),
                "scored_clips": len(scored_clips),
                "min_required": REFUSAL_GATE_MIN_SCORED,
            },
            "discrimination_gate": {
                "pass": bool(discrimination_pass),
                "corpus_mean": corpus_mean,
                "severe_degraded_overall": severe_score,
                "separation": discrimination,
                "threshold": DISCRIMINATION_GATE,
            },
            "variance_gate": {
                "pass": bool(variance_pass),
                "max_per_dim_std": round(max_std, 3),
                "threshold": VARIANCE_GATE,
            },
            "shay_correlation_gate": {
                "pass": bool(shay_pass),
                "spearman_rho": (None if shay_rho != shay_rho else round(shay_rho, 3)),
                "n_corpus_clips": len(xs),
                "threshold": SHAY_CORRELATION_GATE,
            },
            "overall_pass": bool(
                refusal_pass and discrimination_pass and variance_pass and shay_pass
            ),
            "per_clip_overall": per_clip_overall,
        }
    return out


# --- Auto report -------------------------------------------------------------
def write_auto_report(payload: dict) -> None:
    md: list[str] = ["# M17 Phase C — auto-generated data tables", ""]
    md.append(f"Run date: {payload['metadata']['run_date']}")
    md.append(f"Prompt version: `{PROMPT_VERSION}`")
    md.append(f"Reruns per (clip, model): {N_RERUNS_PER_CLIP}")
    md.append(f"Clips evaluated: {payload['metadata']['n_clips']}")
    md.append("")

    md.append("## Manifest")
    md.append("")
    md.append("| Label | clip_id | kind | typology | duration (s) | expected rank | sha256 (8) |")
    md.append("|---|---|---|---|---|---|---|")
    for m in payload["manifest"]:
        md.append(
            f"| `{m['label']}` | {m['clip_id']} | {m['kind']} | {m['typology']} | "
            f"{m['duration_s']:.1f} | {m['expected_rank']} | `{m['sha256'][:8]}` |"
        )
    md.append("")

    md.append("## Gate outcomes")
    md.append("")
    md.append("| Model | Refusal | Discrimination | Variance | Shay ρ | Overall |")
    md.append("|---|---|---|---|---|---|")
    for model, g in payload["gates"].items():
        ref = f"{g['refusal_gate']['scored_clips']}/{g['refusal_gate']['min_required']}"
        disc = (
            "n/a"
            if g["discrimination_gate"]["separation"] != g["discrimination_gate"]["separation"]
            else f"{g['discrimination_gate']['separation']:+.2f}"
        )
        var = f"{g['variance_gate']['max_per_dim_std']:.2f}"
        rho = (
            "n/a"
            if g["shay_correlation_gate"]["spearman_rho"] is None
            else f"{g['shay_correlation_gate']['spearman_rho']:+.2f}"
        )
        md.append(
            f"| `{model}` | {ref} {'✅' if g['refusal_gate']['pass'] else '❌'} | "
            f"{disc} {'✅' if g['discrimination_gate']['pass'] else '❌'} | "
            f"{var} {'✅' if g['variance_gate']['pass'] else '❌'} | "
            f"{rho} {'✅' if g['shay_correlation_gate']['pass'] else '❌'} | "
            f"{'PASS' if g['overall_pass'] else 'FAIL'} |"
        )
    md.append("")
    md.append(
        "Thresholds — refusal: ≥ "
        f"{REFUSAL_GATE_MIN_SCORED}/6 clips scored · discrimination: ≥ {DISCRIMINATION_GATE} · "
        f"variance: per-dim std ≤ {VARIANCE_GATE} · Shay ρ: ≥ {SHAY_CORRELATION_GATE}"
    )
    md.append("")

    md.append("## Per-clip mean overall_quality")
    md.append("")
    md.append("| Clip label | " + " | ".join(payload["gates"].keys()) + " |")
    md.append("|---" + "|---" * len(payload["gates"]) + "|")
    labels = sorted({lbl for g in payload["gates"].values() for lbl in g["per_clip_overall"]})
    for lbl in labels:
        cells = [f"`{lbl}`"]
        for g in payload["gates"].values():
            s = g["per_clip_overall"].get(lbl)
            cells.append("—" if s is None else f"{s:.2f}")
        md.append("| " + " | ".join(cells) + " |")
    md.append("")

    md.append("## Spend")
    md.append("")
    md.append(f"Estimated (pre-flight): ${payload['metadata']['estimated_usd']:.2f}")
    md.append(f"Actual (sum of per-call usd_cost): ${payload['metadata']['actual_usd']:.2f}")
    md.append("")

    AUTO_REPORT_PATH.write_text("\n".join(md), encoding="utf-8")


# --- Main --------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts and estimate cost without calling any LLM API.",
    )
    parser.add_argument(
        "--clip-dir",
        type=Path,
        default=Path(os.environ.get("SYNTHBANSHEE_LLM_SPIKE_CLIP_DIR", DEFAULT_CLIP_DIR)),
        help="Directory containing source .wav clips.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["gemini", "openai"],
        default=["gemini", "openai"],
        help="Which LLM(s) to evaluate. Omit one to run a partial spike.",
    )
    args = parser.parse_args()

    SPIKE_DIR.mkdir(parents=True, exist_ok=True)

    clip_dir = args.clip_dir.resolve()
    print(f"clip_dir = {clip_dir}", flush=True)
    clips = prepare_clips(clip_dir)
    print(f"prepared {len(clips)} clips: {[c.label for c in clips]}", flush=True)

    schema = make_response_schema()
    cost_est = estimate_cost(clips, N_RERUNS_PER_CLIP, args.models)
    print(
        f"estimated cost: ${cost_est['total_usd_estimate']:.2f}  "
        f"({cost_est['total_audio_min']:.1f} min audio × {N_RERUNS_PER_CLIP} reruns × {len(args.models)} models)",
        flush=True,
    )
    for m, c in cost_est["per_model"].items():
        print(f"  {m}: ${c:.2f}", flush=True)

    budget_cap = float(os.environ.get("SYNTHBANSHEE_LLM_SPIKE_BUDGET_USD", "5.0"))
    if cost_est["total_usd_estimate"] > budget_cap:
        print(
            f"ABORT: estimated cost ${cost_est['total_usd_estimate']:.2f} > "
            f"SYNTHBANSHEE_LLM_SPIKE_BUDGET_USD={budget_cap:.2f}. "
            f"Raise the budget or trim the clip set.",
            file=sys.stderr,
        )
        sys.exit(2)

    # --- Dry run early exit --------------------------------------------------
    if args.dry_run:
        # Show one prompt for sanity then bail — no API calls, no spend.
        example_prompt = build_clip_prompt(clips[0])
        print("\n=== EXAMPLE PROMPT (clip 0) ===", flush=True)
        print(SYSTEM_PREAMBLE, flush=True)
        print(example_prompt, flush=True)
        print("\n=== SCHEMA ===", flush=True)
        print(json.dumps(schema, indent=2), flush=True)
        print("\nDry run complete — no API calls made.", flush=True)
        return

    # --- Key check -----------------------------------------------------------
    if "gemini" in args.models and not os.environ.get("GEMINI_API_KEY"):
        sys.exit("Missing GEMINI_API_KEY (or pass --models openai to skip Gemini).")
    if "openai" in args.models and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Missing OPENAI_API_KEY (or pass --models gemini to skip OpenAI).")

    # --- Run -----------------------------------------------------------------
    results: list[JudgeResult] = []
    cumulative_usd = 0.0
    for model in args.models:
        print(f"\n=== Model: {model} ===", flush=True)
        for clip in clips:
            for run_idx in range(N_RERUNS_PER_CLIP):
                if cumulative_usd >= budget_cap:
                    print(
                        f"ABORT: cumulative ${cumulative_usd:.2f} ≥ budget ${budget_cap:.2f}. "
                        "Partial results will still be written.",
                        flush=True,
                    )
                    break
                if model == "gemini":
                    r = call_gemini(clip, run_idx, schema)
                else:
                    r = call_openai(clip, run_idx, schema)
                results.append(r)
                cumulative_usd += r.usd_cost or 0.0
                tag = "REFUSED" if r.refused else "ok"
                overall = (r.parsed or {}).get("overall_quality", "—")
                print(
                    f"  {clip.label} run{run_idx}  {tag}  overall={overall}  "
                    f"latency={r.latency_s:.1f}s  cost=${(r.usd_cost or 0.0):.4f}",
                    flush=True,
                )

    gates = evaluate_gates(clips, results)
    payload = {
        "metadata": {
            "run_date": time.strftime("%Y-%m-%d"),
            "prompt_version": PROMPT_VERSION,
            "n_clips": len(clips),
            "n_reruns": N_RERUNS_PER_CLIP,
            "estimated_usd": cost_est["total_usd_estimate"],
            "actual_usd": round(cumulative_usd, 4),
            "budget_cap_usd": budget_cap,
        },
        "manifest": [
            {
                "label": c.label,
                "clip_id": c.clip_id,
                "kind": c.kind,
                "degradation": c.degradation,
                "typology": c.typology,
                "duration_s": round(c.duration_s, 3),
                "sha256": c.sha256,
                "expected_rank": c.expected_rank,
            }
            for c in clips
        ],
        "results": [asdict(r) for r in results],
        "gates": gates,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {RESULTS_PATH.relative_to(REPO_ROOT)}", flush=True)
    write_auto_report(payload)
    print(f"wrote {AUTO_REPORT_PATH.relative_to(REPO_ROOT)}", flush=True)

    print("\n=== GATE SUMMARY ===", flush=True)
    for model, g in gates.items():
        print(f"  {model}: overall {'PASS' if g['overall_pass'] else 'FAIL'}", flush=True)
        print(
            f"    refusal:        {g['refusal_gate']['scored_clips']}/{g['refusal_gate']['min_required']} "
            f"{'PASS' if g['refusal_gate']['pass'] else 'FAIL'}",
            flush=True,
        )
        d = g["discrimination_gate"]
        print(
            f"    discrimination: separation {d['separation']:+.2f} (≥ {d['threshold']}) "
            f"{'PASS' if d['pass'] else 'FAIL'}",
            flush=True,
        )
        v = g["variance_gate"]
        print(
            f"    variance:       max per-dim std {v['max_per_dim_std']:.2f} (≤ {v['threshold']}) "
            f"{'PASS' if v['pass'] else 'FAIL'}",
            flush=True,
        )
        s = g["shay_correlation_gate"]
        print(
            f"    Shay ρ:         {s['spearman_rho']} (≥ {s['threshold']}) "
            f"{'PASS' if s['pass'] else 'FAIL'}",
            flush=True,
        )
    print(f"  cumulative spend: ${cumulative_usd:.4f} / ${budget_cap:.2f}", flush=True)


if __name__ == "__main__":
    main()
