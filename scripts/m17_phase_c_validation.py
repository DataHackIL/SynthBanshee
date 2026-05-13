"""M17 Phase C validation spike — multimodal LLM judge gates on Hebrew clips.

Implements the Phase C acceptance experiment from
``docs/automated_eval_design.md`` (§E5: Multimodal LLM Judge):

  - Send a 6-clip set (4 corpus typology spans + 2 degraded variants of one
    corpus clip) to two multimodal LLMs (Gemini 2.5 Pro audio, GPT-4o audio).
  - Score each (clip, model) twice, on a fixed structured-output schema.
  - Compute four gate outcomes per model (overall_pass = refusal AND discrimination):
      • Refusal:        ≥ 5/6 clips scored; only content_refusal counts against
                        this — api_error / json_parse_error are infra failures.
      • Discrimination: two independent arms, at least one must clear:
                        (1) noise_corruption: corpus mean − wn_snr_-10db ≥ 0.5
                            (kept for Phase A E2/UTMOS comparability)
                        (2) synth_failure: corpus mean − synth_rate_slow_0.7x ≥ 0.5
                            (the real target: can the model hear synthesis defects?)
      • Variance (advisory): per-dim std across N reruns ≤ 0.5.
                        Trivially PASS at TEMPERATURE=0 — use only with T>0, N≥4.
      • Shay-correlation (advisory): Spearman ρ ≥ 0.3 vs encoded expected ranking.
                        n=4 is not statistically significant; read as directional only.

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
RESULTS_PARTIAL_PATH = SPIKE_DIR / "results_partial.jsonl"
AUTO_REPORT_PATH = SPIKE_DIR / "report_auto.md"

PROMPT_VERSION = "v1"
RNG_SEED = 42
N_RERUNS_PER_CLIP = 2

# Temperature for both models. Keep at 0.0 for structured-output stability with
# greedy decoding. IMPORTANT: at T=0 the variance gate is trivially PASS (std=0
# for any deterministic run). If reproducibility is genuinely uncertain, re-run
# with TEMPERATURE=0.1 and N_RERUNS_PER_CLIP=4. See gate notes below.
TEMPERATURE = 0.0

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
        # Synthesis-failure anchor: resampled to 70% speed then trimmed to original
        # length. This simulates over-slow TTS (rate=0.7) with pitch-shift artefacts
        # and an abruptly cut-off scene. Perceptually: unnatural tempo, wrong pitch,
        # incomplete escalation arc — the failure modes we actually care about.
        # Replaces the former wn_snr_+10db mild-noise anchor which only measured
        # whether the model can hear noise, not whether it can judge synthesis quality.
        "degradation": "synth_rate_slow_0.7x",
        "expected_quality_rank": 1,
    },
    {
        "clip_id": "sp_sv_a_0001_00",
        "typology": "SV",
        "kind": "degraded",
        # Signal-corruption anchor (white noise −10 dB SNR): kept for comparability
        # with the Phase A E2/UTMOS discrimination gate. Any model that passes this
        # but fails the synth_rate_slow anchor is detecting signal corruption, not
        # synthesis quality — record that finding explicitly in the report.
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

# Metadata-bias probe: same template as CLIP_PROMPT_TEMPLATE but without the
# "intended intensity arc" line and without the typology_long description.
# This lets us check whether emotional_expression / escalation_arc scores are
# driven by what the model *hears* or by the label metadata it was *told*.
# A large score gap (with_metadata >> no_arc on those two dimensions) is a
# strong signal that the model is reading the label, not the audio.
CLIP_PROMPT_NO_ARC_TEMPLATE = """\
Clip metadata:
- clip_id: {clip_id}
- typology: {typology}
- duration_seconds: {duration_s:.1f}
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


def apply_rate_slow_trimmed(wav: np.ndarray, rate_factor: float) -> np.ndarray:
    """Simulate over-slow TTS synthesis via resampling + trimming.

    Resamples `wav` so it plays at `rate_factor` speed (< 1.0 = slower), then
    trims back to the original length.  The output sounds like TTS rendered at
    the wrong rate: unnatural tempo, pitch shifted downward, scene cut off
    before the end.  This is a better proxy for TTS synthesis failure than
    white-noise contamination.
    """
    from scipy.signal import resample as sp_resample

    n_out = int(round(len(wav) / rate_factor))
    slowed = sp_resample(wav, n_out).astype(np.float32)
    return slowed[: len(wav)]


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
            spec = src["degradation"]
            label = f"{cid}__{spec}"
            wav_path = SPIKE_DIR / f"{label}.wav"
            wav_path.parent.mkdir(parents=True, exist_ok=True)

            # Only regenerate if the file doesn't already exist; existing files
            # are kept so a --resume run doesn't change the audio that was sent
            # to the API in a partial run.
            if not wav_path.exists():
                if spec.startswith("wn_snr_"):
                    snr_db = float(spec.replace("wn_snr_", "").replace("db", "").replace("+", ""))
                    degraded_wav = apply_white_noise(wav, snr_db, rng)
                elif spec.startswith("synth_rate_slow_"):
                    rate_factor = float(spec.replace("synth_rate_slow_", "").rstrip("x"))
                    degraded_wav = apply_rate_slow_trimmed(wav, rate_factor)
                else:
                    raise ValueError(f"Unknown degradation spec: {spec!r}")
                normalised = rms_normalize_to_match(degraded_wav, wav)
                sf.write(wav_path, normalised, 16000, subtype="PCM_16")
            else:
                # Advance the RNG to keep the sequence consistent even when skipping.
                if spec.startswith("wn_snr_"):
                    snr_db = float(spec.replace("wn_snr_", "").replace("db", "").replace("+", ""))
                    _ = rng.standard_normal(len(wav))  # consume the same RNG draw

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
    """One (clip, model, run, prompt_variant) outcome."""

    clip_label: str
    model: str
    run_idx: int
    # failure_reason is the authoritative failure classifier:
    #   "ok"               — scored successfully
    #   "content_refusal"  — safety filter blocked the response
    #   "json_parse_error" — response returned but was not valid JSON
    #   "api_error"        — network / rate-limit / auth failure
    # `refused` is kept for backwards-compat with Phase A tooling; it is True
    # for all non-"ok" reasons.  Only "content_refusal" counts against the
    # refusal gate; the others are infrastructure failures.
    failure_reason: str  # "ok" | "content_refusal" | "json_parse_error" | "api_error"
    refused: bool  # True iff failure_reason != "ok"
    raw_response: str
    parsed: dict | None
    # prompt_variant distinguishes full-metadata from the metadata-bias probe:
    #   "with_metadata"  — includes intensity arc + typology description (default)
    #   "no_arc"         — omits arc + typology_long; used to test whether
    #                      emotional_expression/escalation_arc are read from the
    #                      audio or inferred from the metadata label
    prompt_variant: str = "with_metadata"
    error: str | None = None
    latency_s: float | None = None
    usd_cost: float | None = None
    extras: dict = field(default_factory=dict)


def _degradation_note(clip: Clip) -> str:
    if not clip.degradation:
        return ""
    if clip.degradation.startswith("wn_snr_"):
        return (
            f"- degradation applied (for evaluation only): {clip.degradation} "
            f"(white noise mixed in, RMS-matched to clean)\n"
        )
    if clip.degradation.startswith("synth_rate_slow_"):
        rate = clip.degradation.replace("synth_rate_slow_", "").rstrip("x")
        return (
            f"- degradation applied (for evaluation only): {clip.degradation} "
            f"(audio resampled to {rate}× speed then trimmed to original length — "
            f"simulates over-slow TTS synthesis)\n"
        )
    return f"- degradation applied (for evaluation only): {clip.degradation}\n"


def build_clip_prompt(clip: Clip, prompt_variant: str = "with_metadata") -> str:
    """Build the per-clip user prompt.

    prompt_variant:
      "with_metadata" — full context including intensity arc (default)
      "no_arc"        — omits arc + typology description (metadata-bias probe)
    """
    deg_note = _degradation_note(clip)
    if prompt_variant == "no_arc":
        return CLIP_PROMPT_NO_ARC_TEMPLATE.format(
            clip_id=clip.label,
            typology=clip.typology,
            duration_s=clip.duration_s,
            speakers_summary=clip.speakers_summary,
            backend=clip.backend,
            degradation_note=deg_note,
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


def call_gemini(
    clip: Clip, run_idx: int, schema: dict, prompt_variant: str = "with_metadata"
) -> JudgeResult:
    """Send one (clip, run) to Gemini 2.5 Pro audio. Lazy-imports `google.genai`.

    Configures safety settings to BLOCK_NONE for the DV-content categories per
    the design doc — this is a research-use exception, never enable it in
    consumer-facing tooling.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    audio_bytes = clip.wav_path.read_bytes()
    prompt = build_clip_prompt(clip, prompt_variant=prompt_variant)

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
                temperature=TEMPERATURE,
                response_mime_type="application/json",
                response_schema=schema,
                safety_settings=safety_settings,
            ),
        )
    except Exception as e:  # noqa: BLE001
        return JudgeResult(
            clip_label=clip.label,
            model="gemini-2.5-pro",
            run_idx=run_idx,
            failure_reason="api_error",
            refused=True,
            raw_response="",
            parsed=None,
            prompt_variant=prompt_variant,
            error=repr(e),
            latency_s=time.time() - t0,
        )
    latency = time.time() - t0

    # Detect content refusal: safety block or empty response despite no exception.
    candidates = getattr(resp, "candidates", None) or []
    finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
    is_safety_block = (
        str(finish_reason).upper() in {"SAFETY", "RECITATION"} or not (resp.text or "").strip()
    )

    raw = resp.text or ""
    if is_safety_block:
        failure_reason = "content_refusal"
        parsed = None
    else:
        try:
            parsed = json.loads(raw)
            failure_reason = "ok"
        except json.JSONDecodeError:
            parsed = None
            failure_reason = "json_parse_error"

    usage = getattr(resp, "usage_metadata", None)
    in_tok = getattr(usage, "prompt_token_count", None) or 0
    out_tok = getattr(usage, "candidates_token_count", None) or 0
    cost = (clip.duration_s / 60.0) * GEMINI_AUDIO_USD_PER_MIN + (
        out_tok / 1000.0
    ) * GEMINI_OUTPUT_USD_PER_KTOK

    return JudgeResult(
        clip_label=clip.label,
        model="gemini-2.5-pro",
        run_idx=run_idx,
        failure_reason=failure_reason,
        refused=failure_reason != "ok",
        raw_response=raw,
        parsed=parsed,
        prompt_variant=prompt_variant,
        latency_s=latency,
        usd_cost=round(cost, 4),
        extras={"in_tok": in_tok, "out_tok": out_tok},
    )


def call_openai(
    clip: Clip, run_idx: int, schema: dict, prompt_variant: str = "with_metadata"
) -> JudgeResult:
    """Send one (clip, run) to gpt-4o-audio-preview. Lazy-imports `openai`."""
    import base64

    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    audio_b64 = base64.b64encode(clip.wav_path.read_bytes()).decode("ascii")
    prompt = build_clip_prompt(clip, prompt_variant=prompt_variant)

    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text"],
            temperature=TEMPERATURE,
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
            failure_reason="api_error",
            refused=True,
            raw_response="",
            parsed=None,
            prompt_variant=prompt_variant,
            error=repr(e),
            latency_s=time.time() - t0,
        )
    latency = time.time() - t0

    choice = resp.choices[0]
    finish_reason = getattr(choice, "finish_reason", "") or ""
    raw = choice.message.content or ""

    if finish_reason == "content_filter" or not raw.strip():
        failure_reason = "content_refusal"
        parsed = None
    else:
        try:
            parsed = json.loads(raw)
            failure_reason = "ok"
        except json.JSONDecodeError:
            parsed = None
            failure_reason = "json_parse_error"

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
        failure_reason=failure_reason,
        refused=failure_reason != "ok",
        raw_response=raw,
        parsed=parsed,
        prompt_variant=prompt_variant,
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
    """Compute per-model gate outcomes.

    Gates are evaluated on `prompt_variant == "with_metadata"` runs only.
    The metadata-bias probe (no_arc runs) is tabulated separately.

    overall_pass = refusal AND discrimination (the two gates testing real
    capability).  variance and shay_correlation are reported but NOT included
    in overall_pass:
      - variance: trivially PASS at TEMPERATURE=0.0 — meaningful only if the
        script is rerun with TEMPERATURE > 0 and N_RERUNS >= 4.
      - shay_correlation: n=4 corpus clips gives a directional signal but not
        a statistically significant one (need n≥7 for p<0.05).  Treat as a
        red-flag check, not a hard gate.
    """
    by_model: dict[str, list[JudgeResult]] = {}
    for r in results:
        by_model.setdefault(r.model, []).append(r)

    out: dict[str, dict] = {}
    for model, all_runs in by_model.items():
        # Gate evaluation uses only the primary (with_metadata) runs.
        runs = [r for r in all_runs if r.prompt_variant == "with_metadata"]

        # --- Refusal gate -------------------------------------------------------
        # Only "content_refusal" counts against this gate; api_error and
        # json_parse_error are infrastructure failures, not content decisions.
        scored_clips = {
            r.clip_label for r in runs if r.failure_reason not in ("content_refusal",) and r.parsed
        }
        content_refusals = [r for r in runs if r.failure_reason == "content_refusal"]
        refusal_pass = len(scored_clips) >= REFUSAL_GATE_MIN_SCORED

        # --- Mean overall_quality per clip across reruns -----------------------
        per_clip_overall: dict[str, float] = {}
        for c in clips:
            scores = [
                r.parsed["overall_quality"] for r in runs if r.clip_label == c.label and r.parsed
            ]
            if scores:
                per_clip_overall[c.label] = float(np.mean(scores))

        # --- Discrimination gate -----------------------------------------------
        # Two discrimination arms are tested independently:
        #   1. signal_corruption: corpus mean − wn_snr_-10db (Phase A comparability)
        #   2. synth_failure: corpus mean − synth_rate_slow (synthesis-defect test)
        # A model must clear at least one arm to pass.  A model that clears only
        # arm 1 can hear noise but not synthesis defects — note that in the report.
        corpus_labels = [c.label for c in clips if c.kind == "corpus"]
        corpus_scores = [per_clip_overall[lbl] for lbl in corpus_labels if lbl in per_clip_overall]
        corpus_mean = float(np.mean(corpus_scores)) if corpus_scores else float("nan")

        noise_label = next((c.label for c in clips if c.degradation == "wn_snr_-10db"), None)
        synth_label = next(
            (c.label for c in clips if (c.degradation or "").startswith("synth_rate_slow_")),
            None,
        )

        noise_sep = (
            corpus_mean - per_clip_overall[noise_label]
            if noise_label is not None and noise_label in per_clip_overall
            else float("nan")
        )
        synth_sep = (
            corpus_mean - per_clip_overall[synth_label]
            if synth_label is not None and synth_label in per_clip_overall
            else float("nan")
        )
        noise_disc_pass = (noise_sep == noise_sep) and noise_sep >= DISCRIMINATION_GATE
        synth_disc_pass = (synth_sep == synth_sep) and synth_sep >= DISCRIMINATION_GATE
        discrimination_pass = noise_disc_pass or synth_disc_pass

        # --- Variance gate (informational at TEMPERATURE=0) -------------------
        # At TEMPERATURE=0 with structured output, greedy decoding is deterministic
        # so std will be 0 across reruns.  This gate is included for completeness
        # but should not be treated as evidence of robustness unless the run used
        # TEMPERATURE > 0 with N_RERUNS >= 4.
        max_std = 0.0
        for c in clips:
            for d in DIMENSIONS:
                vals = [r.parsed[d] for r in runs if r.clip_label == c.label and r.parsed]
                if len(vals) >= 2:
                    max_std = max(max_std, float(np.std(vals, ddof=0)))
        variance_pass = max_std <= VARIANCE_GATE

        # --- Shay-correlation check (informational) ---------------------------
        # n=4 corpus clips → Spearman ρ is directional only, not statistically
        # significant.  A negative ρ is a hard red flag; ρ near zero is
        # inconclusive.  This is excluded from overall_pass — it should not gate
        # a go/no-go decision at n=4.
        xs, ys = [], []
        for c in clips:
            if c.kind == "corpus" and c.label in per_clip_overall:
                xs.append(float(c.expected_rank))
                ys.append(per_clip_overall[c.label])
        shay_rho = spearman(xs, ys) if len(xs) >= 2 else float("nan")
        shay_pass = (shay_rho == shay_rho) and shay_rho >= SHAY_CORRELATION_GATE

        # --- Metadata-bias probe (no_arc vs with_metadata) --------------------
        no_arc_runs = [r for r in all_runs if r.prompt_variant == "no_arc"]
        bias_probe: dict[str, dict] = {}
        for c in clips:
            if c.kind != "corpus":
                continue
            for dim in ("emotional_expression", "escalation_arc"):
                with_scores = [r.parsed[dim] for r in runs if r.clip_label == c.label and r.parsed]
                no_arc_scores = [
                    r.parsed[dim] for r in no_arc_runs if r.clip_label == c.label and r.parsed
                ]
                if with_scores and no_arc_scores:
                    key = f"{c.label}__{dim}"
                    bias_probe[key] = {
                        "with_metadata_mean": round(float(np.mean(with_scores)), 2),
                        "no_arc_mean": round(float(np.mean(no_arc_scores)), 2),
                        "delta": round(
                            float(np.mean(with_scores)) - float(np.mean(no_arc_scores)), 2
                        ),
                    }

        out[model] = {
            "refusal_gate": {
                "pass": bool(refusal_pass),
                "scored_clips": len(scored_clips),
                "content_refusals": len(content_refusals),
                "min_required": REFUSAL_GATE_MIN_SCORED,
            },
            "discrimination_gate": {
                "pass": bool(discrimination_pass),
                "corpus_mean": round(corpus_mean, 3) if corpus_mean == corpus_mean else None,
                "noise_corruption": {
                    "label": noise_label,
                    "score": round(per_clip_overall[noise_label], 3)
                    if noise_label and noise_label in per_clip_overall
                    else None,
                    "separation": round(noise_sep, 3) if noise_sep == noise_sep else None,
                    "pass": bool(noise_disc_pass),
                },
                "synth_failure": {
                    "label": synth_label,
                    "score": round(per_clip_overall[synth_label], 3)
                    if synth_label and synth_label in per_clip_overall
                    else None,
                    "separation": round(synth_sep, 3) if synth_sep == synth_sep else None,
                    "pass": bool(synth_disc_pass),
                },
                "threshold": DISCRIMINATION_GATE,
            },
            "variance_gate": {
                "pass": bool(variance_pass),
                "max_per_dim_std": round(max_std, 3),
                "threshold": VARIANCE_GATE,
                "note": (
                    "TEMPERATURE=0.0 makes this gate trivially PASS via greedy determinism. "
                    "Re-run with TEMPERATURE=0.1 and N_RERUNS=4 for a meaningful estimate."
                    if TEMPERATURE == 0.0
                    else f"TEMPERATURE={TEMPERATURE}"
                ),
            },
            "shay_correlation_check": {
                "informational_only": True,
                "pass": bool(shay_pass),
                "spearman_rho": (None if shay_rho != shay_rho else round(shay_rho, 3)),
                "n_corpus_clips": len(xs),
                "threshold": SHAY_CORRELATION_GATE,
                "note": "n=4 is not statistically significant; treat as directional red-flag check only.",
            },
            "metadata_bias_probe": bias_probe,
            # overall_pass = refusal AND discrimination only.
            # variance and shay_correlation are advisory; see gate notes above.
            "overall_pass": bool(refusal_pass and discrimination_pass),
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
    md.append("| Model | Refusal | Disc (noise) | Disc (synth) | Variance† | Shay ρ† | Overall |")
    md.append("|---|---|---|---|---|---|---|")
    for model, g in payload["gates"].items():
        ref = f"{g['refusal_gate']['scored_clips']}/{g['refusal_gate']['min_required']}"

        def _disc_cell(arm: dict) -> str:
            sep = arm.get("separation")
            if sep is None:
                return "n/a"
            return f"{sep:+.2f} {'✅' if arm['pass'] else '❌'}"

        disc_noise = _disc_cell(g["discrimination_gate"]["noise_corruption"])
        disc_synth = _disc_cell(g["discrimination_gate"]["synth_failure"])
        var = f"{g['variance_gate']['max_per_dim_std']:.2f}"
        rho_val = g["shay_correlation_check"]["spearman_rho"]
        rho = "n/a" if rho_val is None else f"{rho_val:+.2f}"
        md.append(
            f"| `{model}` | {ref} {'✅' if g['refusal_gate']['pass'] else '❌'} | "
            f"{disc_noise} | {disc_synth} | "
            f"{var} {'✅' if g['variance_gate']['pass'] else '❌'} | "
            f"{rho} {'✅' if g['shay_correlation_check']['pass'] else '❌'} | "
            f"{'PASS' if g['overall_pass'] else 'FAIL'} |"
        )
    md.append("")
    md.append(
        f"Thresholds — refusal: ≥ {REFUSAL_GATE_MIN_SCORED}/6 scored · "
        f"discrimination: separation ≥ {DISCRIMINATION_GATE} · "
        f"variance: per-dim std ≤ {VARIANCE_GATE} · Shay ρ: ≥ {SHAY_CORRELATION_GATE}  "
    )
    md.append(
        "† _variance_ and _Shay ρ_ are advisory — not included in overall_pass. "
        "See gate notes in `evaluate_gates()`."
    )
    md.append("")

    # Metadata-bias probe table (only emitted when no_arc runs are present).
    any_bias = any(bool(g.get("metadata_bias_probe")) for g in payload["gates"].values())
    if any_bias:
        md.append("## Metadata-bias probe (no_arc vs with_metadata)")
        md.append("")
        md.append(
            "Delta = with_metadata_mean − no_arc_mean on `emotional_expression` and "
            "`escalation_arc`. Large positive delta → model is reading the label, not the audio."
        )
        md.append("")
        models_with_bias = [m for m, g in payload["gates"].items() if g.get("metadata_bias_probe")]
        md.append("| Clip × Dim | " + " | ".join(f"`{m}` Δ" for m in models_with_bias) + " |")
        md.append("|---" + "|---" * len(models_with_bias) + "|")
        all_keys = sorted(
            {k for g in payload["gates"].values() for k in g.get("metadata_bias_probe", {})}
        )
        for key in all_keys:
            cells = [f"`{key}`"]
            for m in models_with_bias:
                probe = payload["gates"][m].get("metadata_bias_probe", {}).get(key)
                cells.append("—" if probe is None else f"{probe['delta']:+.2f}")
            md.append("| " + " | ".join(cells) + " |")
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
    parser.add_argument(
        "--probe-metadata-bias",
        action="store_true",
        help=(
            "Run an additional no-arc prompt variant on corpus clips (run_idx=0 only) "
            "to measure whether emotional_expression / escalation_arc scores are driven "
            "by the intensity-arc metadata label or by what the model actually hears. "
            "Adds ~4 calls per model (~$0.10–0.40 extra)."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a partial run. Loads completed (clip_label, model, run_idx, "
            "prompt_variant) tuples from results_partial.jsonl and skips them."
        ),
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

    # --- Resume: load prior partial results ----------------------------------
    completed: set[tuple[str, str, int, str]] = set()
    results: list[JudgeResult] = []
    cumulative_usd = 0.0
    if args.resume and RESULTS_PARTIAL_PATH.exists():
        for line in RESULTS_PARTIAL_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            r = JudgeResult(**{k: v for k, v in d.items() if k in JudgeResult.__dataclass_fields__})
            results.append(r)
            cumulative_usd += r.usd_cost or 0.0
            completed.add((r.clip_label, r.model, r.run_idx, r.prompt_variant))
        print(
            f"resumed: loaded {len(results)} prior results (${cumulative_usd:.4f} spent)",
            flush=True,
        )

    # --- Build call plan -----------------------------------------------------
    # Each entry: (clip, model_name, run_idx, prompt_variant)
    call_plan: list[tuple[Clip, str, int, str]] = []
    model_name_map = {"gemini": "gemini-2.5-pro", "openai": "gpt-4o-audio-preview"}
    for model in args.models:
        for clip in clips:
            for run_idx in range(N_RERUNS_PER_CLIP):
                call_plan.append((clip, model, run_idx, "with_metadata"))
            if args.probe_metadata_bias and clip.kind == "corpus":
                call_plan.append((clip, model, 0, "no_arc"))

    # Filter out already-completed calls.
    pending = [
        (clip, model, run_idx, variant)
        for clip, model, run_idx, variant in call_plan
        if (clip.label, model_name_map[model], run_idx, variant) not in completed
    ]
    print(f"call plan: {len(call_plan)} total, {len(pending)} pending", flush=True)

    # --- Run -----------------------------------------------------------------
    partial_f = RESULTS_PARTIAL_PATH.open("a", encoding="utf-8") if not args.dry_run else None
    try:
        current_model = None
        for clip, model, run_idx, prompt_variant in pending:
            if model != current_model:
                current_model = model
                print(f"\n=== Model: {model} ===", flush=True)
            if cumulative_usd >= budget_cap:
                print(
                    f"ABORT: cumulative ${cumulative_usd:.2f} ≥ budget ${budget_cap:.2f}. "
                    "Partial results written to results_partial.jsonl — rerun with --resume.",
                    flush=True,
                )
                break
            if model == "gemini":
                r = call_gemini(clip, run_idx, schema, prompt_variant=prompt_variant)
            else:
                r = call_openai(clip, run_idx, schema, prompt_variant=prompt_variant)
            results.append(r)
            if partial_f is not None:
                partial_f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
                partial_f.flush()
            cumulative_usd += r.usd_cost or 0.0
            tag = r.failure_reason if r.refused else "ok"
            overall = (r.parsed or {}).get("overall_quality", "—")
            variant_tag = f" [{prompt_variant}]" if prompt_variant != "with_metadata" else ""
            print(
                f"  {clip.label} run{run_idx}{variant_tag}  {tag}  overall={overall}  "
                f"latency={r.latency_s:.1f}s  cost=${(r.usd_cost or 0.0):.4f}",
                flush=True,
            )
    finally:
        if partial_f is not None:
            partial_f.close()

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
        ref = g["refusal_gate"]
        print(
            f"    refusal:            {ref['scored_clips']}/{ref['min_required']} scored  "
            f"(content refusals: {ref['content_refusals']})  "
            f"{'PASS' if ref['pass'] else 'FAIL'}",
            flush=True,
        )
        d = g["discrimination_gate"]
        noise = d["noise_corruption"]
        synth = d["synth_failure"]
        noise_sep_s = f"{noise['separation']:+.2f}" if noise["separation"] is not None else "n/a"
        synth_sep_s = f"{synth['separation']:+.2f}" if synth["separation"] is not None else "n/a"
        print(
            f"    discrimination:     noise {noise_sep_s}  synth {synth_sep_s}  "
            f"(threshold ≥ {d['threshold']})  {'PASS' if d['pass'] else 'FAIL'}",
            flush=True,
        )
        v = g["variance_gate"]
        print(
            f"    variance (advisory):max per-dim std {v['max_per_dim_std']:.2f} (≤ {v['threshold']})  "
            f"{'PASS' if v['pass'] else 'FAIL'}  [{v['note'][:50]}…]",
            flush=True,
        )
        s = g["shay_correlation_check"]
        print(
            f"    Shay ρ (advisory):  {s['spearman_rho']} (≥ {s['threshold']}, n={s['n_corpus_clips']})  "
            f"{'PASS' if s['pass'] else 'FAIL'}  [informational only]",
            flush=True,
        )
    print(f"  cumulative spend: ${cumulative_usd:.4f} / ${budget_cap:.2f}", flush=True)


if __name__ == "__main__":
    main()
