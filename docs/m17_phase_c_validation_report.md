# M17 Phase C Validation Report — Multimodal LLM Judge on Hebrew Clips

> **Status:** spike stub. Filled-in numbers land after the script runs against
> real API keys. The gate tables below are skeletons — final values come from
> `state/spikes/m17_phase_c/results.json` and the auto-generated
> `state/spikes/m17_phase_c/report_auto.md`.

This report validates the Phase C acceptance criteria from the M17 design
doc (`docs/automated_eval_design.md` §E5 — Multimodal LLM Judge) before the
E5 evaluator module skeleton lands. Mirrors the Phase A spike protocol that
produced `docs/m17_phase_a_validation_report.md`.

Generated: TBD. Raw data: `state/spikes/m17_phase_c/results.json` (gitignored).
Auto-tables: `state/spikes/m17_phase_c/report_auto.md`. Cached audio (for
degraded variants only) under `state/spikes/m17_phase_c/*.wav`.

## TL;DR

| Evaluator (model) | Refusal | Discrimination | Reproducibility | Shay correlation | Phase C |
|---|---|---|---|---|---|
| **E5a — `gemini-2.5-pro`** | TBD | TBD | TBD | TBD | TBD |
| **E5b — `gpt-4o-audio-preview`** | TBD | TBD | TBD | TBD | TBD |

**Phase C status: TBD.** Per the design doc's per-evaluator gate semantics, each
model is graded independently; one model passing is sufficient to advance E5
to MVP. A model that clears all four of its gates is the recommended primary
backend; the runner-up becomes the refusal-fallback.

## Reproduce

```bash
uv pip install --python .venv/bin/python \
    google-genai openai pydantic soundfile numpy scipy

export GEMINI_API_KEY=...
export OPENAI_API_KEY=...
export SYNTHBANSHEE_LLM_SPIKE_BUDGET_USD=5.0   # hard cap

# Optional dry run — prints the prompt + JSON schema, no API calls, no spend.
.venv/bin/python scripts/m17_phase_c_validation.py --dry-run

# Full run.
.venv/bin/python scripts/m17_phase_c_validation.py
```

The script prepares 6 clip records (4 corpus typology spans + 2 degraded
variants of `sp_sv_a_0001_00`), sends each to each model twice, parses
structured JSON responses against a Pydantic schema, and writes gate
outcomes to `results.json` + an auto-generated markdown summary.

## Clip set

Source dir defaults to `data/m2a_wettest/agg_m_30-45_001/` (8-clip M2a-wettest
batch — same source set as the Phase A spike). Override via
`SYNTHBANSHEE_LLM_SPIKE_CLIP_DIR`.

| Label | Source clip | Kind | Typology | Notes |
|-------|-------------|------|----------|-------|
| `sp_sv_a_0001_00` | sp_sv_a_0001_00 | corpus | SV | The canonical SV scene reviewed by Shay + Gemini + Claude + ChatGPT in `docs/debug_run_1/llm_feedbacks.md` |
| `sp_it_a_0001_00` | sp_it_a_0001_00 | corpus | IT | The IT scene whose I3–I5 distress absence drove issue #97 |
| `sp_neg_a_0001_00` | sp_neg_a_0001_00 | corpus | NEG | Hard-negative class — intense but `has_violence: false` |
| `sp_neu_a_0001_00` | sp_neu_a_0001_00 | corpus | NEU | Neutral baseline; expected highest perceived quality |
| `sp_sv_a_0001_00__wn_snr_+10db` | sp_sv_a_0001_00 + mild white noise | degraded | SV | RMS-matched to clean; mid-anchor |
| `sp_sv_a_0001_00__wn_snr_-10db` | sp_sv_a_0001_00 + severe white noise | degraded | SV | RMS-matched to clean; the discrimination gate's negative anchor |

Shay-correlation gate uses `expected_quality_rank` per clip (encoded in
`CLIP_SOURCES`), grounded in:
- The May-3 listening test memo (M12 breathiness failed gate; systemic
  TTS naturalness issues).
- The May-6 listening test (#92, aggregated naturalness backlog).
- The 2026-04-14 multi-LLM review of `sp_it_a_0001` (`docs/debug_run_1/llm_feedbacks.md`).
- Issue #97 finding that VIC at I3–I5 doesn't sound distressed under
  rate+pitch alone.

If/when new listening tests update the priors, update `CLIP_SOURCES` ranks
and re-run.

## Gate definitions

All four gates are computed **per model**. A model is "Phase C PASS" iff it
clears all four.

| Gate | Criterion | Why |
|------|-----------|-----|
| **Refusal** | ≥ 5/6 clips scored successfully under the DV-research framing | A model that refuses on DV content under research framing can't be a production E5 backend. Mitigation in design doc §E5: BLOCK_NONE on Gemini, multi-model fallback. |
| **Discrimination** | `mean(corpus overall_quality) − mean(–10dB severe-degraded overall_quality) ≥ 0.5` | Same bar as Phase A E2/UTMOS. If the model can't tell clean from heavily noised audio, it can't tell good TTS from broken TTS. |
| **Reproducibility** | Per-dimension std across 2 reruns of the same clip ≤ 0.5 | Without reproducibility, score deltas can't be attributed to pipeline changes — regression detection breaks. |
| **Shay correlation** | Spearman ρ ≥ 0.3 between model `overall_quality` and Shay's encoded expected rank on the 4 corpus clips | n=4 is small; ρ is directional, not statistically significant. A negative ρ here is a hard fail signal (model disagrees with the only human listener available). |

## Per-model results (TBD)

### `gemini-2.5-pro`

| Clip | Run 0 overall | Run 1 overall | Mean | Pronunciation | Prosody | Emotion | Speaker diff. | Dialogue | Escalation | Coherence | Artifacts | Refusal? |
|------|--------------:|--------------:|----:|--------------:|--------:|--------:|--------------:|--------:|-----------:|----------:|:---------:|:--------:|
| sp_sv_a_0001_00 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| sp_it_a_0001_00 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| sp_neg_a_0001_00 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| sp_neu_a_0001_00 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| sp_sv_a_0001_00__wn_snr_+10db | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| sp_sv_a_0001_00__wn_snr_-10db | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### `gpt-4o-audio-preview`

Same table shape — fill from `results.json`.

## Failure-mode notes (TBD post-run)

- **Refusal posture under DV-research framing:** TBD — record any refusals
  here, with the verbatim refusal text and the clip that triggered it. If
  Gemini's BLOCK_NONE setting still refuses, the design doc's content-
  sensitivity mitigation list (alternative framings, GPT-4o fallback)
  becomes the recommended path.
- **Score collapse / no-variance:** TBD — does any model give the same score
  on every dimension across all 6 clips? If so the model isn't actually
  listening to the audio.
- **Length-driven scoring:** TBD — does the model rate longer clips
  systematically higher/lower regardless of content?
- **Anchor recommendation for MVP:** TBD — given the discrimination
  separation and the variance pattern, recommend whether the MVP needs
  in-prompt calibration anchors (the Phase 2b anchor protocol from the
  near-term plan) or can rely on absolute scoring alone.

## Recommendation (TBD)

| Outcome | Next action |
|---------|-------------|
| Both models PASS all gates | Ship E5 MVP behind Gemini primary, GPT-4o fallback. Open follow-up issue for anchor-set protocol + regression CI workflow. |
| One model PASSes, the other fails on a single gate | Ship MVP with the passing model as primary; treat the other as deferred (open a follow-up issue with the specific failure mode). |
| Both models PASS refusal + reproducibility but FAIL discrimination | E5 cannot gate. Use it only as informational scoring. Re-spike with explicit in-prompt anchors (Phase 2b protocol surfaced earlier in the implementation path). |
| Either model fails REFUSAL | Try alternative framings before declaring failure; record framing experiments in this report. |
| Both models FAIL discrimination | Phase C is a NO-GO. Pivot the eval roadmap away from LLM listening; either commit to paid linguistic raters / crowd evaluation, or wait for the next-generation multimodal audio models. |

## Limitations

- **n=4 corpus clips for the Shay-correlation gate.** Spearman ρ at n=4 is
  directional, not statistically significant. A future MVP-phase
  calibration step should use ≥ 20 clips spanning the full corpus once
  more listening-test ground truth is available.
- **n=2 reruns for the reproducibility gate.** A single std estimate from
  2 samples is high-variance; the gate is meant as a sanity check, not a
  precision measurement. If reproducibility looks marginal (std ~ 0.5),
  rerun a subset with more samples before drawing conclusions.
- **One degradation type.** The severe negative anchor is white noise at
  −10 dB SNR. Real TTS failure modes (over-smoothed prosody, wrong gender
  forms, robotic timbre) are perceptually distinct from white-noise
  contamination; a model that discriminates white-noise but not synthesis
  defects would pass this gate while still being useless for our actual
  failure modes. The MVP-phase anchor set must include synthesis-failure
  anchors, not just signal-domain ones.
- **English-trained models on Hebrew audio.** Both Gemini and GPT-4o have
  unknown Hebrew-audio comprehension depth at the time of writing
  (January 2026 knowledge cutoff). If discrimination passes for both, that's
  reassuring; if only one passes, the Hebrew-specific failure mode is
  worth documenting.
- **Prompt v1.** Iterating the prompt is cheap — `PROMPT_VERSION` is
  bumped in the script and the version is recorded in every result row,
  so an A/B between prompt v1 and v2 is straightforward without losing
  the v1 baseline.

## Cost report (TBD)

| Model | Audio min sent | Estimated $ | Actual $ |
|-------|---------------:|------------:|---------:|
| `gemini-2.5-pro` | TBD | TBD | TBD |
| `gpt-4o-audio-preview` | TBD | TBD | TBD |
| **Total** | — | TBD | TBD |

Hard cap was `SYNTHBANSHEE_LLM_SPIKE_BUDGET_USD = $5.00`. Actual spend
within budget: TBD.
