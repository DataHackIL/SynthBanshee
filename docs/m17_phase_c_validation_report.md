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
Auto-tables: `state/spikes/m17_phase_c/report_auto.md`. Degraded audio cached
under `state/spikes/m17_phase_c/*.wav`.

## TL;DR

<!-- report_auto.md §"Gate outcomes" table goes here after the run. -->

| Evaluator (model) | Refusal | Disc (noise) | Disc (synth) | Phase C |
|---|---|---|---|---|
| **E5a — `gemini-2.5-pro`** | TBD | TBD | TBD | TBD |
| **E5b — `gpt-4o-audio-preview`** | TBD | TBD | TBD | TBD |

**Phase C status: TBD.** `overall_pass = refusal AND discrimination`. Variance
and Shay-correlation gates are advisory (see Gate definitions below). One model
passing is sufficient to advance E5 to MVP.

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

# With metadata-bias probe (adds ~4 no-arc calls per model, ~$0.10–0.40 extra).
.venv/bin/python scripts/m17_phase_c_validation.py --probe-metadata-bias

# Resume a partial run after a failure or budget-cap abort.
.venv/bin/python scripts/m17_phase_c_validation.py --resume
```

The script prepares 6 clip records (4 corpus typology spans + 2 degraded
variants of `sp_sv_a_0001_00`), sends each to each model twice under the
`with_metadata` prompt variant, parses structured JSON responses, writes partial
results to `results_partial.jsonl` after each call, and writes final gate
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
| `sp_sv_a_0001_00__synth_rate_slow_0.7x` | sp_sv_a_0001_00 resampled to 0.7× speed, trimmed | degraded | SV | **Synthesis-failure anchor.** Simulates over-slow TTS: unnatural tempo, pitch-shifted down, scene cut off mid-escalation. Tests whether the model can hear synthesis defects, not just noise. |
| `sp_sv_a_0001_00__wn_snr_-10db` | sp_sv_a_0001_00 + severe white noise | degraded | SV | **Signal-corruption anchor.** RMS-matched to clean. Kept for comparability with Phase A E2/UTMOS. If a model passes this but fails the synth anchor, it can detect noise but not synthesis defects. |

Shay-correlation check uses `expected_quality_rank` per clip (encoded in
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

Gates are computed **per model**. `overall_pass = refusal AND discrimination`.

| Gate | Criterion | Included in `overall_pass`? | Why |
|------|-----------|:---:|-----|
| **Refusal** | ≥ 5/6 clips score successfully under the DV-research framing; only `content_refusal` failures count against this gate (`api_error` and `json_parse_error` are retryable infrastructure failures) | ✅ | A model that refuses on DV content under research framing can't be a production E5 backend. |
| **Discrimination** | At least one anchor arm clears: `mean(corpus) − mean(severe degraded) ≥ 0.5`. Two arms: (1) noise-corruption (`wn_snr_-10db`), (2) synthesis-failure (`synth_rate_slow_0.7x`). A model that passes only the noise arm cannot detect synthesis defects. | ✅ | The actual failure modes we need to catch are synthesis defects, not SNR. |
| **Variance** _(advisory)_ | Per-dimension std across 2 reruns ≤ 0.5 | ❌ | At `TEMPERATURE=0.0`, greedy decoding is deterministic → std=0 trivially. Re-run with `TEMPERATURE=0.1` and `N_RERUNS=4` for a meaningful estimate. |
| **Shay correlation** _(advisory)_ | Spearman ρ ≥ 0.3 between model `overall_quality` and Shay's encoded rank on the 4 corpus clips | ❌ | n=4 is not statistically significant (need n≥7 for p<0.05). Treat as a directional red-flag check only — a strongly negative ρ is a hard warning signal. |

## Per-clip mean overall_quality (TBD)

<!-- report_auto.md §"Per-clip mean overall_quality" table goes here. -->

## Metadata-bias probe (TBD)

Run with `--probe-metadata-bias`. The probe sends the 4 corpus clips under a
`no_arc` prompt variant that omits the `intended intensity arc` and
`typology_long` metadata. The delta on `emotional_expression` and
`escalation_arc` between `with_metadata` and `no_arc` measures how much of the
score is read from the label vs. from the audio.

<!-- report_auto.md §"Metadata-bias probe" table goes here. -->

A delta > 0.5 on either dimension is a warning that the model is inflating
those scores based on the label. If the delta is large, the discrimination gate
result on those dimensions should be treated as unreliable.

## Failure-mode notes (TBD post-run)

- **Refusal posture under DV-research framing:** TBD — record any
  `content_refusal` failures here, with the verbatim refusal text and the clip
  that triggered it. Distinguish from `api_error` / `json_parse_error`. If
  Gemini's BLOCK_NONE setting still produces content refusals, the design doc's
  mitigation list (alternative framings, GPT-4o fallback) becomes the
  recommended path.
- **Noise-only discriminator:** TBD — if a model passes the noise arm but fails
  the synth arm, record it here. This model can detect SNR changes but not
  synthesis quality — it cannot serve as E5 backend.
- **Score collapse / no-variance:** TBD — does any model give the same score
  on every dimension across all 6 clips? If so the model isn't actually
  listening to the audio.
- **Metadata-bias:** TBD — compare `emotional_expression` and `escalation_arc`
  deltas from the metadata-bias probe. If delta > 0.5, note which model and
  which clips are affected.
- **Anchor recommendation for MVP:** TBD — given the discrimination separation
  pattern (noise vs synth), recommend whether the MVP needs in-prompt
  calibration anchors (Phase 2b anchor protocol) or can rely on absolute
  scoring alone.

## Recommendation (TBD)

| Outcome | Next action |
|---------|-------------|
| Both models pass refusal + **both** discrimination arms | Ship E5 MVP behind Gemini primary, GPT-4o fallback. Open follow-up issue for anchor-set protocol + regression CI workflow. |
| Both models pass refusal + noise arm only | E5 can detect gross signal corruption but not synthesis defects. Defer MVP; re-spike with synthesis-failure anchors as calibration points in the prompt. |
| One model passes refusal + synthesis discrimination arm | Ship MVP with that model as primary. Document the other model's failure mode in a follow-up issue. |
| Either model fails REFUSAL on content_refusal | Try alternative framings before declaring failure; record framing experiments in this report. |
| Both models FAIL synthesis discrimination | Phase C is a NO-GO for synthesis-quality gating. Pivot the eval roadmap: either commit to paid linguistic raters / crowd evaluation, or wait for next-generation multimodal audio models. |

## Limitations

- **n=4 corpus clips for the Shay-correlation check.** Spearman ρ at n=4 is
  directional, not statistically significant. A future MVP-phase calibration
  step should use ≥ 20 clips spanning the full corpus once more listening-test
  ground truth is available.
- **n=2 reruns at TEMPERATURE=0.** The variance gate trivially passes under
  greedy decoding. To get a real reproducibility estimate, re-run with
  `TEMPERATURE=0.1` and `N_RERUNS_PER_CLIP=4`.
- **Synthesis-failure proxy via resampling.** The `synth_rate_slow_0.7x`
  degradation changes both tempo and pitch simultaneously (a naive resampling
  artifact), which is not exactly how TTS synthesis defects manifest. It is a
  better proxy than white noise but still not identical to over-smoothed
  prosody, wrong gender forms, or robotic timbre. The MVP-phase anchor set
  should include real defective TTS renders, not just signal-domain proxies.
- **English-trained models on Hebrew audio.** Both Gemini and GPT-4o have
  unknown Hebrew-audio comprehension depth. If discrimination passes for both,
  that's reassuring; if only one passes, document the Hebrew-specific failure
  mode.
- **Prompt v1.** Iterating the prompt is cheap — `PROMPT_VERSION` is bumped in
  the script and recorded in every result row, so an A/B between prompt
  versions is straightforward without losing the v1 baseline.

## Cost report (TBD)

<!-- report_auto.md §"Spend" block goes here. -->

| Model | Audio min sent | Estimated $ | Actual $ |
|-------|---------------:|------------:|---------:|
| `gemini-2.5-pro` | TBD | TBD | TBD |
| `gpt-4o-audio-preview` | TBD | TBD | TBD |
| **Total** | — | TBD | TBD |

Hard cap was `SYNTHBANSHEE_LLM_SPIKE_BUDGET_USD = $5.00`. Actual spend
within budget: TBD.
