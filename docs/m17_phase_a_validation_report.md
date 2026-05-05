# M17 Phase A Validation Report — Whisper + UTMOS on Hebrew Clips

This report validates the Phase A acceptance criteria from the M17 design doc
(`docs/automated_eval_design.md` lines 418–422) before the E1/E2 module skeletons land.

Generated: 2026-05-05. Raw data: `state/spikes/m17_phase_a/results.json`. Auto-tables: `state/spikes/m17_phase_a/report_auto.md`. Listening samples: `state/spikes/m17_phase_a/<clip_id>__<degradation>.wav`.

## TL;DR

| Evaluator | Gate | Outcome |
|---|---|---|
| **E1 — ASR** | median WER < 0.40 | **PASS** by ~10× margin on all three Whisper variants |
| **E1 — int8 CT2 in CI** | within ε = 0.02 WER of GPU baseline | **PASS** (Δ = 0.008) |
| **E2 — UTMOS** | clean − degraded mean ≥ 0.5 | **FAIL** (max separation 0.21 across 5 degradations) |

- **E1 — GO.** Adopt `ivrit-ai/whisper-large-v3` (HF, primary). `ivrit-ai/whisper-large-v3-ct2` with `compute_type="int8"` is viable for CI. Add a trigram-repetition hallucination guard.
- **E2 — NO-GO.** UTMOS shows the right direction (degraded < clean) on this set but separation is well below 0.5, and white-noise severity is *inverted* (more noise scores higher). Re-spike at the turn level before any E2 implementation.
- **Design doc fix landed** in this PR: `ivrit-ai/whisper-v2-d3-e3` → `ivrit-ai/whisper-large-v3` at three locations.
- **Carry-over finding:** Azure TTS SSML rejected the LLM-generated script for one of the new IT scenes (`sp_it_a_0003`). Replaced with `sp_neg_a_0003`. PR #67/#71 didn't fully cover this case; tracking as a separate follow-up.

## Reproduce

```bash
uv pip install --python .venv/bin/python torch torchaudio transformers \
    jiwer speechmos accelerate faster-whisper
.venv/bin/python scripts/m17_phase_a_validation.py
```

Inputs: 10 clips at `data/m2a_wettest/agg_m_30-45_001/` — see Manifest below for SHA-256s. Device: MPS (Apple Silicon M4 Max). All decoders are greedy (`num_beams=1`, `do_sample=False`, `temperature=0`); `numpy` and `torch` seeded to 42.

## Clip-set manifest

| Clip | Duration (s) | Ref words | SHA-256 (8) |
|---|---|---|---|
| sp_it_a_0001_00 | 155.9 | 233 | `792d622b` |
| sp_it_a_0002_00 | 146.7 | 291 | `064e9f29` |
| sp_neg_a_0001_00 | 128.5 | 240 | `f403b7db` |
| sp_neg_a_0002_00 | 114.2 | 177 | `bdbf4487` |
| sp_neg_a_0003_00 | 257.5 | 257 | `8988c9ec` (new) |
| sp_neu_a_0001_00 | 121.0 | 189 | `e4782fe6` |
| sp_neu_a_0002_00 | 133.1 | 221 | `9edae0a1` |
| sp_neu_a_0003_00 | 190.4 | 176 | `7a616df5` (new) |
| sp_sv_a_0001_00 | 122.5 | 188 | `682049f7` |
| sp_sv_a_0002_00 | 99.4 | 164 | `44b1e4e4` |

Distribution: 2 IT / 3 NEG / 3 NEU / 2 SV. Spec asked for 10; we have 10.

## Gate Outcomes

| Gate | Criterion | Result | Decision |
|------|-----------|--------|----------|
| ASR — `openai/whisper-large-v3` (HF, MPS) | median WER < 0.40 | **0.064** | ✅ PASS |
| ASR — `ivrit-ai/whisper-large-v3` (HF, MPS) | median WER < 0.40 | **0.036** | ✅ PASS |
| ASR — `ivrit-ai/whisper-large-v3-ct2` (faster-whisper, int8 CPU) | median WER < 0.40 | **0.044** | ✅ PASS |
| CI viability — int8 vs MPS baseline | Δ median WER < ε (0.02) | **0.008** | ✅ PASS |
| UTMOS — primary degradation (white noise −10 dB SNR) | clean − deg mean ≥ 0.50 | separation **+0.135** vs same-5 clean mean | ❌ FAIL |
| UTMOS — best degradation in sweep | any in sweep ≥ 0.50 | max **+0.207** (white noise +10 dB) | ❌ FAIL |
| UTMOS — white-noise monotonic in severity | more noise → lower UTMOS | **inverted** | ❌ FAIL |

**Preferred Whisper run (lowest median WER):** `ivrit-ai/whisper-large-v3` (HF, MPS).

## ASR — Per-Clip WER

| Clip | openai HF | ivrit-ai HF | ivrit-ai CT2 int8 |
|---|---|---|---|
| sp_it_a_0001_00 | 0.056 | 0.039 | 0.047 |
| sp_it_a_0002_00 | 0.065 | **0.127** | 0.062 |
| sp_neg_a_0001_00 | 0.054 | 0.033 | 0.033 |
| sp_neg_a_0002_00 | 0.040 | 0.017 | 0.028 |
| sp_neg_a_0003_00 | 0.535 | 0.016 | 0.016 |
| sp_neu_a_0001_00 | 0.063 | 0.048 | 0.053 |
| sp_neu_a_0002_00 | 0.045 | 0.041 | 0.041 |
| sp_neu_a_0003_00 | 0.341 | 0.051 | 0.057 |
| sp_sv_a_0001_00 | 0.053 | 0.032 | 0.053 |
| sp_sv_a_0002_00 | 0.030 | 0.012 | 0.018 |
| **median** | 0.064 | 0.036 | 0.044 |
| **mean** | 0.096 | 0.042 | 0.041 |

The bolded `0.127` on the ivrit-ai HF run is the known hallucination outlier (see Findings §3). The `0.535`/`0.341` on the new clips through `openai/whisper-large-v3` HF appear to be quality regressions on the Hebrew distribution covered by `sp_neg_a_0003` and `sp_neu_a_0003`; ivrit-ai handles both cleanly. This reinforces the recommendation to prefer the ivrit-ai variant.

## CI viability — int8 CT2 vs MPS baseline

For each clip, |WER(ivrit-ai HF MPS) − WER(ivrit-ai CT2 int8 CPU)|:

| Clip | HF MPS | CT2 int8 | Δ |
|---|---|---|---|
| sp_it_a_0001_00 | 0.039 | 0.047 | +0.008 |
| sp_it_a_0002_00 | **0.127** | 0.062 | −0.065 |
| sp_neg_a_0001_00 | 0.033 | 0.033 | 0.000 |
| sp_neg_a_0002_00 | 0.017 | 0.028 | +0.011 |
| sp_neg_a_0003_00 | 0.016 | 0.016 | 0.000 |
| sp_neu_a_0001_00 | 0.048 | 0.053 | +0.005 |
| sp_neu_a_0002_00 | 0.041 | 0.041 | 0.000 |
| sp_neu_a_0003_00 | 0.051 | 0.057 | +0.006 |
| sp_sv_a_0001_00 | 0.032 | 0.053 | +0.021 |
| sp_sv_a_0002_00 | 0.012 | 0.018 | +0.006 |

Median Δ: **+0.008** — well under the ε = 0.02 tolerance the prompt set. int8 CT2 is viable for CI without revising the design doc's "Whisper small in CI" plan to something smaller. Notable: int8 CT2 actually *avoids* the hallucination loop on `sp_it_a_0002_00` that HF MPS hits (Δ = −0.065 in int8's favour). Either the CT2 conversion or the faster-whisper decoding logic is more resistant to the loop.

## Hallucination detectors — empirical validation

Run on the ivrit-ai HF results (the only run that produces an outlier):

| Clip | WER | length_ratio | hyp 3-gram repeat | trigram_repeat_ratio | RTL marks |
|---|---|---|---|---|---|
| sp_it_a_0001_00 | 0.039 | 1.004 | 2 | 1.00 | 0 |
| **sp_it_a_0002_00** | **0.127** | **1.048** | 2 | **2.00** | **19** |
| sp_neg_a_0001_00 | 0.033 | 1.008 | 2 | 1.00 | 8 |
| sp_neg_a_0002_00 | 0.017 | 1.000 | 1 | 1.00 | 0 |
| sp_neg_a_0003_00 | 0.016 | 1.004 | 2 | 1.00 | 0 |
| sp_neu_a_0001_00 | 0.048 | 1.005 | 1 | 1.00 | 15 |
| sp_neu_a_0002_00 | 0.041 | 1.005 | 1 | 1.00 | 0 |
| sp_neu_a_0003_00 | 0.051 | 1.011 | 1 | 1.00 | 0 |
| sp_sv_a_0001_00 | 0.032 | 0.995 | 2 | 1.00 | 0 |
| sp_sv_a_0002_00 | 0.012 | 1.000 | 2 | 1.00 | 0 |

Empirical detector thresholds (catch the outlier, ignore the rest):

| Detector | Threshold | Catches outlier? | False positives on this set |
|---|---|---|---|
| `length_ratio > 1.04` | 1.04 | yes (1.048) | 0 of 9 (max non-outlier 1.011) |
| `trigram_repeat_ratio > 1.5` | 1.5 | yes (2.00) | 0 of 9 (all non-outliers 1.00) |
| `rtl_mark_count > 16` | 16 | yes (19) | 0 of 9 (others 0–15) |

`trigram_repeat_ratio` is the cleanest signal — non-outliers all sit at exactly 1.0 — so it should be the primary E1 hallucination guard. `length_ratio` is a useful redundant check; the proposed-but-wrong `1.05` from PR #77 review now becomes `1.04`. RTL-mark count is too noisy alone (false positives at 8 and 15 on clean clips); useful only as a tiebreaker.

## UTMOS — clean baseline + degradation sweep

Clean per clip:

| Clip | UTMOS | Notes |
|---|---|---|
| sp_it_a_0001_00 | 1.307 | original |
| sp_it_a_0002_00 | 1.282 | original |
| sp_neg_a_0001_00 | 1.302 | original |
| sp_neg_a_0002_00 | 1.324 | original |
| **sp_neg_a_0003_00** | **2.087** | new render |
| sp_neu_a_0001_00 | 1.299 | original |
| sp_neu_a_0002_00 | 1.327 | original |
| **sp_neu_a_0003_00** | **2.324** | new render |
| sp_sv_a_0001_00 | 1.336 | original |
| sp_sv_a_0002_00 | 1.302 | original |

Mean (n=10): **1.489**. Mean of the 5 clips used in the degradation sweep (first 5 alphabetically — 4 originals + `sp_neg_a_0003_00`): **1.460**.

The new clips (`sp_neg_a_0003`, `sp_neu_a_0003`) score **0.7–1.0 MOS points higher than the originals**. They were rendered by today's TTS pipeline (post-M3a per-turn RMS gain, post-Lombard tilt fix #74, post-#67/#71 SSML escaping); the originals come from the older `m2a_wettest` run. UTMOS *is* sensitive to TTS pipeline version on this data — just not in the way the E2 gate expects.

Degradation sweep (5 clips × 5 degradations, paired against same-5 clean mean of 1.460):

| Degradation | Mean UTMOS | Δ vs clean | Direction |
|---|---|---|---|
| white noise SNR −20 dB | 1.396 | +0.064 | ↓ as expected (smallest gap) |
| white noise SNR −10 dB | 1.325 | +0.135 | ↓ as expected |
| white noise SNR 0 dB | 1.324 | +0.136 | ↓ as expected |
| white noise SNR +10 dB | 1.253 | +0.207 | ↓ as expected (largest gap) |
| 2 kHz brick-wall lowpass | 1.308 | +0.152 | ↓ as expected |

Two observations:

1. **All degradations drop UTMOS below clean** when paired against the same 5 clips. The original PR #77 report's "degraded scored higher than clean" verdict was an artefact of comparing degraded(n=5) against clean(n=8) — once you pair properly, direction is correct.
2. **White-noise severity is inverted.** More noise (lower SNR) → *higher* UTMOS, not lower. UTMOS reads the −20 dB white-noise overlay as more "natural"-sounding than the +10 dB version, possibly because the high-frequency noise floor masks artefacts of the tightly-clean TTS spectrum. This is the harder finding — UTMOS isn't broken in *direction*, but it can't be used to compare degradation magnitudes.

Even with paired comparison, the maximum separation is 0.207 — well below the 0.5 gate. **E2 is not buildable as designed.**

Listening samples for verification: `state/spikes/m17_phase_a/<clip_id>__wn_snr_-20db.wav` (most degraded) through `..__wn_snr_+10db.wav` (least), plus `..__lp_2khz.wav`.

## Findings

### 1. ASR is far better than the gate assumed — across all three runs

The 0.40 WER gate was a wager against a low-resource-language risk. Reality on this data: median WER **0.036** (best), **0.064** (baseline). E1 clip-level pass thresholds in the 0.10–0.15 range are realistic; the design's calibration protocol still applies before any production gate.

### 2. ivrit-ai variants beat openai on Hebrew, with two failure modes worth knowing

- **ivrit-ai HF wins on 9/10 clips** (one outlier: `sp_it_a_0002_00`). Median WER 0.036 vs openai 0.064 (44% relative improvement).
- **openai is unstable on the new clips.** WER 0.535 / 0.341 on `sp_neg_a_0003` / `sp_neu_a_0003` — both rendered by the post-M3a TTS pipeline. ivrit-ai handles them at 0.016 / 0.051. The Hebrew-specialised fine-tune is materially more robust to recent TTS distributional changes.
- **ivrit-ai HF can hallucinate-loop on long-form audio** (the `sp_it_a_0002_00` outlier). The CT2 int8 path *does not* hit the loop on the same clip — possibly because of how faster-whisper's decoder handles seq2seq state across 30 s windows. This is an unexpectedly clean argument for using CT2 int8 even in the dev/QA path, not just CI.

### 3. Trigram-repetition detector cleanly separates the hallucination outlier

Empirical thresholds on this set: `trigram_repeat_ratio > 1.5` catches `sp_it_a_0002_00` (ratio 2.00) and nothing else (all other clips at exactly 1.00). E1 should compute this on every clip and flag as a quality-control signal before WER scoring — the offending clip then either gets re-run on openai (cross-check) or re-run with `condition_on_previous_text=False` and a temperature fallback.

### 4. int8 CT2 is viable for CI

Δ median WER vs MPS baseline: 0.008. Within the prompt's ε = 0.02 tolerance. The design doc's CI cost table at lines 449–456 (which assumes int8 faster-whisper) is validated. Bonus: int8 CT2 dodged the only hallucination loop in the run.

### 5. UTMOS direction is correct — but separation magnitude and severity-monotonicity are not

The original PR #77 report's "UTMOS doesn't discriminate" verdict was an artefact of unpaired comparison. With paired same-5 clean vs same-5 degraded:

- All five degradations (4 SNR levels + 1 lowpass) push UTMOS below clean. ✅
- But max separation is 0.207, not 0.5. ❌
- White-noise severity is inverted (lower SNR → higher UTMOS). ❌

The implications for E2 are unchanged: NO-GO under the current design. But the cause is more nuanced — UTMOS is a noisy signal at the magnitudes we want to gate on, and it's particularly fragile to the type of spectral perturbation we'd need it to flag.

### 6. The original 8 clips score systematically lower on UTMOS than the 2 new ones

Original 8: mean 1.31 (range 1.28–1.34, std 0.018). New 2: mean 2.21 (range 2.09–2.32). This is a ~0.9 MOS-point gap, far larger than any gap UTMOS produces between clean and degraded versions of the same clip. The likely reason is TTS pipeline drift between the m2a_wettest era and now. Two implications:

- UTMOS is *not* fit for run-to-run regression detection — pipeline-version noise dominates the within-clip degradation signal.
- The pipeline-version separation might itself be a useful signal for *human review* (e.g., "the v3-pipeline clips are getting better predicted MOS than v2"), but that's an informational pivot, not a release gate.

### 7. Greedy decoding is reproducible across runs, modulo normalization

WER on every clip is byte-stable run-to-run on MPS with `num_beams=1`, `do_sample=False`, `temperature=0`. The PR #77 → today change in `sp_it_a_0002_00` ivrit-ai WER (0.144 → 0.127) is fully explained by the new RTL-mark stripping in `normalize_for_wer`; the *raw* ASR output is unchanged. Reproducibility is a settled question for E1.

### 8. Azure TTS SSML rejected the LLM-generated script for IT-0003 (twice, two templates)

`sp_it_a_0003.yaml` failed Azure synthesis at "TurnStarted; Received audio size: 0 bytes" with both `intimate_terror_financial_control.j2` and `intimate_terror_jealousy_surveillance.j2` templates under different `script_slots`. PR #67/#71 fixed the adjacent-`<break>` SSML class, but this is a different failure mode — likely either (a) an unescaped character class the LLM emits in IT-typology scripts at this seed range, or (b) something in the prosody/style markup. Replaced with `sp_neg_a_0003`. Tracking as a separate follow-up issue; the spike's gate signal isn't affected.

## Limitations

1. **Single speaker pair.** All 10 clips use `AGG_M_30-45_001` + `VIC_F_25-40_002`. Speaker generalisation untested.
2. **Tier A only.** No room IR, device profiles, or background noise. Tier B/C ASR performance is unknown — likely materially worse and warrants its own validation.
3. **TTS pipeline drift dominates between-clip variance.** New clips score 0.9 MOS points higher than original 8 on UTMOS; we don't yet know how this would interact with E2 calibration on a production batch.
4. **Long-form chunking is HF's experimental path** (`pipeline(chunk_length_s=30)`). Switching E1 to native `model.generate(...)` long-form is a recommended follow-up but doesn't change the gate result.
5. **No CI cost measurement.** Wall-time captured in JSON but not reported — laptop-sleep makes the numbers meaningless. A controlled benchmarking session is required before per-clip latency makes it into the design doc's CI table.
6. **`sp_it_a_0003` not represented.** Azure SSML rejection means one IT scene that we wanted is missing; replaced with NEG. Distribution skew is documented above.

## Recommendation

### E1 — Whisper ASR — GO

1. **Primary model:** `ivrit-ai/whisper-large-v3` (HF transformers, MPS in dev / GPU in prod).
2. **CI model:** `ivrit-ai/whisper-large-v3-ct2` via `faster-whisper` with `compute_type="int8"`. Δ vs primary is within ε = 0.02 on this set.
3. **Hallucination guard:** compute `trigram_repeat_ratio = max_3gram_count(hyp) / max(1, max_3gram_count(ref))`. Threshold `> 1.5`. On flag: re-run with `openai/whisper-large-v3` and prefer the lower-WER hypothesis.
4. **Reference normalisation:** strip Hebrew niqqud and RTL embedding marks before WER. Punctuation → whitespace. The script's `normalize_for_wer` is the canonical implementation.
5. **Defer to follow-up:** switch chunking from HF's `pipeline(chunk_length_s=30)` to native `WhisperForConditionalGeneration.generate(...)`. Doesn't block Phase A.
6. **Calibrate E1 thresholds** on a 50-clip baseline before any WER threshold gates merges, per the design doc's protocol.

### E2 — Predicted MOS — NO-GO

Three options, ordered by cost:

- **Option A — turn-segmented UTMOS re-spike** (one afternoon). Use `ONSET`/`OFFSET` from the reference transcripts to extract per-turn audio (4–15 s windows match UTMOS's training distribution), score each turn, aggregate per clip. Re-test the degradation sweep. If turn-level UTMOS gives ≥ 0.5 separation and severity-monotonicity holds, E2 design proceeds with turn-segmented scoring.
- **Option B — swap UTMOS for a long-form-friendly predictor** (NISQA / WV-MOS). Re-spike on the same 10-clip set + 5 degraded variants per clip.
- **Option C — drop predicted MOS entirely.** Replace E2 with deterministic objective metrics (SNR, peak dBFS, clipping rate, silence ratio, spectral tilt) plus the human listening-test cadence the M12 incident already established as necessary. The TTS-pipeline-version variance we observed makes any predicted-MOS gate fragile to changes that aren't actually quality regressions.

**Recommendation: start with Option A.** If A fails, jump to Option C — the M12 listening-test history says automated MOS isn't a reliable release gate for this project, and Option B is unlikely to cross the gap.

### Phase A overall

Phase A is **partially unblocked**:

- E1 code skeleton can proceed with the model + detector configuration above.
- E2 needs the turn-segmented re-spike (Option A) before any E2 implementation.

The Phase A code-skeleton PR should land E1 only; E2 follows the re-spike.

### Tracked follow-ups

- IT-0003 SSML failure → new issue.
- HF `pipeline(chunk_length_s=30)` → native long-form switch → follow-up after E1 lands.
- M4 Max inference-time calibration session (per-clip latency for the CI cost table).
- E2 turn-segmented re-spike (Option A above).
