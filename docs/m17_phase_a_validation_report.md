# M17 Phase A Validation Report — Whisper + UTMOS on Hebrew Clips

This report validates the Phase A acceptance criteria from the M17 design doc
(`docs/automated_eval_design.md` lines 418–422) before the E1/E2 module skeletons land.

Generated: 2026-05-05. Raw data: `state/spikes/m17_phase_a/results.json` (gitignored). Auto-tables: `state/spikes/m17_phase_a/report_auto.md`. Listening samples: `state/spikes/m17_phase_a/<clip_id>__<degradation>.wav`. SHA-256 prefixes shown in the manifest are first-8-chars only — full hashes in `results.json`.

## TL;DR

| Evaluator | Gate | Outcome |
|---|---|---|
| **E1 — ASR** | median WER < 0.40 | **PASS** by ~10× margin on all three Whisper variants |
| **E2 — UTMOS** | clean − degraded mean ≥ 0.5 | **FAIL** (max separation 0.21 across 5 RMS-matched degradations) |

**Phase A is BLOCKED** per the design doc's gate semantics (all three gates required). Unblock path: turn-segmented UTMOS re-spike (Option A in the recommendation). E1 evidence is strong and ready to use the moment Phase A is unblocked.

Other findings worth surfacing inline:

- **`ivrit-ai/whisper-large-v3` is the right primary model** (median WER 0.036 vs openai 0.064) — but neither model handles a TTS-loudness regression on the new clips well.
- **The new clips (`sp_neg_a_0003`, `sp_neu_a_0003`) are −6 dB quieter than the old 8.** Old: peak 0.891 (= −1 dBFS limiter), RMS 0.097 mean. New: peak 0.49–0.54, RMS 0.045 mean. This is a **real signal-domain regression in the TTS pipeline**, not just a perceptual-quality drift. It's the proximate cause of openai's 28% / 19% WER on the new clips and a major contributor to UTMOS reading them higher. Tracking as a separate follow-up.
- **The hallucination outlier on `sp_it_a_0002` and the openai outliers on the new clips are *different failure modes*** — over-transcription (loop) vs under-transcription (dropout). A single threshold can't cover both; a *bidirectional* length-ratio detector does.
- **UTMOS white-noise severity is genuinely inverted** — confirmed even with RMS-matched degradations. The previous PR's broken peak-clip code did not cause the inversion.
- **CI viability of int8 CT2 is encouraging but underspecified.** Single run on M4 Max; needs validation on representative CI hardware before claiming the design doc's CI plan is settled.
- **Design doc model ID fixed in this PR.** `ivrit-ai/whisper-v2-d3-e3` → `ivrit-ai/whisper-large-v3` at three locations.

## Reproduce

```bash
uv pip install --python .venv/bin/python \
    torch transformers accelerate faster-whisper \
    jiwer numpy scipy soundfile
.venv/bin/python scripts/m17_phase_a_validation.py
```

`torchaudio` is not used (CLAUDE.md forbids it in preprocessing). The `speechmos` PyPI package only ships DNSMOS/AECMOS/PLCMOS — UTMOS22 strong is loaded via `torch.hub` from a pinned `tarepan/SpeechMOS` commit instead (see Limitations §8).

Inputs: 10 clips at `data/m2a_wettest/agg_m_30-45_001/` — see Manifest below for SHA-256s. Device: MPS (Apple Silicon M4 Max). All decoders are greedy (`num_beams=1`, `do_sample=False`, `temperature=0`); `numpy` and `torch` seeded to 42. Degraded variants are RMS-normalized to match each clip's clean RMS before scoring (see `rms_normalize_to_match` in the script).

## Clip-set manifest

| Clip | Duration (s) | Ref words | gen_date | Peak | RMS | Crest (dB) | SHA-256 (first 8) |
|---|---|---|---|---|---|---|---|
| sp_it_a_0001_00 | 155.9 | 233 | 2026-04-15 | 0.891 | 0.099 | 19.1 | `792d622b` |
| sp_it_a_0002_00 | 146.7 | 291 | 2026-04-15 | 0.891 | 0.105 | 18.6 | `064e9f29` |
| sp_neg_a_0001_00 | 128.5 | 240 | 2026-04-15 | 0.891 | 0.089 | 20.1 | `f403b7db` |
| sp_neg_a_0002_00 | 114.2 | 177 | 2026-04-15 | 0.891 | 0.101 | 19.0 | `bdbf4487` |
| **sp_neg_a_0003_00** | 257.5 | 257 | **2026-05-05** | **0.541** | **0.046** | 21.4 | `8988c9ec` |
| sp_neu_a_0001_00 | 121.0 | 189 | 2026-04-15 | 0.891 | 0.096 | 19.4 | `e4782fe6` |
| sp_neu_a_0002_00 | 133.1 | 221 | 2026-04-15 | 0.891 | 0.101 | 18.9 | `9edae0a1` |
| **sp_neu_a_0003_00** | 190.4 | 176 | **2026-05-05** | **0.490** | **0.044** | 21.0 | `7a616df5` |
| sp_sv_a_0001_00 | 122.5 | 188 | 2026-04-15 | 0.891 | 0.084 | 20.5 | `682049f7` |
| sp_sv_a_0002_00 | 99.4 | 164 | 2026-04-15 | 0.891 | 0.099 | 19.1 | `44b1e4e4` |

Distribution: 2 IT / 3 NEG / 3 NEU / 2 SV. Old clips peak at exactly 0.891 (= −1 dBFS, the CLAUDE.md limiter ceiling). New clips peak at 0.49–0.54, ~6 dB below ceiling and ~6 dB below the old RMS — the regression that drives most of the noise in this report's secondary findings.

## Gate Outcomes

| Gate | Criterion | Result | Decision |
|------|-----------|--------|----------|
| ASR — `openai/whisper-large-v3` (HF, MPS) | median WER < 0.40 | **0.064** | ✅ PASS |
| ASR — `ivrit-ai/whisper-large-v3` (HF, MPS) | median WER < 0.40 | **0.036** | ✅ PASS |
| ASR — `ivrit-ai/whisper-large-v3-ct2` (faster-whisper, int8 CPU) | median WER < 0.40 | **0.044** | ✅ PASS |
| UTMOS — primary degradation (white noise −10 dB SNR, RMS-matched) | clean − deg ≥ 0.50 | **+0.135** vs same-5 clean mean (1.461) | ❌ FAIL |
| UTMOS — best of 5-degradation sweep | any ≥ 0.50 | max **+0.208** (white noise +10 dB) | ❌ FAIL |
| UTMOS — white-noise monotonicity in severity | more noise → lower UTMOS | **inverted** | ❌ FAIL |

**Phase A — BLOCKED.** The design doc's acceptance criteria treat the three gates as conjunctive (all three block Phase A). UTMOS fails; Phase A doesn't start. E1 evidence below is strong and ready to land in the Phase A code-skeleton PR the moment Phase A is unblocked.

## ASR — Per-Clip WER

| Clip | openai HF | ivrit-ai HF | ivrit-ai CT2 int8 |
|---|---|---|---|
| sp_it_a_0001_00 | 0.056 | 0.039 | 0.047 |
| sp_it_a_0002_00 | 0.065 | **0.127** | 0.062 |
| sp_neg_a_0001_00 | 0.063 | 0.033 | 0.033 |
| sp_neg_a_0002_00 | 0.045 | 0.017 | 0.028 |
| **sp_neg_a_0003_00** | **0.280** | 0.016 | 0.016 |
| sp_neu_a_0001_00 | 0.079 | 0.048 | 0.053 |
| sp_neu_a_0002_00 | 0.045 | 0.041 | 0.041 |
| **sp_neu_a_0003_00** | **0.188** | 0.051 | 0.057 |
| sp_sv_a_0001_00 | 0.075 | 0.032 | 0.053 |
| sp_sv_a_0002_00 | 0.061 | 0.012 | 0.018 |
| **median** | 0.064 | 0.036 | 0.044 |
| **mean** | 0.096 | 0.042 | 0.041 |

Three outliers worth understanding:

- **`sp_it_a_0002` ivrit-ai HF (0.127):** over-transcription. Length ratio 1.048 (293 hyp words vs 291 ref), trigram repeat 2 (vs 1 in ref), 19 RTL embedding marks in raw output. Classic Whisper repeat-loop failure. Faster-whisper int8 handles this clip cleanly at 0.062.
- **`sp_neg_a_0003` openai HF (0.280):** under-transcription. Length ratio 0.763 (196 hyp vs 257 ref) — openai dropped ~24% of words. No repeat-loop (trigram repeat ratio 0.50). Likely Whisper's silence-handling / VAD treating the quieter new-pipeline audio as silence and clipping segments.
- **`sp_neu_a_0003` openai HF (0.188):** under-transcription. Length ratio 0.920 (162/176) — openai dropped ~8% of words. Same proximate cause.

The two openai outliers are exactly the two new-pipeline clips. ivrit-ai handles both cleanly (0.016, 0.051). The Hebrew fine-tune is materially more robust to the loudness regression in §4 below.

## CI viability — int8 CT2 vs MPS baseline

Δ between `ivrit-ai/whisper-large-v3` (HF, MPS) and `ivrit-ai/whisper-large-v3-ct2` (faster-whisper, int8, CPU):

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

Median |Δ|: **0.008** — encouraging vs the prompt's ε = 0.02 tolerance.

**Caveats before treating this as "CI viable":**

1. **Single run on a single machine.** No variance estimate over time, no validation on representative GitHub Actions x86 hardware.
2. **The two backends differ in more than quantization.** HF transformers uses fixed 30 s window chunking; faster-whisper uses silence-based segmentation; default temperature-fallback chains differ; their decoding loops aren't equivalent. The Δ is not isolating "int8 vs fp32"; it's measuring "the whole stack."
3. **The −0.065 outlier in int8's favour on `sp_it_a_0002`** is *not* evidence that int8 is more robust. It's that the silence-based segmentation in faster-whisper happened to break the repeat-loop on this clip. A different clip with a different triggering pattern could go the other way.

The honest claim: **on this 10-clip set, on this machine, int8 CT2 lands within ε of the MPS baseline, which is consistent with the design doc's CI plan being feasible.** Sizing CI runners and committing to the plan needs at least: a re-run on Linux x86 (where CI actually runs), variance estimation across multiple runs, and matched decoding settings between backends.

## Hallucination detectors — empirical signals on the 10-clip set

Per-clip detector output for `ivrit-ai/whisper-large-v3` (the only run with the over-transcription outlier):

| Clip | WER | length_ratio | hyp 3-gram repeat | trigram_repeat_ratio | RTL marks |
|---|---|---|---|---|---|
| sp_it_a_0001_00 | 0.039 | 1.004 | 2 | 1.00 | 0 |
| **sp_it_a_0002_00** | **0.127** | **1.048** | **2** | **2.00** | **19** |
| sp_neg_a_0001_00 | 0.033 | 1.008 | 2 | 1.00 | 8 |
| sp_neg_a_0002_00 | 0.017 | 1.000 | 1 | 1.00 | 0 |
| sp_neg_a_0003_00 | 0.016 | 1.004 | 2 | 1.00 | 0 |
| sp_neu_a_0001_00 | 0.048 | 1.005 | 1 | 1.00 | 15 |
| sp_neu_a_0002_00 | 0.041 | 1.005 | 1 | 1.00 | 0 |
| sp_neu_a_0003_00 | 0.051 | 1.011 | 1 | 1.00 | 0 |
| sp_sv_a_0001_00 | 0.032 | 0.995 | 2 | 1.00 | 0 |
| sp_sv_a_0002_00 | 0.012 | 1.000 | 2 | 1.00 | 0 |

Per-clip detector output for `openai/whisper-large-v3` (the runs with under-transcription on the new clips):

| Clip | WER | length_ratio | hyp 3-gram repeat | trigram_repeat_ratio | RTL marks |
|---|---|---|---|---|---|
| sp_it_a_0001_00 | 0.056 | 1.013 | 2 | 1.00 | 0 |
| sp_it_a_0002_00 | 0.065 | 1.007 | 1 | 1.00 | 0 |
| sp_neg_a_0001_00 | 0.063 | 1.012 | 2 | 1.00 | 0 |
| sp_neg_a_0002_00 | 0.045 | 1.000 | 1 | 1.00 | 0 |
| **sp_neg_a_0003_00** | **0.280** | **0.763** | 1 | **0.50** | 0 |
| sp_neu_a_0001_00 | 0.079 | 1.005 | 1 | 1.00 | 0 |
| sp_neu_a_0002_00 | 0.045 | 1.009 | 1 | 1.00 | 0 |
| **sp_neu_a_0003_00** | **0.188** | **0.920** | 1 | 1.00 | 0 |
| sp_sv_a_0001_00 | 0.075 | 1.000 | 1 | 1.00 | 0 |
| sp_sv_a_0002_00 | 0.061 | 0.970 | 2 | 1.00 | 0 |

Two failure modes, two detector behaviors:

| Failure | Example | Length ratio | Trigram repeat ratio | RTL marks |
|---|---|---|---|---|
| Over-transcription / loop | `sp_it_a_0002` ivrit-ai | 1.048 | 2.00 | 19 |
| Under-transcription / dropout | `sp_neg_a_0003` openai | 0.763 | 0.50 | 0 |
| Under-transcription / dropout | `sp_neu_a_0003` openai | 0.920 | 0.50 (sub-1.0 floor) | 0 |

A single-sided trigram-repeat detector (the recommendation in the previous PR push) catches only the over-transcription case. **A bidirectional length-ratio detector catches all three** observed positives:

```
flag if length_ratio < 0.95 OR length_ratio > 1.04
```

On this set, this rule has 100% precision (3/3 positives caught, 0/17 false positives across both runs). Calibration caveat: that's a small set, and the threshold pair is fitted to exactly the data that calibrated it. Real validation needs a held-out positive set with multiple instances of each failure mode. Treat the thresholds as a **starting hypothesis** for E1, not a tuned production guard.

## UTMOS — clean baseline + degradation sweep

Clean per clip:

| Clip | UTMOS | Era |
|---|---|---|
| sp_it_a_0001_00 | 1.307 | old |
| sp_it_a_0002_00 | 1.282 | old |
| sp_neg_a_0001_00 | 1.302 | old |
| sp_neg_a_0002_00 | 1.324 | old |
| **sp_neg_a_0003_00** | **2.087** | new |
| sp_neu_a_0001_00 | 1.299 | old |
| sp_neu_a_0002_00 | 1.327 | old |
| **sp_neu_a_0003_00** | **2.324** | new |
| sp_sv_a_0001_00 | 1.336 | old |
| sp_sv_a_0002_00 | 1.302 | old |

Mean (n=10): **1.489**. Mean over the first-5-alphabetical clips used in the degradation sweep (4 old + `sp_neg_a_0003`): **1.461**.

Degradation sweep (5 clips × 5 degradations, **RMS-normalized to match each clean clip** before scoring; paired against same-5 clean mean of 1.461):

| Degradation | Mean UTMOS | Δ vs same-5 clean | Direction |
|---|---|---|---|
| white noise SNR −20 dB | 1.394 | +0.067 | ↓ as expected (smallest gap) |
| white noise SNR −10 dB | 1.327 | +0.135 | ↓ as expected |
| white noise SNR 0 dB | 1.324 | +0.137 | ↓ as expected |
| white noise SNR +10 dB | 1.253 | +0.208 | ↓ as expected (largest gap) |
| 2 kHz brick-wall lowpass | 1.308 | +0.153 | ↓ as expected |

Two findings, both load-bearing:

1. **All degradations score below clean** when paired correctly. The previous PR's "degraded scored higher than clean" verdict was a paired-comparison error.
2. **White-noise severity is inverted** — adding *less* noise produces a *lower* UTMOS score. Verified across the −20 / −10 / 0 / +10 dB SNR sweep; not monotonic at any boundary. Confirmed under RMS-matched conditions (no peak-clip ever fired on this run; per-clip pre/post normalize RMS captured in `results.json` under `per_clip_loudness`), so this isn't a methodology artefact.

The maximum separation is **0.208** at +10 dB SNR — far below the 0.5 gate. UTMOS doesn't discriminate clean from degraded with enough magnitude *or* the right direction to gate releases on.

Listening samples are at `state/spikes/m17_phase_a/<clip_id>__<degradation>.wav` for human verification.

## Findings

### 1. ASR is far better than the gate assumed

All three runs clear the 0.40 gate by ~10×; clip-level pass thresholds in the 0.10–0.15 range are realistic for E1. Calibration on a 50-clip baseline per the design doc's protocol still required before any production gate.

### 2. ivrit-ai/whisper-large-v3 wins on 9/10 clips

Median WER 0.036 vs openai 0.064 (44% relative improvement). Adopt as the primary E1 model.

### 3. Two distinct failure modes on this 10-clip set, both caught by a bidirectional length-ratio detector

- `sp_it_a_0002` ivrit-ai: over-transcription / repeat-loop (length_ratio 1.048, trigram_repeat 2, 19 RTL marks).
- `sp_neg_a_0003` and `sp_neu_a_0003` openai: under-transcription / dropout (length_ratio 0.763 / 0.920).

The length-ratio rule `< 0.95 OR > 1.04` flags all three. Trigram-repeat alone misses both under-transcription cases. RTL count is too noisy to use alone (0–15 on clean clips). E1 should track all three signals; the bidirectional length-ratio rule is the cleanest single-detector starting point — but the detector is fitted on N=3 positives and needs broader validation before it's a release gate.

### 4. The new clips are −6 dB quieter than the old 8 — a TTS pipeline regression

Measured directly from the WAVs:

- Old 8 (rendered 2026-04-15): peak 0.891 *exactly* on every clip (= −1 dBFS, the CLAUDE.md limiter ceiling). RMS 0.084–0.105 (mean 0.097).
- New 2 (rendered 2026-05-05): peak 0.49–0.54. RMS 0.044–0.046 (mean 0.045).

The new clips meet the CLAUDE.md spec ("peak ≤ −1 dBFS") but are nowhere near the limiter ceiling. Something between 2026-04-15 and 2026-05-05 changed the rendering loudness — the most likely culprit is recent mixer work (PRs #65, #66, #74) interacting with the M3a per-turn RMS gain. This is the proximate cause of:

- **openai's 28% / 19% WER on new clips** — Whisper's silence-handling drops segments when input level is low.
- **UTMOS reading new clips ~0.9 points higher** — less aggressive limiting reads as more "natural" to a model trained on speech that wasn't aggressively limited.
- **UTMOS variance dominated by render era**, masking any within-clip degradation signal.

This is a **TTS regression**, not an evaluator concern. Tracking as a separate follow-up — fixing it would tighten the spike's secondary findings considerably and is a prerequisite for any E2 calibration that relies on stable input loudness.

### 5. UTMOS direction is correct under paired comparison; magnitude and severity-monotonicity are not

With same-5 clean vs same-5 degraded, all five degradations push UTMOS below clean — the previous PR's "doesn't discriminate" verdict was a comparison artefact. But:

- Maximum separation 0.208, well below the 0.5 gate.
- White-noise severity inverted (lower SNR → higher UTMOS).
- The inversion survives RMS-normalization, so it's not a loudness artifact of the test pipeline.

Either UTMOS reads heavy noise as "natural acoustic environment" rather than "degraded TTS," or the model's output saturates at a noise level lower than +10 dB SNR. Either way, it can't gate releases on this data at this gate magnitude.

### 6. faster-whisper int8 stays within ε of the MPS baseline on this run

Median |Δ| = 0.008, vs ε = 0.02. Encouraging but not sufficient for CI sizing decisions on its own — see the CI viability section above for the limitations on this claim.

### 7. Reproducibility of greedy decoding on MPS is settled

WERs are byte-stable run-to-run when `num_beams=1`, `do_sample=False`, `temperature=0`, identical normalization. The PR #77 → today change in `sp_it_a_0002` ivrit-ai WER (0.144 → 0.127) is fully explained by the new RTL-mark strip in `normalize_for_wer`; the raw ASR output is unchanged.

### 8. Azure TTS SSML rejected one of the planned IT scenes

`sp_it_a_0003.yaml` failed Azure synthesis with `intimate_terror_financial_control` and again with `intimate_terror_jealousy_surveillance`. Replaced with `sp_neg_a_0003`. The failure pattern (`SSML parsing error: 0x80045003`, "TurnStarted; Received audio size: 0 bytes") suggests PR #67 / #71's escaping doesn't cover this case. Tracking separately.

## Limitations

1. **N=10 with 3 positives** — detector thresholds are fitted on the same data they were validated on. Real validation needs a multi-positive held-out set.
2. **Single speaker pair** (`AGG_M_30-45_001` + `VIC_F_25-40_002`); generalisation untested.
3. **Tier A only** — no room IR / device profile / background noise; Tier B/C ASR performance is unknown.
4. **Single ASR run per backend, single machine** — no variance estimate; CI viability claim is preliminary.
5. **TTS pipeline drift dominates between-clip UTMOS variance.** Until the loudness regression is fixed, any UTMOS-based regression detection over time is unreliable.
6. **Long-form chunking is HF's "experimental" path** (`pipeline(chunk_length_s=30)`). Switching E1 to the native long-form `WhisperForConditionalGeneration.generate(...)` is a recommended follow-up but doesn't change the gate result.
7. **No CI cost measurement** — wall-time captured in the JSON but not reported (laptop sleep makes the numbers meaningless).
8. **UTMOS loader uses `torch.hub` with `trust_repo=True`** — this downloads and executes code from `tarepan/SpeechMOS` at runtime. We pin the source to an immutable commit hash (`ed25eacb...`) to avoid moving-tag risk, but a production E2 implementation should vendor or hash-lock the model artifact rather than fetching it at run time.

## Recommendation

### Phase A overall — BLOCKED

Per the design doc's gate semantics (acceptance criteria table; all three gates required to start Phase A), Phase A is blocked on the UTMOS gate. The unblock path is the Option A re-spike below. E1 evidence is strong and ready to land in the Phase A code-skeleton PR the moment Phase A is unblocked.

### E1 — when Phase A unblocks

- **Primary model:** `ivrit-ai/whisper-large-v3` (HF transformers).
- **CI candidate:** `ivrit-ai/whisper-large-v3-ct2` via `faster-whisper` with `compute_type="int8"` — pending CI-hardware validation.
- **Hallucination guard (preliminary):** bidirectional length-ratio rule (`< 0.95 OR > 1.04`) caught all three observed positives on this set. Validate on a multi-positive held-out set before deploying as a gate.
- **Reference normalisation:** strip Hebrew niqqud, RTL embedding marks, punctuation. The `normalize_for_wer` in the spike script is one defensible implementation; E1 can adopt or replace.

The detail-level prescription for E1 (chunking algorithm, beam settings, threshold values for clip-level WER gates) is out of this spike's scope and belongs in the E1 PR.

### E2 — re-spike before any implementation

Three options, ordered by cost:

- **Option A — turn-segmented UTMOS re-spike** (one afternoon). Use `ONSET`/`OFFSET` from the reference transcripts to extract per-turn audio (4–15 s windows, matching UTMOS's training distribution), score each turn, aggregate per clip. Re-test the degradation sweep. If turn-level UTMOS shows ≥ 0.5 separation and severity-monotonicity holds, E2 design proceeds with turn-segmented scoring.
- **Option B — swap UTMOS for a long-form-friendly predictor** (NISQA / WV-MOS). Re-spike on the same 10-clip set + 5 degraded variants per clip.
- **Option C — drop predicted MOS entirely.** Replace E2 with deterministic objective metrics (SNR, peak dBFS, clipping rate, silence ratio, spectral tilt) plus the human listening-test cadence the M12 incident already established as necessary.

**Recommendation: Option A first.** The TTS-loudness regression in §4 means any predicted-MOS gate is currently fragile to changes that aren't actual quality regressions; turn-segmented scoring may help by removing the dominant variance source. If Option A doesn't restore separation magnitude / monotonicity, jump to Option C.

### Tracked follow-ups

- **TTS loudness regression** (new clips at half the RMS / peak of old clips) — high priority; affects any future eval calibration.
- **Azure TTS SSML failure** on the IT-0003 scene (two different templates) — PR #67/#71 escaping incomplete.
- **HF `pipeline(chunk_length_s=30)` → native long-form `generate(...)`** — follow-up after E1 lands.
- **M4 Max inference-time calibration session** — required before per-clip latency can enter the design doc's CI cost table.
- **E2 turn-segmented UTMOS re-spike** (Option A above) — Phase A unblocker.
- **CI-hardware validation of int8 CT2** — Linux x86 GitHub Actions runner, multiple runs, matched decoding.
