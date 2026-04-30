---
schema_version: '1'
kind: topic
title: Preprocessing Pipeline — Filters, Denoising, and Audio Quality Impact
page_id: topic-preprocessing-pipeline
status: active
review_state: human-authored
source_refs:
  - src-5184b8f4c65ccf204f748e694ad677f7ae3792c2e91a861e636ede81b7ba3c1f
  - src-42711de5f7b0e5a5b70b02d8fcd01097a6901aa5b5d14d61b351b6bbbd94ab89
tags: [preprocessing, audio, filter, denoiser, muffled, quality]
created: '2026-04-30'
updated: '2026-04-30'
---

# Preprocessing Pipeline — Filters, Denoising, and Audio Quality Impact

## Current Pipeline Steps (Stage 3a)

From `synthbanshee/augment/preprocessing.py`, applied in order:

1. **Lowpass filter** — 7500 Hz, order 4 Butterworth
2. **Wiener denoise** — scipy `wiener()` filter
3. **Peak limiter** — ceiling at -1.0 dBFS (limiter, NOT normalizer)
4. **Silence padding** — 0.5 s per side

## Muffled Sound Problem

Human feedback consistently reports "muffled" audio across all clips. This is the most pervasive quality issue.

### Primary Suspect: 7500 Hz Lowpass

- At 16 kHz sample rate, Nyquist is 8000 Hz.
- A 7500 Hz lowpass with order 4 Butterworth has significant rolloff starting well below 7500 Hz.
- Speech intelligibility depends on frequencies up to ~8 kHz for sibilants (/s/, /sh/, /ts/) and breathiness.
- Hebrew has frequent sibilants (shin, samekh, tsadi) — cutting these frequencies directly impacts perceived clarity.
- **The filter is essentially doing nothing useful** since the TTS output is already bandlimited by the 16 kHz output, and it's aggressively removing the top 500 Hz of usable bandwidth.

### Secondary Suspect: Wiener Denoiser

- The Wiener filter is designed for noisy signals. Clean TTS output has no noise to remove.
- Applying Wiener denoising to clean TTS audio will smooth out high-frequency transients and reduce perceptual sharpness.
- The `divide by zero` and `invalid value` RuntimeWarnings seen during preprocessing suggest the filter is operating on near-silent segments where noise estimation breaks down.

### Recommendations

1. **Remove or raise the lowpass filter.** If the spec mandates filtering, raise cutoff to at least 7800–7900 Hz. But since the signal is already at 16 kHz (Nyquist 8 kHz), any lowpass below 8 kHz is removing real signal content.
2. **Make Wiener denoising conditional.** Only apply to Tier B/C clips that have added noise. Skip for Tier A clean TTS output.
3. **A/B test:** Generate the same clip with and without preprocessing to quantify the quality impact.

## Click/Pop Artifacts

Loud clicks heard at sentence boundaries. Possible causes:
- Segment concatenation in mixer without crossfade
- SSML boundary generating a glottal stop or click in TTS output
- Preprocessing filter introducing edge artifacts at segment boundaries

## Spec Constraints

- 16 kHz, mono, 16-bit PCM WAV — non-negotiable
- Peak ≤ -1.0 dBFS — limiter is correct behavior
- ≥ 0.5 s silence padding — correct
- No torchaudio, no lossy formats
