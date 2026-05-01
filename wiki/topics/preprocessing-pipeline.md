---
schema_version: '1'
kind: topic
title: Preprocessing Pipeline — Filters, Denoising, and Audio Quality Impact
page_id: topic-preprocessing-pipeline
status: active
review_state: human-reviewed
source_refs:
  - src-5184b8f4c65ccf204f748e694ad677f7ae3792c2e91a861e636ede81b7ba3c1f
  - src-42711de5f7b0e5a5b70b02d8fcd01097a6901aa5b5d14d61b351b6bbbd94ab89
tags: [preprocessing, audio, filter, denoiser, muffled, quality]
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

### Recommendations (Validated by 3 Independent Research Reports)

All three reports (Gemini, GPT-5.2 thinking, GPT-5.5 Pro) unanimously agree:

1. **Remove Wiener denoising entirely on clean TTS output.** It is the primary cause of muddy/muffled sound. Only apply to Tier B/C clips with real added noise.
2. **Remove the 7.5 kHz lowpass filter entirely.** Replace with:
   - **HPF at 80-120 Hz** (2nd order) for DC/rumble removal
   - **No LPF** or gentle high-shelf -1 to -3 dB above 6.5 kHz (NOT a brick-wall filter)
3. **Add phone-device EQ instead** (for realism, not "cleanup"):
   - Low-mid cut: -1 to -4 dB around 250-400 Hz
   - Presence boost: +2 to +4 dB around 2.5-3.5 kHz
   - Gentle high shelf: -1 to -3 dB above 6.5 kHz
4. **Add 10ms crossfade (160 samples at 16 kHz)** at all turn boundaries to eliminate click artifacts.
5. **A/B test:** Generate the same clip with and without preprocessing to quantify the quality impact.

See [Research Synthesis](research-synthesis.md) for full parameter tables and priority ordering.

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
