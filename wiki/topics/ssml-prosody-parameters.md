---
schema_version: '1'
kind: topic
title: SSML Prosody Parameters — Current Settings and Tuning
page_id: topic-ssml-prosody-params
status: active
review_state: human-reviewed
source_refs:
  - src-3d18f514d3e6bae4424eb2a3a8ff318976d5d5834a19fe6ddc16e9e8e6baebc4
  - src-bdb2f9f939e8892e2c60a517264027e1e7b86c33cfce2f41015c8f0a90c463eb
  - src-42711de5f7b0e5a5b70b02d8fcd01097a6901aa5b5d14d61b351b6bbbd94ab89
tags: [ssml, prosody, tts, azure, pitch, rate, volume]
---

# SSML Prosody Parameters — Current Settings and Tuning

This page tracks the SSML prosody configuration used for Azure TTS rendering, how it maps to intensity levels, and known issues.

## Architecture

1. `SSMLBuilder` (`synthbanshee/tts/ssml_builder.py`) converts a `DialogueTurn` + `SpeakerConfig` into an SSML document.
2. `SpeakerState` (`synthbanshee/tts/speaker_state.py`) maintains per-speaker state across turns (M7), allowing intensity drift and carry-over.
3. `StyleEntry` in speaker config defines per-intensity SSML parameters.

## Recommended Parameters per Intensity Level (from Research Synthesis)

Based on cross-referencing three independent research reports and Hebrew-specific prosody studies (T-RES, Amir et al. 2003, Silber-Varod 2016):

| Intensity | Rate | Pitch (M) | Pitch (F) | Volume | F0 Range | express-as |
|-----------|------|-----------|-----------|--------|----------|------------|
| I1 (calm) | -6% to +2% | -3% to +2% | -3% to +2% | baseline | -10% to 0% | **NONE** |
| I2 (neutral) | 0% to +4% | -2% to +2% | -2% to +3% | baseline | baseline | **NONE** |
| I3 (irritation) | +2% to +8% | -1% to +3% | 0% to +5% | +2 to +4 dB | +10% to +25% | **NONE** |
| I4 (anger) | +4% to +12% | -1% to +5% | +2% to +7% | +6 to +10 dB | +25% to +50% | **NONE** |
| I5 (shouting) | +8% to +15% | 0% to +5% | +4% to +10% | +10 to +15 dB | +35% to +60% | **NONE** |

**Key:** `express-as` is set to NONE because Azure `mstts:express-as` styles are NOT supported for he-IL voices. Use `<prosody>` tags only.

**Break timing:** 150ms after commas; 180-350ms between sentences; 400-700ms at major boundaries.

**Hebrew speaking rate reference:** Neutral baseline ~4.4-5.2 syll/sec; anger ~4.95 syll/sec; professional newscaster ~5.9 syll/sec. Below 4.5 SPS sounds unnaturally slow.

## Known Issues from Listening Feedback

- **AGG pitch rises with anger** — pitch multiplier at I4–I5 should be neutral or slightly negative for male aggressor, not positive.
- **Speech rate too slow at I1–I2** — Hebrew natural speech rate is ~4-5 syllables/sec; current I1 rate may be below this.
- **Voice timbre changes with style switching** — `mstts:express-as` style changes between turns can shift the apparent voice identity.

## Design Constraints (from §4.2a)

- VIC I1 median F0 ≤ 200 Hz
- VIC I4 median F0 < 250 Hz
- VIC I5 median F0 < 250 Hz
- AGG should convey distress via rate/timing, NOT pitch
- AGG I5 − I1 RMS ≥ 8 dB (loudness escalation)

## Research Needed

1. Dump the actual SSML being sent for a representative clip at each intensity level
2. Compare against Azure documentation for he-IL voice prosody ranges
3. Determine if `SpeakerState` carry-over is causing unbounded pitch drift
4. Test: what happens if we remove pitch modulation for AGG entirely and only use rate + volume?
