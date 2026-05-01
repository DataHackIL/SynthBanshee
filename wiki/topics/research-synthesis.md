---
schema_version: '1'
kind: topic
title: Research Synthesis — Cross-Referenced Findings from Three Independent Reports
page_id: topic-research-synthesis
status: active
review_state: human-reviewed
source_refs:
  - src-d7392e10f92c3d658171300d2475c96103de715667bc17b429910da9ba8cea40
  - src-0032ce484cf3a12b55535ba22c39eba309352f6564765fe5f6c2c87431294996
  - src-5539b824c2a485f676fe6097f25b5de407229f8c05d3632e58860a815bb28963
tags: [research, synthesis, tts, prosody, preprocessing, room-acoustics, quality-gates]
---

# Research Synthesis — Cross-Referenced Findings

Three independent research reports (Gemini, GPT-5.2 thinking, GPT-5.5 Pro) were commissioned to investigate realistic Hebrew synthetic speech generation. This page consolidates findings where **two or more reports agree**, with specific parameter values and implementation priorities.

## Unanimous Findings (All 3 Reports Agree)

### 1. Remove Wiener Denoising on Clean TTS

All reports identify Wiener denoising as actively harmful on clean TTS output. It over-smooths high-frequency transients, removes perceptual sharpness, and contributes to "muffled" quality. Only apply to Tier B/C clips with added noise.

### 2. Remove or Soften 7.5 kHz Lowpass Filter

At 16 kHz sample rate (Nyquist 8 kHz), a 7.5 kHz order-4 Butterworth LPF removes the top 500 Hz of usable bandwidth. Hebrew sibilants (shin, samekh, tsadi) and breathiness cues live in this range.

**Recommended replacement:**
- GPT-5.2: gentle lowpass at 7.8 kHz (6 dB/oct) or no LPF
- GPT-5.5: no default LPF; use high-shelf -1 to -3 dB above 6.5 kHz if needed
- Gemini: replace with 100 Hz HPF + gentle codec simulation
- **Consensus:** Remove the current LPF entirely. Add HPF at 80-120 Hz for DC/rumble removal.

### 3. Anger Prosody: Range + Intensity, NOT Pitch Rise

Critical Hebrew-specific finding from Amir et al. (2003): F0 mean is the strongest anger indicator, but F0 *range* exhibits a **negative correlation** with anger intensity. The pitch floor rises faster than the ceiling, creating a "constricted" but high-pitched delivery.

**Do:**
- Keep male mean F0 stable (+0 to +5%)
- Increase F0 range multiplier [1.3, 1.8]
- Increase rate (+4% to +12%)
- Increase volume/intensity (+6 to +15 dB for hot anger)
- Use flatter spectral tilt (more high-frequency energy)

**Don't:**
- Raise pitch globally for male AGG speakers
- Expand F0 range (it should narrow with real anger)
- Use `mstts:express-as style="angry"` (causes identity shifts)

### 4. Add 10ms Crossfades at Turn Boundaries

All reports identify DC-offset jumps at SSML block boundaries as the cause of click/pop artifacts. Fix: apply 5-20ms crossfade (consensus: 10ms = 160 samples at 16 kHz) between every turn.

### 5. express-as NOT Supported for Hebrew Azure Voices

Azure `mstts:express-as` styles are either unavailable or "blended" for he-IL voices, causing timbre/identity shifts. All reports recommend abandoning express-as in favor of `<prosody>` tags with explicit rate, pitch, volume, and contour control.

**Recommended config:** `use_express_as: false`

## Hebrew F0 Reference Ranges

All reports converge on these ranges (sourced from Gelfer 2005, Israeli transgender voice study):

| Gender | Mean F0 (Hz) | F0 SD (Hz) | Typical F1 (Hz) |
|--------|-------------|------------|-----------------|
| Male | 100-146 | ~15 | 400-600 |
| Female | 188-221 | ~30 | 500-800 |
| Boundary/Neutral | 155-180 | - | - |

**Quality gate guardrails:**
- Male: reject if sustained median F0 outside [80, 180] Hz
- Female: reject if sustained median F0 outside [150, 290] Hz
- Max unexplained F0 drift: 2.0 semitones across clip

## Hebrew Speaking Rate (T-RES + Amir)

| Emotion | Mean F0 (Norm.) | F0 Range (Norm.) | Speech Rate (Syll/Sec) |
|---------|----------------|------------------|----------------------|
| Anger | 2.47 | 3.21 | 4.95 |
| Happiness | 2.11 | 2.41 | 4.14 |
| Sadness | 1.33 | 2.85 | 3.77 |
| Neutral | 0.53 | 1.53 | 4.41 |

Professional Hebrew newscaster baseline: 5.90 syll/sec; articulation rate 6.42 syll/sec (Amir).

**Implication:** Current I1-I2 rate is below 4.5 SPS, which sounds unnaturally slow.

## SSML Prosody Parameters (Consensus Values)

| Parameter | Calm/I1 | Neutral/I2 | Irritation/I3 | Anger/I4 | Shouting/I5 |
|-----------|---------|-----------|---------------|----------|-------------|
| Rate | -6% to +2% | 0% to +4% | +2% to +8% | +4% to +12% | +8% to +15% |
| Pitch (male) | -3% to +2% | -2% to +2% | -1% to +3% | -1% to +5% | 0% to +5% |
| Pitch (female) | -3% to +2% | -2% to +3% | 0% to +5% | +2% to +7% | +4% to +10% |
| Volume | baseline | baseline | +2 to +4 dB | +6 to +10 dB | +10 to +15 dB |
| F0 Range | -10% to 0% | baseline | +10% to +25% | +25% to +50% | +35% to +60% |

**Break timing:**
- After commas: 150ms
- Between sentences: 180-350ms
- Between paragraphs/major boundaries: 400-700ms

## Room Acoustics Parameters

| Environment | RT60 (s) | Source-Mic Distance (m) | DRR (dB) |
|-------------|---------|------------------------|----------|
| Treated office | 0.20-0.35 | 0.5-1.0 | +3 to +10 |
| Apartment/furnished | 0.30-0.60 | 1.0-2.5 | -2 to +6 |
| Kitchen/hard surfaces | 0.45-0.90 | 2.0-4.0 | -8 to 0 |

**Pyroomacoustics config:**
- Shoebox dimensions: [3-6, 3-8, 2.4-3.0] m
- Absorption: frequency-dependent material (alpha ~0.15 for typical)
- max_order: 3-10 (hybrid ISM+ray tracing)
- RIR source mix: 65% real, 35% simulated

**Phone-on-table early reflection:**
- Delay: 0.6-1.2ms (~10-20 samples at 16 kHz)
- Attenuation: -6 to -10 dB relative to direct
- LPF on reflected copy: 4-6 kHz

## SNR Distribution

| Scenario | SNR (dB) | Profile |
|----------|---------|---------|
| Phone on table | 15-35 | Room reverb + table reflections |
| Phone in pocket | 0-15 | Low-frequency rubbing, HF attenuation |
| Quiet room | 25-40 | Faint HVAC, low room tone |
| Office | 10-25 | Keyboard, HVAC, distant talk |

**Recommended training distribution:**
- 50% clips: 18-30 dB (normal)
- 30% clips: 10-18 dB (noisy)
- 10% clips: 5-10 dB (very noisy)
- 10% clips: 30-40 dB (nearly clean)

## Phone/Device EQ (Wideband at 16 kHz)

- HPF: 80-120 Hz (2nd order) — rumble removal
- Low-mid cut: -1 to -4 dB around 250-400 Hz if muddy
- Presence boost: +2 to +4 dB around 2.5-3.5 kHz (Q ~0.8-1.2)
- High shelf: -1 to -3 dB above 6.5 kHz (gentle, NOT brick-wall)
- Optional: 1-3 comb/notch filters, Q 3-12, depth -1 to -5 dB between 500 Hz and 5 kHz

## Codec Simulation

| Codec | Bandwidth | Bitrate | Use Case |
|-------|----------|---------|----------|
| AMR-NB | 300-3400 Hz @ 8 kHz | 4.75-12.2 kbps | Classic "phone muffled" |
| AMR-WB | 50-7000 Hz @ 16 kHz | 6.6-23.85 kbps | HD Voice / cellular |
| Opus 24k | Wideband | 24 kbps | WhatsApp/VoIP |
| Opus 12k | Narrowband | 12 kbps | Low-quality VoIP |

**Recommended distribution:** 80% no codec; 20% with codec (50% Opus 24k, 25% Opus 12k, 15% AMR-NB, 10% GSM).

## Quality Gates

### Turn-Level (After TTS Render)

| Metric | Reject/Flag Threshold |
|--------|----------------------|
| Max sustained voiced segment | >2.8 s |
| Peak level | >-0.1 dBFS |
| Male median F0 | outside [80, 180] Hz |
| Female median F0 | outside [150, 290] Hz |
| F0 range | <2 st flag; >18 st flag |
| Duration | >30 s per turn |
| UTMOS score | <3.5 |

### Clip-Level (After Post-Processing)

| Metric | Target |
|--------|--------|
| Integrated loudness | -26 to -18 LUFS |
| True peak | <= -1 dBTP |
| Speech activity ratio | 35-85% |
| Speaker embedding cosine | >0.74 per speaker |
| Silence ratio | <10% absolute silence below -60 dBFS |
| Spectral anomaly | No constant energy >2 s |

## Recommended Tools

### Must-Have
- **Parselmouth** (Praat wrapper) — jitter/shimmer injection, F0 analysis
- **pyroomacoustics** — room IR simulation
- **audiomentations** — CPU augmentation (noise, RIR, gain)
- **UTMOS** — synthetic speech MOS predictor (quality gate)
- **audio-slicer** — silence detection, threshold -40 dB

### RIR Databases
- **BUT ReverbDB** — 8 real rooms (small/medium/large)
- **OpenSLR SLR28** — RIR + noise database
- **EchoThief** — diverse non-traditional spaces
- **dEchorate** — annotated multichannel RIRs

### Noise Databases
- **MUSAN** — music, speech, noise
- **DEMAND** — diverse environments
- **FSD50K** — Freesound dataset

### Hebrew-Specific
- **HebTTS** — Hebrew TTS toolkit
- **Phonikud** — Hebrew G2P
- **SASPEECH** — 30h single-speaker Hebrew @ 44.1 kHz (OpenSLR/134)
- **ivrit.ai** — large Hebrew speech dataset

## Key Papers

1. **Amir et al. (2003)** — "Characteristics of authentic anger in Hebrew speech" — F0 range narrows with anger intensity
2. **Silber-Varod et al. (2016)** — Acoustic correlates of lexical stress in Israeli Hebrew — duration is dominant stress marker
3. **T-RES cross-linguistic validation** — Gold standard for Hebrew emotional prosody parameters
4. **Bat-El, Cohen & Silber-Varod (2019)** — Modern Hebrew stress: 75-85% iambic (word-final)
5. **Gelfer (2005)** — Speaking F0 ranges: male 100-146 Hz, female 188-221 Hz
6. **Banse & Scherer (1996)** — Acoustic profiles in vocal emotion expression
7. **Gobl & Ni Chasaide (2003)** — Voice quality in communicating emotion

## Priority Implementation Order

### Phase 0: Stop Degrading Quality (Immediate)
1. Remove Wiener denoising on clean TTS
2. Remove 7.5 kHz LPF; add 80 Hz HPF
3. Stop using pitch-level shifts for male anger; switch to range + intensity
4. Set `use_express_as: false` for Hebrew voices

### Phase 1: Fix Core TTS (High Impact)
5. Implement per-turn synthesis with crossfade assembly (10ms)
6. Add SSML pacing rules using consensus values above
7. Reset pitch state per turn: `<prosody pitch="default">`
8. Add F0/duration/sustained-vowel quality gates

### Phase 2: Add Realism (Medium Effort)
9. Room acoustics via pyroomacoustics (RT60 0.25-0.7s)
10. Phone-on-table early reflection model
11. Device EQ (HPF + presence boost + gentle high shelf)
12. Background noise at realistic SNR distribution

### Phase 3: Polish (Higher Effort)
13. Codec simulation (Opus/AMR-NB/AMR-WB)
14. Jitter/shimmer injection via Parselmouth
15. Multi-candidate generation (3+ per turn) with UTMOS ranking
16. Speaker embedding consistency gate (ECAPA-TDNN)
