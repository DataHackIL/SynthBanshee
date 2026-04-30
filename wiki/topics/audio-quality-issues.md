---
schema_version: '1'
kind: topic
title: Audio Quality Issues — Known Problems and Root Causes
page_id: topic-audio-quality-issues
status: active
review_state: human-authored
source_refs:
  - src-b98ddf4495b0916b6c9cdc3804daf32554f40b26dc0ba217c5e3d1664ee5ddf4
  - src-42711de5f7b0e5a5b70b02d8fcd01097a6901aa5b5d14d61b351b6bbbd94ab89
tags: [audio-quality, tts, ssml, preprocessing, feedback]
created: '2026-04-30'
updated: '2026-04-30'
---

# Audio Quality Issues — Known Problems and Root Causes

Consolidated tracker for audio quality problems observed during human listening review, with hypothesized root causes and investigation status.

## P0 — Broken Output

### 16-Second TTS Sustain ("ka mehhhhh")

- **Clip:** sp_sv_a_0001_00.wav, ~1:00 mark
- **Symptom:** Male speaker holds a single vowel sound for ~16 seconds
- **Hypothesized cause:** SSML `<prosody>` rate/duration tag causing Azure TTS to sustain a phoneme indefinitely. Possibly a `duration` attribute on a `<break>` or a rate multiplier near zero.
- **Investigation:** Need to dump the SSML sent to Azure for the failing turn and inspect.
- **Status:** Open

### Stub/Beep Clips

- **Clips:** sp_neu_a_0000_00, sp_neu_a_000_00, sp_neu_a_0002_00 (agg_m_30-45_001 dir)
- **Symptom:** 5-second beep sounds instead of speech
- **Cause:** Leftover from failed earlier generation runs; the files were written but never completed.
- **Fix:** Clean stale output directory before batch runs; add output-dir cleanup step to `generate-batch`.
- **Status:** Known, needs cleanup logic

## P1 — Pitch & Prosody

### Male Pitch Rises with Anger

- **Symptom:** AGG speaker pitch increases at I4–I5 instead of staying stable or dropping.
- **Design intent (§4.2a):** "distress via rate/timing, not pitch." VIC I4 median F0 < 250 Hz, VIC I5 < 250 Hz. AGG should convey aggression through rate increase and loudness, not pitch rise.
- **Root cause:** SSML `<prosody pitch="+X%">` applied per intensity level in `SSMLBuilder`. The pitch multiplier for AGG at high intensity is positive when it should be neutral or slightly negative.
- **File:** `synthbanshee/tts/ssml_builder.py` — look at `_intensity_to_prosody()` or equivalent.
- **Status:** Open — needs SSML parameter audit

### Upward Pitch Drift Across Clip

- **Symptom:** Male speaker's pitch at last utterance is much higher than first, creating an unrealistic progression.
- **Root cause:** `SpeakerState` (M7) carries intensity drift between turns. If the state accumulates pitch offsets without a ceiling, drift is unbounded.
- **File:** `synthbanshee/tts/speaker_state.py`
- **Status:** Open — needs state carry-over audit

### Female Pitch Unrealistically Low

- **Symptom:** VIC female voice has unnaturally low pitch in some NEU clips.
- **Possible cause:** SSML pitch offset for low-intensity VIC is negative, pulling pitch below the natural range. Or the base voice (he-IL-HilaNeural) has a lower register than expected.
- **Status:** Open

### Both Pitches Too High (SV clip)

- **Symptom:** Both AGG and VIC sound higher-pitched than natural speech.
- **Possible cause:** Base pitch offset applied globally, or `express-as` style interacting with prosody tags.
- **Status:** Open

## P2 — Naturalness

### Speech Too Slow

- **Symptom:** Pacing is deliberate and robotic, especially at low intensity.
- **Possible cause:** SSML `<prosody rate="...">` for I1–I2 is too conservative. Real baseline speech is faster than what we're generating.
- **Status:** Open — need to compare rate params against natural Hebrew speech rate (~4-5 syllables/sec)

### Muffled Sound

- **Symptom:** Everything sounds like it's behind a wall.
- **Root cause candidates:**
  1. **7500 Hz lowpass filter** in preprocessing — this is very aggressive; natural speech has significant energy up to 8-10 kHz. Cutting at 7.5 kHz removes breathiness, sibilants, and clarity.
  2. **Wiener denoiser** may be over-smoothing, especially on clean TTS output that has no real noise to remove.
  3. **16 kHz sample rate** — Nyquist is 8 kHz, so the 7.5 kHz lowpass is nearly at the limit.
- **File:** `synthbanshee/augment/preprocessing.py`
- **Status:** Open — **most likely the 7500 Hz lowpass is the primary culprit**

### Loud Clicks at Sentence Boundaries

- **Symptom:** Audible click/pop at the end of each turn.
- **Possible cause:** TTS audio segments not properly crossfaded or zero-padded at boundaries. Or the mixer concatenation creates discontinuities.
- **File:** `synthbanshee/tts/mixer.py` — look at segment joining logic
- **Status:** Open

### Voice Identity Changes Between Sentences

- **Symptom:** Same speaker sounds like a different person across turns.
- **Possible cause:** `express-as` style changes per turn (e.g., "calm" → "angry" → "sad") cause the Azure voice model to shift timbre. Azure's `mstts:express-as` can dramatically change voice quality.
- **Status:** Open — may be inherent to the style-switching approach

## P3 — Spatial Realism

### No Room Acoustic Feel

- **Symptom:** Sounds like studio recording, not a phone on a table in a room.
- **Note:** Tier A is intentionally clean (no augmentation). Room acoustics are Tier B (pyroomacoustics). But the muffled quality from preprocessing makes even clean clips sound artificial.
- **Status:** Expected for Tier A; Tier B should address this.

## Research Questions

1. What are the actual SSML `<prosody>` parameters being sent for each intensity level? Need a dump.
2. What does the `SpeakerState` pitch carry-over curve look like across a 12-turn clip?
3. Would removing the 7500 Hz lowpass (or raising to spec max of 8 kHz) fix muffled quality without violating the 16 kHz sample rate constraint?
4. Can we crossfade turn boundaries in the mixer to eliminate clicks?
5. Is the Wiener denoiser doing more harm than good on clean TTS output?
