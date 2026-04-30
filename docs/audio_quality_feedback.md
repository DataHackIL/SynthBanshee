# Audio Quality Feedback Log

Human listening feedback on generated clips. Used to track issues and drive improvements.

---

## Session: 2026-04-30 — First batch generation (Tier A, She-Proves)

**Listener:** Shay (project lead)
**Clips reviewed:** 6 clips from `/avdp-synth-corpus/data/he/`

### sp_sv_a_0001_00.wav (Situational Violence, Tier A)

- **Pitch too high** for both speakers.
- **Male voice pitch goes UP when angry** — should go down or stay stable with more intensity coming from rate/volume, not pitch.
- **Upward pitch drift** across the clip: male pitch at last utterance is ridiculously higher than at the first. This is a persistent issue.
- **16-second stuck sound** around 1:00 — male speaker produces "ka mehhhhh" where the "eh" sound sustains for ~16 seconds like a broken robot. This is a TTS rendering artifact, possibly from SSML prosody tags or a malformed phoneme.
- **Volume variation is good** — intensity-driven loudness differences are working.
- **Intonation and emotion are much better** than previous attempts.

### sp_neu_a_0001_00.wav (Neutral, Tier A)

- **Female pitch unrealistically low** for the character.
- **Speech too slow and overly paced** — sounds robotic/deliberate rather than natural.
- **Text content is great** — script generation is working well.
- **Muffled sound** — everything sounds like it's behind a wall or through a bad codec.
- **Loud click at end of each sentence** — likely a TTS artifact or SSML boundary issue.

### sp_neu_a_0002_00.wav (Neutral, Tier A, agg_m_30-45_003)

- **Too slow.** Pacing is still unnatural.
- **Muffled sound** persists.
- **Female voice changes between sentences** — inconsistent voice identity, possibly from different TTS style/prosody settings per turn.
- **Sounds like two people next to a muffled mic**, not a realistic room acoustic. Missing the "phone on a table in a room" spatial feel.

### sp_neu_a_0000_00.wav, sp_neu_a_000_00.wav, sp_neu_a_0002_00.wav (agg_m_30-45_001)

- **Just 5-second beep sounds** — these are stub/failed clips from earlier runs, not real audio. Need to be cleaned up.

---

## Recurring Issues (Priority Order)

### P0 — Broken clips
1. **16-second stuck TTS sound** — "ka mehhhhh" sustain. Must investigate SSML generation for this turn. Possibly a prosody rate/duration tag causing the TTS to hold a phoneme.
2. **Stub/beep clips** in the output directory from failed runs. Need cleanup and prevention.

### P1 — Pitch & Prosody
3. **Male pitch goes UP with anger** — design doc says distress via rate/timing, not pitch. The SSML prosody pitch multiplier for high-intensity AGG turns may be too aggressive or going in the wrong direction.
4. **Upward pitch drift across clip** — `SpeakerState` intensity carry-over may be accumulating pitch shifts instead of resetting or capping.
5. **Female pitch unrealistically low** in some clips.
6. **Both pitches too high** in SV clip.

### P2 — Naturalness
7. **Speech too slow** — TTS rate parameters may need adjustment, especially for low-intensity turns.
8. **Muffled sound** — likely from the Wiener denoiser or lowpass filter in preprocessing. The 7500 Hz lowpass is aggressive for speech clarity.
9. **Loud clicks at sentence boundaries** — TTS concatenation artifact or SSML boundary issue.
10. **Voice identity changes between sentences** — inconsistent `express-as` style settings or prosody parameters causing the TTS model to shift voice characteristics.

### P3 — Spatial Realism
11. **No room acoustic feel** — Tier A is clean (no augmentation), but even clean clips should sound like they were recorded in a space, not in a studio. This is by design for Tier A, but the muffled quality makes it worse.

---

## What's Working Well

- **Script content** — Hebrew dialogue is natural and contextually appropriate.
- **Volume/loudness variation** — intensity-driven RMS differences are perceptible and good.
- **Emotion and intonation** — noticeably improved over previous iterations.
- **Pipeline reliability** — all stages complete, validation passes, labels are correct.
