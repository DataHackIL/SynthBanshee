# Audio Generation V2 — Design Update and Implementation Plan

**Status:** Draft — for review before finalizing design/spec/implementation docs
**Date:** 2026-04-13
**Context:** Based on four independent reviews of clip `debug_run_1`: user (Shay), Gemini, Claude, and ChatGPT.
**Trustworthiness note:** Where Shay's feedback conflicts with an LLM review, Shay's assessment is authoritative.

---

## 1. Executive Summary: What Is Wrong and Why It Matters

The current TTS pipeline produces audio that is acoustically unsuitable as primary training data for intensity I3–I5 domestic violence detection. The audio is acceptable as a smoke-test artifact and as a structural template, but has at least eight distinct failure modes that will teach a downstream classifier the wrong cues.

All four reviewers converged on the same root cause: **the pipeline treats each utterance as an independent TTS call with per-utterance prosody parameters, and concatenates the results with fixed silence gaps**. This means:

- Speaker state resets at every turn boundary.
- Emotional escalation is expressed only through scripted SSML parameters, not through the phonation-quality, rate-instability, and amplitude-discontinuity features that real escalating confrontations produce.
- The turn-taking pattern has no conversational logic — no interruption, no overlap, no psychologically-motivated timing.

The problem is not a single bug to fix. It requires changes across three layers: **text preprocessing**, **TTS synthesis control**, and **post-synthesis audio assembly**.

The good news: the script generation, label schema, pipeline structure, and clip metadata system are all sound and need no architectural change.

---

## 2. Findings by Priority (All Four Reviews Combined)

The following table consolidates all findings, ranked by harm to ML training quality. Where Shay identified a problem, his observation is marked **(S)**; LLM-measured values are included where they add quantitative grounding.

| # | Problem | Evidence | ML Harm |
|---|---------|----------|---------|
| 1 | **Near-zero amplitude variation across intensity levels** | 1.7 dB RMS range for AGG across I1–I5. Shouted vs. conversational speech differs by 15–20 dB. **(S)** "too slow, robotic twang, emotional tone lacking" all partly caused by this. | Classifier cannot learn loudness-escalation cues. These will be suppressed by peak normalization applied uniformly across a scene. |
| 2 | **VIC (HilaNeural) baseline F0 too high; rises into child-voice range** | Measured baseline ~215 Hz, rising to 285–287 Hz at I5. Adult female natural speech is 165–220 Hz; distress adds F0 *variability*, not just elevation. **(S)** "pitch is too high for a woman". | Model learns "high pitch = female distress" at ranges it will never see in real recordings. Systematic OOD at test time. |
| 3 | **No creaky/breathy phonation in VIC at high intensity** | HNR 0.31–0.41 (too clean/modal) at all intensities. Real distress produces aperiodic, breathy, or creaky phonation (HNR ~0.15–0.25). | Model cannot learn the modal→creaky voice-quality shift. HNR-based features are uninformative. |
| 4 | **Hebrew gender-agreement errors** | Azure TTS defaults to masculine inflection for unvocalized ambiguous forms. High-risk tokens in current scripts: **שלך, עשית, היית, חשבת, הלכת, תשמעי, תראי, יוצאת, נפגשת**. **(S)** heard directly. | Acoustic noise at relational lexical targets; unnatural enough to attract model attention as an artifact class. |
| 5 | **AGG F0 variance too narrow at I4–I5** | AGG F0 mean rises only to ~128–133 Hz at I5. Std ≤ 25 Hz. Real threatening speech carries large F0 excursions on key accusatory words. | Model underweights F0 variance as an aggressor cue. |
| 6 | **Uniform speech rate within turns; no burst-pause microstructure** | Male rate 1.5→2.9 words/sec (directionally correct). Female rate 1.2–1.6 words/sec across all intensities, too slow and too smooth. **(S)** "way too slow". | Rate-based features trained here will be too smooth to generalize. |
| 7 | **Non-motivated, mechanically clean inter-turn silence** | Turn gap range 0.3–1.5 s with no conversational logic. 1.5 s post-threat gap before VIC I5 reads as editorial pause, not stunned silence. | Model learns artificial turn-timing signatures as class features. |
| 8 | **No overlap or interruption at any intensity** | Every turn boundary is a clean silence. Real coercive confrontations have barge-in behavior at I3–I5. | Missing one of the strongest structural features distinguishing DV from ordinary conversation. |
| 9 | **Peak normalization erases within-scene loudness trajectory** | Current pipeline applies `−1.0 dBFS` peak normalize to the full scene after mixing. This collapses the loudness difference between I1 and I5 turns to approximately 2 dB. | Loudness escalation, one of the most reliable real-world DV cues, is trained away. |
| 10 | **Missing paralinguistic events in VIC** | No breath catches, sighs, swallows, or voice tremor. Only one filled-pause "אה..." at I2; absent thereafter. | Gap in victim-state features. Not actively misleading, but leaves model unable to learn distress-breathing markers. |
| 11 | **Metallic / overly-clean TTS artifacts** | Systematic harmonic smoothing artifacts, especially at extreme SSML parameters (I5 AGG, I4–5 VIC). **(S)** "metallic/robotic twang". | Model may learn TTS acoustic fingerprint rather than underlying prosodic features. Partly mitigated if training data is mixed with real speech. |

---

## 3. What Does NOT Need to Change

The following components are architecturally sound and require no redesign:

- **Script generation** (LLM + Jinja2 templates): produces structurally correct, semantically appropriate dialogue. The scripts are good as structural templates; the problem is in how they are rendered to audio.
- **Label schema and taxonomy**: the hierarchical `PHYS`/`VERB`/`DIST`/`EMOT`/`ACOU`/`NONE` taxonomy is appropriate.
- **Pipeline stage separation** (Script → TTS → Preprocess → Augment → Labels → Validate): the stage boundaries are correct.
- **Clip metadata schema** (`ClipMetadata`, `EventLabel`): sound.
- **TTS cache system** (SHA-256 of full SSML string): correct.
- **Acoustic augmentation** (room IR, device profiles, noise overlays): already planned for Tier B; still needed and architecturally correct.
- **Preprocessing stack** (resample, mono, Butterworth LP, Wiener denoise, silence pad): correct.
- **Validation and QA** (`validate_clip`, `qa-report`): correct, needs extension (see §7.3).

---

## 4. Architecture Changes Required

### 4.1 Text Pre-processing: Hebrew Disambiguation Layer

**Problem:** Azure TTS synthesizes unvocalized Hebrew text and defaults to masculine inflection when the written form is morphologically ambiguous. This is a hard-coded TTS behavior that cannot be corrected with SSML alone.

**Solution:** Insert a new pipeline stage between script generation and SSML construction. This stage rewrites known ambiguous tokens with niqqud or with phonetically unambiguous alternatives before the text reaches the SSML builder.

**New module:** `synthbanshee/script/hebrew_disambiguator.py`

Interface:
```python
def disambiguate_for_speaker(
    text: str,
    speaker_role: str,     # "AGG" or "VIC"
    addressee_role: str,   # "AGG" or "VIC"
) -> str:
```

Approach (in priority order):
1. **Rule-based niqqud insertion** for a lexicon of known high-risk second-person address forms that appear in AGG→VIC speech. Initial lexicon: `{שלך, עשית, היית, חשבת, הלכת, תשמעי, תראי, יוצאת, נפגשת, שלי (context-dep.)}`. Map each to its feminine-directed vocalized form.
2. **Morphological rewrite** for forms where niqqud is insufficient (e.g., full-form rewrites that are unambiguous by orthography: "היית" → "היית" with niqqud; "עשית" → "עשית" with niqqud for feminine-feminine address).
3. **Optional integration with Nakdan/Dicta API** for full-text niqqud on scripts that contain unanticipated forms. This is a quality improvement for later; the rule-based layer is the required step.

Where it sits in the pipeline: called inside `ScriptGenerator._post_process_turn()` (or equivalent) before turns are passed to `TTSRenderer`. The `DialogueTurn.text` field that enters the TTS stage must already be disambiguated.

**Risk:** Automated niqqud tools have ~5–10% error rates. The rule-based lexicon approach covers the 10–12 highest-frequency error sources and is safer than full automation. Human review of new scripts for ambiguous forms should remain in the QA checklist.

---

### 4.2 SSML Prosody Redesign: Per-Phrase Control and Volume Escalation

**Problem:** Current SSML applies a single rate/pitch/volume value to the entire utterance. This produces flat, uniform delivery. Real escalating speech has:
- acceleration bursts on key accusatory phrases
- deliberate low-rate deliveries on threat utterances ("עכשיו", "ברור?")
- volume spikes on specific words
- pre-pause breaks before and after impact words

**Solution:** Extend `SSMLBuilder` to support intra-utterance prosody markup. This requires:

1. **Script-level phrase annotations** — the LLM script template should optionally annotate specific phrases with prosody hints. These will be passed as structured data in `DialogueTurn` and used by the SSML builder to inject `<break>` and nested `<prosody>` tags.

2. **Intensity-to-SSML parameter table redesign** — the current `style_map` in speaker configs maps intensity→(style, rate, pitch, volume) as single values. This table must be expanded:
   - Add `volume_delta_db` that actually escalates (target: +0 dB at I1, +4 dB at I3, +8 dB at I4, +12 dB at I5 for AGG)
   - Lower VIC baseline pitch: target `pitch_delta_st` of `−3 st` to `−5 st` from current default to bring HilaNeural from ~215 Hz to ~185–195 Hz at I1
   - Cap VIC pitch escalation at I4–I5: cap delta at +2 st to prevent 285 Hz peaks; instead shift weight to rate-instability

3. **Per-phrase prosody injection** — `SSMLBuilder.build_from_speaker_config()` gains an optional `phrase_prosody: list[PhraseProsody]` parameter. Each `PhraseProsody` item specifies a text span and a local override (rate, pitch, break before/after). The builder wraps that span in a nested `<prosody>` element.

**New data type** (`synthbanshee/tts/ssml_types.py`):
```python
@dataclass
class PhraseProsody:
    text_span: str          # exact substring to match in text
    rate: str | None        # SSML rate value, e.g. "+15%" or "fast"
    pitch: str | None       # e.g. "+5st"
    volume: str | None      # e.g. "+6dB"
    break_before_ms: int = 0
    break_after_ms: int = 0
```

**Annotation source:** The LLM prompt template gains an optional `phrase_hints` field in the structured turn output. For high-intensity turns (I3–I5), the LLM is instructed to identify the 1–2 most emotionally loaded phrases and mark them. These are passed through the script pipeline and consumed by the SSML builder.

---

### 4.3 Stateful Cross-Turn Emotional Controller

**Problem:** Each turn is synthesized independently. The speaker's "physiological state" resets at every turn boundary. There is no representation of emotional carryover — a speaker who has been shouting for 30 seconds should sound different at the start of their next turn than at the start of the scene.

**Solution:** Introduce a lightweight `SpeakerState` object that is maintained across turns in `TTSRenderer.render_scene()`. It tracks:
- Current rate offset (accumulates as intensity rises)
- Current pitch-variance target (increases with intensity; affects randomization bounds)
- Current volume offset (accumulates)
- Current "breathiness proxy" level (for VIC; drives post-synthesis processing, see §4.4)

The state evolves via a simple transition rule applied before each turn: if this turn's intensity is higher than the previous turn by the same speaker, drift state variables toward the higher-intensity targets; if intensity drops, decay back slowly.

This is not a machine learning model — it is a deterministic state machine with configurable drift rates. It replaces the current stateless per-turn parameter lookup.

**New class** (`synthbanshee/tts/speaker_state.py`):
```python
@dataclass
class SpeakerState:
    intensity_history: list[int] = field(default_factory=list)
    rate_offset: float = 1.0
    pitch_offset_st: float = 0.0
    volume_offset_db: float = 0.0
    breathiness_level: float = 0.0   # 0.0 = modal; 1.0 = maximum breathiness

    def update(self, new_intensity: int, speaker_role: str) -> None:
        ...
```

The state's outputs feed into `render_utterance()` as additional offsets that stack on top of the SSML parameters derived from `style_map`.

---

### 4.4 Post-Synthesis Voice Quality Modification (Breathiness / Phonation Texture)

**Problem:** HilaNeural has HNR 0.31–0.41 at all intensities — clean, modal phonation throughout. Real victim distress should shift voice quality toward breathy/creaky (HNR ~0.15–0.25 under threat).

Azure SSML cannot control phonation quality directly (no breathiness or creak style tag for he-IL voices). This must be done as a post-synthesis audio transformation.

**Solution:** New module `synthbanshee/augment/voice_texture.py` with a `add_breathiness()` function:
```python
def add_breathiness(
    wav_array: np.ndarray,
    sample_rate: int,
    level: float,    # 0.0 = no effect; 1.0 = maximum
) -> np.ndarray:
```

Implementation: apply band-pass filtered white noise (approximately 3–8 kHz, −30 dBFS scaled by `level`) mixed into the signal. This simulates the spectral-tilt change associated with breathy phonation and reduces the effective harmonic ratio.

This function is called inside `TTSRenderer.render_scene()` for VIC turns at I3 and above, with `level` derived from `SpeakerState.breathiness_level`. It runs after the raw WAV bytes are decoded from the TTS response but before the segment is passed to the mixer.

**AGG post-processing (optional, deferred):** Consider adding a mild low-frequency emphasis filter (+2–3 dB below 300 Hz) for AGG I4–I5 to add "chest register" weight to threatening speech. Lower priority than VIC breathiness.

---

### 4.5 Inter-Turn Timing Redesign

**Problem:** Current pause values in scene configs are either fixed or drawn from a simple uniform range with no conversational logic. This produces the observed mechanical gap pattern (0.3–1.5 s, no relationship to intensity, conversational role, or turn context).

**Solution:** Replace the single `pause_before_s` scalar in `DialogueTurn` with a structured `TurnGap` computed by a new `TurnGapController` that is called during scene assembly, not during script generation.

**Psychologically-motivated gap table:**

| Context | Suggested range | Rationale |
|---------|----------------|-----------|
| VIC responding to I1–I2 AGG turn | 300–600 ms | Normal conversational latency |
| VIC responding to I3 AGG accusation | 150–350 ms | Defensive, quick self-justification |
| VIC responding to I4 command | 50–150 ms | Compliance, fear-driven immediacy |
| VIC responding to I5 threat | 100–300 ms | Shock; not 1.5 s editorial pause |
| AGG responding to VIC at I1–I2 | 200–500 ms | Normal |
| AGG responding at I3–I5 (escalating) | 50–200 ms | Aggressor cuts in; loading for next volley |
| AGG after own previous I4–I5 turn | 800–1500 ms | Deliberate threatening pause before next statement |

The controller draws from these ranges using a seeded RNG for reproducibility, conditioned on `(current_speaker_role, prev_speaker_role, current_intensity, prev_intensity)`.

**New class** (`synthbanshee/tts/gap_controller.py`):
```python
class TurnGapController:
    def gap_seconds(
        self,
        current_turn: DialogueTurn,
        prev_turn: DialogueTurn | None,
        rng: random.Random,
    ) -> float:
        ...
```

---

### 4.6 Overlap and Interruption Simulation

**Problem:** All turn boundaries are clean silence gaps. Real coercive confrontations have barge-in behavior (AGG cuts off VIC) and anticipatory response starts (VIC responds before AGG finishes).

**Solution:** Extend `SceneMixer` to support two new mixing patterns in addition to the current sequential concatenation:

1. **Overlap segment**: begin the next speaker's turn `overlap_ms` before the previous turn ends. Both signals are present simultaneously in the mix for that window. The overlap region is typically 100–400 ms.

2. **Barge-in hard cut**: the previous speaker's audio is cut off (amplitude fade to silence over 50 ms) at the barge-in point; the next speaker's audio starts immediately.

A new `MixMode` enum in `SceneMixer`:
```python
class MixMode(Enum):
    SEQUENTIAL = "sequential"    # current behavior
    OVERLAP = "overlap"          # next starts before prev ends
    BARGE_IN = "barge_in"        # prev is cut off; next starts over
```

`TurnGapController` returns both a gap value and a `MixMode`. High-intensity transitions use `BARGE_IN` probabilistically (e.g., 30% of AGG→VIC transitions at I4+, 50% at I5).

This is the most structurally invasive change and should be treated as a separate milestone.

---

### 4.7 Normalization Strategy Change

**Problem:** Peak normalization applied to the full mixed scene after assembly erases within-scene loudness trajectory. A scene that starts at −20 dBFS and escalates to −5 dBFS will be compressed to −1 dBFS at the peak, collapsing the escalation signal.

**Solution:** Change the normalization approach in `preprocess()`:

- **Remove scene-level peak normalization** from the preprocessing step.
- **Add per-turn RMS targeting** at the mixing stage: each turn's gain is set such that its RMS is at a target level for its intensity (e.g., I1 target −24 dBFS, I2 −22 dBFS, I3 −20 dBFS, I4 −17 dBFS, I5 −14 dBFS for AGG). The VIC scale is shifted down by approximately 2 dB.
- **Apply true-peak limiting** (not normalization) at −1.0 dBFS after mixing to ensure the clip is technically compliant without destroying the loudness trajectory.

This means `PreprocessingResult.peak_dbfs` will reflect the result of the limiter, not a normalization operation. The clip will be spec-compliant (no clipping) but the within-scene dynamic range will be preserved.

The spec currently requires "−1.0 dBFS peak normalized." This constraint must be updated in `docs/spec.md` to "peak-limited to ≤ −1.0 dBFS (true-peak); per-turn RMS targeting applied at mix stage." This is a spec change that requires updating `validate_audio()` accordingly.

---

## 5. What Stays in the Current Architecture (Confirmation)

| Component | Verdict |
|-----------|---------|
| `ScriptGenerator` + Jinja2 templates | Keep as-is. Output quality is good. |
| `TTSRenderer.render_utterance()` API | Keep. Extend with `SpeakerState` input and `phrase_prosody` parameter. |
| Azure provider (`AzureProvider`) | Keep. No provider change needed for V2. |
| TTS cache (SHA-256 of SSML) | Keep. Cache will miss on any SSML change, which is correct. |
| `SceneMixer` sequential path | Keep as default. Add overlap/barge-in as opt-in. |
| Preprocessing pipeline | Modify normalization only (§4.7). Everything else stays. |
| `preprocess()` → `PreprocessingResult` API | Keep. Extend with per-turn loudness metadata. |
| Acoustic augmentation (Tier B) | Keep. Runs after the V2 audio is assembled. |
| Label generation | Keep. No schema changes. |
| Metadata schema (`ClipMetadata`) | Keep. Add optional `audio_v2` flag for tracking. |
| Validator (`validate_clip`) | Keep. Update `validate_audio()` for new normalization spec. |
| QA report | Keep. Extend with acoustic metrics (§5.3 below). |
| Split strategy (speaker-disjoint) | Keep. |

---

## 6. Implementation Plan — Milestones

The milestones are sequenced so that each builds on the previous and produces testable audio at each step. They are deliberately ordered by impact-per-implementation-cost, not by component type.

---

### Milestone 1: Hebrew Gender Disambiguation Layer
**Expected impact:** Eliminates gender errors heard directly by Shay; removes lexical/acoustic noise at relational words.
**Dependencies:** None. Can be done before any other milestone.
**Scope:**
- Create `synthbanshee/script/hebrew_disambiguator.py`
- Implement `disambiguate_for_speaker(text, speaker_role, addressee_role) → str`
- Build initial lexicon of ~15 high-risk second-person forms with their feminine-directed niqqud equivalents
- Wire into script pipeline: apply after `ScriptGenerator.generate()` returns turns, before turns reach `TTSRenderer`
- Unit tests: cover each lexicon entry; test that non-listed words are passed through unchanged
- Integration test: generate a script, verify none of the high-risk tokens appear in unvocalized form in the AGG→VIC speech

**Deliverable:** All AGG→VIC second-person forms are correctly gendered in synthesized audio.

---

### Milestone 2: SSML Prosody Parameter Redesign
**Expected impact:** Reduces VIC baseline pitch to natural adult female range; introduces meaningful amplitude escalation across intensity levels; adds within-utterance prosody variation for high-intensity AGG turns.
**Dependencies:** None for the SSML parameter changes. `PhraseProsody` annotation requires LLM template update.
**Scope:**

**2a — Parameter table update (speaker configs):**
- Update all speaker YAML configs: revise `style_map` entries to:
  - AGG: volume increases by +4 dB per intensity level above I2 (absolute target: I5 = +12 dB over I1)
  - VIC: add `pitch_delta_st: -3` baseline offset to bring HilaNeural F0 to ~185–195 Hz; cap pitch escalation at +2 st max above I3
  - Both: rate variation becomes intensity-linear rather than discrete steps
- Re-run a test clip and measure F0 and RMS per turn to confirm targets are met

**2b — Per-phrase SSML injection:**
- Add `PhraseProsody` dataclass to `synthbanshee/tts/ssml_types.py`
- Extend `SSMLBuilder.build_from_speaker_config()` with `phrase_prosody: list[PhraseProsody] | None`
- Builder logic: for each `PhraseProsody`, find the text span in the full utterance and wrap it in a nested `<prosody>` element with optional `<break>` elements
- Add `phrase_prosody: list[PhraseProsody] = []` field to `DialogueTurn`
- Update LLM prompt templates (Jinja2) to instruct the LLM to optionally mark 1–2 emotionally loaded phrases per turn at I3–I5 with a JSON field `phrase_hints: [{text: ..., hint: "stress"|"slow"|"break_before"}, ...]`
- Map LLM hints to `PhraseProsody` objects in `ScriptGenerator._post_process_turn()`

**2c — Validation:**
- Measure F0 and RMS of new clip per turn; confirm AGG I5 is ≥ 8 dB louder than I1; VIC baseline F0 ≤ 200 Hz

**Deliverable:** Re-generated clip with measurably wider loudness range and lower VIC baseline pitch.

---

### Milestone 3: Normalization Strategy Change
**Expected impact:** Preserves the within-scene loudness trajectory established in M2 from being erased at post-processing.
**Dependencies:** M2 (the loudness gains need to exist before we can preserve them).
**Scope:**
- Add `per_turn_rms_target_db: dict[int, float]` to `SpeakerConfig` (intensity → RMS target in dBFS)
- Apply per-turn RMS gain at mix stage in `SceneMixer.mix_sequential()`: before inserting each segment, scale it to its target RMS
- Replace `preprocess()` peak normalization with a true-peak limiter (threshold −1.0 dBFS, fast attack, no lookahead needed at these levels; scipy can implement this as a simple clip + gentle slope)
- Update `validate_audio()` to check that the clip's true peak is ≤ −1.0 dBFS (not that it is exactly −1.0 dBFS)
- Update `docs/spec.md` §3 to reflect the new normalization spec

**Deliverable:** Generated clips have measurable loudness escalation from I1 to I5 with no normalization-induced compression.

---

### Milestone 4: Stateful Cross-Turn Emotional Controller
**Expected impact:** Speaker's acoustic state carries over across turns; eliminates the "state reset" artifact at every turn boundary.
**Dependencies:** M2 (new parameter baseline), M3 (loudness preservation).
**Scope:**
- Create `synthbanshee/tts/speaker_state.py` with `SpeakerState` dataclass and `update()` method
- Transition rules: if current intensity > previous intensity by same speaker, drift rate/pitch/volume offsets toward higher-intensity targets by 50–70% of the gap; if intensity drops, decay at 30% per turn
- Wire into `TTSRenderer.render_scene()`: maintain one `SpeakerState` per speaker_id across the turn loop; before calling `render_utterance()`, apply state offsets to rate/pitch/volume
- Unit tests: verify state evolution across a 5-turn sequence with known intensity pattern

**Deliverable:** Clip in which a speaker's prosody at the start of a turn is influenced by their preceding turns, not just the current turn's intensity level.

---

### Milestone 5: Post-Synthesis Voice Quality Modification (Breathiness)
**Expected impact:** VIC I3–I5 audio sounds like genuine distress rather than elevated-pitch neutral speech; reduces HNR from ~0.35 to ~0.20 in those turns.
**Dependencies:** M4 (state controller provides `breathiness_level`).
**Scope:**
- Create `synthbanshee/augment/voice_texture.py` with `add_breathiness(wav_array, sample_rate, level) → np.ndarray`
- Implementation: bandpass filter (3–8 kHz, scipy Butterworth) applied to white noise; mix at level-dependent amplitude (approximately −30 dBFS at level=0.5)
- Wire into `TTSRenderer.render_scene()` for VIC speaker: decode WAV bytes to numpy array after TTS response, apply breathiness based on `SpeakerState.breathiness_level`, re-encode to WAV bytes before passing to mixer
- `breathiness_level` schedule: 0.0 at I1–I2; 0.3 at I3; 0.6 at I4; 0.9 at I5
- Verify HNR reduction using librosa in a standalone test

**Deliverable:** VIC turns at I3–I5 audibly exhibit breathier voice quality.

---

### Milestone 6: Inter-Turn Timing Redesign
**Expected impact:** Turn-taking pauses have conversational logic; eliminates 1.5 s post-threat pause and the mechanical timing pattern.
**Dependencies:** None (independent of audio content changes). Can run in parallel with M5.
**Scope:**
- Create `synthbanshee/tts/gap_controller.py` with `TurnGapController` class
- Implement gap table (§4.5) with configurable per-context ranges in a YAML config or as class constants
- Replace `DialogueTurn.pause_before_s` assignment in `ScriptGenerator.generate()` with a call to `TurnGapController.gap_seconds()` during scene assembly in `TTSRenderer.render_scene()`
- `pause_before_s` on `DialogueTurn` becomes a suggested value from the LLM; the controller overrides it based on context
- Unit tests: verify that I4 command → VIC response gap is in [50, 150] ms; verify post-threat AGG self-pause is in [800, 1500] ms

**Deliverable:** Turn gap pattern that passes a timing-logic review (no mechanical 0.6 s uniform gaps).

---

### Milestone 7: Overlap and Interruption Simulation
**Expected impact:** High-intensity turns contain barge-in behavior; dialogue sounds like a shared acoustic scene rather than alternating independent utterances.
**Dependencies:** M6 (gap controller provides the gap/overlap decision).
**Scope:**
- Add `MixMode` enum to `synthbanshee/tts/mixer.py`
- Extend `SceneMixer.mix_sequential()` to handle `MixMode.OVERLAP` and `MixMode.BARGE_IN`
  - `OVERLAP`: sum the two audio arrays with a crossfade window at the overlap region
  - `BARGE_IN`: truncate the preceding turn at the barge-in point with a 50 ms amplitude ramp-down; start the new turn immediately
- `TurnGapController` returns `(gap_s: float, mix_mode: MixMode)`; negative gap implies overlap
- Update turn onset/offset metadata to account for overlap (the offset of the interrupted turn must reflect the actual end of audible audio)
- `MixedScene.turn_onsets_s` and `turn_offsets_s` must be updated to reflect actual audio boundaries, not nominal TTS durations
- Unit tests: verify that an overlapped segment produces correctly merged audio; verify onset/offset metadata is consistent

**Deliverable:** Generated clips at I4–I5 contain occasional barge-in and overlap behavior consistent with heated confrontation.

---

### Milestone 8: Extended QA Metrics
**Expected impact:** Enables automated detection of regressions in each of the above parameters; gives engineering team a per-clip dashboard.
**Dependencies:** None (can be developed in parallel with any milestone); most useful once M1–M7 produce improved clips.
**Scope:**
- Extend `synthbanshee/package/qa.py` with acoustic metric computation:
  - Median and std F0 by speaker by intensity level (using librosa pyin)
  - RMS energy by turn
  - Inter-turn gap distribution
  - Presence of any niqqud-less high-risk tokens (regex scan of transcript)
  - Overlap ratio (turns with `MixMode != SEQUENTIAL`)
  - Flag: VIC median F0 at I4–I5 > 250 Hz → warning
  - Flag: AGG RMS range (I5 − I1) < 6 dB → warning
  - Flag: any high-risk gender-ambiguous token present unvocalized in AGG speech → warning
- Add `acoustic_metrics` section to `QAReport`
- Extend `qa-report` CLI output to show per-speaker per-intensity F0 and RMS table

**Deliverable:** `qa-report` produces actionable acoustic metric output per clip; regressions are automatically flagged.

---

## 7. Deferred / Out of Scope for V2

The following improvements were raised in the reviews but are deferred due to higher implementation cost or dependency on external resources:

### 7.1 Paralinguistic Event Injection (Sighs, Breath Catches)
Requires either: (a) a library of real or TTS-generated short breath clips to be inserted at VIC turn openings, or (b) a vocoder-based approach that can add breath noise to synthesized audio without artifacts. Neither is trivially available for he-IL. Deferred to V3 or to the real-data pipeline.

### 7.2 Alternative TTS Voices / Multi-Voice Diversity
ChatGPT recommended adding multiple TTS voices to avoid model learning Azure voice identity. This is valid but requires:
- Evaluation of other he-IL voices (Eleven Labs, Coqui TTS with Hebrew fine-tuning)
- Integration testing with the existing SSML/cache pipeline
- Speaker config YAML updates for each new voice

This is planned for Phase 2 scale-up but is not a V2 redesign item.

### 7.3 Real Speech Mixing
Claude's review recommended mixing synthetic clips with 20% real Hebrew conversational speech to regularize feature distributions. This requires sourcing (CoSICorpus or equivalent), licensing, and integration work. Deferred; depends on actor recording pipeline (6–12 months out per project plan).

### 7.4 Nakdan/Dicta API Integration for Full-Text Niqqud
The rule-based disambiguation layer (M1) covers the top ~15 high-risk forms. Full-text niqqud via Nakdan/Dicta would catch unanticipated forms. Deferred to a follow-up after M1 is validated.

### 7.5 Custom Neural Voice Training
Azure Custom Neural Voice could be fine-tuned on actor speech to produce a more natural distress voice. This requires actor recording data, which is the real-data pipeline deliverable at Phase 2.

---

## 8. Sequencing and Dependencies

```
M1 (Hebrew disambiguation) ──────────────────────────────────────────┐
                                                                       │
M2 (SSML prosody redesign) ──► M3 (normalization) ──► M4 (state) ──► M7 (overlap)
                                                         │
                                                         └──► M5 (breathiness)

M6 (gap controller) ──────────────────────────────────────────────────┘

M8 (QA metrics) — parallel, most useful after M2–M5
```

Recommended order for minimum time to improved audio:
**M1 → M2a → M3 → M6 → M2b → M4 → M5 → M7 → M8**

M1 and M2a are the quickest wins with the highest combined impact. M6 is structurally independent and can be done at any point.

---

## 9. Open Questions for Review

The following design decisions require explicit confirmation before implementation begins:

1. **Spec change for normalization (§4.7):** The current spec requires "−1.0 dBFS peak normalized." The V2 approach changes this to "true-peak limited." Does the AI team have a preference for how loudness normalization is applied to the final dataset, or is the preservation of within-scene dynamic range acceptable?

2. **Niqqud lexicon validation:** The initial lexicon in M1 should be reviewed by a native Hebrew speaker before implementation. Who reviews this?

3. **LLM phrase annotation (M2b):** Adding `phrase_hints` to the LLM output JSON requires a prompt change and potentially a schema change in `DialogueTurn`. Confirm this is acceptable overhead (it adds ~50–100 tokens to each LLM response).

4. **Breathiness level calibration (M5):** The suggested schedule (level 0.3 at I3, 0.6 at I4, 0.9 at I5) is a starting estimate. It should be validated by listening tests and HNR measurement before being committed to speaker config YAMLs.

5. **Overlap probability (M7):** The suggested barge-in probability (30% at I4, 50% at I5) is a starting estimate. It should be validated against the label schema (overlapping turns affect event onset/offset timestamps and require careful `MixedScene` updates).

---

## 10. Summary Table

| Milestone | Primary benefit | Estimated effort | Spec change? |
|-----------|----------------|-----------------|--------------|
| M1 — Hebrew disambiguation | Eliminates gender errors | Small | No |
| M2a — SSML parameters | Lowers VIC F0; adds loudness escalation | Small | No |
| M2b — Per-phrase SSML | Burst-pause microstructure in AGG speech | Medium | No (new optional field) |
| M3 — Normalization strategy | Preserves loudness trajectory | Small | Yes — spec.md §3 |
| M4 — Stateful speaker controller | Cross-turn continuity | Medium | No |
| M5 — Voice breathiness | VIC distress phonation texture | Medium | No |
| M6 — Gap controller | Psychologically-motivated timing | Small | No |
| M7 — Overlap/barge-in | Interruption behavior in heated turns | Large | No (new MixMode) |
| M8 — QA metrics | Automated acoustic regression detection | Medium | No |
