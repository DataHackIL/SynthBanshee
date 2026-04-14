# Feedback on `audio_generation_v2_design.md` for Final Synthesis

## Executive Verdict

The V2 direction is materially better than the current pipeline and is pointed at the right root causes. It correctly identifies that the current failure is not “bad scripts” but a structural audio-generation problem: stateless per-turn TTS, sequential concatenation, and post-mix normalization flattening the very cues the downstream models need.

That said, the plan still has several important weaknesses that should be fixed before it is treated as the final design:

1. It is still too TTS-centric and not yet strong enough on domain randomization and voice diversity, which is one of the biggest protections against learning Azure-specific fingerprints.
2. It overcommits to some single-clip-derived numeric targets as if they were stable truth rather than initial calibration points.
3. Its proposed breathiness / phonation workaround is the riskiest part of the plan and should be framed as experimental, not as a core V2 pillar.
4. It under-specifies several metadata / QA / labeling implications of overlap, interruption, loudness targeting, and text disambiguation.
5. It is too eager to preserve parts of the current preprocessing stack that likely need to become configurable or tier-specific, not globally fixed.

My recommendation: keep the overall V2 structure, but revise it into a more explicit Audio Realism V2 design with three priorities:

- P0: fix clearly harmful deterministic artifacts
- P1: add stateful conversational dynamics and preserve escalation cues
- P2: widen diversity and observability so the model does not overfit to one synthesis engine or one cue family

## What V2 Gets Right

### 1. Correct diagnosis of the current root cause
The document is right that the main problem is not a single bug. The current repo really does synthesize each utterance independently, then concatenate them with silence, which guarantees speaker-state reset and non-conversational turn-taking. The current `TTSRenderer` is stateless at the utterance level, `SceneMixer` is sequential-only, and preprocessing peak-normalizes after the fact. That diagnosis is correct and should remain central.

### 2. Correct prioritization of loudness preservation
Changing normalization strategy is necessary. The current preprocessing pipeline peak-normalizes to `-1.0 dBFS`, and that is almost certainly collapsing meaningful within-scene escalation. Moving to “preserve intra-scene dynamics, then limit for compliance” is the right direction.

### 3. Correct separation of text, synthesis, and assembly layers
The plan properly treats Hebrew disambiguation, prosody control, and scene assembly as distinct problems. That separation is good architecture and should remain.

### 4. Correct emphasis on conversational timing and interruption
The absence of overlap / barge-in is a major realism problem. Adding timing logic and interruption simulation is not cosmetic; it is one of the most important improvements.

### 5. Correct instinct to extend QA beyond packaging validity
The current QA pipeline is mostly packaging/spec compliance. The proposed move toward acoustic QA is the right one.

## Main Problems in the Current V2 Plan

### 1. Voice diversity is still too deferred
This is the biggest strategic weakness in the V2 plan.

The plan acknowledges that models may learn Azure fingerprints, but then defers alternative voice diversity to a later phase. That is too late. Even if all prosody fixes land, training primarily on two Azure Hebrew voices remains dangerous.

#### What to change
Move multi-voice diversification forward from “deferred / Phase 2” into core V2. It does not need to block all other work, but it should be a first-class milestone.

Minimum acceptable V2 target:
- at least 3–5 male and 3–5 female synthetic voice variants
- ideally from more than one TTS backend
- if multi-backend is too expensive immediately, at least multiple Azure Hebrew voices plus structured perturbations

The final design should say explicitly that no model-training dataset intended for meaningful experiments should be generated from only one AGG voice and one VIC voice.

### 2. The proposed phonation fix is too brittle and too optimistic
The `add_breathiness()` proposal is the weakest technical component in the document.

Adding band-passed white noise to simulate breathiness may reduce HNR numerically, but it is not the same thing as human breathy or distressed phonation. It risks creating a synthetic artifact that is easy for the model to learn and that does not transfer to real victim speech.

#### What to change
Do not remove this idea entirely, but downgrade it from “core V2 milestone” to experimental branch / optional augmentation module.

Reframe it like this:
- Primary distress realism should come first from:
  - lower-level pitch calibration
  - rate instability
  - phrase shaping
  - dynamic loudness
  - pause compression
  - overlap / interruption
  - voice diversity
- Post-synthesis phonation modification should be optional and validated only if listening tests and classifier behavior show benefit.

The final design should explicitly warn that “HNR reduction alone is not proof of realistic distress.”

### 3. The plan is too confident in exact numeric targets from one debug clip
The V2 document contains several useful numbers, but some of them are treated too much like fixed truth rather than initial calibration from a single review artifact.

Examples:
- exact target dB ladders by intensity
- exact target F0 ranges by role
- exact breathiness schedules
- exact barge-in probabilities

These are fine as initial defaults, but not as final design truths.

#### What to change
Convert hard targets into:
- initial defaults
- QA warning thresholds
- tunable config values
- validated by listening plus distribution review

### 4. Phrase-level prosody via exact text-span matching is fragile
The proposed `PhraseProsody(text_span=...)` design is directionally good, but exact substring matching is fragile in this pipeline because text may change at multiple stages:
- LLM output
- disfluency insertion
- Hebrew disambiguation / niqqud insertion
- punctuation normalization
- later phrase rewriting

#### What to change
Do not make phrase prosody depend primarily on exact text-string matching.

Use one of these instead:
1. Character offsets derived after final text normalization
2. Phrase IDs attached during script generation
3. Structured span list stored on the turn object after final text preprocessing

Also: do not make the LLM the only source of phrase hints. For some turn classes, deterministic heuristics are better.

### 5. Hebrew disambiguation should preserve both original and spoken text
The current proposal is right to add a disambiguation layer, but it needs one important architectural addition.

For each turn, preserve:
- canonical/original text
- spoken/rendered text actually sent to TTS after normalization

If the TTS-facing text diverges from the canonical text, that must be visible in metadata.

Suggested fields:
- `text_original`
- `text_spoken`
- `text_normalization_applied`
- `normalization_rules_triggered`

### 6. Overlap / interruption changes require a stronger labeling redesign
Once turns overlap, there are at least three different time concepts:
- canonical scripted turn interval
- actual rendered audio interval
- audible dominance interval in the mixed waveform

At minimum preserve:
- `script_onset` / `script_offset`
- `rendered_onset` / `rendered_offset`
- `audible_end_if_interrupted`
- `mix_mode`

If barge-in truncates a speaker, the label generator must not keep pretending the full uncut utterance remained audible.

### 7. Current preprocessing should not be treated as universally sound
The V2 document says the current preprocessing stack is architecturally correct except for normalization. That is too generous.

The current pipeline always applies:
- resample
- mono
- low-pass
- Wiener denoise
- peak normalize
- silence pad

For clean synthetic TTS, unconditional low-pass filtering and Wiener denoising may smear or simplify the signal in ways that are not obviously beneficial.

#### Recommendation
Make preprocessing tier-aware and configurable:
- Tier A: minimal destructive preprocessing
- Tier B/C: scene-dependent processing
- post-mix compliance processing separate from realism augmentations

### 8. Emotional-state propagation is currently broken or underused
The current artifacts show a mismatch: the script/asset JSON has emotional-state values like `fearful`, `pleading`, `threatening`, but the strong-label JSONL for the same clip shows many turns as `neutral`. That is a data-consistency smell.

Add a direct design note:
- emotional-state fields must remain consistent across script, rendering metadata, and labels, or the schema should be simplified intentionally
- do not silently downgrade turn-level emotional state during packaging

### 9. The plan needs stronger run-level observability, not just clip-level QA
You also need run-level distribution QA.

Recommended additions:
- F0 distributions by role, voice, intensity, typology, and project
- RMS / LUFS distributions by role and intensity
- gap distributions and overlap ratios
- distribution of normalization rewrites / Hebrew disambiguation triggers
- phrase-prosody usage rates
- speaker diversity counts
- backend diversity counts
- fraction of clips with interruption
- fraction of clips with no measurable escalation
- outlier clips by each metric family

### 10. The plan should distinguish She-Proves and Elephant defaults much more aggressively
The two projects have different acoustic and interaction regimes.

She-Proves likely needs more:
- long stretches of low salience
- off-axis / distant capture
- domestic ambiences
- intermittent escalation
- partial audibility
- phone placement variation

Elephant likely needs more:
- front-facing / room-mic speech
- shorter escalation horizons
- different interruption structure
- more third-party voices
- clinic / office ambient confounders
- event-like alert conditions

The final design should explicitly say that timing, overlap, loudness, phrase-prosody, and augmentation defaults are project-specific, not global.

## Section-by-Section Feedback on the V2 Design

### Section 4.1: Hebrew disambiguation layer
Keep:
- dedicated module
- rule-based first
- lexicon-driven approach
- optional fuller-text approach later

Change:
- preserve both original and spoken text
- make normalization output auditable
- avoid making external Nakdan/Dicta a core-path dependency
- consider phoneme/lexicon override support if Azure allows it

### Section 4.2: SSML prosody redesign
Keep:
- moving beyond utterance-global prosody
- per-phrase controls
- lowering female baseline pitch
- adding actual loudness escalation

Change:
- do not rely on exact text matching
- do not assume loudness ladders should be the same across typologies or projects
- make phrase-prosody optional and partly heuristic
- avoid overfitting the whole design around HilaNeural / AvriNeural only

### Section 4.3: Stateful cross-turn emotional controller
This is one of the best parts of the plan.

Keep:
- per-speaker state
- drift rather than reset
- deterministic state machine
- state stacked over style-map defaults

Add:
- serialize state-related parameters into metadata for reproducibility
- separate speaker persistent baseline, scene-level state, and turn-level requested intensity
- optional scene archetype parameters such as “suppressed fear,” “explosive confrontation,” “cold coercion”

### Section 4.4: Post-synthesis voice-quality modification
Keep:
- acknowledging SSML limitations
- recognizing that voice quality matters, not just pitch and rate

Change:
- frame as experimental
- do not promise that noise injection equals realistic breathiness
- do not make HNR reduction the main success criterion

Add:
- listening-test acceptance gate
- classifier-side sanity check
- milder spectral-envelope variation before explicit noise addition

### Section 4.5 / 4.6: Timing, overlap, interruption
Keep:
- psychologically motivated timing logic
- mix modes
- overlap / barge-in as explicit constructs
- seeded stochasticity

Change:
- make timing tables project-specific
- define metadata and label semantics clearly for interrupted turns

Add:
- overlap should be role-asymmetric and type-asymmetric
- negative/confusor scenes also need overlap, otherwise overlap itself becomes a violence cue

### Section 4.7: Normalization strategy
Strong section. Keep the spec change.

Add:
1. Consider LUFS / short-term loudness logging in QA, not just RMS
2. Be careful with the term “true-peak limiting” if implementation is only sample-peak limiting

### Section 5: What stays
Keep:
- script generation
- taxonomy / metadata foundations
- cache design
- broad pipeline decomposition

Modify:
- preprocessing verdict should change from “keep except normalization” to “keep architecture, but revisit default low-pass and denoise behavior for Tier A”
- `DialogueTurn` and `MixedScene` schema definitely need extension

## Revised Milestone Order

### P0 / immediate
1. Hebrew disambiguation
2. SSML baseline redesign (pitch/rate/volume defaults)
3. normalization strategy change
4. run-level acoustic QA extension

### P1 / realism core
5. stateful speaker controller
6. timing controller
7. overlap / interruption
8. schema + metadata extension for rendered vs audible timing

### P2 / diversity and advanced realism
9. multi-voice / multi-backend diversification
10. optional phrase-prosody hints
11. experimental voice-quality transforms
12. project-specific default profiles

## Missing or Underdeveloped Design Elements

### 1. Dataset provenance needs stronger versioning
Add explicit metadata such as:
- `generation_variant`
- `tts_backend`
- `voice_family`
- `text_normalization_version`
- `timing_controller_version`
- `mix_mode_statistics`
- `prosody_controller_version`

### 2. Negative and confusor scenes need the same realism machinery
Do not improve overlap, timing realism, loudness variation, or statefulness only for violent scenes. Otherwise the model learns “realistic assembly = positive class.”

### 3. Add listening-test gates
Before declaring V2 done, add a small review loop:
- 10–20 clips per milestone
- reviewed by at least one native Hebrew listener
- scored on:
  - gender correctness
  - adultness / age plausibility
  - distress plausibility
  - conversational naturalness
  - obvious TTS-ness

### 4. Add “do not use for training yet” gates
The final plan should state explicit release gates for when a generated set is allowed to be used beyond smoke tests.

Examples:
- if only 2 voices are present, training use is restricted
- if overlap ratio is zero, training use is restricted
- if VIC I4/I5 F0 median exceeds a warning threshold too often, training use is restricted
- if unresolved ambiguous Hebrew forms remain, training use is restricted

## Concrete Changes I Would Ask Claude to Make in the Final Version

1. Promote voice diversity into core V2, not deferred Phase 2.
2. Preserve both canonical text and spoken/rendered text in the schema and metadata.
3. Replace exact phrase text-span matching with a more stable span/ID mechanism.
4. Downgrade breathiness injection from core milestone to experimental augmentation.
5. Add explicit label semantics for overlap/barge-in, including rendered vs audible timing.
6. Make preprocessing tier-aware/configurable, especially low-pass and Wiener denoise.
7. Add run-level QA distributions, not just clip-level warnings.
8. Make realism machinery apply to negative/confusor scenes too, so realism itself does not become a class cue.
9. Differentiate project defaults for She-Proves vs Elephant more explicitly.
10. Add stronger provenance/version metadata for generated clips and runs.
11. Add listening-test acceptance gates before treating V2 as ready for training-scale generation.
12. Call out and fix emotional-state metadata consistency across script, rendering, and packaged labels.

## Final Bottom Line

Claude’s V2 plan is fundamentally on the right track and is much closer to a workable redesign than the current pipeline. But before finalization, it should become:
- less brittle
- less single-voice / single-backend dependent
- less confident in single-clip numeric targets
- more explicit about metadata, labels, and QA consequences
- more cautious about synthetic phonation hacks

If those revisions are made, the result will be a substantially stronger final design document.
