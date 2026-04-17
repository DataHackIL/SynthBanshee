# Audio Generation V3 — Revised Design and Implementation Plan

**Status:** Draft — revised from V2 incorporating ChatGPT review feedback
**Date:** 2026-04-14
**Supersedes:** `audio_generation_v2_design.md`
**Context:** Four independent reviews of `debug_run_1` (Shay, Gemini, Claude, ChatGPT) plus ChatGPT review of V2 design.
**Trustworthiness note:** Where Shay's feedback conflicts with an LLM review, Shay's assessment is authoritative.

**What changed from V2:**
- Phrase prosody redesigned to use phrase IDs, not exact text-span matching (fragile after niqqud + disfluency)
- `DialogueTurn` extended with `text_original` / `text_spoken` / `normalization_rules_triggered`
- Overlap/barge-in label schema extended to three distinct time concepts (scripted, rendered, audible)
- NEG/NEU confusor scenes now explicitly required to receive the same realism machinery as violent scenes
- Emotional-state metadata consistency added as an explicit first-class design requirement (fixes the `_normalize_emotion()` hack)
- Run-level distribution QA added as a new section
- Release gates / training-use restriction criteria added as a new section
- Multi-voice diversification promoted from "deferred" into core milestones (P2)
- Breathiness injection downgraded from core milestone to experimental augmentation with a listening-test gate
- All numeric targets explicitly labeled as configurable starting points, not fixed truth
- Preprocessing made tier-aware: Wiener denoising is now a configurable flag, not unconditional
- Project-specific default profiles added for She-Proves and Elephant
- Provenance/versioning metadata added to `ClipMetadata`
- **(Gemini)** SSML `<phoneme>` tag injection added as a third layer in M1 for the highest-priority gender-sensitive tokens, bypassing the TTS G2P model entirely
- **(Gemini)** `ProviderCapabilities` matrix added to §4.10 / M9: SSML-driven and slider-driven backends have distinct control surfaces; the provider ABC must declare capabilities so prosody parameters are mapped correctly rather than silently dropped

---

## Implementation Tracker

All P0–P2 work is tracked here. Milestones that span multiple PRs are listed one row per PR, with sub-suffixes (e.g. M3a, M3b). Status key: ✅ Done · 🔲 Not started · 🧪 Experimental (gate required before scale use).

| PR | Priority | Status | Primary benefit | Scope (this PR) | Effort | Spec change? |
|----|----------|--------|----------------|-----------------|--------|--------------|
| **M1** | P0 | ✅ Done | Eliminates AGG→VIC gender errors | `hebrew_disambiguator.py` — niqqud lexicon + SSML `<phoneme>` for top tokens; `DialogueTurn.text_spoken` / `text_original` / `normalization_rules_triggered`; wired after `ScriptGenerator.generate()`; QA flag for unvocalized high-risk tokens in `text_spoken` | Small | No |
| **M2a** | P0 | ✅ Done | Lowers VIC F0 to adult range; caps AGG pitch escalation | Update `style_map` in speaker YAMLs per §4.2a table — VIC pitch −4→−1 st, AGG pitch capped 0/0/0/+1/+1 across I2–I5 | Small | No |
| **M3a** | P0 | ✅ Done | Preserves within-scene loudness trajectory | `rms_target_dbfs` on `StyleEntry`; `_apply_rms_gain()` helper in `SceneMixer`; 4-tuple segment API; AGG −28→−15 dBFS (I1→I5), VIC −26→−30 dBFS; FLOAT temp WAV write to avoid PCM_16 hard-clip | Small | No |
| **M3b** | P0 | ✅ Done | Prevents peak-normalize from erasing inter-scene RMS contrast | Replace `_normalize_peak()` in `preprocess()` with sample-peak limiter (only clips above −1.0 dBFS, no forced scale-up); update `validate_audio()` to check peak ≤ −1.0 dBFS instead of == −1.0 dBFS; update `spec.md §3` | Small | Yes — spec.md §3 |
| **M4** | P0 | ✅ Done | Fixes silent label corruption in strong-label JSONL | Audit `debug_run_1` emotional states; extend `taxonomy.yaml` `emotional_states` to cover all legitimate LLM outputs; replace `_normalize_emotion()` silent fallback with explicit mapping table + logged warning (hard fail for completely unknown states); add clip-level `quality_flags` value `"emotion_downgrade"` | Small | No |
| **M5** | P1 | ✅ Done | Protects Tier A phonetic quality from over-denoising | `PreprocessingConfig` dataclass; update `preprocess()` to accept it (default unchanged); scene YAMLs gain optional `preprocessing` block; Wiener step skipped when `wiener_denoise=False`; unit tests verify Wiener skip | Small | No |
| **M6** | P1 | 🔲 Not started | Replaces mechanical silence gaps with psychologically-motivated turn latencies | `TurnGapController` in `synthbanshee/tts/gap_controller.py`; project-specific gap tables (§4.5); wire into `TTSRenderer.render_scene()` replacing `turn.pause_before_s`; NEG/NEU confusor gap ranges; unit tests per context type and project | Small | No |
| **M7** | P1 | 🔲 Not started | Eliminates speaker-state reset at turn boundaries | `SpeakerState` with `update()` / `to_metadata_dict()`; intensity-drift rules per role; wire into `TTSRenderer.render_scene()` — one `SpeakerState` per speaker; state offsets applied to render params; state written to per-turn generation metadata | Medium | No |
| **M2b** | P1 | 🔲 Not started | Burst-pause microstructure within high-intensity turns | `PhraseHint` / `PhraseProsody` dataclasses in `ssml_types.py`; `DialogueTurn.phrase_hints`; char-offset resolver (maps `text_original` offsets → `text_spoken` after niqqud); `SSMLBuilder` extension for nested `<prosody>` + `<break>`; Jinja2 template update requesting `phrase_hints` for I3–I5 turns; deterministic sentence-final imperative heuristics | Medium | No |
| **M8a** | P1 | 🔲 Not started | Adds interruption behavior to audio assembly | `MixMode` enum (`SEQUENTIAL`, `OVERLAP`, `BARGE_IN`); extend `SceneMixer.mix_sequential()` to handle overlap/barge-in segment mixing; `TurnGapController.gap_seconds()` extended to return `(gap_s, MixMode)`; unit tests for overlapped segment merge | Medium | No |
| **M8b** | P1 | 🔲 Not started | Precise event timestamps for overlapping turns; NEG/NEU confusor overlap | `MixedScene` gains `script_onset_s`, `rendered_onset_s`, `audible_onset_s` per turn; label generator uses `audible_onset_s`/`audible_end_s` exclusively; truncated turns get `truncated: true`; NEG/NEU overlap rates; integration tests | Medium | No |
| **M9a** | P2 | 🔲 Not started | Second TTS backend; eliminates single-backend fingerprint risk | `google_provider.py` — Google Cloud TTS Chirp 3 HD he-IL; `ProviderCapabilities` ABC so SSML-driven and slider-driven backends declare their control surface; `TTSRenderer` backend dispatch; speaker YAMLs for Google he-IL male and female voices | Medium | No |
| **M9b** | P2 | 🔲 Not started | ≥ 3 voice variants per gender across dataset | ≥ 4 additional speaker YAMLs in `configs/speakers/` (≥ 2 male, ≥ 2 female, including ≥ 1 Google variant per gender); update scene configs to distribute speaker IDs across scenes; `voice_family` column in manifest CSV; `generate-batch` distributes across voice variants | Small | No |
| **M10a** | P2 | 🔲 Not started | Catches clip-level prosody regressions | Extend `qa.py` with per-clip acoustic metrics: F0 median/std by speaker by intensity (librosa pyin), RMS and LUFS by turn (pyloudnorm), ambiguous Hebrew token count, `normalization_rules_triggered` frequency; per-clip WARN flags: `WARN_VIC_F0_HIGH`, `WARN_AGG_NO_ESCALATION`, `WARN_GENDER_AMBIGUITY` | Medium | No |
| **M10b** | P2 | 🔲 Not started | Catches systematic run-level biases | Run-level aggregation in `QAReport`: F0/RMS/LUFS distributions by role/typology/project; voice and backend diversity counts; `mix_mode` distribution; outlier clips; `qa-report --run-summary` CLI table; `WARN_NO_OVERLAP` and `WARN_EMOTION_DOWNGRADE` run-level flags | Medium | No |
| **M11** | P2 | 🔲 Not started | Enables ablation studies and debugging across pipeline versions | `GenerationMetadata` dataclass (§4.11); written to `{clip_id}.json` under `generation_metadata` key; backward-compatible — existing V1 clips not invalidated | Small | No |
| **M12** | P2 | 🧪 Experimental | Reduces VIC HNR at I3–I5 if listening-test gate passes | `add_breathiness()` in `synthbanshee/augment/voice_texture.py`; wire for VIC turns via `SpeakerState.breathiness_level`; listening-test gate — 20 clips, native Hebrew listener, blind A/B (§8.3); `breathiness_applied` flag in `GenerationMetadata`; disabled by default if gate fails | Medium | No |
| **M13** | P2 | 🔲 Not started | She-Proves / Elephant generate audio appropriate to their distinct acoustic regimes | `project_profile` field in `RunConfig`; gap, overlap probability, loudness targets, preprocessing config, and augmentation config all carry project-specific defaults; two profile YAML files in `configs/run_configs/`; new profiles addable without code changes | Small | No |

---

## 1. Executive Summary

The current pipeline treats each utterance as an independent TTS call with per-utterance prosody parameters and concatenates the results with fixed silence gaps. Speaker state resets at every turn boundary. Emotional escalation is expressed only through scripted SSML parameters, not through the phonation-quality, rate-instability, and amplitude-discontinuity features that real escalating confrontations produce. Turn-taking has no conversational logic.

The problem is structural, not a single bug. It requires changes across four layers:
1. **Text preprocessing** — Hebrew gender disambiguation
2. **TTS synthesis control** — prosody parameter redesign, stateful speaker model
3. **Post-synthesis audio assembly** — normalization strategy, timing logic, overlap/barge-in
4. **Data model and QA** — dual text fields, three-timeline labels, run-level metrics, release gates

The scripts, label taxonomy, pipeline stage decomposition, and cache system are sound and do not need redesign.

**Priority structure:**
- **P0 (immediate):** Fix deterministic artifacts that actively harm training data
- **P1 (realism core):** Add stateful conversational dynamics; preserve escalation cues
- **P2 (diversity and observability):** Widen voice diversity; harden QA; add release gates

---

## 2. Findings by Priority (Unchanged from V2)

| # | Problem | Evidence | ML Harm |
|---|---------|----------|---------|
| 1 | Near-zero amplitude variation across intensity levels | 1.7 dB RMS range for AGG across I1–I5 | Classifier cannot learn loudness-escalation cues |
| 2 | VIC (HilaNeural) baseline F0 too high; rises into child-voice range | Baseline ~215 Hz, rising to 285–287 Hz at I5. **(S)** | Systematic OOD at test time; model learns wrong distress proxy |
| 3 | No creaky/breathy phonation in VIC at high intensity | HNR 0.31–0.41 (clean modal) at all intensities | Model cannot learn modal→creaky voice-quality shift |
| 4 | Hebrew gender-agreement errors | Azure defaults masculine for ambiguous forms. **(S)** heard directly | Acoustic noise at relational lexical targets |
| 5 | AGG F0 variance too narrow at I4–I5 | F0 mean ~133 Hz at I5; std ≤ 25 Hz | Model underweights F0 variance as an aggressor cue |
| 6 | Uniform speech rate within turns | Female rate 1.2–1.6 words/sec across all intensities. **(S)** "way too slow" | Rate-based features too smooth to generalize |
| 7 | Non-motivated inter-turn silence | Gap range 0.3–1.5 s with no conversational logic | Model learns artificial timing signatures as class features |
| 8 | No overlap or interruption | Every boundary is a clean silence | Missing strongest structural DV indicator |
| 9 | Peak normalization erases loudness trajectory | Scene-level peak normalize collapses I1–I5 to ~2 dB range | Loudness escalation trained away |
| 10 | Missing paralinguistic events in VIC | No breath catches, sighs, tremor | Gap in victim-state features |
| 11 | Metallic TTS artifacts | Harmonic smoothing at extreme SSML parameters. **(S)** | Model may learn Azure fingerprint rather than prosodic features |
| 12 | **[V3-added]** Only two voices used in any generated dataset | AvriNeural + HilaNeural only | Model learns voice identity as a class cue |
| 13 | **[V3-added]** Emotional-state metadata inconsistency | `_normalize_emotion()` silently downgrades emotional states in packaged labels | Script and JSONL disagree; corrupts training signal for emotion-conditioned models |

---

## 3. What Does NOT Need to Change

- **Script generation** (LLM + Jinja2 templates) — produces structurally correct dialogue; the problem is in rendering, not generation
- **Label taxonomy** (`configs/taxonomy.yaml`) — hierarchical PHYS/VERB/DIST/EMOT/ACOU/NONE is appropriate
- **Pipeline stage decomposition** — the five stages are correctly separated
- **TTS cache** (SHA-256 of full SSML string) — correct; will naturally miss on any SSML change
- **Acoustic augmentation** (Tier B) — still needed and architecturally correct; runs after V3 audio assembly
- **Silence pad, resample, mono, low-pass** in preprocessing — correct and unconditional *(Wiener denoising is now configurable — see §4.9)*
- **Validation** (`validate_clip`) — correct; needs minor update for new normalization spec
- **`ClipMetadata` core schema** — correct; extend with provenance fields (see §4.10)

---

## 4. Architecture Changes

### 4.1 Text Preprocessing: Hebrew Gender Disambiguation Layer

**Problem:** Azure TTS defaults to masculine inflection for morphologically ambiguous unvocalized Hebrew second-person forms. This is a TTS engine behavior that cannot be corrected with SSML alone.

**New module:** `synthbanshee/script/hebrew_disambiguator.py`

```python
@dataclass
class NormalizationResult:
    text_spoken: str                       # text actually sent to TTS
    normalization_rules_triggered: list[str]  # IDs of rules that fired

def disambiguate_for_speaker(
    text: str,
    speaker_role: str,       # "AGG" or "VIC"
    addressee_role: str,     # "AGG" or "VIC"
) -> NormalizationResult:
```

Approach (three layers, applied in order):
1. **Rule-based niqqud insertion** for a lexicon of known high-risk second-person address forms in AGG→VIC speech. Initial lexicon: `{שלך, עשית, היית, חשבת, הלכת, תשמעי, תראי, יוצאת, נפגשת}`. Each entry maps to its correctly vocalized feminine-directed form.
2. **SSML `<phoneme>` tag injection** for the highest-priority tokens where niqqud alone may be misread by the TTS G2P model. For the 3–4 most critical words (שלך being the clearest case), wrapping the token in `<phoneme alphabet="ipa" ph="ʃɛˈlaχ">שלך</phoneme>` bypasses the engine's grapheme-to-phoneme model entirely and guarantees the correct phoneme sequence regardless of vowel-pointing interpretation. This is a last-resort layer: applied only to the tokens where a wrong pronunciation is most audibly harmful and most reliably producible via IPA. Phoneme strings must be validated against Azure's he-IL IPA coverage before deployment.
3. **Phonetic rewrite fallback** for forms where niqqud alone is insufficient and no reliable IPA string is known — produce an orthographically unambiguous alternative.
4. **Optional full-text Nakdan/Dicta integration** — not a core-path dependency; opt-in for scripts containing unanticipated forms. Note: LLM-generated niqqud must not be used — LLMs hallucinate Hebrew vowel pointing at a rate that would produce worse output than the current masculine default.

**Schema change — `DialogueTurn` gains two new fields:**

```python
text_original: str          # canonical LLM output, never modified
text_spoken: str            # text actually sent to TTS after all normalization
normalization_rules_triggered: list[str]  # rule IDs that fired, empty if none
```

`text_original` is set at script generation time and never mutated. Niqqud insertion, disfluency injection, and any other text normalization write to `text_spoken` only. If no normalization is applied, `text_spoken == text_original`.

**Why this matters for debugging:** When a gender error is reported, the engineering team can inspect `text_spoken` to see exactly what was synthesized, and `normalization_rules_triggered` to know which rules fired. Without this, diagnosing post-deployment gender errors is guesswork.

**Risk:** Automated niqqud tools have ~5–10% error rates. The rule-based lexicon covers the 10–12 highest-frequency error sources. New scripts should be checked against the lexicon at QA time. Human review of ambiguous forms before large-scale runs is in the release gate criteria (§8).

**Where it runs:** After `ScriptGenerator.generate()` returns turns, before `TTSRenderer` is called. The `text_spoken` field that enters the TTS stage must already be fully normalized.

---

### 4.2 SSML Prosody Redesign: Volume Escalation and Per-Phrase Control

**Problem:** Current SSML applies a single rate/pitch/volume to the entire utterance. The intensity-to-SSML parameter table has essentially no volume escalation (1.7 dB range across I1–I5). VIC baseline pitch is too high (~215 Hz, should be ~185–195 Hz).

**Changes:**

**4.2a — Parameter table redesign (speaker configs)**

Update `style_map` entries in speaker YAML configs. The values below are **initial calibration defaults derived from one debug clip — treat as configurable starting points, validate by listening and measurement before committing**:

| Intensity | AGG volume target | VIC volume target | VIC pitch offset |
|-----------|------------------|------------------|-----------------|
| I1 | 0 dB (baseline) | 0 dB | −3 st to −5 st from default |
| I2 | +2 dB | +1 dB | −3 st |
| I3 | +5 dB | +2 dB | −3 st |
| I4 | +9 dB | +4 dB | −2 st (cap pitch escalation) |
| I5 | +13 dB | +5 dB | −1 st (hard cap) |

Note: "0 dB baseline" means the per-turn RMS target for I1; the absolute SSML `volume` value is determined by the speaker config and the normalization strategy (§4.7). The dB offsets are relative to I1, not absolute SSML values. VIC pitch cap at I4–I5 prevents the child-voice-range problem; additional distress at those levels should come from rate instability, timing compression, and breathiness (§4.4), not further pitch elevation.

These values are per-project and per-typology configurable — She-Proves apartment confrontations and Elephant clinic incidents have different baseline loudness expectations. Do not assume the same table is correct across all scene types.

**4.2b — Per-phrase SSML injection (phrase ID mechanism)**

The V2 design proposed `PhraseProsody(text_span=...)` — matching by exact substring. This is fragile: niqqud insertion, disfluency injection, and punctuation normalization all mutate `text_spoken` after the LLM generates phrase hints. A substring match on the original LLM text breaks after any normalization step.

**Revised approach: phrase IDs attached at script generation time, resolved against `text_spoken` after final normalization.**

New data types:

```python
@dataclass
class PhraseHint:
    phrase_id: str          # unique within the turn, e.g. "t3_p1"
    hint: Literal["stress", "slow", "break_before", "break_after", "menace"]
    char_start_original: int   # offset in text_original
    char_end_original: int

@dataclass
class PhraseProsody:
    phrase_id: str
    rate: str | None        # SSML rate, e.g. "+15%" or "slow"
    pitch: str | None       # e.g. "+3st"
    volume: str | None      # e.g. "+6dB"
    break_before_ms: int = 0
    break_after_ms: int = 0
```

`DialogueTurn` gains `phrase_hints: list[PhraseHint] = []`. At script generation time, the LLM (for I3–I5 turns) annotates 1–2 emotionally loaded phrases by their character offsets in `text_original`. After all text normalization is applied to produce `text_spoken`, a resolver maps those original offsets to offsets in `text_spoken` (accounting for niqqud insertion length changes). The SSML builder receives resolved `PhraseProsody` objects referencing `text_spoken` positions.

**Two sources of phrase prosody** (not just LLM):
1. LLM-generated hints (I3–I5 turns): accusatory phrase acceleration, command deliberate slowing
2. Deterministic heuristics: sentence-final single-word imperatives (like "עכשיו", "ברור?") always get a `<break>` before them and a `slow` hint; this does not require LLM annotation

Both sources produce `PhraseProsody` objects; the SSML builder does not distinguish their origin.

---

### 4.3 Stateful Cross-Turn Emotional Controller

**Problem:** Speaker state resets at every turn boundary. A speaker who has been shouting for 30 seconds should not sound identical at the start of their next turn to how they sounded at the start of the scene.

**New class:** `synthbanshee/tts/speaker_state.py`

```python
@dataclass
class SpeakerState:
    intensity_history: list[int] = field(default_factory=list)
    rate_offset: float = 1.0
    pitch_offset_st: float = 0.0
    volume_offset_db: float = 0.0
    breathiness_level: float = 0.0   # 0.0 = modal; 1.0 = maximum

    def update(self, new_intensity: int, speaker_role: str) -> None:
        ...

    def to_metadata_dict(self) -> dict[str, float]:
        # Serialized into per-turn metadata for reproducibility
        ...
```

Transition rule: if current intensity > previous intensity by same speaker, drift state variables toward higher-intensity targets by 50–70% of the gap; if intensity drops, decay at 30% per turn. Drift rates are configurable.

**State serialization:** `SpeakerState.to_metadata_dict()` is written into per-turn rendering metadata so the state that produced each turn's audio is reproducible. Without this, debugging "why does this turn sound wrong" is impossible if the state evolved stochastically.

**Scene archetypes (lightweight):** The speaker configs can optionally carry a `scene_archetype` field (e.g., `"explosive_confrontation"`, `"cold_coercion"`, `"suppressed_fear"`) that sets initial drift rates and breathiness schedule. This is a one-field addition to `SpeakerConfig`, not a separate system. Defer full implementation to P2 if needed.

---

### 4.4 Post-Synthesis Voice Quality Modification (Experimental)

**Status: Experimental augmentation, not a core V2/V3 milestone. Subject to listening-test acceptance gate before any scale use.**

**Problem:** HilaNeural has HNR 0.31–0.41 (clean modal) at all intensities. Azure SSML cannot control phonation quality for he-IL voices.

**Proposed approach (experimental):** `synthbanshee/augment/voice_texture.py` — `add_breathiness(wav_array, sample_rate, level)` applies bandpass-filtered noise (3–8 kHz, level-scaled) mixed into the VIC signal at I3+.

**Important caveats — explicitly part of the design:**
- Bandpassed white noise ≠ human breathy phonation. HNR reduction is not proof of realistic distress.
- This creates a synthetic artifact. If a downstream classifier learns to detect it, the training data has negative value for that feature.
- This must not be the primary distress signal. Primary distress must come from prosodic improvements (§4.2, §4.3), rate instability, timing compression, and overlap (§4.6).
- Before any dataset generated with this feature is used for model training, it must pass a listening-test gate (§8.3) and a classifier-side sanity check that breathiness-affected turns are not being flagged via HNR artifacts rather than prosodic features.

**Acceptance gate:** Generate 20 VIC turns with and without breathiness processing. Have a native Hebrew listener rate each on a 1–5 distress-plausibility scale blind to condition. If the breathy condition does not score ≥ 0.5 points higher on average, the feature is disabled.

---

### 4.5 Inter-Turn Timing Redesign

**Problem:** Inter-turn silence has no conversational logic.

**New class:** `synthbanshee/tts/gap_controller.py` — `TurnGapController`

```python
def gap_seconds(
    self,
    current_turn: DialogueTurn,
    prev_turn: DialogueTurn | None,
    rng: random.Random,
) -> float:
```

**Psychologically-motivated gap table (initial calibration defaults — all values configurable per project):**

| Context | She-Proves range | Elephant range | Rationale |
|---------|-----------------|---------------|-----------|
| VIC responding to I1–I2 AGG | 300–600 ms | 250–500 ms | Normal conversational latency |
| VIC responding to I3 accusation | 150–350 ms | 100–300 ms | Defensive, quick self-justification |
| VIC responding to I4 command | 50–150 ms | 50–100 ms | Compliance; fear-driven immediacy |
| VIC responding to I5 threat | 100–300 ms | 80–200 ms | Shock; not a 1.5 s editorial pause |
| AGG responding at I1–I2 | 200–500 ms | 150–400 ms | Normal |
| AGG responding at I3–I5 | 50–200 ms | 50–150 ms | Cutting in; loading for next volley |
| AGG deliberate self-pause at I4–I5 | 600–1400 ms | 500–1200 ms | Menacing pause before threat |

Values are seeded-RNG draws for reproducibility. The `TurnGapController` is instantiated with a project identifier and returns project-appropriate ranges. She-Proves and Elephant are the two initial project configs; adding a new project requires only a new config block.

**Important:** confusor scenes (NEG typology) must use the same gap controller with psychologically-motivated ranges appropriate for their scenario type (heated argument, sports excitement, clinic altercation). Do not leave NEG scenes with fixed mechanical gaps while violent scenes have timing logic — the model will learn timing realism as a violence cue.

---

### 4.6 Overlap and Interruption Simulation

**Problem:** All turn boundaries are clean silence gaps. Real coercive confrontations have barge-in behavior.

**New `MixMode` enum in `SceneMixer`:**

```python
class MixMode(Enum):
    SEQUENTIAL = "sequential"   # current behavior
    OVERLAP = "overlap"         # next starts before prev ends
    BARGE_IN = "barge_in"       # prev is cut off; next starts immediately
```

**Three distinct time concepts** — once turns overlap, the label generator must track all three:

| Concept | Field name | Definition |
|---------|-----------|-----------|
| Scripted interval | `script_onset_s` / `script_offset_s` | What the script intended |
| Rendered interval | `rendered_onset_s` / `rendered_offset_s` | What the TTS actually produced before mixing |
| Audible interval | `audible_onset_s` / `audible_end_s` | What is audible in the final waveform after overlap/barge-in |

For `BARGE_IN`, `audible_end_s` of the interrupted turn is earlier than `rendered_offset_s`. The label generator **must use `audible_onset_s` / `audible_end_s`** for all event timestamp outputs — not the scripted or rendered values. If a speaker is cut off mid-utterance, any label for an event in the tail of that utterance must either be dropped or marked as `truncated: true`.

`MixedScene` must expose all three sets of timestamps.

**Asymmetric overlap probabilities (initial defaults — configurable):**

| Transition | BARGE_IN prob | OVERLAP prob | Notes |
|-----------|--------------|-------------|-------|
| AGG cuts off VIC at I3 | 10% | 15% | Starts to impose |
| AGG cuts off VIC at I4 | 25% | 20% | Dominant behavior |
| AGG cuts off VIC at I5 | 40% | 15% | Maximum dominance |
| VIC cuts off AGG at any I | 5% | 10% | Rare; only in defiance moments |

**Confusor scenes must also use overlap.** A heated but non-violent argument (NONE_ARGU) should have overlap at a rate comparable to a mild confrontation. If violent scenes have overlap and confusor scenes do not, the model will learn "overlap = violence." Match the overlap probability to the acoustic intensity of the confusor, not to zero.

---

### 4.7 Normalization Strategy

**Problem:** Scene-level peak normalization erases the within-scene loudness trajectory that §4.2 works to create.

**Changes:**

1. **Per-turn RMS targeting at the mix stage:** Each turn is gain-adjusted before being inserted into `SceneMixer` so its RMS matches the intensity-level target from §4.2. This creates the loudness escalation.

2. **Replace scene-level peak normalization with a true-peak limiter:** After mixing, apply a limiter at ≤ −1.0 dBFS to ensure technical compliance without collapsing the trajectory. Note: the current `preprocess()` implementation uses sample-peak normalization (a simple divide-by-max). A true-peak limiter accounts for inter-sample peaks (relevant for lossy re-encoding, less critical here but more correct). If implementation uses sample-peak limiting, document it as such in the spec rather than calling it "true-peak."

3. **LUFS logging in QA:** Log short-term LUFS by turn (via pyloudnorm or equivalent) in addition to RMS. LUFS is more perceptually accurate and is the standard used by broadcast and streaming. Include it in the acoustic QA metrics (§7) for trend monitoring.

4. **Spec change required:** `docs/spec.md` §3 currently states "−1.0 dBFS peak normalized." Must be updated to: "Peak-limited to ≤ −1.0 dBFS after mixing; per-turn RMS targeting applied at mix stage to preserve within-scene loudness trajectory."

---

### 4.8 Tier-Aware Preprocessing

**Problem:** The current preprocessing pipeline unconditionally applies resample → mono → low-pass → Wiener denoise → peak normalize → silence pad. For clean synthetic TTS audio (Tier A), Wiener denoising may remove fine phonetic detail that is not actually noise. The pipeline should be configurable per tier.

**Changes:**

Add a `PreprocessingConfig` dataclass controlling which steps are applied:

```python
@dataclass
class PreprocessingConfig:
    resample: bool = True           # always True in practice
    downmix_mono: bool = True       # always True
    lowpass_hz: float | None = 7500 # None to skip; 7500 Hz is correct for 16 kHz output
    wiener_denoise: bool = True     # configurable; default off for Tier A
    normalization: NormalizationMode = NormalizationMode.PER_TURN_RMS
    silence_pad_s: float = 0.5
```

**Default per tier:**
- **Tier A (synthetic only):** `wiener_denoise=False`. The audio is already clean; Wiener filtering is likely to harm phonetic quality.
- **Tier B (augmented):** `wiener_denoise=True`. After room IR convolution and noise overlay, actual denoising may improve quality.
- **Tier C (confusor):** `wiener_denoise=True`. Scaper-mixed audio may benefit.

The low-pass at 7.5 kHz is appropriate for all tiers (it is the Nyquist-adjacent band for 16 kHz output) and remains unconditional.

---

### 4.9 Emotional-State Metadata Consistency

**Problem:** The current pipeline silently downgrades emotional states in packaged labels. LLM generates `calm`, `fearful`, `pleading` for turn emotional states; `_normalize_emotion()` in `cli.py` maps unknown values to `neutral`. The strong-label JSONL shows many turns as `neutral` even though the script has them as `fearful` or `pleading`. Script and labels disagree.

This is not just a data cleanliness issue — an emotion-conditioned model that learns from the label JSONL will have a corrupted supervision signal.

**Changes:**

1. **Remove the silent downgrade.** `_normalize_emotion()` should log a warning and either (a) fail loudly for known invalid emotional states, or (b) map only to the nearest valid state with an explicit mapping table, not to `neutral` as a catch-all.

2. **Extend `configs/taxonomy.yaml` `emotional_states` list** to cover all emotional states that the LLM plausibly generates for each intensity level. Run the debug_run_1 script through the pipeline, collect all unique emotional states, and add any missing ones to the taxonomy before removing the fallback.

3. **Add a QA check:** `qa.py` should flag any clip where the number of turns with emotional_state = `neutral` in the JSONL exceeds the number of turns with intensity ≤ 2 in the script. If I4 and I5 turns are being labeled as `neutral`, the normalization fallback has been triggered.

4. **Consistency invariant in the validator:** `validate_clip()` should optionally check that `emotional_state` values in the strong-label JSONL are consistent with the intensity and speaker_role values in the same turn. Gross mismatches (e.g., AGG I5 labeled `neutral`) are a data integrity signal.

---

### 4.10 Multi-Voice Diversification (Core V2, not Deferred)

**Problem:** Training on only AvriNeural + HilaNeural means the model will learn Azure voice identity as a class cue. This is a systematic generalization failure regardless of how good the prosody improvements are.

**Why this was deferred in V2 and why that was wrong:** V2 deferred this to "Phase 2 scale-up." The review correctly identifies that the fingerprint risk is present from the first training run. Even a bootstrap/ablation dataset should have at least 3 voice variants per gender.

**What is available for he-IL:**
- **Azure he-IL voices:** Currently `he-IL-AvriNeural` (male), `he-IL-HilaNeural` (female). Azure may have additional he-IL voices in preview; check the Azure voice catalog at run time, not at design time.
- **Structured perturbations of existing voices:** Applying deterministic pitch/rate/timbre offsets via SSML parameters to the same voice effectively produces a different perceptual identity without a new voice. This is the minimum viable diversification if no additional voices are available.
- **Alternative TTS backends:** Google Cloud TTS Chirp 3 HD he-IL is already listed as a secondary provider in the project. It produces acoustically distinct output from Azure and immediately doubles voice diversity with no new integration work beyond the existing `[google-tts]` optional extra.

**Minimum acceptable target for any dataset used beyond smoke tests:**
- ≥ 3 distinct male voice variants (AGG role)
- ≥ 3 distinct female voice variants (VIC role)
- ≥ 2 TTS backends represented

If this target cannot be met, the dataset is restricted to bootstrap/smoke-test use only — see §8.

**Speaker config changes:** `SpeakerConfig.tts_voice_id` already supports any voice ID string. To add a voice variant, add a new speaker YAML with the new voice ID. No code changes required — this is a configuration expansion.

**Provider capabilities matrix:** Azure and non-SSML backends (ElevenLabs, Bark) have fundamentally different control surfaces. Azure accepts explicit SSML prosody tags; ElevenLabs accepts API sliders (`stability`, `style_exaggeration`); Bark relies on implicit textual cues. A generic `text_in/audio_out` interface that sends the same SSML to all backends silently loses expressiveness on non-SSML providers — Azure's prosody tags become no-ops, and ElevenLabs's emotional range is never accessed.

The `AzureProvider` / `TTSProvider` ABC must be extended with a `ProviderCapabilities` declaration:

```python
@dataclass
class ProviderCapabilities:
    supports_ssml: bool
    supports_style_tags: bool          # <mstts:express-as>
    supports_phoneme_tags: bool        # <phoneme alphabet="ipa">
    supports_api_emotion_sliders: bool # ElevenLabs stability / style_exaggeration
    max_volume_delta_db: float | None  # None = unlimited
```

Each provider declares its capabilities at instantiation. The render pipeline queries capabilities before building the SSML or API payload:
- SSML-capable providers (Azure, Google) receive full SSML with prosody, phoneme, and style tags.
- Slider-capable providers (ElevenLabs) receive plain text with intensity mapped to API parameters — e.g., intensity 4–5 maps to `stability=0.3, style_exaggeration=0.7` to induce the voice cracking and pitch instability SSML cannot produce on that backend.
- The `SpeakerState` and prosody parameters flow into whichever control surface the provider supports; no capability is silently dropped.

This does not require a new pipeline stage. It is a refactor of `SSMLBuilder` and the provider base class, to be done as part of M9 when the second backend is integrated.

---

### 4.11 Dataset Provenance and Version Metadata

**Problem:** Once V2/V3 changes are deployed, datasets generated at different pipeline versions will have different acoustic properties. Without provenance metadata, a model trained on a mix of V1 and V3 clips cannot be debugged.

**Changes to `ClipMetadata`:** Add an optional `generation_metadata` block:

```python
@dataclass
class GenerationMetadata:
    pipeline_version: str          # e.g. "v3.0"
    tts_backend: str               # e.g. "azure", "google"
    voice_family: str              # e.g. "he-IL-AvriNeural"
    text_normalization_version: str  # version of the disambiguation lexicon
    prosody_controller_version: str
    timing_controller_version: str
    mix_mode_used: str             # dominant MixMode for the scene
    normalization_strategy: str    # e.g. "per_turn_rms_v1"
    breathiness_applied: bool
    speaker_state_serialized: dict  # final SpeakerState for each speaker
```

This block is included in `{clip_id}.json` under key `generation_metadata`. It is optional (backward-compatible) so existing V1 clips are not invalidated.

---

## 5. What Stays (Updated from V2)

| Component | Verdict | Change from V2 |
|-----------|---------|---------------|
| Script generation (LLM + Jinja2) | Keep | None |
| Label taxonomy | Keep | Extend `emotional_states` list (§4.9) |
| Pipeline stage decomposition | Keep | None |
| TTS cache (SHA-256 of SSML) | Keep | Will miss on any SSML change; correct |
| `SceneMixer` sequential path | Keep as default | Add overlap/barge-in as opt-in (§4.6) |
| Resample, mono, low-pass | Keep unconditionally | None |
| Wiener denoising | **Make configurable** | V2 said "keep"; V3 makes it tier-aware (§4.8) |
| Acoustic augmentation (Tier B) | Keep | Runs after V3 audio assembly |
| Label generation | Keep | Extend for three-timeline overlap semantics (§4.6) |
| `ClipMetadata` core schema | Keep | Add optional `generation_metadata` (§4.11) |
| Validation (`validate_clip`) | Keep | Update `validate_audio()` for new normalization spec |
| QA report | Keep | Extend with acoustic and run-level metrics (§7) |
| Split strategy (speaker-disjoint) | Keep | None |

---

## 6. Implementation Milestones

### P0 — Fix Deterministic Artifacts (Most Urgent)

#### M1: Hebrew Gender Disambiguation
**Impact:** Eliminates gender errors heard by Shay; removes acoustic noise at relational lexical targets.
**Scope:**
- `synthbanshee/script/hebrew_disambiguator.py` with `NormalizationResult` return type
- Initial lexicon of ~15 high-risk second-person forms with feminine-directed niqqud
- `DialogueTurn` schema: add `text_original`, `text_spoken`, `normalization_rules_triggered`
- Wire into pipeline after `ScriptGenerator.generate()`, before `TTSRenderer`
- Unit tests: each lexicon entry; pass-through for unlisted words; `NormalizationResult` fields populated correctly
- QA check: flag any AGG→VIC turn where a high-risk token appears unvocalized in `text_spoken`

#### M2a: SSML Parameter Redesign (Prosody Defaults)
**Impact:** Lowers VIC baseline F0 to adult range; introduces meaningful loudness escalation across intensities.
**Scope:**
- Update `style_map` in speaker YAML configs per §4.2a table
- Regenerate one test clip; measure F0 and RMS per turn; confirm VIC I1 baseline ≤ 200 Hz and AGG I5 is ≥ 8 dB above I1
- All numeric targets stored in speaker YAML, not hardcoded — configurable without code changes

#### M3: Normalization Strategy
**Impact:** Preserves within-scene loudness trajectory created by M2a.
**Scope:**
- Add `per_turn_rms_target_db: dict[int, float]` to `SpeakerConfig`
- Apply per-turn RMS gain in `SceneMixer` before inserting each segment
- Replace `preprocess()` peak normalization with sample-peak limiter (≤ −1.0 dBFS)
- Update `validate_audio()` to check peak ≤ −1.0 dBFS (not == −1.0 dBFS)
- Update `docs/spec.md` §3

#### M4: Emotional-State Metadata Consistency
**Impact:** Fixes data corruption in strong-label JSONL; removes the `_normalize_emotion()` silent downgrade.
**Scope:**
- Audit `debug_run_1` for all unique emotional states generated by LLM
- Extend `taxonomy.yaml` `emotional_states` list to cover all legitimate LLM outputs
- Replace `_normalize_emotion()` fallback with logged warning + explicit mapping table (or hard fail for completely unknown states)
- Add QA check: flag clips with more `neutral` labels than low-intensity turns warrant

---

### P1 — Realism Core

#### M5: Tier-Aware Preprocessing
**Impact:** Prevents Wiener denoising from removing phonetic detail from clean Tier A audio.
**Scope:**
- Add `PreprocessingConfig` dataclass (§4.8)
- Update `preprocess()` to accept `PreprocessingConfig`; default behavior unchanged for backward compatibility
- Scene YAML configs gain optional `preprocessing` block; defaults are tier-appropriate
- Unit tests: verify Wiener step is skipped when `wiener_denoise=False`

#### M6: Inter-Turn Timing Redesign
**Impact:** Replaces mechanical gaps with psychologically-motivated turn latencies.
**Scope:**
- `synthbanshee/tts/gap_controller.py` — `TurnGapController` with project-specific gap tables (§4.5)
- Wire into `TTSRenderer.render_scene()`: replace `turn.pause_before_s` with `TurnGapController.gap_seconds()`
- NEG/NEU confusor scenes must use the gap controller with appropriate scenario ranges — not left with fixed gaps
- Unit tests: verify gap ranges for each context type and each project

#### M7: Stateful Cross-Turn Emotional Controller
**Impact:** Eliminates speaker-state reset at turn boundaries.
**Scope:**
- `synthbanshee/tts/speaker_state.py` — `SpeakerState` with `update()` and `to_metadata_dict()`
- Wire into `TTSRenderer.render_scene()`: one `SpeakerState` per speaker_id; apply state offsets to render params
- Write state to per-turn `generation_metadata`
- Unit tests: verify drift behavior across known intensity sequences

#### M2b: Per-Phrase SSML Injection (Phrase IDs)
**Impact:** Burst-pause microstructure within high-intensity turns.
**Dependencies:** M1 (text normalization must be final before phrase offsets are resolved).
**Scope:**
- `PhraseHint` and `PhraseProsody` dataclasses in `synthbanshee/tts/ssml_types.py`
- `DialogueTurn.phrase_hints` field
- Character-offset resolver: maps `char_start_original` / `char_end_original` in `text_original` to offsets in `text_spoken` (accounting for niqqud insertion length changes)
- `SSMLBuilder` extension: wrap resolved phrase spans in nested `<prosody>` + `<break>` elements
- LLM Jinja2 template update for I3–I5 turns: request `phrase_hints` JSON array
- Deterministic heuristics for sentence-final imperatives (always added, no LLM annotation needed)

#### M8: Overlap and Interruption Simulation
**Impact:** Dialogue sounds like a shared acoustic scene; adds barge-in behavior at high intensities.
**Dependencies:** M6 (gap controller must be in place; it returns `(gap_s, MixMode)`).
**Scope:**
- `MixMode` enum added to `synthbanshee/tts/mixer.py`
- `SceneMixer.mix_sequential()` extended to handle `OVERLAP` and `BARGE_IN` (§4.6)
- `MixedScene` extended with three-timeline fields per turn: `script_onset_s`, `rendered_onset_s`, `audible_onset_s` (and corresponding offsets)
- Label generator updated to use `audible_onset_s` / `audible_end_s` exclusively for event timestamps; truncated turns get `truncated: true` on their labels
- NEG/NEU confusor scenes use overlap at scenario-appropriate rates
- Unit tests: overlapped segment produces correctly merged audio; onset/offset metadata is consistent

---

### P2 — Diversity and Observability

#### M9: Multi-Voice Diversification
**Impact:** Prevents model from learning Azure voice identity as a class cue.
**Scope:**
- Audit available he-IL voices across Azure (check current catalog, not 2024 snapshot) and Google Cloud TTS Chirp 3 HD
- Add speaker YAMLs for ≥ 2 additional male variants and ≥ 2 additional female variants
- At minimum: one Google TTS he-IL voice per gender (backend diversity) + one SSML-perturbed Azure variant per gender (within-backend diversity)
- Update `generate-batch` to distribute speaker configs across voice variants
- Dataset-level manifest: record `voice_family` per speaker per clip

#### M10: Run-Level Distribution QA
**Impact:** Catches systematic biases that clip-level warnings miss.
**Scope:**
- Extend `synthbanshee/package/qa.py` with acoustic metric computation per clip:
  - Median and std F0 by speaker by intensity (librosa pyin)
  - RMS and short-term LUFS by turn (pyloudnorm)
  - Gap distribution and overlap ratio
  - Ambiguous Hebrew token presence in `text_spoken`
  - `mix_mode` distribution per scene
  - `normalization_rules_triggered` frequency
- Add run-level aggregation in `QAReport`:
  - F0 distributions by role / voice / intensity / typology / project
  - RMS/LUFS distributions by role and intensity
  - Speaker and backend diversity counts
  - Fraction of clips with interruption
  - Fraction of clips with no measurable loudness escalation (I5 − I1 < 4 dB)
  - Outlier clips by each metric family
- Extend `qa-report` CLI to output run-level summary table

Per-clip warning flags:
- VIC median F0 at I4–I5 > 250 Hz → `WARN_VIC_F0_HIGH`
- AGG RMS range (I5 − I1) < 6 dB → `WARN_AGG_NO_ESCALATION`
- Unvocalized high-risk token in AGG `text_spoken` → `WARN_GENDER_AMBIGUITY`
- Overlap ratio = 0 in scene with I4+ turns → `WARN_NO_OVERLAP`
- Emotional state `neutral` count > low-intensity turn count → `WARN_EMOTION_DOWNGRADE`

#### M11: Dataset Provenance Metadata
**Impact:** Enables ablation studies and debugging across pipeline versions.
**Scope:**
- `GenerationMetadata` dataclass (§4.11)
- Write to `{clip_id}.json` under `generation_metadata` key
- Backward compatible (existing V1 clips are not invalidated)

#### M12: Experimental Breathiness Augmentation
**Impact:** Reduces VIC HNR at I3–I5 if the listening-test gate passes.
**Status:** Experimental. Do not deploy at scale without passing §8.3.
**Scope:**
- `synthbanshee/augment/voice_texture.py` — `add_breathiness(wav_array, sample_rate, level)`
- Wire into `TTSRenderer.render_scene()` for VIC turns, controlled by `SpeakerState.breathiness_level`
- Listening-test gate: 20 clips, native Hebrew listener, blind A/B scoring
- If gate passes: deploy with `breathiness_applied: true` in `GenerationMetadata`
- If gate fails: module remains in codebase as opt-in, disabled by default

#### M13: Project-Specific Default Profiles
**Impact:** She-Proves and Elephant generate audio appropriate to their distinct acoustic regimes.
**Scope:**
- Add `project_profile: Literal["she_proves", "elephant", "generic"] = "generic"` to `RunConfig`
- Gap controller, overlap probabilities, loudness targets, preprocessing config, and augmentation config all have project-specific defaults
- `configs/run_configs/` gains two default profile YAMLs — one per project
- New project profiles can be added without code changes

---

## 7. Run-Level QA and Observability

Beyond per-clip validation, every batch run should produce a distribution summary. This is what the QA pipeline currently lacks entirely.

**Required run-level metrics (produced by `qa-report --run-summary`):**

| Metric | What a bad value means |
|--------|----------------------|
| F0 mean by speaker / intensity | VIC I4+ mean > 250 Hz → female pitch problem persists |
| F0 std by speaker / intensity | AGG I4+ std < 20 Hz → no F0 variance in threats |
| RMS by intensity (per project) | AGG I5 − I1 < 6 dB → loudness escalation not working |
| Overlap ratio | = 0 → barge-in not firing; model will not see interruption |
| Gender ambiguity token count | > 0 → disambiguation not firing or lexicon gap |
| Emotional state distribution | `neutral` dominates high-intensity turns → downgrade still occurring |
| Voice diversity count | Male variants < 3 or female variants < 3 → diversity insufficient |
| Backend diversity | Only one backend → fingerprint risk |
| Clips flagged with any WARN_ | > 10% → systematic pipeline problem |

---

## 8. Release Gates and Training-Use Restrictions

Generated datasets are not interchangeable. A dataset must meet explicit criteria before it is used for a given purpose.

### 8.1 Use Classification

| Class | Requirements | Permitted uses |
|-------|-------------|---------------|
| **Smoke-test only** | Any pipeline version | Pipeline integration checks; label schema testing |
| **Bootstrap / ablation** | M1–M4 complete; ≥ 1 voice per gender | Early ablation studies; feature importance; architecture debugging |
| **Restricted training** | M1–M8 complete; ≥ 3 voice variants per gender; ≥ 2 backends | Training with explicit caveat that results may not generalize |
| **Full training** | All P0–P2 milestones complete; listening-test gate passed; run-level QA clean | Production model training |

### 8.2 Hard Release Gates

A dataset must fail the following checks before training use. If any gate fails, the dataset class is downgraded:
- Unresolved ambiguous Hebrew forms in AGG→VIC `text_spoken` → restricted to smoke-test
- VIC median F0 > 250 Hz in ≥ 20% of I4–I5 turns → bootstrap only
- AGG RMS escalation < 4 dB in ≥ 30% of scenes → bootstrap only
- < 3 distinct voice variants for either gender → bootstrap only (≥ 3 required for restricted training)
- Overlap ratio = 0 across entire run → restricted training only
- `WARN_EMOTION_DOWNGRADE` in > 5% of clips → restricted training only

### 8.3 Listening-Test Gate

Before any dataset version is promoted from "restricted training" to "full training," a listening test must be conducted:
- 20–30 clips per milestone: sampled across intensities, typologies, and projects
- Reviewed by ≥ 1 native Hebrew listener
- Scored on: gender correctness (yes/no), adult-voice plausibility (1–5), distress plausibility at high intensity (1–5), conversational naturalness (1–5), TTS obviousness (1–5, lower is better)
- Pass criteria: gender correctness ≥ 95%; distress plausibility ≥ 3.0 mean at I4–I5; no single clip scores 1 on adult-voice plausibility

---

## 9. Sequencing and Dependencies

```
M1  (disambiguation)   ✅ ─────────────────────────────────────────── P0
M2a (SSML params)      ✅ ─────────────────────────────────────────── P0
M3a (per-turn gain)    ✅ requires M2a ──────────────────────────────── P0
M3b (peak-limiter)     ✅ requires M3a ──────────────────────────────── P0
M4  (emotion metadata)    ─────────────────────────────────────────── P0

M5  (preprocessing)       ──────────────────────────────── P1
M6  (gap controller)      ──────────────────────────────── P1
M7  (speaker state)       requires M2a, M3a ─────────────── P1
M2b (phrase prosody)      requires M1 (text finalized) ──── P1
M8a (overlap mixing)      requires M6 ──────────────────── P1
M8b (overlap labels)      requires M8a ─────────────────── P1

M9a (Google TTS)          ──────────────────────────────── P2
M9b (speaker diversity)   ──────────────────────────────── P2  (independent of M9a; Azure variants can land first)
M10a (per-clip metrics)   parallel, most useful after P0 ── P2
M10b (run-level QA)       requires M10a ───────────────────── P2
M11 (provenance meta)     ──────────────────────────────── P2
M12 (breathiness)         requires M7 (state), gate ──────── P2
M13 (project profiles)    requires M6, M8b ──────────────── P2
```

Fastest path to improved audio: **M3b → M4** (P0 remaining), then **M6 → M7 → M8a → M8b**.

M4 and M5 are structurally independent of the audio quality improvements and can be done at any time.

---

## 10. Summary Table

See **Implementation Tracker** at the top of this document for the full per-PR breakdown with status. Quick reference (milestone-level):

| Milestone | Priority | Primary benefit | Effort | Spec change? |
|-----------|----------|----------------|--------|--------------|
| M1 — Hebrew disambiguation | P0 | Eliminates gender errors | Small | No |
| M2a — SSML parameter redesign | P0 | Lowers VIC F0; adds loudness escalation | Small | No |
| M3a — Per-turn RMS gain | P0 | Preserves loudness trajectory | Small | No |
| M3b — Peak-limiter + spec update | P0 | Prevents peak-normalize from erasing inter-scene contrast | Small | Yes — spec.md §3 |
| M4 — Emotional-state consistency | P0 | Fixes label corruption | Small | No |
| M5 — Tier-aware preprocessing | P1 | Protects Tier A phonetic quality | Small | No |
| M6 — Gap controller | P1 | Psychologically-motivated timing | Small | No |
| M7 — Stateful speaker controller | P1 | Cross-turn continuity | Medium | No |
| M2b — Per-phrase SSML | P1 | Burst-pause microstructure | Medium | No |
| M8a — Overlap audio mixing | P1 | Interruption behavior in audio assembly | Medium | No |
| M8b — Three-timeline labels | P1 | Precise event timestamps for overlapping turns | Medium | No |
| M9a — Google TTS backend | P2 | Second TTS backend; eliminates fingerprint risk | Medium | No |
| M9b — Speaker diversification | P2 | ≥ 3 voice variants per gender | Small | No |
| M10a — Per-clip acoustic metrics | P2 | Catches clip-level prosody regressions | Medium | No |
| M10b — Run-level QA aggregation | P2 | Catches systematic run-level biases | Medium | No |
| M11 — Provenance metadata | P2 | Ablation studies; debugging | Small | No |
| M12 — Breathiness (experimental) | P2 | VIC distress phonation; gate required | Medium | No |
| M13 — Project-specific profiles | P2 | She-Proves / Elephant acoustic fit | Small | No |
