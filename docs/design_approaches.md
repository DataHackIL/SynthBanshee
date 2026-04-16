# Synthetic Dataset Generation Framework — Design Approaches

**Project:** Audio Violence Dataset Project (AVDP)
**Initiatives:** She-Proves · Elephant in the Room
**Organization:** DataHack / DataForBetter (datahack.org.il)
**Status:** Draft — for review by AI team leads (Bar, Livnat, Asya)
**Date:** 2026-04-06

---

## Context & Constraints

Both She-Proves and Elephant in the Room require a labeled Hebrew audio dataset to bootstrap model development. A real-data pipeline (actor recordings guided by domain experts from Rakman Institute and field social workers) is 6–12 months away. This framework must therefore produce synthetic data good enough to:

- Stand up and stress-test the full ingestion → labeling → training → evaluation stack
- Enable architecture comparisons and baseline model training
- Produce a deliberately wide distribution of nuisance factors (voices, rooms, devices, noise, pacing)
- Comply with the schema and labeling standards defined in the AVDP Recording & Tagging Field Specification

The framework is **not** intended to replace actor recordings or real-world data for deployment thresholds or forensic use.

### Hard Requirements from the AVDP Tech Spec

All generated datasets must conform to:

| Requirement | Specification |
|---|---|
| Audio format | WAV, 16 kHz, lossless |
| Directory structure | `data/{Language}/{Speaker_ID}/{clip_id}.wav` + matching `{clip_id}.txt` |
| Filenames | ASCII only, no spaces or characters above U+00A1 |
| Silence padding | ≥ 0.5 s digital silence/ambient baseline before and after target speech |
| SNR | ≥ 15 dB at acquisition; degrade controllably in augmentation |
| Annotation granularity | Weak labels (30–120 s clips) **and** strong labels (onset/offset timestamps), 3.0 s analysis windows, 1.0 s hop |
| Label taxonomy | Hierarchical tiers — `has_violence` is a derived convenience field; the taxonomy (`violence_typology`, `tier1_category`, `tier2_subtype`, `max_intensity`) is the ground truth and must not be replaced by a single binary flag |
| Metadata | JSON per clip with: `event_id`, `onset`, `offset`, `primary_label`, `intensity` (1–5), `speaker_role`, `emotional_state`, `confidence` (0.0–1.0) |
| IAA targets | Cohen's Kappa ≥ 0.65 (physical), ≥ 0.60 (verbal aggression) on 20% second-pass |
| Preprocessing | Resample → normalize to −1.0 dBFS → spectral filter → denoise; always retain original "dirty" file |

### Dataset Tiers (Both Projects)

| Tier | Description | Purpose |
|---|---|---|
| A | Clean synthetic Hebrew dialogue, 2–3 speakers, minimal noise | Get backbone working, architecture exploration |
| B | Same scripts with room acoustics, overlap, phone/device placement effects | Robustness training |
| C | Acoustically intense non-target scenes (false-positive confusors) | Reduce false alarm rate |

### Project-Specific Framing

**She-Proves**: Long-form passive phone audio; sparse, low-base-rate incidents; near-field to far-field variation; optimize for high recall on early-warning cues.

**Elephant in the Room**: Short-form fixed-device (Raspberry Pi mic) interactions; shorter escalation windows; near-real-time alerting; optimize for precision and low false-alarm rate.

---

## Three Design Approaches

---

### Approach 1 — Config-Driven Modular Pipeline ⭐ *Recommended*

#### Philosophy

Every element of a generated clip — script, voices, acoustic environment, label — is determined by a declarative YAML/JSON configuration. The pipeline is a pure function of its config: given the same config and a fixed random seed, you get the same dataset. This makes it straightforward to reproduce, diff, and improve.

The architecture decomposes into five sequential stages (Stage 3 and Stage 4 each have sub-stages):

```
SceneConfig (YAML)
  → [1]  Script Generator      → DialogueTurns      (LLM fills Jinja2 template)
  → [2]  TTS Renderer          → MixedScene WAV     (per-speaker segments mixed)
  → [3a] Preprocessing         → spec-compliant WAV (16 kHz, mono, normalized, silence-padded)
  → [3b] Acoustic Augmentation → augmented WAV      (room IR, device profile, noise) — Tier B only
  → [4a] Transcript Writer     → {clip_id}.txt      (per-turn text with onset/offset markers)
  → [4b] Strong Label Writer   → {clip_id}.jsonl    (per-event EventLabel records)
  → [4c] Metadata Writer       → {clip_id}.json     (ClipMetadata — weak labels, speaker info)
  → [5]  Validator             → pass/fail + warnings
```

Each stage is a Python module with a clean interface; teams can develop and test stages in isolation.

#### Tech Stack

| Stage | Primary Tools | Notes |
|---|---|---|
| 1 — Script generation | Claude API / GPT-4 API, Jinja2 templates, hand-authored Hebrew scene templates | LLM fills parameterized template slots; always constrained by a config schema |
| 2 — Hebrew TTS | Azure Cognitive Services (he-IL voices, SSML), Google Cloud TTS (Chirp 3 HD, he-IL) | Azure SSML supports multi-speaker, style tags, pitch/rate/volume control |
| 3a — Preprocessing | `scipy`, `soundfile` (no torchaudio) | Resample → downmix → low-pass → Wiener denoise → normalize → silence pad |
| 3b — Acoustic augmentation | `pyroomacoustics`, `synthbanshee.augment` | Room IR simulation (ISM), device codec emulation, SNR degradation — Tier B only |
| 4a–4c — Label generation | Python, auto-derived from script + augmentation logs | Onset/offset precise because TTS mixer + Stage 3b are fully logged |
| 5 — Validation & packaging | Python, JSONL, CSV manifests | validate_clip(), manifest CSV, speaker-disjoint splits |

#### Config Schema Sketch

```yaml
# scene_config.yaml
scene_id: she_proves_DV_005
project: she_proves          # she_proves | elephant_in_the_room
violence_type: intimate_terrorism   # situational_violence | intimate_terrorism
language: he                 # he (Hebrew)
speakers:
  - id: SPK_A
    role: aggressor
    gender: male
    age_range: 30-45
  - id: SPK_B
    role: victim
    gender: female
    age_range: 25-40
  - id: SPK_C
    role: bystander
    gender: female
    age_range: 6-10           # child
script_template: templates/she_proves/intimate_terror_escalation_v2.j2
intensity_arc: [1, 2, 2, 3, 4, 5, 4]  # per-segment intensity
duration_minutes: 4.5
tier: B                      # A | B | C
acoustic_scene:
  room_type: apartment_kitchen
  device: phone_in_pocket     # phone_in_hand | phone_on_table | pi_budget_mic
  snr_db: 18
  ir_source: pyroomacoustics  # or a recorded IR file path
  background_events:
    - type: tv_ambient
      onset: 0.0
      level_db: -30
    - type: door_slam
      onset: 112.4
random_seed: 42
```

#### Label Generation

Because the script template encodes the intensity arc and event markers (e.g., `[שבירת זכוכית]`, `[צרחת פאניקה]`), and the TTS renderer logs exact onset/offset of each segment, the pipeline can auto-generate AVDP-compliant strong labels with no human annotation for Tier A clips. Tier B/C clips require a single human validation pass rather than full annotation, substantially reducing the annotation burden.

#### Strengths

- Fastest path to a working dataset; AI teams can start within days
- Fully reproducible — the config IS the dataset provenance
- Easy to extend: add a new violence type by writing a new template and adding an entry to the type registry
- Auto-labeling reduces annotation bottleneck for Phase 0/1
- Clean separation of concerns enables parallel development across teams

#### Weaknesses

- Quality ceiling set by Hebrew TTS expressiveness (Azure/Google he-IL voices are good but not actor-grade)
- Synthetic-to-real gap is highest of the three approaches; models trained here will need significant fine-tuning on actor and real data
- Script diversity depends on LLM + template quality; early outputs may feel formulaic without expert review

#### Suitability Score

| Criterion | Score |
|---|---|
| Time-to-first-dataset | ★★★★★ |
| Acoustic realism | ★★☆☆☆ |
| Label accuracy | ★★★★☆ |
| Scalability | ★★★★★ |
| Maintainability | ★★★★★ |
| **Overall** | **★★★★☆** |

---

### Approach 2 — Graph-Based Scene Orchestrator

#### Philosophy

Instead of treating a scene as a linear script filled from a template, this approach models each scene as a **directed state machine** where nodes represent emotional/intensity states and edges represent transitions (escalation, de-escalation, interruption, bystander entry). The graph is the semantic core of the dataset.

```
Scene Graph (DOT/NetworkX) → State Sequence Sampler
  → per-state Dialogue Generator (LLM)
  → per-state TTS Renderer
  → State Stitcher (cross-fade, overlap, silence)
  → Acoustic Augmenter (shared with Approach 1)
  → Label Aligner (labels derive directly from node/edge metadata)
```

#### Tech Stack

| Component | Primary Tools |
|---|---|
| Scene graph definition | NetworkX, YAML/DOT graph files |
| State machine execution | Custom Python state machine runner |
| Per-state dialogue generation | Claude/GPT-4 with state-aware prompts |
| TTS and augmentation | Same as Approach 1 |
| Label alignment | Node metadata → AVDP JSON; escalation arcs are explicit in graph structure |

#### Scene Graph Example

```
[neutral_conversation] --tension_rises--> [verbal_provocation]
[verbal_provocation]   --escalates-->     [verbal_aggression_L3]
[verbal_provocation]   --de-escalates-->  [neutral_conversation]
[verbal_aggression_L3] --triggers-->      [physical_contact_L4]
[physical_contact_L4]  --peaks-->         [physical_violence_L5]
[physical_violence_L5] --aftermath-->     [distress_crying]
[distress_crying]      --child_enters-->  [bystander_present_distress]
```

Each node carries: intensity level, violence type tags, speaker state (emotional label per speaker), expected duration range, and TTS style directives. Each edge carries: transition trigger type, acoustic event (if any), and probability weight (for stochastic sampling).

#### Why This Matters for AVDP

Escalation arc is one of the most clinically important dimensions in both She-Proves and Elephant in the Room. The graph approach makes escalation **a first-class citizen of the dataset structure** rather than something implied by sequential templates. This means:

- You can explicitly generate datasets balanced across escalation patterns (rapid → slow, single-peak → multi-peak, de-escalating → re-escalating)
- Labels for "escalation onset" are automatically and precisely defined as edge traversal timestamps
- You can deliberately sample rare transition patterns that templates might underrepresent

#### Strengths

- Semantically richest representation; escalation dynamics are explicit and precise
- Labels for escalation onset/peak/recovery are automatically derived from graph traversal
- Easier to reason about dataset balance across scene types
- Aligns naturally with the clinical typology (situational vs. intimate terrorism have different graph topologies)

#### Weaknesses

- More complex to implement than Approach 1; estimated 3–4× more engineering effort for the orchestrator
- Scene graph design requires involvement of domain experts (Shelly, Shiri, field social workers) to validate topologies — this is a bottleneck
- Per-state LLM generation can produce unnatural dialogue transitions at state boundaries without careful prompt engineering
- Returns similar TTS quality to Approach 1

#### Suitability Score

| Criterion | Score |
|---|---|
| Time-to-first-dataset | ★★★☆☆ |
| Acoustic realism | ★★☆☆☆ |
| Label accuracy | ★★★★★ |
| Scalability | ★★★★☆ |
| Maintainability | ★★★☆☆ |
| **Overall** | **★★★☆☆** |

---

### Approach 3 — Asset Library + Soundscape Composer

#### Philosophy

Rather than synthesizing everything from text, this approach treats audio generation as a **sound design problem**: curate a rich library of atomic audio assets (TTS speech segments, sound effects, ambient recordings, noise textures), then compose scenes by layering and arranging those assets using a structured soundscape synthesis tool. Realism comes from the assets themselves, not from a single end-to-end synthesis model.

```
Asset Library (speech + SFX + ambience)
  → Scene Score (Scaper event list or custom YAML timeline)
  → Soundscape Composer (Scaper + pydub)
  → Post-Processor (room IR, device codec, normalization)
  → Label Mapper (event list → AVDP JSON)
```

#### Tech Stack

| Component | Primary Tools |
|---|---|
| Speech assets | Azure/Google TTS (pre-rendered per utterance), optional short human recordings |
| Sound effect assets | MUSAN (noise/music/speech), FSD50K (annotated sound events), Freesound (licensed), in-house safe recordings |
| Ambient assets | Recorded benign environments (office chatter, clinic waiting room, kitchen) |
| Soundscape composition | `Scaper` (DCASE-standard tool for structured soundscape synthesis) |
| Acoustic post-processing | `pyroomacoustics`, `torchaudio`, `ffmpeg` |
| Label mapping | Scaper event manifests → AVDP JSON (direct mapping since Scaper logs exact onsets) |

#### Asset Library Structure

```
assets/
  speech/
    he/
      SPK_aggressor_M_30-45/     # pre-rendered TTS utterances by speaker persona
      SPK_victim_F_25-40/
      SPK_bystander_child_F_8/
  sfx/
    physical/
      glass_break_01.wav
      door_slam_02.wav
      object_throw_03.wav
    vocal/
      scream_panic_F_01.wav
      sob_distress_F_01.wav
      grunt_struggle_M_01.wav
  ambient/
    apartment_kitchen_day.wav
    clinic_waiting_room.wav
    office_corridor.wav
  noise/
    musan_noise_subset/
    traffic_street_tel_aviv.wav
```

#### Why Tier C (Hard Negatives) Is Best Here

The asset library approach is uniquely powerful for generating **acoustic confusors** (Tier C): children crying ≠ fear crying, sports yelling ≠ aggression, breaking dishes while cooking ≠ thrown in violence. Because every element in a scene is a labeled asset, you can compose a scene that is acoustically intense but semantically non-violent with complete label precision. This is where Approaches 1 and 2 are most likely to produce mislabeled or ambiguous confusor examples.

#### Strengths

- Best acoustic realism — real SFX and ambient assets produce more convincing scenes
- Best Tier C (hard negative) generation — precise control over acoustic-vs-semantic alignment
- Label accuracy is very high for non-speech events because asset onsets are known exactly
- Aligns with DCASE methodology (well-validated for sound event detection datasets)

#### Weaknesses

- Highest upfront effort: asset library must be curated, cleared for licensing, and quality-controlled before generation begins
- Slower iteration: adding a new scene type requires sourcing and processing new assets
- Speech quality at scene junctions is lower than Approaches 1/2 — pre-rendered utterances are less contextually adaptive
- Scaper has a learning curve; team must onboard to DCASE-style scene description formats

#### Suitability Score

| Criterion | Score |
|---|---|
| Time-to-first-dataset | ★★☆☆☆ |
| Acoustic realism | ★★★★☆ |
| Label accuracy | ★★★★☆ |
| Scalability | ★★★☆☆ |
| Maintainability | ★★★☆☆ |
| **Overall** | **★★★☆☆** |

---

## Ranking & Recommendation

| Rank | Approach | Overall | Primary Reason |
|---|---|---|---|
| 🥇 1 | Config-Driven Modular Pipeline | ★★★★☆ | Fastest time-to-working-dataset; fully reproducible; low implementation overhead; AI teams can start within days of spec finalization |
| 🥈 2 | Graph-Based Scene Orchestrator | ★★★☆☆ | Semantically superior for escalation modeling, but implementation complexity and domain-expert dependency create a bottleneck for Phase 0 |
| 🥉 3 | Asset Library + Soundscape Composer | ★★★☆☆ | Best acoustic realism and hard-negative quality, but asset curation cost makes it impractical as the primary bootstrap approach |

### Recommended Hybrid Path

The three approaches are not mutually exclusive. The recommended strategy is:

**Phase 0 (Months 0–2): Approach 1** — implement the Config-Driven Pipeline end-to-end. Produce Tier A/B datasets for both projects. Get AI teams training baselines.

**Phase 1 (Months 2–4): Layer in Approach 2 concepts** — add a lightweight scene-graph layer on top of Approach 1's pipeline. Use it to improve escalation arc balance and produce precise escalation-onset labels. Domain experts can review/edit graphs without touching code.

**Phase 1–2 (Ongoing): Layer in Approach 3 assets** — curate a targeted sound effects library (physical violence SFX, Hebrew vocal distress events) and integrate with the Approach 1 compositor. Prioritize Tier C hard-negative generation where Approach 1 is weakest.

---

## Next Documents

This design document will be followed by:

1. **`spec.md`** — Full dataset specification: directory structure, metadata schema, label taxonomy, preprocessing checklist, split strategy, per-project label variants (She-Proves vs. Elephant in the Room)
2. **`implementation_plan.md`** — Phased implementation plan with milestones, module breakdown, dependencies, and team assignments

---

*Document prepared for DataHack AVDP — not for distribution outside the project team.*
