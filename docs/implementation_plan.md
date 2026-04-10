# AVDP Synthetic Framework — Implementation Plan

**Project:** Audio Violence Dataset Project (AVDP)
**Initiatives:** She-Proves · Elephant in the Room
**Organization:** DataHack / DataForBetter (datahack.org.il)
**Status:** Phase 0 complete (2026-04-07) — Phase 1 complete (2026-04-08)
**Date:** 2026-04-06 (updated 2026-04-08)
**Companion documents:** `design_approaches.md`, `spec.md`

---

## Guiding Principles

**Domain randomization over realism theater.** The goal is not to produce clips that sound perfectly real. The goal is a wide, deliberate distribution of voices, acoustic conditions, pacing patterns, interruption types, and non-target confounders. Synthetic-to-real gap is real and expected; design for it from the start.

**The config is the provenance.** Every generated clip must be fully reproducible from its YAML scene config and a fixed random seed. If you can't reproduce a clip, you can't understand where your dataset came from.

**Labels derive from structure, not from post-hoc review.** Tier A labels are auto-generated from the scene script and augmentation log. Human review is reserved for validation, not primary labeling. This keeps the labeling bottleneck out of the critical path.

**Develop in modules, test in integration.** Each pipeline stage (Stage 1: script generation, Stage 2: TTS rendering, Stage 3a/3b: preprocessing and acoustic augmentation, Stage 4a–4c: labeling and transcript, Stage 5: validation and packaging) has its own interface and tests. Integration is a separate step.

---

## Repository Structure

```
SynthBanshee/
├── synthbanshee/                 ← main Python package
│   ├── __init__.py
│   ├── config/
│   │   ├── scene_config.py       # Pydantic models for scene YAML configs
│   │   ├── speaker_config.py     # Speaker persona definitions
│   │   ├── acoustic_config.py    # Acoustic scene definitions
│   │   ├── run_config.py         # Batch run configuration (RunConfig)
│   │   └── taxonomy.py           # Taxonomy loader (single source of truth)
│   ├── script/
│   │   ├── generator.py          # LLM-based script generation (ScriptGenerator)
│   │   ├── types.py              # DialogueTurn, MixedScene dataclasses
│   │   └── templates/            # Jinja2 scene templates (.j2 files)
│   │       ├── she_proves/
│   │       └── elephant/
│   ├── tts/
│   │   ├── renderer.py           # TTSRenderer (single + multi-speaker)
│   │   ├── mixer.py              # SceneMixer (concatenate segments → MixedScene)
│   │   ├── azure_provider.py     # Azure Cognitive Services (he-IL SSML)
│   │   ├── google_provider.py    # Google Cloud TTS (he-IL Chirp 3 HD)
│   │   └── ssml_builder.py       # SSML document generation
│   ├── augment/
│   │   ├── preprocessing.py      # Resample → downmix → filter → denoise → normalize → pad
│   │   ├── room_sim.py           # pyroomacoustics room IR simulation (Phase 2)
│   │   ├── device_profiles.py    # Device codec / placement emulation (Phase 2)
│   │   └── noise_mixer.py        # Background noise / SFX layering (Phase 2)
│   ├── labels/
│   │   ├── generator.py          # LabelGenerator (ScriptEvent → EventLabel → ClipMetadata)
│   │   ├── schema.py             # AVDP label schema (Pydantic)
│   │   └── iaa.py                # Inter-annotator agreement utilities
│   ├── package/
│   │   ├── validator.py          # validate_clip() — three-file spec check
│   │   ├── manifest.py           # generate_manifest() → CSV
│   │   ├── splitter.py           # assign_splits() — speaker-disjoint Union-Find
│   │   └── qa.py                 # run_qa() → QAReport
│   └── cli.py                    # Click CLI (generate, generate-batch, validate, qa-report)
├── configs/
│   ├── scenes/                   # Per-scene YAML configs
│   ├── speakers/                 # Speaker persona YAMLs
│   ├── acoustic_scenes/          # Room/device YAML configs
│   ├── run_configs/              # Full generation run definitions (RunConfig YAML)
│   ├── taxonomy.yaml             # Label taxonomy — single source of truth
│   └── examples/                 # Worked examples for each config type
├── assets/                       # Source assets (gitignored; populated at runtime)
│   ├── speech/                   # TTS utterance cache (SHA-256 keyed)
│   ├── scripts/                  # LLM script generation cache
│   ├── sfx/                      # Sound effects (licensed CC0/CC-BY)
│   ├── ambient/                  # Ambient/background audio
│   └── noise/                    # MUSAN noise subset
├── data/                         # Generated dataset output (gitignored)
├── tests/
│   ├── unit/
│   └── integration/
├── docs/                         # This directory
├── pyproject.toml
└── README.md
```

---

## Phase 0 — Foundation (Weeks 1–3)

**Goal:** Everything needed to generate a single valid, spec-compliant clip. No scale, no variety — just correctness.

### 0.1 Environment & Tooling Setup

**Owner:** Any AI team lead
**Duration:** 2 days

- Set up Python ≥ 3.11 environment, `pyproject.toml`, `ruff` linting, `pytest`
- Install core dependencies: `pydantic`, `jinja2`, `click`, `soundfile`, `numpy`, `scipy` (note: **no torchaudio** — preprocessing uses scipy+soundfile exclusively to avoid torch version incompatibilities; `pyroomacoustics` and `scaper` are Phase 2 dependencies)
- Provision TTS API access: Azure Cognitive Services (he-IL voices: `he-IL-AvriNeural`, `he-IL-HilaNeural`) and Google Cloud TTS (he-IL)
- Provision LLM API access (Claude or GPT-4) for script generation
- Acquire MUSAN noise corpus (freely available): `wget http://www.openslr.org/17/`
- Set up project secrets management (`python-dotenv` or environment variables)

**Acceptance criteria:** `pytest tests/unit/test_setup.py` passes; Azure TTS returns audio for a Hebrew test sentence.

**✓ Complete** — environment configured with uv; CI workflow runs ruff, mypy, and pytest across Python 3.11 and 3.12.

### 0.2 Config Schema & Validation

**Owner:** Bar / Livnat
**Duration:** 3 days

- Implement Pydantic models for `SceneConfig`, `SpeakerConfig`, `AcousticConfig` (see `spec.md §2`)
- Write config YAML loaders with validation
- Write 5–10 hand-authored minimal scene configs (not yet used for generation — for validation testing only)
- Unit tests: invalid configs are rejected with descriptive errors; valid configs round-trip serialize/deserialize

**Acceptance criteria:** All hand-authored configs parse without errors; invalid configs (bad intensity range, missing required fields) raise `ValidationError`.

**✓ Complete** — `SceneConfig`, `SpeakerConfig`, `AcousticSceneConfig` implemented with full taxonomy-backed validation. Example configs in `configs/examples/`.

### 0.3 TTS Renderer (Single Speaker)

**Owner:** Asya
**Duration:** 4 days

- Implement `azure_provider.py`: accepts a list of `(text, speaker_persona, style_directives)` tuples, returns WAV byte streams
- Implement SSML builder: generates SSML documents with `<voice>`, `<prosody>` (rate, pitch, volume), `<mstts:express-as>` style tags for he-IL voices
- Implement rendering cache: skip re-rendering utterances that already exist in `assets/speech/` (keyed on **SHA-256 of the full rendered SSML string**, which captures voice, text, style, prosody, and randomization — not just `(voice_id, text)`)
- Implement single-speaker rendering test: render 3 Hebrew utterances at different intensity levels

**Acceptance criteria:** Three Hebrew utterances rendered as valid 16-bit PCM WAV files at 24 kHz (pre-downsampling); cache prevents re-render on second run.

**✓ Complete** — `SSMLBuilder`, `AzureProvider` (with `sdk_factory` injection for testing), and `TTSRenderer` implemented with SSML-keyed filesystem cache.

### 0.4 Preprocessing Pipeline

**Owner:** Bar
**Duration:** 2 days

- Implement `preprocessing.py`: resample → downmix → low-pass filter → Wiener denoise → normalize → pad → validate
- Retain "dirty" originals in `assets/` before processing, named `{output_stem}_dirty{original_suffix}` to avoid filename collisions across pipeline runs
- Validate output against spec constraints (sample rate, bit depth, mono, normalization, SNR check, silence padding)

**Acceptance criteria:** A noisy TTS output passes through pipeline and emerges as a valid 16 kHz mono WAV at −1.0 dBFS with ≥ 0.5 s padding; dirty file retained.

**✓ Complete** — pure scipy+soundfile implementation (no torchaudio); `validate_audio()` uses the file's actual sample rate for all duration and padding calculations.

### 0.5 Label Schema & Auto-Generator (Stub)

**Owner:** Livnat
**Duration:** 2 days

- Implement Pydantic models for `EventLabel`, `WeakLabel`, `ClipMetadata` matching `spec.md §5`
- Implement stub auto-label generator: given a script structure (list of events with types, onset/offset from TTS render log), produce valid AVDP-schema JSONL
- Write round-trip test: generate labels, serialize to JSONL, deserialize, assert equality

**Acceptance criteria:** Labels for a 3-event script serialize to valid JSONL and round-trip correctly; schema validation rejects malformed labels.

**✓ Complete** — `EventLabel`, `WeakLabel`, `ClipMetadata` implemented; `LabelGenerator` produces JSONL and JSON with full round-trip fidelity. ASCII-safe validation covers `clip_id`, `project`, `tts_engine`, `violence_typology`, and `generator_version`.

### 0.6 End-to-End "Happy Path" Test

**Owner:** All
**Duration:** 1 day

- Wire together: config → TTS render → preprocessing → label generation → write output files
- Assert: `.wav` + `.txt` + `.json` all present with correct naming; manifest CSV row is valid; label JSONL parses

**Acceptance criteria:** Running `synthbanshee generate --config configs/scenes/test_scene_001.yaml` produces a complete, spec-compliant clip.

**✓ Complete** — `validate_clip()` checks: all three files present, WAV passes `validate_audio()`, JSON parses as `ClipMetadata` with `is_synthetic=True`, filename stem is ASCII-only lowercase. 80 unit + integration tests pass on Python 3.11 and 3.12.

---

## Phase 1 — Tier A Dataset (Weeks 4–7)

**Goal:** Generate 500–1,000 clean (Tier A) clips per project, sufficient for teams to train and compare first baseline models.

### 1.1 Script Template Library

**Owner:** Livnat (engineering lead) + field expert review
**Duration:** 1 week

Write **20–30 Jinja2 scene templates per project** covering the required violence typologies:

**She-Proves templates (20+):**
- Intimate terrorism: coercive control, financial threat, isolation demand, gaslighting exchange, chronic threat cycle
- Situational violence: argument escalation (kitchen), argument escalation (bedroom), return-home confrontation
- Neutral controls: everyday conversation, planning discussion, light disagreement
- Confusor stubs (for Tier C later): argument that de-escalates, TV drama playing

**Elephant in the Room templates (20+):**
- Client attack: benefits denial → escalation → physical assault
- Intake gone wrong: routine intake → coercive demand → attack
- Crisis visit: agitated client, escalation, social worker at risk
- Animated non-violent: frustrated client resolves without violence
- Neutral controls: routine intake, follow-up visit, calm crisis support

Each template defines:
- Parameterized slots: relationship type, names, specific grievance, outcome
- Intensity arc: ordered list of per-segment intensity values
- Event markers: `[ACTION: TYPE | INTENSITY: N]` inline in dialogue
- Stage labels: `[PHASE: escalation]` at phase boundaries

**Domain expert review:** Share templates with Rakman Institute contacts (Dr. Shelly Blat-Zak, Dr. Omer Shaked) for plausibility review. This is not a blocking dependency for Phase 1 generation — use initial AI-reviewed templates and flag for expert review in parallel.

**Acceptance criteria:** 20 templates per project; each template renders a syntactically valid script; event markers map to AVDP taxonomy.

**✓ Complete** — 26 Jinja2 templates across both projects in `synthbanshee/script/templates/`; each template produces a syntactically valid script with AVDP-taxonomy event markers and intensity arcs.

### 1.2 Multi-Speaker TTS Rendering

**Owner:** Asya
**Duration:** 4 days

- Extend TTS renderer to handle multi-speaker SSML documents (Azure supports multiple `<voice>` blocks in one SSML document)
- Implement speaker turn interleaving: respect timing from script (silence gaps between turns, overlap markers)
- Implement prosody variation: randomize pitch ±10%, rate ±15%, volume ±5 dB within persona bounds
- Add disfluency injection: occasional filled pauses (`אממ`, `אה`), false starts, truncations (configurable probability)

**Acceptance criteria:** A 2-speaker dialogue renders with distinct voices, natural timing, and audible prosody variation across multiple renders with different random seeds.

**✓ Complete** — `TTSRenderer.render_scene()` renders each turn individually (with SSML-keyed cache), then `SceneMixer.mix_sequential()` concatenates segments into a `MixedScene` with per-turn onset/offset metadata. Disfluency injection and prosody randomization via `rng_seed`.

### 1.3 LLM Script Generator

**Owner:** Bar
**Duration:** 5 days

- Implement `generator.py`: given a scene config (template path + slot values), calls LLM API to fill template slots and generate a complete Hebrew dialogue
- Implement Hebrew constraint validation: check output for transcript redundancy violations (no 4+ consecutive identical tokens), ASCII-safe metadata
- Implement generation cache (keyed on template + config hash) to avoid re-generating identical scenes
- Generate 50 scripts per project using the template library; review 10% manually for plausibility

**LLM prompting strategy:**
- System prompt: Hebrew social work domain expert role, with constraints from scripting instructions doc
- User prompt: template with slot descriptions + intensity arc + clinical notes from template
- Post-process: validate event markers are present and map to taxonomy; regenerate if validation fails (max 3 attempts)

**Acceptance criteria:** 50 scripts per project generated; zero transcript redundancy violations; 10% manual spot-check passes plausibility review.

**✓ Complete** — `ScriptGenerator` in `synthbanshee/script/generator.py`; SHA-256 generation cache in `assets/scripts/`; output validated as `list[DialogueTurn]`; wired end-to-end into `_run_generate_pipeline()` in `cli.py`. `_TYPOLOGY_INTENSITY_MAP` in `cli.py` maps (typology, intensity) → taxonomy codes, validated against `configs/taxonomy.yaml` at import time.

### 1.4 Tier A Dataset Generation Run

**Owner:** All
**Duration:** 3 days

- Configure a full generation run: 500 clips per project, Tier A (clean, no acoustic augmentation), stratified across violence typologies and intensity bands
- Assign speaker personas to clips; enforce speaker-disjoint splits at assignment time
- Run end-to-end pipeline; collect any errors or quality flags
- Generate manifest CSV and label JSONL files

**Scale targets per project (Tier A):**

| Violence typology | Clips | Notes |
|---|---|---|
| Intimate Terrorism (IT) | 150 | |
| Situational Violence (SV) | 150 | |
| Negative / Confusor (NEG) | 150 | Tier C stubs at Tier A quality |
| Neutral (NEU) | 50 | |
| **Total** | **500** | |

**Acceptance criteria:** 500 clips per project; full manifest and label files present; < 2% clips with quality flags requiring exclusion.

**✓ Complete** — `RunConfig` Pydantic model (`synthbanshee/config/run_config.py`) with per-typology `TypologyTarget` and `SplitFractions` validator. `generate-batch` CLI command: discovers scene configs by typology, applies per-typology count caps, renders clips, assigns speaker-disjoint splits via Union-Find, and writes manifest CSV. Run configs for both projects at `configs/run_configs/`. `assign_splits()` in `synthbanshee/package/splitter.py` uses path-halving Union-Find to guarantee speaker-disjoint train/val/test partitions.

### 1.5 Dataset QA & Delivery

**Owner:** Bar + Livnat
**Duration:** 2 days

- Run automated QA suite: sample rate, channel, duration, normalization, silence padding, filename constraints, label schema validity, manifest completeness
- Compute dataset statistics: duration distribution, intensity distribution, violence category distribution, speaker distribution
- Produce QA report notebook
- Package and deliver to AI teams as a versioned archive (`avdp_synth_v0.1_tier_a.tar.gz`)

**Acceptance criteria:** QA suite passes with < 1% clip failure rate; statistics report shows balanced distribution across stratification variables; AI teams confirm they can load the dataset.

**✓ Complete** — `run_qa()` in `synthbanshee/package/qa.py` validates every clip via `validate_clip()`, re-parses `ClipMetadata`, accumulates `DatasetStats` (total/failed clips, duration, typology/split/speaker counts, quality-flagged clips), and returns a `QAReport` (pass/fail based on configurable `max_failure_rate`, default 2%). `qa-report` CLI command prints a Rich table summary and optionally writes a JSON report file; exits 1 if the report fails. `generate_manifest()` in `synthbanshee/package/manifest.py` produces a flat CSV with 12 columns (clip_id, project, violence_typology, tier, duration_seconds, speaker_ids, has_violence, max_intensity, quality_flags, split, wav_path, strong_labels_path).

---

## Phase 2 — Tier B & Acoustic Augmentation (Weeks 8–11)

**Goal:** Add acoustic realism to Tier A scenes. Produce 1,000–2,000 additional Tier B clips per project. These are the primary robustness training data.

### 2.1 Room Simulation Module

**Owner:** Asya
**Duration:** 5 days

- Implement `room_sim.py` using `pyroomacoustics`:
  - Support 6 room presets: `small_bedroom`, `kitchen`, `living_room`, `clinic_office`, `welfare_office`, `open_office_corridor`
  - Support 3 source distances: near (0.5 m), mid (1.5 m), far (3.5 m)
  - Randomize room dimensions, absorption coefficients, and source/mic positions within preset bounds
  - Generate room impulse response (RIR) and convolve with clean speech
- Implement device profiles in `device_profiles.py`:
  - `phone_in_hand`: slight bandpass, low compression
  - `phone_in_pocket`: strong muffling (cloth occlusion), reduced high-frequency content
  - `phone_on_table`: moderate room pickup, surface reflection
  - `pi_budget_mic`: budget mic frequency response, electrical hum floor, AGC-like level pumping

**Acceptance criteria:** 6 room presets × 3 distances × 4 device profiles produce audibly distinct outputs; SNR is logged for each augmented clip.

### 2.2 Background Noise & SFX Mixer

**Owner:** Bar
**Duration:** 4 days

- Implement `noise_mixer.py` using `Scaper` or direct `pydub` composition:
  - Layer MUSAN noise (traffic, HVAC, chatter) at configurable SNR
  - Insert SFX events (glass break, door slam, object throw) at script-specified onsets
  - Mix ambient loops (TV audio, radio, children playing) at configurable levels
  - Ensure SFX onset times are logged for automatic label alignment

**SFX asset requirements (to be licensed/recorded before Phase 2):**
- Glass breaking: 5 variants
- Door slam: 5 variants
- Object impact/throw: 5 variants
- Footsteps (rapid): 3 variants

Licensed sources: Freesound.org (CC0/CC-BY), BBC Sound Effects library (open for research), in-house safe recordings.

**Acceptance criteria:** A Tier B clip contains audible background noise, at least one SFX event matching its label, and all SFX onsets are present in the strong label JSONL.

### 2.3 Tier B Generation Run

**Owner:** All
**Duration:** 4 days

Re-render all Tier A scene configs through the full augmentation pipeline. Each Tier A clip produces 2–4 Tier B variants with different room/device/noise combinations.

**Scale targets per project (Tier B):**

| Augmentation dimension | Variants |
|---|---|
| Room × distance | 6 rooms × 3 distances = 18 combinations |
| Device profile | 3–4 profiles |
| Background noise level | 3 SNR levels (−5 dB, 0 dB, +10 dB relative to speech) |
| Per-scene sample | 2–4 variants drawn from above |

Target: **1,000–1,500 Tier B clips per project.**

**Acceptance criteria:** Labels for Tier B clips are identical to their Tier A parent at the event level, with updated `acoustic_scene` metadata block; QA suite passes.

---

## Phase 3 — Tier C Hard Negatives & Scale-Up (Weeks 12–16)

**Goal:** Build a comprehensive hard-negative (confusor) set. Scale total dataset to 5,000–20,000 clips per project. Deliver the complete Phase 1 dataset to AI teams.

### 3.1 Tier C Template Library

**Owner:** Livnat + field expert review
**Duration:** 1 week

Write **15–20 dedicated confusor templates per project** based on the confusor taxonomy in `spec.md §9`. Key templates:

- Argument that de-escalates before any violence (both projects)
- Child tantrum / loud play in background (She-Proves)
- Sports yelling (TV on in background, She-Proves)
- Grief crying (non-violence context, both)
- Animated clinic intake with agitated but non-violent client (Elephant)
- Social worker supporting client in distress — no aggression (Elephant)
- Cooking accident (crockery breaking, non-violent, She-Proves)
- Laughter burst (acoustically resembles screaming)

These templates require the most careful clinical review because they define the boundary between "alert" and "no alert." Expert review by Rakman Institute is strongly recommended before generation.

**Acceptance criteria:** 15 confusor templates per project; field expert has reviewed ≥ 50% of templates; each template has at least one "why this is NOT violence" annotation note.

### 3.2 Scale-Up Generation

**Owner:** All
**Duration:** 2 weeks

- Scale to full dataset targets using the complete template library + augmentation pipeline
- Implement parallelized generation (multiprocessing or async, rate-limited by TTS API quotas)
- Monitor TTS API costs; cache aggressively

**Final scale targets (combined Tier A + B + C):**

| Project | Tier A | Tier B | Tier C | Total |
|---|---|---|---|---|
| She-Proves | 1,000 | 2,000 | 1,000 | 4,000 |
| Elephant in the Room | 1,000 | 2,000 | 1,000 | 4,000 |
| **Combined** | **2,000** | **4,000** | **2,000** | **8,000** |

These are Phase 3 targets. Phase 1 (Tier A only, 500/project) is sufficient to start model development.

### 3.3 Full IAA Pass

**Owner:** Livnat + field expert annotators
**Duration:** 2 weeks (parallel with scale-up)

- Sample 20% of Tier B/C clips for human annotation review
- Compute Cohen's Kappa for each event category
- Flag and exclude clips below minimum acceptable κ
- Produce IAA report; escalate disagreements to Rakman Institute expert

**Acceptance criteria:** IAA targets from `spec.md §6.2` met for ≥ 80% of reviewed categories; disagreement rate < 10%.

### 3.4 Final Dataset Package

**Owner:** Bar
**Duration:** 3 days

- Full QA suite on complete dataset
- Generate versioned archive: `avdp_synth_v1.0.tar.gz`
- Write dataset card (HuggingFace format) documenting: generation methodology, class distribution, known limitations, intended use, license
- Upload to shared storage; notify AI teams

---

## Phase 4 — Scene Graph Layer (Months 4–6, Parallel Track)

This phase implements the Approach 2 scene graph concepts on top of the Approach 1 pipeline. It is a parallel enhancement, not a blocker for Phase 1–3.

### 4.1 Scene Graph Definition Format

- Define scene graphs as NetworkX DiGraphs serialized to YAML
- Node schema: `state_id`, `intensity`, `phase`, `violence_categories`, `typical_duration_range`, `tts_style_directives`
- Edge schema: `trigger_type`, `acoustic_event`, `transition_probability`

### 4.2 Graph-Driven Scene Sampler

- Replace linear intensity arc in scene config with a graph traversal sampler
- Stochastic path sampling: sample paths weighted by transition probabilities
- Guarantee coverage: ensure each scene type's graph has all critical paths covered across the dataset

### 4.3 Escalation Arc Labels

- Auto-generate `scene_phase` and escalation onset/offset labels from graph traversal logs
- Produce per-clip escalation arc as a sequence: `[(phase, intensity, onset, offset), ...]`
- Store in clip JSON metadata under `escalation_arc` key

---

## Dependency Map

```
Phase 0.1 (env setup)
  └─→ Phase 0.2 (config schema)
        ├─→ Phase 0.3 (TTS renderer)
        │     └─→ Phase 0.4 (preprocessing)
        │           └─→ Phase 0.5 (label schema)
        │                 └─→ Phase 0.6 (happy path test)
        │                       └─→ Phase 1.1 (templates)
        │                             ├─→ Phase 1.3 (LLM generator)
        │                             │     └─→ Phase 1.4 (Tier A generation)
        │                             │           └─→ Phase 1.5 (QA & delivery) ← TEAMS CAN START HERE
        │                             │                 └─→ Phase 2.1 (room sim)
        │                             │                       └─→ Phase 2.2 (noise mixer)
        │                             │                             └─→ Phase 2.3 (Tier B)
        │                             │                                   └─→ Phase 3.2 (scale-up)
        │                             │                                         └─→ Phase 3.4 (final package)
        │                             └─→ Phase 1.2 (multi-speaker TTS)
        └─→ Phase 3.1 (Tier C templates) → Phase 3.2 (scale-up)
                                         → Phase 3.3 (IAA)
Phase 4 (scene graph) runs in parallel with Phases 2–3
```

**AI teams can begin model development after Phase 1.5** (Tier A delivery). They do not need to wait for Tier B or Tier C.

---

## API Cost Estimates

These are rough estimates for planning purposes. Cache aggressively to reduce costs.

| Service | Volume | Estimated cost |
|---|---|---|
| Azure TTS (he-IL) | ~50 hours of audio (8,000 clips × 4 min avg) | ~$240 (at standard pricing) |
| LLM script generation (Claude/GPT-4) | ~8,000 scripts × ~500 tokens avg | ~$40–80 |
| Google TTS (comparison runs) | ~5 hours for evaluation | ~$30 |
| **Total** | | **~$300–350** |

Caching TTS outputs (same text + same voice = same file) will substantially reduce re-generation costs across dataset versions.

---

## Known Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Hebrew TTS expressiveness insufficient for distress events | Medium | Medium | Evaluate both Azure and Google; post-process with prosody manipulation; flag as known limitation in dataset card |
| LLM-generated Hebrew scripts contain unnatural phrasing | Medium | Low–Medium | Domain expert spot-check of 10% of scripts; iterative prompt improvement |
| TTS API rate limits slow scale-up | Medium | Low | Implement async generation with rate limiting; cache aggressively; consider parallel Azure regions |
| SFX asset licensing issues | Low | Medium | Use only CC0/CC-BY assets; document license for every asset in manifest |
| Domain expert availability bottleneck (Rakman Institute) | Medium | Medium | De-scope expert review from the critical path; flag reviewed vs. unreviewed templates in metadata |
| Synthetic-to-real gap larger than expected | High | Medium | This is expected and acceptable for Phase 0–1; document explicitly in dataset card; plan actor recording phase |

---

## Milestones

| Milestone | Target date | Deliverable | Status |
|---|---|---|---|
| M0: Happy path test passing | Week 3 | Single spec-compliant clip generated end-to-end | **Done** |
| M1: Tier A dataset delivered | Week 7 | 500 clips/project, Tier A, AI teams can train baselines | **Done** |
| M2: Template expert review complete | Week 9 | 20+ templates/project reviewed by Rakman Institute |
| M3: Tier B dataset delivered | Week 11 | 1,000–1,500 clips/project, Tier B augmented |
| M4: Phase 1 complete dataset | Week 16 | 4,000 clips/project, all tiers, IAA done, dataset card published |
| M5: Scene graph layer (Phase 4) | Month 6 | Graph-driven escalation labels available |

---

*Document prepared for DataHack AVDP — not for distribution outside the project team.*
