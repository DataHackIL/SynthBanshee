# AVDP Synthetic Dataset Framework — Codex Context

## What this repo is

A framework for generating large-scale synthetic Hebrew audio datasets for two AI safety initiatives run by DataHack (datahack.org.il):

- **She-Proves** — passively monitors a smartphone for domestic violence incidents and preserves audio evidence for legal use
- **Elephant in the Room (הפיל שבחדר)** — a Raspberry Pi–class device in clinic/welfare offices that alerts security when a social worker is being attacked

This is Phase 0 / Phase 1 bootstrap data. The goal is a wide, deliberate distribution of voices, acoustic conditions, and violence types — not forensic-grade realism. A real-data pipeline (actor recordings) is 6–12 months away. These docs and this code give the AI teams something to train on now.

## Read these first

All design decisions are documented. Read before changing anything structural:

| Document | What it covers |
|---|---|
| `docs/design_approaches.md` | The three design approaches considered; why Approach 1 (Config-Driven Pipeline) was chosen; the recommended hybrid path |
| `docs/spec.md` | The authoritative schema: file naming, audio format, label taxonomy, metadata JSON schema, IAA protocol, split strategy, per-project variants |
| `docs/implementation_plan.md` | Phased milestones, module breakdown, dependency map, API cost estimates, risk register |

## Repo structure

```
synthbanshee/              ← main Python package
  config/                  ← Pydantic config models
  script/                  ← LLM-based script generation + Jinja2 templates
  tts/                     ← TTS rendering (Azure he-IL, Google he-IL)
  augment/                 ← acoustic augmentation (room sim, device profiles, noise)
  labels/                  ← auto-label generation, schema, IAA utilities
  package/                 ← dataset assembly, manifests, splits
  cli.py                   ← Click CLI entry points

configs/
  scenes/                  ← per-scene YAML configs (see examples/)
  speakers/                ← speaker persona YAMLs (see examples/)
  acoustic_scenes/         ← room/device YAML configs
  examples/                ← worked examples for each config type

assets/                    ← source assets (gitignored; populated at runtime)
  speech/                  ← TTS utterance cache
  sfx/                     ← sound effects (licensed CC0/CC-BY)
  ambient/                 ← ambient/background audio
  noise/                   ← MUSAN noise subset

data/                      ← generated dataset output (gitignored)
docs/                      ← design, spec, implementation plan
tests/
  unit/
  integration/
```

## Language

All generated audio is **Hebrew (he-IL)**. Transcripts are UTF-8 Hebrew. Filenames and metadata strings must be ASCII only (no UTF-8 above U+00A1 in filenames or JSON keys/values — Hebrew text goes in transcript `.txt` files only, not in filenames or metadata string fields).

## The four pipeline stages

Every clip is produced by four sequential stages. Keep them modular with clean interfaces.

```
SceneConfig (YAML)
  → [1] Script Generator   → dialogue JSON  (LLM fills Jinja2 template)
  → [2] TTS Renderer       → per-speaker WAV segments (Azure/Google he-IL)
  → [3] Acoustic Augmenter → augmented scene WAV (room IR, device profile, noise)
  → [4] Label Generator    → AVDP-schema JSONL (auto-derived from script + augmentation log)
```

The augmentation log from Stage 3 is the source of truth for SFX onset/offset times in Stage 4 labels. Don't derive label timings from anything else.

## Key schema constraints (from spec.md)

- Audio: **16 kHz, mono, 16-bit PCM WAV, −1.0 dBFS peak normalized**
- Directory: `data/{language_code}/{speaker_id}/{clip_id}.wav` + matching `.txt` + `.json`
- Filenames: **ASCII only**, no spaces, no UTF-8 above U+00A1, lowercase
- Every `.wav` must have a matching `.txt` (transcript) and `.json` (metadata)
- **No binary Violence/Non-Violence labels** — use the hierarchical taxonomy in `configs/taxonomy.yaml`
- Silence padding: **≥ 0.5 s** ambient baseline before and after target speech
- Retain "dirty" (pre-preprocessing) files in `assets/` always
- `is_synthetic: true` in all generated clip metadata

## Label taxonomy

The full hierarchical taxonomy is in `configs/taxonomy.yaml`. Always import from there — do not hardcode label strings in application code. The taxonomy has three levels:

1. **Violence typology** (scene-level): `SV`, `IT`, `NEG`, `NEU`
2. **Tier 1 category** (event-level): `PHYS`, `VERB`, `DIST`, `ACOU`, `EMOT`, `NONE`
3. **Tier 2 subtype** (event-level): e.g., `PHYS_HARD`, `VERB_THREAT`, `DIST_SCREAM`

See `spec.md §4` for full definitions.

## TTS providers

- **Primary:** Azure Cognitive Services he-IL voices — `he-IL-AvriNeural` (male), `he-IL-HilaNeural` (female)
- **Secondary / evaluation:** Google Cloud TTS Chirp 3 HD he-IL
- Azure SSML supports multi-speaker documents, prosody control, and `<mstts:express-as>` style tags — use these
- Cache all TTS outputs in `assets/speech/` keyed on `(voice_id, text_hash)` to avoid redundant API calls

## TTS API credentials

Expected as environment variables (`.env` file or shell environment):
```
AZURE_TTS_KEY=...
AZURE_TTS_REGION=...
GOOGLE_APPLICATION_CREDENTIALS=path/to/service_account.json
OPENAI_API_KEY=...   # or ANTHROPIC_API_KEY for Codex script generation
```

## Two projects — different priorities

| Dimension | She-Proves | Elephant in the Room |
|---|---|---|
| Scene length | 3–6 min | 1–4 min |
| Device profile | phone_in_pocket / phone_on_table / phone_in_hand | pi_budget_mic |
| Room types | apartment rooms (kitchen, bedroom, living room) | clinic_office, welfare_office, open_office |
| Optimize for | high recall, incident window detection | high precision, low false alarms, real-time alerting |
| Incident sparsity | ≥ 60% of scene is pre-incident | alert in final 40% of scene |
| Extra metadata key | `she_proves_meta` | `elephant_meta` |

## Phase targets

| Phase | Deliverable | When teams can start |
|---|---|---|
| Phase 0 (Weeks 1–3) | Single spec-compliant clip end-to-end | — |
| Phase 1 (Weeks 4–7) | 500 Tier A clips/project | **AI teams start here** |
| Phase 2 (Weeks 8–11) | 1,000–1,500 Tier B clips/project | Robustness training |
| Phase 3 (Weeks 12–16) | 4,000 clips/project, all tiers | Full Phase 1 dataset |

## Testing conventions

- Unit tests in `tests/unit/`, integration tests in `tests/integration/`
- Run with `pytest`
- Every module must have unit tests before the corresponding integration test is written
- A generated clip is valid if and only if it passes `synthbanshee.package.validator.validate_clip(clip_path)`

## What NOT to do

- Don't mix speaker personas across train/val/test splits (speaker-disjoint splits are required)
- Don't hardcode Hebrew text in Python source — it goes in template `.j2` files or transcript `.txt` files
- Don't write binary (Violence/Non-Violence) labels — the spec explicitly prohibits this
- Don't discard "dirty" pre-processing audio files — they're needed for robustness testing
- Don't use lossy audio formats (MP3, AAC) anywhere in the pipeline
- Don't generate clips shorter than 3.0 s (below the minimum label window)
