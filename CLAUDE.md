# AVDP Synthetic Dataset Framework — Claude Code Context

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
  run_configs/             ← full generation run definitions (RunConfig YAML)
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

## The five pipeline stages

Every clip is produced by five sequential stages. Keep them modular with clean interfaces.

```
SceneConfig (YAML)
  → [1]  Script Generator      → DialogueTurns  (LLM fills Jinja2 template)
  → [2]  TTS Renderer          → MixedScene WAV (Azure/Google he-IL, per-speaker segments mixed)
  → [3a] Preprocessing         → spec-compliant WAV (16 kHz, mono, normalized, silence-padded)
  → [3b] Acoustic Augmentation → augmented WAV (room IR, device profile, noise) — Tier B only
  → [4a] Transcript Writer     → {clip_id}.txt  (per-turn text with onset/offset markers)
  → [4b] Strong Label Writer   → {clip_id}.jsonl (per-event EventLabel records)
  → [4c] Metadata Writer       → {clip_id}.json  (ClipMetadata — weak labels, speaker info, etc.)
  → [5]  Validator             → pass/fail + warnings (validate_clip)
```

Stage 3b runs only for Tier B scenes (those with an `acoustic_scene` block). For Tier A scenes the pipeline goes directly from Stage 3a to Stage 4a.

The augmentation log produced by Stage 3b is the source of truth for SFX onset/offset times used in Stage 4b labels. Don't derive SFX label timings from anything else — only speech-turn timings come from the TTS mixer output (Stage 2).

## Key schema constraints (from spec.md)

- Audio: **16 kHz, mono, 16-bit PCM WAV, −1.0 dBFS peak normalized**
- Directory: `data/{language_code}/{speaker_id}/{clip_id}.wav` + matching `.txt` + `.json`
- Filenames: **ASCII only**, no spaces, no UTF-8 above U+00A1, lowercase
- Every `.wav` must have a matching `.txt` (transcript) and `.json` (metadata)
- **`has_violence` is a derived convenience field** (`weak_label.has_violence` in `.json`, column in `manifest.csv`), computed from the hierarchical taxonomy — never assigned independently. Keep it; AI teams need it for baseline models and stratified sampling. The taxonomy columns (`violence_typology`, `tier1_category`, `tier2_subtype`, `max_intensity`) are the ground truth.
- Silence padding: **≥ 0.5 s** ambient baseline before and after target speech
- Retain "dirty" (pre-preprocessing) files in `assets/` always — named `{clip_id}_dirty{original_suffix}` (e.g. `clip_001_dirty.wav`), NOT the raw input filename, to avoid collisions when multiple inputs share the same basename
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
- Cache all TTS outputs in `assets/speech/` keyed on **SHA-256 of the full rendered SSML string**. This captures voice, text, style, prosody parameters, and any randomization seed — so calls that differ only in intensity or rate will never share a cache entry. The key is NOT derived from `(voice_id, text)` alone.

## TTS API credentials

Expected as environment variables (`.env` file or shell environment):
```
AZURE_TTS_KEY=...
AZURE_TTS_REGION=...
GOOGLE_APPLICATION_CREDENTIALS=path/to/service_account.json
OPENAI_API_KEY=...   # or ANTHROPIC_API_KEY for Claude script generation
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

## ASCII-safe metadata fields

`ClipMetadata` enforces the "no Hebrew in JSON string fields" rule via a `@field_validator` on:
`clip_id`, `project`, `tts_engine`, `violence_typology`, `generator_version`.
Path fields (`transcript_path`, `scene_config`) and date/version fields are intentionally excluded — they are always ASCII by construction.

## Preprocessing stack

**No torchaudio.** The preprocessing pipeline (`synthbanshee/augment/preprocessing.py`) uses only `scipy` + `soundfile` to avoid torch version incompatibilities. Operations in order: resample (polyphase), downmix to mono, Butterworth low-pass at 7.5 kHz, Wiener denoise, peak-normalize to −1.0 dBFS, silence pad ≥ 0.5 s.

`preprocess()` returns `PreprocessingResult` which includes a `silence_pad_applied_s` field (currently always `0.5`). That value is applied **per side**: preprocessing pads the audio with the same amount of silence at both the start and the end, so total clip duration increases by `2 * result.silence_pad_applied_s`. Any code that writes per-turn transcript or label timings derived from `MixedScene.turn_onsets_s/offsets_s` **must add only the leading pad, `result.silence_pad_applied_s`, to every onset and offset** — the mixer starts at `t=0` with no head silence, and the trailing pad extends clip length but does not shift event timestamps.

## Event type mapping

`_TYPOLOGY_INTENSITY_MAP` in `synthbanshee/cli.py` maps `(violence_typology, intensity)` to `(tier1_category, tier2_subtype)`. Every code in the map is validated against `configs/taxonomy.yaml` at import time via `_validate_event_type_codes()`. If you add a new typology to the taxonomy, add a corresponding entry to the map — a missing entry will fall through to `("NONE", "NONE_AMBIENT")` silently, which is wrong. If you change a code name in the taxonomy, the import will fail loudly with a `ValueError`.

## Testing conventions

- Unit tests in `tests/unit/`, integration tests in `tests/integration/`
- Run with `pytest`
- Every module must have unit tests before the corresponding integration test is written
- A generated clip is valid if and only if it passes `synthbanshee.package.validator.validate_clip(clip_path)`
- `validate_clip` checks: (1) all three files present, (2) WAV passes `validate_audio()`, (3) JSON parses as `ClipMetadata` with `is_synthetic=True`, (4) filename stem is ASCII-only lowercase

## Phase 0 status (complete as of 2026-04-07)

All Phase 0 milestones (0.2–0.6) are implemented and tested:

| Milestone | Module(s) | Status |
|---|---|---|
| 0.2 Config schema | `synthbanshee/config/` | Done |
| 0.3 TTS renderer | `synthbanshee/tts/` | Done |
| 0.4 Preprocessing pipeline | `synthbanshee/augment/preprocessing.py` | Done |
| 0.5 Label schema & generator | `synthbanshee/labels/` | Done |
| 0.6 Happy path + validator + CLI | `synthbanshee/package/validator.py`, `synthbanshee/cli.py` | Done |

CI runs ruff, mypy, and pytest (Python 3.11 + 3.12) on every PR and push to main.

## Phase 1 status (complete as of 2026-04-08)

All Phase 1 milestones are implemented and tested:

| Milestone | Module(s) | Status |
|---|---|---|
| 1.1 Jinja2 template library | `synthbanshee/script/templates/` | Done |
| 1.2 Multi-speaker TTS renderer | `synthbanshee/tts/renderer.py`, `synthbanshee/tts/mixer.py` | Done |
| 1.3 LLM script generator | `synthbanshee/script/generator.py` | Done |
| 1.3 Full pipeline integration | `synthbanshee/cli.py` (`_run_generate_pipeline`) | Done |
| 1.4 Run config schema | `synthbanshee/config/run_config.py` | Done |
| 1.4 Speaker-disjoint splits | `synthbanshee/package/splitter.py` | Done |
| 1.4 Batch generation CLI | `synthbanshee/cli.py` (`generate-batch`) | Done |
| 1.4 Run configs (Tier A) | `configs/run_configs/` | Done |
| 1.5 Manifest CSV | `synthbanshee/package/manifest.py` | Done |
| 1.5 Automated QA suite | `synthbanshee/package/qa.py` | Done |
| 1.5 QA report CLI | `synthbanshee/cli.py` (`qa-report`) | Done |

The full pipeline is now wired end-to-end: `ScriptGenerator.generate()` → `TTSRenderer.render_scene()` → `preprocess()` → per-turn event labels → `ClipMetadata`. AI teams can use the Tier A dataset to start baseline model development.

## Phase 3 IAA utilities (complete as of 2026-04-15)

| Milestone | Module(s) | Status |
|---|---|---|
| 3.1 Per-turn RMS gain (M3) | `synthbanshee/tts/mixer.py`, `synthbanshee/config/speaker_config.py` | Done |
| 3.2 Per-category kappa | `synthbanshee/labels/iaa.py` | Done |
| 3.3 IAAReport | `synthbanshee/labels/iaa.py` | Done |
| 3.4 iaa-report CLI | `synthbanshee/cli.py` | Done |

### M3 — per-turn RMS gain

Azure TTS normalizes WAV output, so `volume_delta_db` in SSML has no effect on amplitude.
M3 fixes this by applying post-TTS RMS gain inside `SceneMixer.mix_sequential()`:

- `StyleEntry` gains an optional `rms_target_dbfs: float | None` field (in `synthbanshee/config/speaker_config.py`).
- `SceneMixer.mix_sequential()` now takes 4-tuples `(wav_bytes, pause_before_s, speaker_id, rms_target_dbfs)`. If `rms_target_dbfs` is not `None`, `_apply_rms_gain()` scales the decoded segment so its RMS matches the target before mixing.
- `TTSRenderer.render_scene()` passes `style_entry.rms_target_dbfs` as the 4th element of each segment tuple.
- Near-silent frames (RMS < −80 dBFS) are skipped to avoid noise-floor amplification.
- The gain-adjusted signal is not clipped in the mixer; downstream peak-normalization in `preprocessing.py` handles clipping.

AGG/BEN speaker YAMLs: `rms_target_dbfs` escalates −28 → −15 dBFS across I1–I5 (+13 dB, satisfying the ≥ 8 dB AGG escalation spec target).
VIC/SW speaker YAMLs: `rms_target_dbfs` descends −26 → −30 dBFS across I1–I5 (victim's voice retreats under pressure).

## What NOT to do

- Don't mix speaker personas across train/val/test splits (speaker-disjoint splits are required)
- Don't hardcode Hebrew text in Python source — it goes in template `.j2` files or transcript `.txt` files
- Don't treat `has_violence` as the primary or sole label — it is a derived summary field. Always preserve the full hierarchical taxonomy (`violence_typology`, `tier1_category`, `tier2_subtype`, `max_intensity`) alongside it. Never replace the taxonomy with a single binary flag.
- Don't discard "dirty" pre-processing audio files — they're needed for robustness testing
- Don't use lossy audio formats (MP3, AAC) anywhere in the pipeline
- Don't generate clips shorter than 3.0 s (below the minimum label window)
