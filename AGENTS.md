# AVDP Synthetic Dataset Framework — Agent Rules

**Detailed design:** `docs/spec.md` (schema), `docs/audio_generation_v3_design.md` (Implementation Tracker), `docs/implementation_plan.md` (milestones, risk register). Read these before any structural change.

## Repo layout

```
synthbanshee/       ← main package
  config/           ← Pydantic models (scene, speaker, acoustic, run)
  script/           ← LLM script generation + Jinja2 templates
  tts/              ← TTS rendering (Azure/Google) + SceneMixer
  augment/          ← preprocessing, room IR, device profiles, noise
  labels/           ← auto-label generation, schema, IAA utilities
  package/          ← manifest, QA, splitter, validator, archiver
  cli.py            ← Click entry points
configs/
  scenes/ speakers/ acoustic_scenes/ run_configs/ examples/
assets/             ← gitignored; populated at runtime
data/               ← gitignored; generated output
docs/ tests/unit/ tests/integration/
```

## Language / encoding

- All audio: **Hebrew (he-IL)**
- **Filenames + JSON string fields: ASCII only** — no UTF-8 above U+00A1
- Hebrew text goes in `.j2` templates or `.txt` transcripts only
- `ClipMetadata` enforces the rule via `@field_validator` on `clip_id`, `project`, `tts_engine`, `violence_typology`, `generator_version`

## Audio format (hard constraints)

- **16 kHz, mono, 16-bit PCM WAV**
- **Peak limiter ceiling: −1.0 dBFS** — never scale up quiet clips (limiter, not normalizer)
- **Silence padding: ≥ 0.5 s** at head and tail — `silence_pad_applied_s` is per-side
- Onset/offset times from `MixedScene` must be shifted by the leading pad only
- **No torchaudio** — preprocessing uses `scipy` + `soundfile` exclusively
- **No lossy formats** (MP3, AAC) anywhere in the pipeline
- **Retain dirty files**: named `{clip_id}_dirty{ext}` in `assets/`; never overwrite

## Pipeline stages (keep modular, clean interfaces)

```
SceneConfig (YAML)
  → [1]  Script Generator      → DialogueTurns   (LLM + Jinja2)
  → [2]  TTS Renderer          → MixedScene WAV  (Azure/Google he-IL)
  → [3a] Preprocessing         → spec-compliant WAV
  → [3b] Acoustic Augmentation → augmented WAV   (Tier B only)
  → [4a] Transcript Writer     → {clip_id}.txt
  → [4b] Strong Label Writer   → {clip_id}.jsonl
  → [4c] Metadata Writer       → {clip_id}.json
  → [5]  Validator             → validate_clip()
```

SFX onset/offset times come **only** from the Stage 3b augmentation log. Speech-turn times come from Stage 2 mixer output.

## Label taxonomy

- **Source of truth: `configs/taxonomy.yaml`** — never hardcode label strings in application code
- Three levels: `violence_typology` → `tier1_category` → `tier2_subtype`
- `has_violence` is a **derived convenience field**; the full taxonomy columns are ground truth
- `_TYPOLOGY_INTENSITY_MAP` in `cli.py` maps `(typology, intensity)` → `(tier1, tier2)`. If you add a typology to the taxonomy, add a row to the map — missing entries fall through to `("NONE", "NONE_AMBIENT")` silently
- Codes are validated against `taxonomy.yaml` at import; renaming a code causes an import `ValueError`

## TTS

- **Primary:** Azure `he-IL-AvriNeural` (M), `he-IL-HilaNeural` (F); SSML `<mstts:express-as>` + prosody tags
- **Secondary:** Google Cloud TTS Chirp 3 HD he-IL
- **Cache key: SHA-256 of the full rendered SSML string** — not `(voice_id, text)`
- **M3a — per-turn RMS gain:** `StyleEntry.rms_target_dbfs` → `SceneMixer._apply_rms_gain()`; 4-tuple segment API `(wav_bytes, pause_s, speaker_id, rms_target_dbfs)`
- **Temp WAV from mixer must be `subtype="FLOAT"`** — never `PCM_16` before `preprocess()` to avoid hard-clipping M3 gain

## Splits

- **Speaker-disjoint**: never place the same speaker persona in more than one of train / val / test_synth

## Testing conventions

- Unit tests (`tests/unit/`) before integration tests (`tests/integration/`)
- Run: `pytest`; CI also runs `ruff` and `mypy` (Python 3.11 + 3.12)
- A clip is valid iff it passes `synthbanshee.package.validator.validate_clip(clip_path)`:
  1. All three files present (`.wav`, `.txt`, `.json`)
  2. WAV passes `validate_audio()` — 16 kHz, mono, peak ≤ −1.0 dBFS, duration ≥ 3 s
  3. JSON parses as `ClipMetadata` with `is_synthetic=True`
  4. Filename stem is ASCII-only lowercase

## Environment variables

```
AZURE_TTS_KEY            AZURE_TTS_REGION
GOOGLE_APPLICATION_CREDENTIALS
OPENAI_API_KEY           # or ANTHROPIC_API_KEY for Claude script generation
```

## Projects

| Dimension | She-Proves | Elephant in the Room |
|---|---|---|
| Scene length | 3–6 min | 1–4 min |
| Device profile | `phone_in_pocket / _on_table / _in_hand` | `pi_budget_mic` |
| Room types | apartment rooms | `clinic_office`, `welfare_office`, `open_office` |
| Optimize for | high recall, incident window detection | high precision, low false alarms |
| Incident sparsity | ≥ 60% pre-incident | alert in final 40% |
| Extra metadata key | `she_proves_meta` | `elephant_meta` |

## What NOT to do

- Don't mix speaker personas across train/val/test splits
- Don't hardcode Hebrew text in Python source
- Don't treat `has_violence` as the primary label — preserve the full taxonomy hierarchy
- Don't discard dirty pre-processing files
- Don't use lossy audio formats
- Don't generate clips shorter than 3.0 s
- Don't add a new typology to `taxonomy.yaml` without adding it to `_TYPOLOGY_INTENSITY_MAP`
