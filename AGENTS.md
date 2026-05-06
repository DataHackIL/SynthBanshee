# AVDP Synthetic Dataset Framework — Agent Rules

**Detailed design:** `docs/spec.md` (schema), `docs/audio_generation_v3_design.md` (Implementation Tracker), `docs/implementation_plan.md` (milestones, risk register). Read these before any structural change.

## Repo layout

```
synthbanshee/       ← main package
  config/           ← Pydantic models (scene, speaker, acoustic, run)
  data/             ← bundled data files (taxonomy.yaml)
  script/           ← LLM script generation + Jinja2 templates
  tts/              ← TTS rendering (Azure/Google) + SceneMixer
  augment/          ← preprocessing, room IR, device profiles, noise
  labels/           ← auto-label generation, schema, IAA utilities
  package/          ← manifest, QA, splitter, validator, archiver
  cli.py            ← Click entry points
configs/
  scenes/ run_configs/ examples/    ← checked-in; speakers/ and acoustic_scenes/ are user-created if needed
assets/             ← gitignored; populated at runtime
data/               ← gitignored; generated output
docs/ tests/unit/ tests/integration/
```

## Language / encoding

- All audio: **Hebrew (he-IL)**
- **Filenames:** strictly lowercase `[a-z0-9_-]`
- **JSON string fields validated by `ClipMetadata`:** no characters above U+00A1
- Hebrew text goes in `.j2` templates or `.txt` transcripts only
- `ClipMetadata` enforces the string-field rule via `@field_validator` on `clip_id`, `project`, `tts_engine`, `violence_typology`, `generator_version`

## Audio format (hard constraints)

- **16 kHz, mono, 16-bit PCM WAV**
- **Loudness: peak-normalize to target (−2.0 dBFS default, range [−12.0, −1.5]), then peak-limit at ≤ −1.0 dBFS** — `preprocess()` applies a single global gain so the clip's absolute peak lands at `PreprocessingConfig.target_peak_dbfs`; the −1.0 dBFS limiter is a safety ceiling that is a guaranteed no-op for in-spec target values. The single-gain step preserves per-turn RMS contrast (M3a) — it shifts absolute level but every per-turn RMS *ratio* survives unchanged. Tier B/C augmentation re-applies the same target after room-IR/noise mixing through the shared `peak_normalize_to_target` helper, so all tiers exit at the same absolute peak. **Behaviour change vs. pre-#78:** Tier B/C clips previously normalized to −1.0 dBFS (zero headroom over the limiter); they now exit at the configured target (−2.0 dBFS by default). The `loudness_target_peak_dbfs` field in `GenerationMetadata` records the exact value used so any future loudness regression can be diagnosed from metadata alone — without this trail, #78 hid for three weeks behind unchanged `generator_version`. **Empirical note:** peak/RMS in the spec range does not affect Whisper WER (Whisper's log-mel feature extractor internally normalizes); the M17 ASR regression has a separate prosody-level cause being tracked elsewhere. (#78)
- **Silence padding: ≥ 0.5 s** at head and tail — `silence_pad_applied_s` is per-side
- Onset/offset times from `MixedScene` must be shifted by the leading pad only
- **No torchaudio** — preprocessing uses `scipy` + `soundfile` exclusively
- **No lossy formats** (MP3, AAC) anywhere in the pipeline
- **Retain dirty files**: `preprocess()` writes them as `{output_path.stem}_dirty{input_suffix}` in `assets/` (CLI produces e.g. `{clip_id}_00_dirty.wav`); never overwrite

## Pipeline stages (keep modular, clean interfaces)

```
SceneConfig (YAML)
  → [1]  Script Generator      → DialogueTurns   (LLM + Jinja2)
  → [2]  TTS Renderer          → MixedScene WAV  (Azure/Google he-IL)
  → [3a] Preprocessing         → spec-compliant WAV
  → [3b] Acoustic Augmentation → augmented WAV   (Tier B and Tier C only)
  → [4a] Transcript Writer     → {clip_id}.txt
  → [4b] Strong Label Writer   → {clip_id}.jsonl
  → [4c] Metadata Writer       → {clip_id}.json
  → [5]  Validator             → validate_clip()
```

SFX onset/offset times come **only** from the Stage 3b augmentation log. Speech-turn times come from Stage 2 mixer output.

## Label taxonomy

- **Source of truth: `synthbanshee/data/taxonomy.yaml`** — never hardcode label strings in application code
- Three levels: `violence_typology` → `tier1_category` → `tier2_subtype`
- `has_violence` is a **derived convenience field**; the full taxonomy columns are ground truth
- `_TYPOLOGY_INTENSITY_MAP` in `cli.py` maps `(typology, intensity)` → `(tier1, tier2)`. If you add a typology to the taxonomy, add a row to the map — missing entries fall through to `("NONE", "NONE_AMBIENT")` silently
- Codes are validated against `taxonomy.yaml` at import; renaming a code causes an import `ValueError`

## TTS

- **Primary:** Azure `he-IL-AvriNeural` (M), `he-IL-HilaNeural` (F); SSML `<mstts:express-as>` + prosody tags
- **Secondary:** Google Cloud TTS Chirp 3 HD he-IL
- **Cache key: SHA-256 of the full rendered SSML string** — not `(voice_id, text)`
- **M3a — per-turn RMS gain:** `StyleEntry.rms_target_dbfs` → module-level `_apply_rms_gain()` in `synthbanshee/tts/mixer.py`; 4-tuple segment API `(wav_bytes, pause_s, speaker_id, rms_target_dbfs)`
- **Temp WAV from mixer must be `subtype="FLOAT"`** — never `PCM_16` before `preprocess()` to avoid hard-clipping M3 gain
- **#87 effective-prosody cap:** `synthbanshee/tts/renderer._apply_effective_prosody_cap` clamps post-state, post-randomization prosody to `pitch ∈ [−3.0, +2.0] st`, `rate ∈ [0.85, 1.20]`. Volume is left to `_volume_to_string`'s ±50% Azure clamp (Whisper internally normalizes loudness; #82 lever probe). Cap activations are recorded per-turn in `DialogueTurn.effective_prosody_caps` and rolled up into `ClipMetadata.generation_metadata.effective_prosody_caps`. Caps are anchored to the pre-#51 effective envelope (the 04-15 reference baseline that Whisper handles with WER 0.04–0.08); changing them requires a paired listening test + Whisper sanity check (see "ASR sanity check policy" below).

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

## ASR sanity check policy (#87)

The Whisper-large-v3 sanity check (`synthbanshee qa-report --asr`) detects audio that synthesizes fine to a listener but trips Whisper's silence-detection / segmentation heuristic — the failure mode that surfaced in #87 and went undetected on `sp_it_a_0001` for weeks. The canonical fingerprint is **length-ratio collapse** (`hyp_words / ref_words < 0.85`).

**Cost:** ~5–20 s per clip on M-series MPS, ~3 GB Whisper-large-v3 download on first use. Real Azure render to populate the cache is an additional ~$0.05–$0.20 per scene.

**Policy — for the foreseeable future, this check is _local-only_, not in CI:**

1. **Unit tests** for the cap logic (in `tests/unit/test_effective_prosody_cap.py`) run on every PR. They cover ~90 % of regressions with no Azure / Whisper cost.
2. **`qa-report --asr` (Tier-3)** is run **locally**, **once**, **after all PR reviews and fixes are done**, **and only for PRs that might affect audio rendering** (anything touching `synthbanshee/tts/`, `synthbanshee/script/`, `synthbanshee/augment/`, or the speaker / scene / acoustic / project YAML configs). PRs that only touch labels, packaging, docs, or tests skip it.
3. **Cost discipline:** use this command **very sparingly** — each invocation incurs Azure spend (~$1–$2 per real-render run). Prefer running against a directory of already-cached clips when possible; only re-render when a code change invalidates the SSML cache.
4. **Result documentation:** when run, paste the `qa-report --asr` output (table of any clips with `length_ratio < 0.85`, plus the green-tick "no asr warnings" line if all clean) into the PR body's "Test plan" section under a "Tier-3 ASR sanity (local)" sub-heading.

**Future CI consideration:** see GH issue tracking the addition of a scheduled / labeled Whisper-based CI workflow, including repo-secret management for `AZURE_TTS_KEY` and Azure spend caps. Re-evaluate when scene volume increases enough that local manual runs become impractical.

**Install:** `uv pip install --python .venv/bin/python -e ".[eval-asr]"` (heavyweight optional extra).

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

## Feature / milestone completion checklist

A feature or milestone is **not complete** until all of the following are true:

1. All code changes are committed and pushed to the feature branch.
2. A **non-draft** PR is open on GitHub targeting `main`, with:
   - A descriptive title following the `type(scope): subject` convention.
   - A body that includes: problem statement, solution summary, table of changes per file, and a test plan checklist with pass/fail results.
   - At least one label (e.g. `enhancement`, `bugfix`, `tests & testing`).
   - A milestone assigned (create one if none exists for this milestone ID).
3. The PR has been pushed — not just the branch.

Do not consider your work done after a commit. The PR is the deliverable.

## PR review workflow

When addressing PR review comments (Copilot, human, or bot):

1. Triage each open thread: resolve as already treated / implement fix / open issue + resolve as out-of-scope.
2. Land all code changes in a single commit on the feature branch and push.
3. **After the commit lands, resolve every addressed thread on GitHub** using the GraphQL API:
   ```bash
   gh api graphql -f query='mutation { resolveReviewThread(input: {threadId: "PRRT_..."}) { thread { isResolved } } }'
   ```
   Get thread node IDs with:
   ```bash
   gh api graphql -f query='{ repository(owner: "OWNER", name: "REPO") { pullRequest(number: N) { reviewThreads(first: 50) { nodes { id isResolved comments(first: 1) { nodes { databaseId } } } } } } }'
   ```
4. Do **not** leave threads open after the corresponding change is committed — unresolved threads imply unfinished work.

## CI / Workflow conventions

- `pr-agent-context` is wired in two workflow files:
  - `.github/workflows/ci.yml` — initial CI run (produces the baseline context comment)
  - `.github/workflows/pr-agent-context-refresh.yml` — refresh run (review/check/schedule-driven)
- Both workflows call the reusable workflow via **`@v4` (floating major tag)** — do **not** pin to a specific patch version (e.g. `@v4.0.18`). The floating tag is intentional: it picks up compatible patch releases automatically, and the `tool_ref: v4` input mirrors it.
- Coverage artifacts are named `pr-agent-context-coverage-py<version>` (uploaded by the matrix `test` job). Both the CI `pr-agent-context` job and the refresh workflow reference them via `coverage_artifact_prefix: pr-agent-context-coverage` — do not change this prefix or the naming convention without updating both workflow files.
- The refresh workflow includes a `dispatch-scheduled-refreshes` job that runs every 15 minutes and fans out `workflow_dispatch` runs to open same-repo PRs that lack a current refresh comment. This is the fallback for bot-authored review events that do not fire a `pull_request_review` GitHub event.

## What NOT to do

- Don't mix speaker personas across train/val/test splits
- Don't hardcode Hebrew text in Python source
- Don't treat `has_violence` as the primary label — preserve the full taxonomy hierarchy
- Don't discard dirty pre-processing files
- Don't use lossy audio formats
- Don't generate clips shorter than 3.0 s
- Don't add a new typology to `taxonomy.yaml` without adding it to `_TYPOLOGY_INTENSITY_MAP`
- Don't pin `pr-agent-context` workflow references to a specific patch version — keep `@v4` floating
- Don't relax the #87 effective-prosody cap (`_EFFECTIVE_PITCH_*`, `_EFFECTIVE_RATE_*` in `synthbanshee/tts/renderer.py`) without a paired listening test + `qa-report --asr` Tier-3 run; the cap defends both naturalness (May-3 listening test) and Whisper transcription
- Don't merge a PR that touches audio rendering (`synthbanshee/tts/`, `synthbanshee/script/`, `synthbanshee/augment/`, or speaker / scene / acoustic / project YAMLs) without running `qa-report --asr` locally first and pasting the result into the PR's test plan (see "ASR sanity check policy")
