# SynthBanshee

**SynthBanshee** is a config-driven pipeline for generating large-scale synthetic Hebrew audio
datasets. It is part of the [AVDP](https://datahack.org.il) (Audio Violence Dataset Project)
initiative run by DataHack, which produces training data for two AI safety products:

- **She-Proves** — a smartphone app that passively monitors for domestic violence incidents and
  preserves audio evidence for legal use
- **Elephant in the Room (הפיל שבחדר)** — a Raspberry Pi–class device in social-work offices that
  alerts security when a social worker is being physically threatened

The goal of SynthBanshee is to supply AI teams with a wide, deliberate distribution of voices,
acoustic conditions, and violence types while real-data (actor recording) pipelines are built in
parallel. Synthetic-to-real gap is expected and documented.

---

## How it works

Every clip is produced by four sequential pipeline stages:

```
SceneConfig (YAML)
  → [1] Script Generator   LLM fills a Jinja2 template → dialogue JSON
  → [2] TTS Renderer       Azure he-IL SSML → per-speaker WAV segments
  → [3] Acoustic Augmenter Room IR + device profile + noise → scene WAV
  → [4] Label Generator    Script structure + augmentation log → AVDP-schema JSONL
```

Output per clip: `{clip_id}.wav` + `{clip_id}.txt` (Hebrew transcript) + `{clip_id}.json` (metadata).

Audio spec: **16 kHz · mono · 16-bit PCM · −1.0 dBFS peak · ≥ 0.5 s silence pad**.

---

## Dataset tiers

| Tier | Description | Target (per project) |
|---|---|---|
| A | Clean TTS, no acoustic augmentation | 1,000 clips |
| B | Room simulation + device profile + background noise | 2,000 clips |
| C | Hard negatives / confusors (arguments that de-escalate, ambient sounds) | 1,000 clips |

Two projects — **She-Proves** (3–6 min clips, apartment rooms) and **Elephant in the Room**
(1–4 min clips, clinic/welfare offices) — each receive the full tier stack.

---

## Label taxonomy

Labels use a three-level hierarchy (no binary Violence/Non-Violence):

1. **Violence typology** (scene): `SV` · `IT` · `NEG` · `NEU`
2. **Tier 1 category** (event): `PHYS` · `VERB` · `DIST` · `ACOU` · `EMOT` · `NONE`
3. **Tier 2 subtype** (event): e.g. `PHYS_HARD` · `VERB_THREAT` · `DIST_SCREAM`

Full taxonomy: `configs/taxonomy.yaml`.

---

## Current status

Phase 0 (pipeline foundation) is complete. Phase 1 batch generation and QA infrastructure (milestones 1.4 & 1.5) is complete — AI teams can now use the delivered Tier A dataset to start baseline model development.

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Single spec-compliant clip end-to-end | **Done** |
| 1 (1.4–1.5) | Batch generation run + QA suite | **Done** |
| 1 (1.1–1.3) | Script templates + multi-speaker TTS + LLM generator | In progress |
| 2 | 1,000–1,500 Tier B clips/project | Planned |
| 3 | 4,000 clips/project, all tiers | Planned |

---

## Quick start

```bash
# Install (requires Python ≥ 3.11 and uv)
uv pip install -e .

# Generate a single clip from a scene config
synthbanshee generate --config configs/scenes/test_scene_001.yaml

# Generate a full Tier A batch from a run config
synthbanshee generate-batch \
  --run-config configs/run_configs/tier_a_500_she_proves.yaml \
  --output-dir data/he

# Run the automated QA suite on a dataset directory
synthbanshee qa-report data/he

# Validate an existing clip
synthbanshee validate data/he/clip_001.wav
```

API credentials required (set in environment or `.env`):

```
AZURE_TTS_KEY=...
AZURE_TTS_REGION=...
ANTHROPIC_API_KEY=...   # for script generation
```

---

## Docs

| Document | Contents |
|---|---|
| `docs/spec.md` | Audio format, file naming, label schema, IAA protocol |
| `docs/implementation_plan.md` | Phased milestones, module map, API cost estimates |
| `docs/design_approaches.md` | Design decisions and rationale |
| `CLAUDE.md` | Claude Code context guide (pipeline constraints, conventions) |
