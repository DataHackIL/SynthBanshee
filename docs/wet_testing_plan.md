# SynthBanshee — Wet Testing & Local Operation Plan

This document walks you through the first end-to-end local run of the SynthBanshee
pipeline. Nothing below assumes any prior run. Work through the phases in order, paste
the results or error output back after each phase, and we will decide whether to proceed
or fix before continuing.

---

## Prerequisites (read before touching anything)

| What | Minimum version |
|---|---|
| Python | 3.11 or 3.12 |
| `uv` | any recent (≥ 0.4) |
| Azure account | Subscription with access to Azure Cognitive Services |
| Anthropic account | API key with Claude access |

You do **not** need a Google Cloud account for this plan. Google TTS is the evaluation
provider and is not exercised here.

---

## Phase 0 — Environment setup

### 0.1 Confirm Python and uv

```bash
python3 --version     # must be 3.11 or 3.12
uv --version
```

**Report back:** both version strings, or the error if either is missing.

### 0.2 Install the package

From the repo root:

```bash
cd /path/to/SynthBanshee    # wherever you cloned it
uv pip install -e .
```

This installs all dependencies (Azure speech SDK, scipy, soundfile, anthropic, etc.)
into your active environment. The first install will take 1–3 minutes.

**Report back:** the last few lines of output, and whether the command succeeded.

### 0.3 Smoke-test the CLI

```bash
synthbanshee --help
synthbanshee generate --help
synthbanshee generate-batch --help
```

**Report back:** whether the help text printed, or the exact error.

---

## Phase 1 — API credentials

You need two credentials for basic operation. Set them now before running anything else.

### 1.1 Azure Cognitive Services Speech

1. Go to [portal.azure.com](https://portal.azure.com).
2. Create a **Speech** resource (or use an existing one).
   - Tier: Free (F0) is fine for testing — 0.5 M characters/month.
   - Region: pick the region closest to you (e.g. `westeurope`, `eastus`).
3. After the resource is created, go to **Keys and Endpoint**.
4. Copy **Key 1** and the **Location/Region** value (e.g. `westeurope`).

Create a `.env` file in the repo root (this file is gitignored):

```bash
cat > .env <<'EOF'
AZURE_TTS_KEY=paste_key_1_here
AZURE_TTS_REGION=paste_region_here
ANTHROPIC_API_KEY=paste_anthropic_key_here
EOF
```

Then load it into your shell (or add this line to your shell rc file):

```bash
set -a && source .env && set +a
```

Verify the variables are live:

```bash
echo $AZURE_TTS_KEY | head -c 10    # should print first 10 chars, not empty
echo $AZURE_TTS_REGION              # e.g. westeurope
echo $ANTHROPIC_API_KEY | head -c 10
```

### 1.2 Anthropic API key

If you do not have one:

1. Go to [console.anthropic.com](https://console.anthropic.com).
2. Create an API key under **API Keys**.
3. Add it to `.env` as shown above.

**Report back:** that all three `echo` commands printed non-empty values. Do not paste the
actual keys.

---

## Phase 2 — Config validation (no API calls)

This phase exercises config parsing only. No network calls are made.

### 2.1 Dry-run the test scene

```bash
synthbanshee generate \
  --config configs/scenes/test_scene_001.yaml \
  --dry-run
```

Expected output (approximately):

```
Loading config: configs/scenes/test_scene_001.yaml
╭─ Scene Config ────────────────────────────────────────────────────────────────╮
│ Scene: SP_IT_A_0001  Project: she_proves  Typology: IT  Tier: A              │
╰───────────────────────────────────────────────────────────────────────────────╯
Dry run — config is valid. Exiting.
```

**Report back:** the full terminal output.

### 2.2 Confirm speaker configs resolve

The test scene uses two speakers from `configs/examples/`. Verify they exist:

```bash
ls configs/examples/speaker_AGG_M_30-45_001.yaml \
      configs/examples/speaker_VIC_F_25-40_002.yaml
```

Both files must be present. If either is missing, stop and report back.

---

## Phase 3 — Script generation (LLM only, no TTS)

This phase calls the Anthropic API to generate a Hebrew dialogue script. No audio is
produced yet. Costs roughly **$0.01–0.05** per run.

### 3.1 Run a quick Python snippet

Save the following to a temporary file and run it:

```bash
cat > /tmp/test_script_gen.py << 'PYEOF'
from pathlib import Path
from synthbanshee.script.generator import ScriptGenerator

gen = ScriptGenerator(cache_dir=Path("assets/scripts"))
turns = gen.generate(
    scene_id="SP_IT_A_0001",
    project="she_proves",
    violence_typology="IT",
    script_template="synthbanshee/script/templates/she_proves/intimate_terror_coercive_control.j2",
    script_slots={
        "relationship": "spouse",
        "setting": "apartment_kitchen",
        "grievance": "going_out_without_permission",
    },
    intensity_arc=[1, 2, 3, 4, 5],
    target_duration_minutes=3.0,
    speakers=[
        {"speaker_id": "AGG_M_30-45_001", "role": "AGG", "gender": "male", "age_range": "30-45"},
        {"speaker_id": "VIC_F_25-40_002", "role": "VIC", "gender": "female", "age_range": "25-40"},
    ],
    random_seed=42,
)
print(f"Generated {len(turns)} turns")
for t in turns[:3]:
    print(f"  [{t.speaker_id}] {t.text[:80]}")
PYEOF

python3 /tmp/test_script_gen.py
```

Make sure the env vars from Phase 1 are loaded in the same shell before running.

**Expected output:**
- `Generated N turns` (typically 8–15 turns)
- 3 sample turns showing Hebrew text with speaker IDs
- A cache file written to `assets/scripts/`

**Report back:** the full output including any errors, plus `ls assets/scripts/` to confirm
the cache file was written.

---

## Phase 4 — First full clip (LLM + TTS + preprocessing + labels)

This is the main event. The pipeline calls Anthropic (Stage 1) then Azure TTS
(Stage 2) then runs local preprocessing (Stage 3) then writes labels and metadata
(Stage 4) and finally validates the output (Stage 5).

**Cost estimate:** ~$0.05 Anthropic + ~$0.10–0.30 Azure TTS (depending on dialogue
length). Total: under $0.50 for one 3-minute clip.

**Time estimate:** 3–8 minutes on first run (no cache). Subsequent runs of the same
scene are instant (both LLM output and TTS utterances are cached).

### 4.1 Create the output directory

```bash
mkdir -p data/he
```

### 4.2 Generate the test clip

```bash
synthbanshee generate \
  --config configs/scenes/test_scene_001.yaml \
  --output-dir data/he \
  --cache-dir assets/speech \
  --script-cache-dir assets/scripts
```

Watch the console. You should see:

1. Scene config panel printed
2. `Running generate pipeline...`
3. Stage 1: LLM script generation (or cache hit)
4. Stage 2: TTS rendering per speaker turn (multiple Azure calls — this is the slow part)
5. Stage 3: preprocessing (fast, local)
6. Stage 4: labels and metadata written
7. Stage 5: validation
8. `Clip generated and valid: data/he/<speaker_id>/<clip_id>.wav`

### 4.3 Confirm output files exist

```bash
find data/he -type f | sort
```

You should see at least three files sharing the same stem:

```
data/he/<speaker_id>/<clip_id>.wav
data/he/<speaker_id>/<clip_id>.txt
data/he/<speaker_id>/<clip_id>.json
```

And in `assets/`:

```bash
ls assets/speech/ | wc -l      # count of cached TTS utterances
ls assets/scripts/ | wc -l     # count of cached LLM scripts
```

**Report back:**
- Full terminal output from `generate` (or the error and traceback)
- Output of `find data/he -type f | sort`
- Output of the two `ls | wc -l` commands

---

## Phase 5 — Validate the generated clip

```bash
synthbanshee validate data/he/<speaker_id>/<clip_id>.wav
```

Replace the path with the actual path from Phase 4. You can also use shell glob:

```bash
synthbanshee validate $(find data/he -name "*.wav" | head -1)
```

Expected output: a green `PASS` panel. If it prints red `FAIL`, report the full output.

**Report back:** the full terminal output.

---

## Phase 6 — QA report on the single-clip dataset

```bash
synthbanshee qa-report data/he/
```

With only one clip this will produce minimal stats, but it exercises the full QA code
path and confirms the metadata schema parses correctly.

Expected output:
- `QA: PASS — 1 clips attempted, 1 passed, 0 failed (0.0% failure rate)`
- A table of stats (total duration, speakers, typology/tier/split distributions)

**Report back:** full terminal output.

---

## Phase 7 — Generate a second clip (different typology)

Now generate one more clip using the Elephant in the Room neutral scene. This exercises
a different template and a different set of speaker configs.

### 7.1 Check which Elephant configs exist

```bash
ls configs/scenes/elephant_tier_b/ 2>/dev/null || echo "no tier b scenes yet"
```

If no scene configs exist there yet, we will generate one. Report back what you see.

### 7.2 Generate a second clip (if scenes exist)

If `configs/scenes/elephant_tier_b/` has YAML files:

```bash
synthbanshee generate \
  --config configs/scenes/elephant_tier_b/<first_yaml_filename> \
  --output-dir data/he
```

If it is empty, skip to Phase 8 — we will add scene configs before the next batch run.

**Report back:** directory listing + generation output (or confirmation that the dir is empty).

---

## Phase 8 — Dataset card (offline, no APIs)

This command only reads the already-generated data. No API calls.

```bash
synthbanshee dataset-card data/he/ --version v0.0.1-test
```

Expected: a HuggingFace-format Markdown card printed to stdout, with stats tables
populated from your generated clip(s).

To save it:

```bash
synthbanshee dataset-card data/he/ --version v0.0.1-test \
  --output releases/DATASET_CARD.md
```

**Report back:** first 20 lines of the printed card, and any errors.

---

## Phase 9 — Package the dataset

```bash
mkdir -p releases
synthbanshee package-dataset data/he/ releases/ --version v0.0.1-test
```

Expected output:
- `Running QA on data/he/ …`
- `QA: PASS — N clips attempted …`
- `Creating archive releases/avdp_synth_v0.0.1-test.tar.gz …`
- `Archive written: …  Files: N  Uncompressed: X.X MB`
- `SHA-256: <hex>`
- `Manifest: releases/avdp_synth_v0.0.1-test_SHA256SUMS.txt`

Verify the artifacts:

```bash
ls -lh releases/
tar -tzf releases/avdp_synth_v0.0.1-test.tar.gz | head -20
cat releases/avdp_synth_v0.0.1-test_SHA256SUMS.txt
```

**Report back:** full terminal output and the `ls -lh releases/` listing.

---

## Phase 10 — Cache behaviour test (re-run the same scene)

Run Phase 4's generate command again, unchanged:

```bash
synthbanshee generate \
  --config configs/scenes/test_scene_001.yaml \
  --output-dir data/he \
  --cache-dir assets/speech \
  --script-cache-dir assets/scripts
```

This should complete in **under 5 seconds** — both the LLM script and all TTS utterances
are cached. If it makes network calls again or takes longer than ~10 seconds, something
is wrong with the cache.

**Report back:** elapsed time (e.g. `time synthbanshee generate ...`) and whether any
"generating" or "synthesizing" messages appeared (they should not on a cache hit).

---

## Summary of what to report back at each phase

| Phase | What to paste |
|---|---|
| 0.1–0.3 | Version strings or errors; help text confirmation |
| 1 | Confirmation all three env vars printed non-empty |
| 2.1 | Full dry-run output |
| 3 | Script generation output + `ls assets/scripts/` |
| 4 | Full `generate` output + `find data/he -type f` + cache counts |
| 5 | Full `validate` output |
| 6 | Full `qa-report` output |
| 7 | `ls configs/scenes/elephant_tier_b/` + second clip output (if any) |
| 8 | First 20 lines of dataset card |
| 9 | Full `package-dataset` output + `ls -lh releases/` |
| 10 | Elapsed time for re-run |

Work through them one at a time. After each phase report, I will either say "proceed to
next phase" or identify what needs fixing first.

---

## Known unknowns to watch for

These are the parts of the pipeline that have **never been exercised end-to-end** in a
live environment (only unit-tested with mocks):

1. **Azure TTS SDK on macOS** — the `azure-cognitiveservices-speech` wheel has
   platform-specific native libraries. If it fails to import, report the full traceback.

2. **Hebrew SSML prosody tags** — the SSML builder adds `<mstts:express-as>` style tags
   and prosody ranges. Azure may reject malformed SSML. The error will come from Stage 2.

3. **`pydub` + `soundfile` interaction** — audio mixing uses pydub internally; some
   macOS installs need `ffmpeg`. If pydub fails, install it:
   ```bash
   brew install ffmpeg
   ```

4. **`pyroomacoustics` import** — only required for Tier B/C scenes; the test scene is
   Tier A so this should not trigger. If it does import-error during Tier A generation,
   report it.

5. **LLM output parsing** — the script generator expects the LLM to return structured
   JSON. Occasionally the model returns markdown fences or prose. If Stage 1 fails with a
   JSON parse error, report the raw LLM output (it will be in the traceback or cache).

6. **Output directory collision** — if you run the same scene twice it may try to write
   the same clip ID twice. The current code may or may not handle this gracefully. Phase
   10 will tell us.
