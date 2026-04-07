---
name: Phase 0 implementation status
description: Phase 0 of AVDP synthetic dataset framework is complete — all 80 tests pass, ruff clean
type: project
---

Phase 0 (Foundation) is implemented and passing as of 2026-04-06.

**Completed milestones:**
- 0.2 Config schema: SceneConfig, SpeakerConfig, AcousticSceneConfig (Pydantic v2) in avdp/config/
- 0.3 TTS renderer: SSMLBuilder, AzureProvider (mock-injectable), TTSRenderer with SHA-256 cache in avdp/tts/
- 0.4 Preprocessing pipeline: scipy-based resample→downmix→lowpass→Wiener→normalize→pad→validate in avdp/augment/preprocessing.py
- 0.5 Label schema: EventLabel, WeakLabel, ClipMetadata (Pydantic), LabelGenerator with JSONL round-trip in avdp/labels/
- 0.6 Happy path: validate_clip() in avdp/package/validator.py; CLI in avdp/cli.py; integration test passes

**Key decisions:**
- torchaudio replaced with scipy for resampling (torch/torchaudio version mismatch in venv)
- Azure TTS mocked via sdk_factory injection — no real credentials needed in tests

**Next:** Phase 1 — Jinja2 script templates, LLM script generator, multi-speaker TTS, Tier A dataset run

**Why:** Phase 0 goal is a single valid spec-compliant clip end-to-end. All 80 tests pass, ruff clean.

**How to apply:** When resuming work, start from Phase 1.1 (script template library).
