# Automated Evaluation System — Design Document

**Project:** AVDP Synthetic Framework (SynthBanshee)
**Status:** Proposal (2026-05-04)
**Companion docs:** `implementation_plan.md`, `audio_generation_v3_design.md`, `spec.md`

---

## Motivation

Human listening tests are the gold standard for evaluating synthetic speech quality, but they are expensive, slow, and happen infrequently in this project. The current quality pipeline uses acoustic signal processing (F0, RMS, click detection) which catches gross errors but cannot assess:

- **Pronunciation correctness** — is each Hebrew word intelligible?
- **Prosody naturalness** — does intonation match the emotional intent?
- **Emotional authenticity** — does a turn labeled "anger at I4" actually sound angry?
- **Dialogue coherence** — does the scene flow naturally as a conversation?
- **Regression detection** — did a code change degrade output quality?

This document proposes an automated evaluation layer using two complementary tool families:

1. **TTS/Speech neural models** — specialized models for ASR, MOS prediction, emotion recognition, speaker verification
2. **Multimodal LLMs** — general-purpose audio+text reasoning for holistic quality judgment

Together they provide continuous, scalable quality assessment that runs after every generation and gates dataset releases.

---

## Architecture Overview

```
Pipeline output (WAV + transcript + metadata)
        │
        ├──→ [E1] ASR Transcript Verification
        ├──→ [E2] MOS Prediction (UTMOS / NISQA)
        ├──→ [E3] Emotion Recognition
        ├──→ [E4] Speaker Consistency Verification
        ├──→ [E5] Multimodal LLM Judge (audio + text)
        │
        ▼
   EvalReport (per-clip scores + run-level aggregation)
        │
        ▼
   Release Gate (pass/fail decision for dataset delivery)
```

---

## E1: ASR Transcript Verification

**Goal:** Verify that every word in the ground-truth transcript is intelligible in the audio.

**Approach:**
- Run an ASR model (Whisper large-v3 or ivrit-ai/whisper-v2-d3-e3) on each turn's audio segment
- Compute Word Error Rate (WER) and Character Error Rate (CER) against `text_spoken`
- Hebrew-specific: strip niqqud before comparison; normalize final-form letters

**Metrics:**
- `wer_per_turn`: per-turn WER (0.0 = perfect)
- `cer_per_turn`: per-turn CER
- `unintelligible_words`: list of words in transcript not recovered by ASR

**Thresholds (proposed):**
- Turn-level: WER < 0.30 (Hebrew ASR is imperfect; allow headroom)
- Clip-level: mean WER < 0.20
- Run-level: < 5% of turns exceed WER 0.40

**Tools:**
- Primary: `openai/whisper-large-v3` (he-IL)
- Alternative: `ivrit-ai/whisper-v2-d3-e3` (Hebrew-specialized, potentially better WER)
- Fallback: Google Cloud Speech-to-Text v2 (he-IL)

**Implementation notes:**
- Segment audio per turn using `MixedScene.turn_onsets_s` / `turn_offsets_s`
- Run ASR on individual turn segments, not the full clip (avoids cross-talk confusion)
- Cache ASR results keyed on `(audio_hash, model_version)` for reproducibility

---

## E2: MOS Prediction (Naturalness Score)

**Goal:** Predict perceptual quality (Mean Opinion Score) without human listeners.

**Approach:**
- Run a neural MOS predictor on each turn's audio
- Produces a continuous score 1.0–5.0 mimicking human MOS ratings
- Detects: robotic artifacts, unnatural prosody, audio glitches, over-processing

**Models (ranked by suitability):**
1. **UTMOS** (SpeechMOS) — lightweight, trained on diverse TTS systems, works on 16 kHz
2. **NISQA** — ITU-T P.808-aligned, speech quality + naturalness dual prediction
3. **DNSMOS** (Microsoft) — designed for speech enhancement eval; good for detecting processing artifacts

**Metrics:**
- `mos_per_turn`: predicted MOS (1.0–5.0)
- `mos_clip_mean`: average across all turns in a clip
- `mos_below_threshold_count`: turns scoring < 3.0

**Thresholds (proposed):**
- Turn-level: MOS >= 3.0 (below = flagged for review)
- Clip-level: mean MOS >= 3.5
- Run-level: < 10% of turns below 3.0

**Implementation notes:**
- UTMOS accepts 16 kHz mono WAV directly (matches our format)
- Run on preprocessed audio (post Stage 3a) to measure what the downstream model sees
- MOS prediction is language-agnostic (evaluates signal quality, not content)

---

## E3: Emotion Recognition

**Goal:** Verify that the perceived emotion in the audio matches the intended `emotional_state` label.

**Approach:**
- Run a speech emotion recognition (SER) model on each turn
- Compare predicted emotion against the turn's `emotional_state` metadata
- Flag mismatches (e.g., turn labeled "anger" but SER predicts "neutral")

**Models:**
1. **wav2vec2-based SER** (e.g., `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`) — multilingual, works on Hebrew prosody even without Hebrew training data
2. **emotion2vec** — Alibaba's universal speech emotion model, multi-lingual
3. **Custom fine-tuned** — fine-tune on a small set of human-annotated Hebrew emotional speech (future)

**Metrics:**
- `emotion_match_per_turn`: boolean (predicted matches intended)
- `emotion_confidence`: model confidence for the predicted class
- `emotion_confusion_matrix`: run-level confusion matrix (intended vs. predicted)

**Thresholds (proposed):**
- Turn-level: match rate > 60% for I3–I5 turns (lower bar because SER models are imperfect on Hebrew)
- Run-level: per-emotion recall > 50% (no emotion category should be systematically unrecognized)
- Escalation correlation: for AGG turns, predicted arousal should increase with intensity

**Implementation notes:**
- Map project emotion labels to SER model labels (e.g., "threatening" → "anger" bucket)
- Focus on arousal dimension (high/low) rather than fine-grained categories for initial deployment
- Skip I1–I2 turns (neutral) — only evaluate I3–I5 where emotion expression matters

---

## E4: Speaker Consistency Verification

**Goal:** Verify that the same speaker sounds consistent across turns within a scene, and distinct from other speakers.

**Approach:**
- Extract speaker embeddings from each turn using a speaker verification model
- Compute intra-speaker similarity (all turns by same speaker should cluster)
- Compute inter-speaker distance (different speakers should be separable)

**Models:**
1. **ECAPA-TDNN** (SpeechBrain) — state-of-the-art speaker verification
2. **WavLM-based speaker embedding** — robust to prosody variation
3. **Resemblyzer** — lightweight, good for same/different-speaker detection

**Metrics:**
- `intra_speaker_cosine_sim`: mean pairwise cosine similarity between turns of same speaker
- `inter_speaker_cosine_dist`: mean distance between speaker centroids
- `speaker_confusion_rate`: % of turn pairs where wrong-speaker is closer than same-speaker

**Thresholds (proposed):**
- Intra-speaker similarity > 0.75 (same voice identity despite prosody changes)
- Inter-speaker distance > 0.30 (speakers are perceptually distinct)
- Speaker confusion rate < 5%

**Implementation notes:**
- Prosody variation (rate/pitch/volume across intensities) will reduce intra-speaker similarity; thresholds must account for this
- Extract embeddings from the first 3 seconds of each turn (most stable region)
- This validates that our prosody-baseline speaker differentiation (voice_family + baseline offsets) actually works

---

## E5: Multimodal LLM Judge

**Goal:** Holistic quality assessment combining audio perception with semantic understanding.

**Approach:**
- Send audio + transcript + metadata context to a multimodal LLM
- Ask structured evaluation questions across multiple dimensions
- Aggregate scores into a per-clip quality report

**Models:**
1. **Gemini 2.5 Pro** (audio-native multimodal) — primary choice for audio understanding
2. **GPT-4o** (audio input) — alternative multimodal judge
3. **Claude with audio** (when available) — future option

**Evaluation dimensions (per turn):**

| Dimension | Prompt focus | Score range |
|-----------|-------------|-------------|
| Pronunciation clarity | "Are all Hebrew words clearly pronounced and intelligible?" | 1–5 |
| Prosody naturalness | "Does the intonation sound natural for spoken Hebrew?" | 1–5 |
| Emotional expression | "Does the speaker's tone match [emotional_state] at intensity [N]?" | 1–5 |
| Speaker differentiation | "Do the two speakers sound like distinct people?" | 1–5 |
| Dialogue flow | "Does the conversation flow naturally between turns?" | 1–5 |
| Artifact detection | "Are there any audio glitches, clicks, or robotic artifacts?" | binary |

**Evaluation dimensions (per clip):**

| Dimension | Prompt focus | Score range |
|-----------|-------------|-------------|
| Escalation arc | "Does tension build naturally across the scene?" | 1–5 |
| Scene coherence | "Does this sound like a plausible real conversation?" | 1–5 |
| Overall quality | "Rate the overall production quality of this audio clip" | 1–5 |

**Metrics:**
- Per-dimension scores (1–5 or binary)
- `llm_overall_score`: weighted average across dimensions
- `llm_artifact_flags`: list of detected issues with timestamps

**Thresholds (proposed):**
- Per-clip overall >= 3.0 (below = review queue)
- Run-level mean >= 3.5 (below = generation parameters need tuning)
- Artifact rate < 10% of turns

**Implementation notes:**
- Use structured output (JSON schema) to enforce consistent scoring
- Include 2–3 calibration examples in the prompt (anchor scores to known quality levels)
- Rate limit: evaluate a stratified sample (e.g., 20% of clips per run) rather than all clips
- Cost control: Gemini 2.5 Pro audio input is ~$0.01/min → full 4-min clip ≈ $0.04 → 100 clips sample = $4/run
- Cache results keyed on `(clip_audio_hash, prompt_version, model_version)`
- Hebrew-specific instructions: prompt must specify the language and cultural context

**Prompt template structure:**
```
You are evaluating synthetic Hebrew speech generated for a domestic violence
audio detection dataset. The clip contains a {typology} scene between
{num_speakers} speakers at intensity levels {intensity_arc}.

Listen to the audio and evaluate each dimension below. Score 1-5 where:
1 = Unacceptable, 2 = Poor, 3 = Acceptable, 4 = Good, 5 = Excellent.

Return JSON matching this schema: {schema}
```

---

## Integration into Pipeline

### Where evaluators run

```
Stage 5 — Validation (existing: validate_clip)
  ↓
Stage 6 — Automated Evaluation (NEW)
  ├── E1: ASR verification  (every clip)
  ├── E2: MOS prediction    (every clip)
  ├── E3: Emotion check     (I3–I5 turns only)
  ├── E4: Speaker verify    (every clip)
  └── E5: LLM judge         (stratified sample)
  ↓
Stage 7 — Release Gate (NEW)
  └── Aggregated EvalReport → pass/fail
```

### CLI integration

```bash
# Run eval on a single clip
synthbanshee eval -c data/clip_001.wav --all

# Run eval on a dataset directory (stratified sample for LLM judge)
synthbanshee eval-batch -d data/v0.2/ --sample-rate 0.2 --output eval_report.json

# Check release gate
synthbanshee release-gate -d data/v0.2/ --eval-report eval_report.json
```

### CI/CD integration

- **Post-generation hook**: after `generate-batch`, automatically run E1+E2 on all clips
- **PR gate**: for PRs touching TTS/augment code, run eval on a reference set of 10 clips and compare against baseline scores (regression detection)
- **Nightly**: full eval run on latest generated dataset; results posted to monitoring dashboard

---

## Run-Level Aggregation: EvalReport

```python
@dataclass
class EvalReport:
    """Aggregated evaluation results for a generation run."""

    # Per-evaluator summaries
    asr_mean_wer: float
    asr_mean_cer: float
    asr_unintelligible_rate: float  # fraction of turns with WER > 0.40

    mos_mean: float
    mos_below_3_rate: float

    emotion_match_rate: float  # I3-I5 turns only
    emotion_arousal_correlation: float  # Spearman correlation: intensity vs. arousal

    speaker_intra_sim_mean: float
    speaker_inter_dist_mean: float
    speaker_confusion_rate: float

    llm_overall_mean: float  # sampled clips only
    llm_artifact_rate: float
    llm_pronunciation_mean: float
    llm_prosody_mean: float

    # Release gate decision
    passed: bool
    gate_failures: list[str]  # which gates failed and why
```

---

## Release Gate Criteria

A dataset passes the release gate when ALL of the following hold:

| Gate | Criterion | Rationale |
|------|-----------|-----------|
| ASR intelligibility | mean WER < 0.20 | Words must be recognizable |
| Naturalness | mean MOS >= 3.5 | Audio must not sound broken |
| Emotion expression | match rate > 50% for I3–I5 | High-intensity turns must carry emotion |
| Speaker identity | confusion rate < 5% | Speakers must be distinguishable |
| LLM overall | mean score >= 3.5 | Holistic quality bar |
| Artifact rate | < 10% of turns flagged | No systematic audio defects |

Gates are additive — a dataset can fail on one dimension while passing others. Gate failures produce actionable diagnostics (which clips failed, why, suggested fix).

---

## Regression Detection

**Reference baseline:** After each successful generation run that passes release gates, snapshot the eval scores as the new baseline.

**Regression check:** On PRs touching TTS/augment/preprocessing code:
1. Generate 10 reference clips (fixed seeds, covering all intensity levels)
2. Run E1–E4 on the reference set
3. Compare against baseline scores
4. Flag if any metric degrades by > 1 standard deviation

**Integration:** GitHub Actions workflow triggered on PR label `needs-eval` or automatically for files matching `synthbanshee/tts/**`, `synthbanshee/augment/**`.

---

## Implementation Phases

### Phase A: Foundation (E1 + E2)
- Integrate Whisper ASR for transcript verification
- Integrate UTMOS for MOS prediction
- Add `synthbanshee/eval/` module with `asr.py`, `mos.py`
- Add `eval` CLI command
- Add EvalReport dataclass and JSON serialization
- Unit tests with synthetic reference clips

### Phase B: Emotion + Speaker (E3 + E4)
- Integrate emotion recognition model
- Integrate speaker embedding extraction
- Add emotion match and speaker consistency metrics
- Extend EvalReport

### Phase C: LLM Judge (E5)
- Implement Gemini audio evaluation pipeline
- Design and calibrate prompt templates
- Implement stratified sampling and caching
- Add cost tracking and budget controls

### Phase D: Release Gate + CI
- Implement release gate logic
- Add regression detection workflow
- Add `eval-batch` and `release-gate` CLI commands
- Integrate into CI (reference clip baseline check on PRs)

---

## Cost Estimates

| Evaluator | Cost per clip | 500-clip run | Notes |
|-----------|---------------|--------------|-------|
| E1: Whisper ASR | ~$0 (local) | $0 | Runs on GPU locally |
| E2: UTMOS | ~$0 (local) | $0 | Lightweight PyTorch model |
| E3: Emotion | ~$0 (local) | $0 | wav2vec2-based, local |
| E4: Speaker | ~$0 (local) | $0 | ECAPA-TDNN, local |
| E5: LLM Judge | ~$0.04/clip | $4 (20% sample) | Gemini 2.5 Pro audio |

**Total per run:** ~$4 for 500-clip dataset (only LLM judge has API cost).

---

## Dependencies

| Dependency | Purpose | License |
|------------|---------|---------|
| `openai-whisper` or `faster-whisper` | ASR (E1) | MIT |
| `speechmos` (UTMOS) | MOS prediction (E2) | Apache 2.0 |
| `transformers` + wav2vec2 SER | Emotion recognition (E3) | Apache 2.0 |
| `speechbrain` (ECAPA-TDNN) | Speaker embeddings (E4) | Apache 2.0 |
| `google-genai` | Gemini API for LLM judge (E5) | Proprietary API |

All local models require a CUDA GPU for reasonable inference speed. CPU fallback is supported but slow (suitable for CI with small reference sets).

---

## Relationship to Human Listening Tests

Automated evaluation does NOT replace human listening. It provides:
- **Continuous monitoring** between infrequent human reviews
- **Regression detection** on every code change
- **Triage** — humans review only clips flagged by automated eval
- **Calibration** — periodic human MOS ratings calibrate the UTMOS thresholds

Human listening test protocol remains:
1. Sample N clips stratified by typology/intensity
2. Rate on standard dimensions (naturalness, intelligibility, etc.)
3. Compare human ratings against automated scores
4. Adjust automated thresholds based on correlation

---

*Document prepared for DataHack AVDP — not for distribution outside the project team.*
