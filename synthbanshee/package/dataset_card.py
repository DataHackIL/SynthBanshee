"""HuggingFace-format dataset card generator for AVDP (Phase 3.4).

Renders a dataset card from a :class:`~synthbanshee.package.qa.QAReport`,
including YAML frontmatter, dataset statistics, generation methodology,
known limitations, data format description, license, and BibTeX citation.
"""

from __future__ import annotations

from synthbanshee.package.qa import QAReport


def generate_dataset_card(
    qa_report: QAReport,
    version: str,
    *,
    projects: list[str] | None = None,
) -> str:
    """Render a HuggingFace-format dataset card from a :class:`QAReport`.

    Args:
        qa_report: Completed QA report from :func:`~synthbanshee.package.qa.run_qa`.
        version: Dataset version string (e.g. ``"v1.0"``).
        projects: Human-readable project names to mention in the description.
            Defaults to ``["She-Proves", "Elephant in the Room"]``.

    Returns:
        Dataset card as a UTF-8 string (YAML frontmatter + Markdown body).
    """
    if projects is None:
        projects = ["She-Proves", "Elephant in the Room"]

    stats = qa_report.stats
    total_clips = stats.total_clips
    total_hours = stats.total_duration_seconds / 3600

    def _pct(n: int) -> str:
        return f"{100 * n / total_clips:.1f}%" if total_clips else "—"

    typology_rows = "".join(
        f"| `{typ}` | {cnt:,} | {_pct(cnt)} |\n"
        for typ, cnt in sorted(stats.clips_by_typology.items())
    )
    tier_rows = "".join(
        f"| Tier {tier} | {cnt:,} | {_pct(cnt)} |\n"
        for tier, cnt in sorted(stats.clips_by_tier.items())
    )
    split_rows = "".join(
        f"| {split} | {cnt:,} | {_pct(cnt)} |\n"
        for split, cnt in sorted(stats.clips_by_split.items())
    )

    projects_str = " and ".join(f"**{p}**" for p in projects)
    cite_version = version.replace(".", "_")

    return f"""\
---
language:
- he
license: cc-by-4.0
task_categories:
- audio-classification
tags:
- synthetic
- hebrew
- violence-detection
- audio
- safety-ai
pretty_name: AVDP Synthetic Dataset {version}
size_categories:
- {_size_category(total_clips)}
---

# AVDP Synthetic Dataset — {version}

## Dataset Description

Synthetic Hebrew audio dataset for two AI safety initiatives run by
[DataHack](https://datahack.org.il):

- **She-Proves** — passively monitors a smartphone for domestic violence
  incidents and preserves audio evidence for legal use.
- **Elephant in the Room** — a Raspberry Pi–class device in clinic/welfare
  offices that alerts security when a social worker is being attacked.

All audio is **fully synthetic** (`is_synthetic: true` in every clip's
metadata). A real-data pipeline using actor recordings is planned for a future
release. This dataset provides bootstrap training data for the {projects_str}
models.

## Dataset Statistics ({version})

| Metric | Value |
|--------|-------|
| Total clips | {total_clips:,} |
| Total duration | {total_hours:.1f} h ({stats.total_duration_seconds:,.0f} s) |
| Unique speakers | {stats.speaker_count:,} |
| Failed clips (QA) | {stats.failed_clips:,} |
| Quality-flagged clips | {stats.quality_flagged_clips:,} |
| Language | Hebrew (he-IL) |
| Sample rate | 16 kHz |
| Channels | Mono |
| Bit depth | 16-bit PCM |
| Peak level | −1.0 dBFS |

### Violence Typology Distribution

| Typology | Clips | % |
|----------|-------|---|
{typology_rows}
### Tier Distribution

| Tier | Clips | % |
|------|-------|---|
{tier_rows}
### Train / Val / Test Split

| Split | Clips | % |
|-------|-------|---|
{split_rows}
## Generation Methodology

Clips are produced by a five-stage pipeline:

1. **Script Generator** — LLM (Claude/GPT-4) fills Jinja2 scene templates to
   produce Hebrew dialogue with inline event markers.
2. **TTS Renderer** — Azure Cognitive Services he-IL voices
   (`he-IL-AvriNeural`, `he-IL-HilaNeural`) render per-speaker segments;
   a scene mixer concatenates them with natural timing.
3. **Preprocessing** — Polyphase resampling to 16 kHz, mono downmix, 7.5 kHz
   Butterworth low-pass, Wiener denoising, −1.0 dBFS peak normalization,
   ≥ 0.5 s silence padding. No torchaudio — implemented with scipy + soundfile.
4. **Acoustic Augmentation (Tier B/C only)** — Room impulse response simulation
   (pyroomacoustics), device codec profiles (phone/Pi mic), and background noise
   layering (MUSAN subset + licensed SFX).
5. **Labeling** — Auto-generated from script structure and augmentation log;
   no post-hoc human labeling in the primary pipeline. IAA-reviewed clips are
   flagged with `iaa_reviewed: true` in the strong label JSONL.

### Label Taxonomy

Labels use a three-level hierarchy (defined in `configs/taxonomy.yaml`):

- **Violence typology** (scene-level): `SV`, `IT`, `NEG`, `NEU`
- **Tier 1 category** (event-level): `PHYS`, `VERB`, `DIST`, `ACOU`, `EMOT`, `NONE`
- **Tier 2 subtype** (event-level): e.g. `PHYS_HARD`, `VERB_THREAT`, `DIST_SCREAM`

Binary Violence/Non-Violence labels are **not provided** by design.

## Known Limitations

- **Synthetic-to-real gap**: Hebrew TTS voices lack the acoustic variability of
  real speech under stress. Distress vocalisations (`DIST_SCREAM`, `DIST_CRY`)
  are the weakest TTS category. Models trained solely on this data should be
  evaluated on real recordings before deployment.
- **Speaker diversity**: Speaker personas are drawn from a finite set of
  parameterized profiles. The dataset does not capture dialect variation,
  elderly speakers, or children's voices.
- **Script plausibility**: Scene templates have been reviewed by the engineering
  team. Formal clinical review by Rakman Institute domain experts is ongoing for
  a subset of templates.
- **No environmental co-occurring events**: Tier A clips have clean audio with
  no background interference. Tier B/C clips add acoustic augmentation but do
  not capture the full range of real domestic or clinic environments.

## Data Format

Each clip consists of three required files sharing the same stem:

```
data/{{language_code}}/{{speaker_id}}/{{clip_id}}.wav   ← 16 kHz mono 16-bit PCM
data/{{language_code}}/{{speaker_id}}/{{clip_id}}.txt   ← Hebrew transcript (UTF-8)
data/{{language_code}}/{{speaker_id}}/{{clip_id}}.json  ← ClipMetadata (JSON)
data/{{language_code}}/{{speaker_id}}/{{clip_id}}.jsonl ← EventLabel records (JSONL, Tier B/C)
```

A flat `manifest.csv` at the dataset root indexes all clips with columns:
`clip_id`, `project`, `violence_typology`, `tier`, `duration_seconds`,
`speaker_ids`, `has_violence`, `max_intensity`, `quality_flags`, `split`,
`wav_path`, `strong_labels_path`.

## License

[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

Audio generated via Azure Cognitive Services TTS is subject to Microsoft's
[Cognitive Services terms](https://azure.microsoft.com/en-us/support/legal/).
MUSAN noise clips are used under CC BY 4.0 ([OpenSLR](https://openslr.org/17/)).

## Citation

```bibtex
@dataset{{avdp_synth_{cite_version},
  title  = {{AVDP Synthetic Hebrew Audio Dataset {version}}},
  author = {{DataHack / DataForBetter}},
  year   = {{2026}},
  url    = {{https://datahack.org.il}},
  note   = {{Synthetic bootstrap dataset for She-Proves and Elephant in the Room
            AI safety projects}}
}}
```

## Contact

DataHack / DataForBetter — [datahack.org.il](https://datahack.org.il)
"""


def _size_category(n_clips: int) -> str:
    """Return the HuggingFace ``size_categories`` tag for *n_clips*."""
    if n_clips < 1_000:
        return "n<1K"
    if n_clips < 10_000:
        return "1K<n<10K"
    if n_clips < 100_000:
        return "10K<n<100K"
    return "100K<n<1M"
