"""SynthBanshee CLI entry points.

Usage:
    synthbanshee generate --config configs/scenes/test_scene_001.yaml
    synthbanshee generate-batch --run-config configs/run_configs/tier_a_500_she_proves.yaml
    synthbanshee validate data/he/agg_m_30-45_001/sp_it_a_0001_00.wav
    synthbanshee qa-report data/he --output qa_report.json
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Shared pipeline helper
# ---------------------------------------------------------------------------


# Semantic mapping: (violence_typology) → ordered list of (max_intensity, tier1, tier2).
# Each entry matches intensities ≤ max_intensity; the last entry is the catch-all.
# Validated against configs/taxonomy.yaml at import time (see _validate_event_type_codes).
_TYPOLOGY_INTENSITY_MAP: dict[str, list[tuple[int, str, str]]] = {
    "NEU": [(5, "NONE", "NONE_AMBIENT")],
    "NEG": [(5, "NONE", "NONE_ARGU")],
    "IT": [
        (2, "EMOT", "EMOT_GASLIT"),
        (3, "EMOT", "EMOT_ISOL"),
        (5, "VERB", "VERB_THREAT"),
    ],
    "SV": [
        (2, "VERB", "VERB_SHOUT"),
        (3, "VERB", "VERB_THREAT"),
        (4, "DIST", "DIST_SCREAM"),
        (5, "PHYS", "PHYS_HARD"),
    ],
}
_DEFAULT_EVENT_TYPE: tuple[str, str] = ("NONE", "NONE_AMBIENT")


def _validate_event_type_codes() -> None:
    """Assert every code in _TYPOLOGY_INTENSITY_MAP is valid per configs/taxonomy.yaml."""
    from synthbanshee.config.taxonomy import tier1_category_codes, tier2_subtype_codes

    valid_tier1 = tier1_category_codes()
    valid_tier2 = tier2_subtype_codes()
    for typology, entries in _TYPOLOGY_INTENSITY_MAP.items():
        for _max_int, t1, t2 in entries:
            if t1 not in valid_tier1:
                raise ValueError(
                    f"_TYPOLOGY_INTENSITY_MAP[{typology!r}]: tier1 {t1!r} not in taxonomy"
                )
            if t2 not in valid_tier2:
                raise ValueError(
                    f"_TYPOLOGY_INTENSITY_MAP[{typology!r}]: tier2 {t2!r} not in taxonomy"
                )
    t1_default, t2_default = _DEFAULT_EVENT_TYPE
    if t1_default not in valid_tier1 or t2_default not in valid_tier2:
        raise ValueError(f"_DEFAULT_EVENT_TYPE {_DEFAULT_EVENT_TYPE!r} not in taxonomy")


_validate_event_type_codes()


def _derive_event_type(violence_typology: str, intensity: int) -> tuple[str, str]:
    """Map (violence_typology, intensity) to (tier1_category, tier2_subtype)."""
    for max_intensity, tier1, tier2 in _TYPOLOGY_INTENSITY_MAP.get(violence_typology, []):
        if intensity <= max_intensity:
            return tier1, tier2
    return _DEFAULT_EVENT_TYPE


def _run_generate_pipeline(
    config: Path,
    output_dir: Path,
    cache_dir: Path,
    dirty_dir: Path,
    script_cache_dir: Path,
    verbose: bool = False,
) -> tuple[Path | None, list[str]]:
    """Run the full single-clip generate pipeline.

    Returns (wav_path, errors/warnings). wav_path is None on failure.
    """

    def vlog(msg: str) -> None:
        if verbose:
            console.print(msg)

    import soundfile as sf

    from synthbanshee.augment.preprocessing import preprocess
    from synthbanshee.config.scene_config import SceneConfig
    from synthbanshee.config.speaker_config import SpeakerConfig
    from synthbanshee.labels.generator import LabelGenerator, ScriptEvent
    from synthbanshee.labels.schema import PreprocessingApplied, SpeakerInfo
    from synthbanshee.package.validator import validate_clip
    from synthbanshee.script.generator import ScriptGenerator
    from synthbanshee.tts.renderer import TTSRenderer

    # Pre-pipeline: load configs (not a numbered stage — no data transformation occurs)
    # 1. Load scene config
    try:
        scene = SceneConfig.from_yaml(config)
    except Exception as exc:
        return None, [f"Config parse error: {exc}"]

    # 2. Load ALL speaker configs into a dict keyed by speaker_id
    speakers: dict[str, SpeakerConfig] = {}
    for spk_ref in scene.speakers:
        spk_path = Path("configs/speakers") / f"{spk_ref.speaker_id}.yaml"
        if not spk_path.exists():
            spk_path = Path("configs/examples") / f"speaker_{spk_ref.speaker_id}.yaml"
        if not spk_path.exists():
            return None, [f"Speaker config not found: {spk_ref.speaker_id}"]
        try:
            speakers[spk_ref.speaker_id] = SpeakerConfig.from_yaml(spk_path)
        except Exception as exc:
            return None, [f"Speaker config parse error: {exc}"]

    # Stage 1 — Script Generation
    # 3. Generate script
    vlog("[bold]Stage 1[/bold] — Script generation")
    script_gen = ScriptGenerator(cache_dir=script_cache_dir)
    try:
        turns = script_gen.generate(
            scene_id=scene.scene_id,
            project=scene.project,
            violence_typology=scene.violence_typology,
            script_template=scene.script_template,
            script_slots=scene.script_slots,
            intensity_arc=scene.intensity_arc,
            target_duration_minutes=scene.target_duration_minutes,
            speakers=[
                {
                    "speaker_id": spk.speaker_id,
                    "role": spk.role,
                    "gender": spk.gender,
                    "age_range": spk.age_range,
                    "tts_voice_id": spk.tts_voice_id,
                }
                for spk in speakers.values()
            ],
            random_seed=scene.random_seed,
            verbose_log=vlog if verbose else None,
        )
    except Exception as exc:
        return None, [f"Script generation error: {exc}"]

    vlog(f"  [dim]script: {len(turns)} turns[/dim]")

    # Stage 2 — TTS Rendering
    # 4. Render multi-speaker scene
    vlog(f"[bold]Stage 2[/bold] — TTS rendering ({len(turns)} turns)")
    renderer = TTSRenderer(cache_dir=cache_dir)
    try:
        mixed = renderer.render_scene(
            turns=turns,
            speakers=speakers,
            disfluency=True,
            rng_seed=scene.random_seed,
            verbose_log=vlog if verbose else None,
        )
    except Exception as exc:
        return None, [f"TTS render error: {exc}"]

    vlog(f"  [dim]scene mixed: {mixed.duration_s:.1f} s[/dim]")

    # Stage 3a — Preprocessing
    vlog("[bold]Stage 3a[/bold] — Preprocessing")
    # 5. Write raw mix to temp WAV, preprocess to final output path
    clip_id = scene.scene_id.lower().replace("-", "_")
    first_speaker_ref = scene.speakers[0]
    speaker_dir = output_dir / first_speaker_ref.speaker_id.lower()
    clip_wav = speaker_dir / f"{clip_id}_00.wav"
    clip_txt = speaker_dir / f"{clip_id}_00.txt"
    clip_json = speaker_dir / f"{clip_id}_00.json"

    try:
        with tempfile.TemporaryDirectory() as tmp:
            raw_wav = Path(tmp) / "raw.wav"
            sf.write(str(raw_wav), mixed.samples, mixed.sample_rate, subtype="PCM_16")
            result = preprocess(raw_wav, clip_wav, dirty_dir=dirty_dir)
    except Exception as exc:
        return None, [f"Pipeline error: {exc}"]

    vlog(
        f"  [dim]preprocess: {result.sample_rate} Hz, {result.duration_seconds:.1f} s,"
        f" peak={result.peak_dbfs:.1f} dBFS, pad={result.silence_pad_applied_s:.2f} s/side"
        f" — steps: {', '.join(result.steps_applied)}[/dim]"
    )

    # Stage 3b — Acoustic Augmentation (Tier B and Tier C)
    # 5b. Room simulation → device profile → noise mix
    #
    # Stage 3 receives the fully preprocessed (16 kHz, mono, silence-padded) WAV.
    # NoiseMixer therefore operates on padded audio, so its returned AugmentedEvent
    # onset/offset values are measured from the start of the *padded* clip — they are
    # already in final clip coordinates and must NOT be shifted by pad_s again.
    # (This differs from the AugmentedEvent docstring convention, which assumes
    # NoiseMixer receives the raw un-padded MixedScene.  Here we pass padded audio
    # intentionally so the placed events align with audible speech, not with silence.)
    acoustic_scene_meta = None
    _aug_acou_events: list = []  # ACOU_* SFX events for strong-label generation
    if scene.tier in ("B", "C") and scene.acoustic_scene is not None:
        vlog("[bold]Stage 3b[/bold] — Acoustic augmentation")
        try:
            import numpy as _np
            import soundfile as _sf

            from synthbanshee.augment.device_profiles import DeviceProfiler
            from synthbanshee.augment.noise_mixer import NoiseMixer
            from synthbanshee.augment.room_sim import RoomSimulator
            from synthbanshee.config.taxonomy import tier2_subtype_codes
            from synthbanshee.labels.schema import ClipAcousticScene

            audio_data, audio_sr = _sf.read(str(clip_wav), dtype="float32", always_2d=False)
            n_orig = len(audio_data)
            reverbed = RoomSimulator().apply(
                audio_data, audio_sr, scene.acoustic_scene, rng_seed=scene.random_seed
            )
            device_colored = DeviceProfiler().apply(
                reverbed, audio_sr, scene.acoustic_scene.device, rng_seed=scene.random_seed
            )
            aug_samples, aug_events, snr_actual = NoiseMixer().mix(
                device_colored, audio_sr, scene.acoustic_scene, rng_seed=scene.random_seed
            )

            # Enforce exact length match with the input WAV so metadata.duration_seconds
            # and transcript timings remain consistent after the write-back.
            if len(aug_samples) > n_orig:
                aug_samples = aug_samples[:n_orig]
            elif len(aug_samples) < n_orig:
                aug_samples = _np.pad(aug_samples, (0, n_orig - len(aug_samples))).astype(
                    _np.float32
                )

            # Restore silence padding regions that ambient mixing or room reverb tails
            # may have contaminated.  validate_audio() requires the first and last
            # silence_pad_applied_s seconds to remain below −40 dBFS.
            pad_n = int(result.silence_pad_applied_s * audio_sr)
            aug_samples[:pad_n] = 0.0
            aug_samples[-pad_n:] = 0.0

            # Peak-normalize augmented signal to −1.0 dBFS
            peak = float(_np.max(_np.abs(aug_samples)))
            if peak > 0.0:
                target_peak = 10.0 ** (-1.0 / 20.0)
                aug_samples = (aug_samples * (target_peak / peak)).astype(_np.float32)
            _sf.write(str(clip_wav), aug_samples, audio_sr, subtype="PCM_16")

            acoustic_scene_meta = ClipAcousticScene(
                room_type=scene.acoustic_scene.room_type,
                device=scene.acoustic_scene.device,
                ir_source=scene.acoustic_scene.ir_source,
                speaker_distance_meters=scene.acoustic_scene.speaker_distance_meters,
                snr_db_actual=round(snr_actual, 2),
                background_events=[
                    {
                        "type": ev.type,
                        "onset": round(ev.onset_s, 3),
                        "offset": round(ev.offset_s, 3),
                        "level_db": round(ev.level_db, 1),
                    }
                    for ev in aug_events
                ],
            )
            # Keep foreground ACOU_* SFX that are valid taxonomy tier2 codes.
            # Unknown/mis-typed ACOU_* values are silently dropped rather than
            # crashing label generation with a cryptic validation error.
            _valid_tier2 = tier2_subtype_codes()
            _aug_acou_events = [
                ev for ev in aug_events if ev.type.startswith("ACOU_") and ev.type in _valid_tier2
            ]
        except Exception as exc:
            return None, [f"Acoustic augmentation error: {exc}"]

    # Stage 4a — Transcript
    vlog("[bold]Stage 4[/bold] — Transcript, labels, metadata")
    # 6. Write structured multi-speaker transcript
    # preprocess() prepends result.silence_pad_applied_s of silence unconditionally,
    # so all MixedScene timings must be shifted forward by that amount.
    pad_s = result.silence_pad_applied_s
    label_gen = LabelGenerator()
    clip_txt.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"[CLIP_ID: {clip_id}_00]"]
    for i, turn in enumerate(turns):
        raw_onset = mixed.turn_onsets_s[i] if i < len(mixed.turn_onsets_s) else 0.0
        raw_offset = mixed.turn_offsets_s[i] if i < len(mixed.turn_offsets_s) else mixed.duration_s
        onset = raw_onset + pad_s
        offset = raw_offset + pad_s
        spk = speakers.get(turn.speaker_id)
        role = spk.role if spk else "UNK"
        _, tier2 = _derive_event_type(scene.violence_typology, turn.intensity)
        lines.append(
            f"[SPEAKER: {turn.speaker_id} | ROLE: {role} | ONSET: {onset:.2f} | OFFSET: {offset:.2f}]"
        )
        lines.append(turn.text)
        lines.append(f"[ACTION: {tier2} | INTENSITY: {turn.intensity}]")
    clip_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Stage 4b — Strong Labels
    # 7. Build per-turn event labels from MixedScene timing (authoritative per spec)
    # Same pad_s offset applied here to match the processed audio.
    from synthbanshee.config.taxonomy import emotional_state_values as _valid_emotions

    _known_emotions = set(_valid_emotions())

    def _normalize_emotion(state: str) -> tuple[str, bool]:
        """Map LLM-generated emotional states to valid taxonomy values.

        Strips whitespace and lowercases before lookup so that common LLM
        output variations like 'Calm', 'ANGER', or 'neutral ' are handled
        correctly. Falls back to 'neutral' for values not in the taxonomy
        (e.g. 'relaxed', 'happy', 'worried').

        Returns (normalized_state, fallback_used).
        """
        normalized = state.strip().lower()
        if normalized in _known_emotions:
            return normalized, False
        return "neutral", True

    messages: list[str] = []
    events: list[ScriptEvent] = []
    for i, turn in enumerate(turns):
        raw_onset = mixed.turn_onsets_s[i] if i < len(mixed.turn_onsets_s) else 0.0
        raw_offset = mixed.turn_offsets_s[i] if i < len(mixed.turn_offsets_s) else mixed.duration_s
        onset = raw_onset + pad_s
        offset = raw_offset + pad_s
        spk = speakers.get(turn.speaker_id)
        role = spk.role if spk else "UNK"
        tier1, tier2 = _derive_event_type(scene.violence_typology, turn.intensity)
        emotion, fallback = _normalize_emotion(turn.emotional_state)
        if fallback:
            messages.append(
                f"turn[{i}]: emotional_state {turn.emotional_state!r} not in taxonomy"
                f" — mapped to 'neutral'"
            )
        events.append(
            ScriptEvent(
                tier1_category=tier1,
                tier2_subtype=tier2,
                onset=onset,
                offset=max(offset, onset + 0.1),
                intensity=turn.intensity,
                speaker_id=turn.speaker_id,
                speaker_role=role,
                emotional_state=emotion,
            )
        )
    # Stage 3b ACOU_* SFX events (Tier B and Tier C). Their onset/offset times are
    # already in padded-audio coordinates — no pad_s shift needed.
    for aug_ev in _aug_acou_events:
        events.append(
            ScriptEvent(
                tier1_category="ACOU",
                tier2_subtype=aug_ev.type,
                onset=aug_ev.onset_s,
                offset=max(aug_ev.offset_s, aug_ev.onset_s + 0.1),
                intensity=3,  # default mid-intensity for physical SFX events
                notes="sfx_augmentation",
            )
        )

    event_labels = label_gen.generate_event_labels(f"{clip_id}_00", events)

    # 7b. Write per-clip strong labels JSONL alongside the WAV (Stage 4b output)
    clip_jsonl = speaker_dir / f"{clip_id}_00.jsonl"
    label_gen.write_strong_labels_jsonl(event_labels, clip_jsonl)

    # Stage 4c — Clip Metadata JSON
    # 8. Write clip metadata JSON
    speakers_meta = [
        SpeakerInfo(
            speaker_id=spk.speaker_id,
            role=spk.role,
            gender=spk.gender,
            age_range=spk.age_range,
            tts_voice_id=spk.tts_voice_id,
        )
        for spk in speakers.values()
    ]
    preprocessing_meta = PreprocessingApplied(
        resampled_to_16k=True,
        downmixed_to_mono=True,
        spectral_filtered=True,
        denoised=True,
        normalized_dbfs=-1.0,
        silence_padded=True,
    )
    metadata = label_gen.generate_clip_metadata(
        clip_id=f"{clip_id}_00",
        project=scene.project,
        violence_typology=scene.violence_typology,
        tier=scene.tier,
        duration_seconds=result.duration_seconds,
        events=event_labels,
        speakers=speakers_meta,
        scene_config_path=str(config),
        random_seed=scene.random_seed,
        preprocessing=preprocessing_meta,
        dirty_file_path=str(result.dirty_path) if result.dirty_path else None,
        transcript_path=str(clip_txt),
        acoustic_scene=acoustic_scene_meta,
    )
    label_gen.write_clip_metadata_json(metadata, clip_json)

    vlog(
        f"  [dim]wrote: {clip_txt.name} ({len(turns)} turns),"
        f" {clip_jsonl.name} ({len(event_labels)} events),"
        f" {clip_json.name}[/dim]"
    )

    # Stage 5 — Validation
    vlog("[bold]Stage 5[/bold] — Validation")
    validation = validate_clip(clip_wav)
    if not validation.is_valid:
        return None, validation.errors

    vlog(f"  [dim]clip valid: {clip_wav}[/dim]")
    return clip_wav, messages + list(validation.warnings)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option()
def cli() -> None:
    """AVDP Synthetic Dataset Generation Framework."""


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to scene YAML config.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Override output directory (default: from config).",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=Path("assets/speech"),
    show_default=True,
    help="TTS render cache directory.",
)
@click.option(
    "--dirty-dir",
    type=click.Path(path_type=Path),
    default=Path("assets/speech/dirty"),
    show_default=True,
    help="Directory to retain pre-preprocessing (dirty) audio files.",
)
@click.option(
    "--script-cache-dir",
    type=click.Path(path_type=Path),
    default=Path("assets/scripts"),
    show_default=True,
    help="LLM script generation cache directory.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Parse and validate config only; do not render.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Print per-stage and per-turn progress (cache hits, preprocessing steps, etc.).",
)
def generate(
    config: Path,
    output_dir: Path | None,
    cache_dir: Path,
    dirty_dir: Path,
    script_cache_dir: Path,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Generate a synthetic clip from a scene YAML config."""
    from synthbanshee.config.scene_config import SceneConfig

    console.print(f"[bold]Loading config:[/bold] {config}")
    scene = SceneConfig.from_yaml(config)
    console.print(
        Panel(
            f"Scene: [bold]{scene.scene_id}[/bold]  "
            f"Project: {scene.project}  "
            f"Typology: {scene.violence_typology}  "
            f"Tier: {scene.tier}",
            title="Scene Config",
        )
    )

    if dry_run:
        console.print("[green]Dry run — config is valid. Exiting.[/green]")
        return

    out_dir = Path(output_dir or scene.output_dir)

    console.print("[cyan]Running generate pipeline...[/cyan]")
    wav_path, messages = _run_generate_pipeline(
        config, out_dir, cache_dir, dirty_dir, script_cache_dir, verbose=verbose
    )

    if wav_path is None:
        console.print("[bold red]Generation FAILED:[/bold red]")
        for err in messages:
            console.print(f"  [red]• {err}[/red]")
        sys.exit(1)

    console.print(f"[bold green]Clip generated and valid:[/bold green] {wav_path}")
    for w in messages:
        console.print(f"  [yellow]• {w}[/yellow]")


# ---------------------------------------------------------------------------
# generate-batch
# ---------------------------------------------------------------------------


def _discover_scene_configs(
    scene_configs_dir: Path,
    project: str,
    tier: str,
) -> list[Path]:
    """Return all scene config YAML paths in scene_configs_dir matching project + tier."""
    from synthbanshee.config.scene_config import SceneConfig

    results: list[Path] = []
    for yaml_path in sorted(scene_configs_dir.rglob("*.yaml")):
        try:
            scene = SceneConfig.from_yaml(yaml_path)
        except Exception:
            continue
        if scene.project == project and scene.tier == tier:
            results.append(yaml_path)
    return results


def _select_configs_by_typology(
    all_configs: list[Path],
    targets: dict[str, int],
    rng_seed: int,
) -> list[Path]:
    """Select scene configs stratified by typology targets.

    For each typology, configs are shuffled with rng_seed then capped at the
    target count.  This ensures reproducible selection across runs.
    """
    import random

    from synthbanshee.config.scene_config import SceneConfig

    by_typology: dict[str, list[Path]] = {}
    for yaml_path in all_configs:
        try:
            scene = SceneConfig.from_yaml(yaml_path)
        except Exception:
            continue
        typology = scene.violence_typology
        by_typology.setdefault(typology, []).append(yaml_path)

    rng = random.Random(rng_seed)
    selected: list[Path] = []
    for typology, limit in targets.items():
        pool = list(by_typology.get(typology, []))
        rng.shuffle(pool)
        selected.extend(pool[:limit])

    return selected


def _render_one(
    scene_yaml: Path,
    out_dir: Path,
    cache_dir: Path,
    dirty_dir: Path,
    script_cache_dir: Path,
    max_retries: int,
    stop_event: threading.Event,
    verbose: bool = False,
) -> tuple[Path | None, list[str]]:
    """Render a single clip with retries.

    Thread-safety: this function itself holds no shared in-memory mutable state.
    On-disk cache directories (TTS, scripts) are shared across workers; cache
    writes are keyed by content hash so concurrent writes for the same key are
    idempotent (same bytes written by multiple threads produce the same result).
    Output paths are unique per clip_id, so file writes never collide.

    Returns (wav_path, messages).  Returns (None, ["cancelled"]) immediately if
    *stop_event* is set before the first attempt, or (None, messages) on
    exhausted retries.
    """
    messages: list[str] = []
    for _attempt in range(max_retries):
        if stop_event.is_set():
            return None, ["cancelled"]
        wav_path, messages = _run_generate_pipeline(
            scene_yaml, out_dir, cache_dir, dirty_dir, script_cache_dir, verbose=verbose
        )
        if wav_path is not None:
            return wav_path, messages
    return None, messages


@cli.command("generate-batch")
@click.option(
    "--run-config",
    "-r",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to run configuration YAML.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Override output directory (default: from run config).",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=Path("assets/speech"),
    show_default=True,
    help="TTS render cache directory.",
)
@click.option(
    "--dirty-dir",
    type=click.Path(path_type=Path),
    default=Path("assets/speech/dirty"),
    show_default=True,
    help="Directory to retain pre-preprocessing (dirty) audio files.",
)
@click.option(
    "--script-cache-dir",
    type=click.Path(path_type=Path),
    default=Path("assets/scripts"),
    show_default=True,
    help="LLM script generation cache directory.",
)
@click.option(
    "--manifest-out",
    "-m",
    type=click.Path(path_type=Path),
    default=None,
    help="Path for manifest CSV (default: <output-dir>/manifest.csv).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Discover and count scene configs without rendering.",
)
@click.option(
    "--workers",
    "-j",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Number of parallel worker threads for clip rendering.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Print per-turn TTS and script generation details.",
)
def generate_batch(
    run_config: Path,
    output_dir: Path | None,
    cache_dir: Path,
    dirty_dir: Path,
    script_cache_dir: Path,
    manifest_out: Path | None,
    dry_run: bool,
    workers: int,
    verbose: bool,
) -> None:
    """Run a full batch generation from a run configuration YAML.

    Discovers scene configs in the run config's scene_configs_dir, applies
    per-typology count limits, renders each clip, assigns speaker-disjoint
    splits, and writes a manifest CSV.
    """
    from synthbanshee.config.run_config import RunConfig
    from synthbanshee.package.manifest import generate_manifest
    from synthbanshee.package.splitter import assign_splits

    console.print(f"[bold]Loading run config:[/bold] {run_config}")
    run_cfg = RunConfig.from_yaml(run_config)
    console.print(
        Panel(
            f"Run: [bold]{run_cfg.run_id}[/bold]  "
            f"Project: {run_cfg.project}  "
            f"Tier: {run_cfg.tier}  "
            f"Total target: {run_cfg.total_target} clips",
            title="Run Config",
        )
    )

    scene_cfg_dir = Path(run_cfg.scene_configs_dir)
    if not scene_cfg_dir.exists():
        console.print(f"[red]scene_configs_dir not found: {scene_cfg_dir}[/red]")
        sys.exit(1)

    console.print(f"[cyan]Discovering scene configs in:[/cyan] {scene_cfg_dir}")
    all_configs = _discover_scene_configs(scene_cfg_dir, run_cfg.project, run_cfg.tier)
    selected = _select_configs_by_typology(
        all_configs, run_cfg.targets_by_typology(), run_cfg.random_seed
    )

    console.print(
        f"Found [bold]{len(all_configs)}[/bold] matching configs; "
        f"selected [bold]{len(selected)}[/bold] for this run."
    )

    if dry_run:
        _print_selection_summary(selected)
        console.print("[green]Dry run — no clips rendered.[/green]")
        return

    out_dir = Path(output_dir or run_cfg.output_dir)
    manifest_path = manifest_out or (out_dir / "manifest.csv")

    if workers > 1:
        console.print(f"[cyan]Parallel rendering:[/cyan] {workers} workers")

    # --- Render loop ---
    succeeded: list[Path] = []
    failed: list[tuple[Path, list[str]]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Rendering clips", total=len(selected))

        if workers == 1:
            # Sequential path — exact pre-parallelisation behaviour; no thread
            # overhead, no signal-handling differences, debugger-friendly.
            _no_stop = threading.Event()
            for scene_yaml in selected:
                wav_path, messages = _render_one(
                    scene_yaml,
                    out_dir,
                    cache_dir,
                    dirty_dir,
                    script_cache_dir,
                    run_cfg.max_retries,
                    _no_stop,
                    verbose=verbose,
                )
                progress.advance(task_id)
                if wav_path is None:
                    failed.append((scene_yaml, messages))
                    if run_cfg.fail_fast:
                        progress.stop()
                        console.print(
                            f"[red]fail_fast: aborting after failure on {scene_yaml.name}[/red]"
                        )
                        for msg in messages:
                            console.print(f"  [red]• {msg}[/red]")
                        sys.exit(1)
                else:
                    succeeded.append(wav_path)
        else:
            # Parallel path — ThreadPoolExecutor; workers > 1 only.
            stop_event = threading.Event()
            lock = threading.Lock()

            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_scene = {
                    executor.submit(
                        _render_one,
                        scene_yaml,
                        out_dir,
                        cache_dir,
                        dirty_dir,
                        script_cache_dir,
                        run_cfg.max_retries,
                        stop_event,
                        verbose,
                    ): scene_yaml
                    for scene_yaml in selected
                }

                for future in as_completed(future_to_scene):
                    scene_yaml = future_to_scene[future]
                    progress.advance(task_id)
                    try:
                        wav_path, messages = future.result()
                    except CancelledError:
                        continue
                    with lock:
                        if wav_path is None and not stop_event.is_set():
                            failed.append((scene_yaml, messages))
                            if run_cfg.fail_fast:
                                stop_event.set()
                                for f in future_to_scene:
                                    f.cancel()
                        elif wav_path is not None:
                            succeeded.append(wav_path)

            if stop_event.is_set() and failed:
                scene_yaml, messages = failed[0]
                console.print(f"[red]fail_fast: aborting after failure on {scene_yaml.name}[/red]")
                for msg in messages:
                    console.print(f"  [red]• {msg}[/red]")
                sys.exit(1)

    # --- Split assignment ---
    from synthbanshee.labels.schema import ClipMetadata

    clip_speaker_map: dict[str, list[str]] = {}
    for wav_path in succeeded:
        json_path = wav_path.with_suffix(".json")
        try:
            meta = ClipMetadata.model_validate_json(json_path.read_text(encoding="utf-8"))
            clip_speaker_map[meta.clip_id] = [s.speaker_id for s in meta.speakers]
        except Exception:
            clip_speaker_map[wav_path.stem] = []

    splits = assign_splits(
        clip_speaker_map,
        train_frac=run_cfg.splits.train,
        val_frac=run_cfg.splits.val,
        test_frac=run_cfg.splits.test,
        rng_seed=run_cfg.random_seed,
    )

    # --- Manifest ---
    rows = generate_manifest(out_dir, manifest_path, splits=splits, clip_ids=set(splits.keys()))
    console.print(f"[bold green]Manifest written:[/bold green] {manifest_path} ({len(rows)} rows)")

    # --- Summary ---
    failure_rate = len(failed) / len(selected) if selected else 0.0
    _print_batch_summary(succeeded, failed, splits, failure_rate)

    if failed:
        sys.exit(1)


def _print_selection_summary(configs: list[Path]) -> None:
    """Print a Rich table showing selected configs by typology."""
    from synthbanshee.config.scene_config import SceneConfig

    counts: dict[str, int] = {}
    for p in configs:
        try:
            scene = SceneConfig.from_yaml(p)
            counts[scene.violence_typology] = counts.get(scene.violence_typology, 0) + 1
        except Exception:
            pass

    table = Table(title="Selected configs by typology")
    table.add_column("Typology")
    table.add_column("Count", justify="right")
    for typology, count in sorted(counts.items()):
        table.add_row(typology, str(count))
    console.print(table)


def _print_batch_summary(
    succeeded: list[Path],
    failed: list[tuple[Path, list[str]]],
    splits: dict[str, str],
    failure_rate: float,
) -> None:
    """Print a Rich summary table for a completed batch run."""
    split_counts: dict[str, int] = {}
    for s in splits.values():
        split_counts[s] = split_counts.get(s, 0) + 1

    table = Table(title="Batch Generation Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Succeeded", str(len(succeeded)))
    table.add_row("Failed", str(len(failed)))
    table.add_row("Failure rate", f"{failure_rate:.1%}")
    for split_name in ("train", "val", "test"):
        table.add_row(f"  {split_name} clips", str(split_counts.get(split_name, 0)))
    console.print(table)

    if failed:
        console.print("[bold red]Failed configs:[/bold red]")
        for cfg_path, errors in failed:
            console.print(f"  [red]{cfg_path.name}[/red]")
            for err in errors:
                console.print(f"    • {err}")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("clip", type=click.Path(exists=True, path_type=Path))
def validate(clip: Path) -> None:
    """Validate an existing clip against the AVDP spec."""
    from synthbanshee.package.validator import validate_clip

    result = validate_clip(clip)
    if result.is_valid:
        console.print(f"[bold green]VALID[/bold green] {clip}")
    else:
        console.print(f"[bold red]INVALID[/bold red] {clip}")
        for err in result.errors:
            console.print(f"  [red]• {err}[/red]")
        sys.exit(1)

    if result.warnings:
        for w in result.warnings:
            console.print(f"  [yellow]• {w}[/yellow]")


# ---------------------------------------------------------------------------
# qa-report
# ---------------------------------------------------------------------------


@cli.command("qa-report")
@click.argument("data_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Write QA report to this JSON file (default: print to console only).",
)
@click.option(
    "--max-failure-rate",
    type=float,
    default=0.02,
    show_default=True,
    help="Maximum acceptable failure rate (fraction, default 2%).",
)
def qa_report(data_dir: Path, output: Path | None, max_failure_rate: float) -> None:
    """Run the automated QA suite on a dataset directory.

    Validates every WAV/TXT/JSON triplet, computes dataset statistics, and
    reports pass/fail based on the clip failure rate.
    """
    from synthbanshee.package.qa import run_qa

    console.print(f"[cyan]Running QA on:[/cyan] {data_dir}")
    report = run_qa(data_dir, max_failure_rate=max_failure_rate)
    stats = report.stats

    table = Table(title="Dataset Statistics")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Total clips (valid)", str(stats.total_clips))
    table.add_row("Failed clips", str(stats.failed_clips))
    table.add_row("Failure rate", f"{report.failure_rate:.1%}")
    table.add_row(
        "Total duration",
        f"{stats.total_duration_seconds / 60:.1f} min"
        if stats.total_duration_seconds > 0
        else "0 min",
    )
    table.add_row("Unique speakers", str(stats.speaker_count))
    table.add_row("Quality-flagged", str(stats.quality_flagged_clips))
    missing_jsonl_label = (
        f"[yellow]{stats.clips_missing_strong_labels}[/yellow]"
        if stats.clips_missing_strong_labels > 0
        else "0"
    )
    table.add_row("Missing strong-labels JSONL (Stage 4b)", missing_jsonl_label)
    console.print(table)

    if stats.clips_by_typology:
        t2 = Table(title="Clips by Typology")
        t2.add_column("Typology")
        t2.add_column("Count", justify="right")
        for typology, count in sorted(stats.clips_by_typology.items()):
            t2.add_row(typology, str(count))
        console.print(t2)

    if stats.clips_by_split:
        t3 = Table(title="Clips by Split")
        t3.add_column("Split")
        t3.add_column("Count", justify="right")
        for split_name, count in sorted(stats.clips_by_split.items()):
            t3.add_row(split_name, str(count))
        console.print(t3)

    if report.failed_clip_ids:
        console.print("[bold red]Failed clip IDs:[/bold red]")
        for clip_id in report.failed_clip_ids[:20]:
            console.print(f"  [red]• {clip_id}[/red]")
        if len(report.failed_clip_ids) > 20:
            console.print(f"  ... and {len(report.failed_clip_ids) - 20} more")

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        report_dict = {
            "data_dir": report.data_dir,
            "passed": report.passed,
            "failure_rate": report.failure_rate,
            "stats": {
                "total_clips": stats.total_clips,
                "failed_clips": stats.failed_clips,
                "total_duration_seconds": stats.total_duration_seconds,
                "speaker_count": stats.speaker_count,
                "quality_flagged_clips": stats.quality_flagged_clips,
                "clips_missing_strong_labels": stats.clips_missing_strong_labels,
                "clips_by_typology": stats.clips_by_typology,
                "clips_by_split": stats.clips_by_split,
                "clips_by_tier": stats.clips_by_tier,
                "intensity_distribution": {
                    str(k): v for k, v in stats.intensity_distribution.items()
                },
            },
            "failed_clip_ids": report.failed_clip_ids,
            "quality_flagged": report.quality_flagged,
        }
        output.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
        console.print(f"[bold green]Report written:[/bold green] {output}")

    if report.passed:
        console.print(
            f"[bold green]QA PASSED[/bold green] — failure rate "
            f"{report.failure_rate:.1%} ≤ {max_failure_rate:.1%}"
        )
    else:
        console.print(
            f"[bold red]QA FAILED[/bold red] — failure rate "
            f"{report.failure_rate:.1%} > {max_failure_rate:.1%}"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# iaa-report
# ---------------------------------------------------------------------------


@cli.command("iaa-report")
@click.argument("annotations_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--total-clips",
    type=click.IntRange(min=1),
    required=True,
    help="Total number of clips in the dataset (for coverage fraction calculation).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Write IAA report to this JSON file (default: print to console only).",
)
def iaa_report(annotations_dir: Path, total_clips: int, output: Path | None) -> None:
    """Compute inter-annotator agreement from paired JSONL annotation files.

    ANNOTATIONS_DIR must contain subdirectories named after clip IDs, each
    with exactly two JSONL files — one per annotator.  The command reads every
    EventLabel from each file, computes Cohen's Kappa per event-category group
    (PHYS, VERB, DIST, ACOU, EMOT) and linearly-weighted kappa for intensity
    ratings, then reports pass/fail against the spec.md §6.2 thresholds.

    Example layout::

        annotations/
          clip_001/
            annotator_a.jsonl
            annotator_b.jsonl
          clip_002/
            annotator_a.jsonl
            annotator_b.jsonl
    """
    import jsonlines

    from synthbanshee.labels.iaa import run_iaa
    from synthbanshee.labels.schema import EventLabel

    def _read_jsonl(path: Path) -> list[EventLabel]:
        with jsonlines.open(path) as reader:
            return [EventLabel.model_validate(record) for record in reader]

    pairs: list[tuple[list[EventLabel], list[EventLabel]]] = []
    clip_ids: list[str] = []

    for clip_dir in sorted(annotations_dir.iterdir()):
        if not clip_dir.is_dir():
            continue
        jsonl_files = sorted(clip_dir.glob("*.jsonl"))
        if len(jsonl_files) != 2:
            console.print(
                f"[yellow]Skipping {clip_dir.name}: expected 2 JSONL files,"
                f" found {len(jsonl_files)}[/yellow]"
            )
            continue
        try:
            events_a = _read_jsonl(jsonl_files[0])
            events_b = _read_jsonl(jsonl_files[1])
        except Exception as exc:
            console.print(f"[yellow]Skipping {clip_dir.name}: parse error — {exc}[/yellow]")
            continue
        pairs.append((events_a, events_b))
        clip_ids.append(clip_dir.name)

    if not pairs:
        console.print("[red]No valid annotation pairs found.[/red]")
        sys.exit(1)

    console.print(f"[cyan]Computing IAA for {len(pairs)} clip pairs...[/cyan]")
    report = run_iaa(pairs, clip_ids, total_clips=total_clips)

    console.print(report.summary())

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        report_dict = {
            "n_clips_reviewed": report.n_clips_reviewed,
            "total_clips": report.total_clips,
            "coverage_fraction": report.coverage_fraction,
            "meets_coverage": report.meets_coverage,
            "passes": report.passes,
            "disagreement_clip_ids": report.disagreement_clip_ids,
            "category_results": [
                {
                    "category": r.category,
                    "kappa": r.kappa,
                    "n_observations": r.n_observations,
                    "target_kappa": r.target_kappa,
                    "min_kappa": r.min_kappa,
                    "meets_target": r.meets_target,
                    "meets_minimum": r.meets_minimum,
                }
                for r in report.category_results
            ],
        }
        output.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
        console.print(f"[bold green]IAA report written:[/bold green] {output}")

    if not report.passes:
        sys.exit(1)


# ---------------------------------------------------------------------------
# dataset-card
# ---------------------------------------------------------------------------


@cli.command("dataset-card")
@click.argument("data_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--version", "-v", required=True, help="Dataset version string, e.g. 'v1.0'.")
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the dataset card to this file (default: print to stdout).",
)
def dataset_card(data_dir: Path, version: str, output: Path | None) -> None:
    """Generate a HuggingFace-format dataset card from DATA_DIR.

    Runs the QA suite on DATA_DIR to gather dataset statistics, then renders
    a Markdown dataset card with YAML frontmatter suitable for uploading to
    the HuggingFace Hub.
    """
    import re

    from synthbanshee.package.dataset_card import generate_dataset_card
    from synthbanshee.package.qa import run_qa

    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", version):
        raise click.BadParameter(
            "version must contain only alphanumerics, dots, hyphens, and underscores (e.g. 'v1.0')",
            param_hint="'--version'",
        )

    console.print(f"[cyan]Running QA on {data_dir} …[/cyan]")
    qa_report = run_qa(data_dir)

    card_text = generate_dataset_card(qa_report, version)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(card_text, encoding="utf-8")
        console.print(f"[bold green]Dataset card written:[/bold green] {output}")
    else:
        console.print(card_text, markup=False)


# ---------------------------------------------------------------------------
# package-dataset
# ---------------------------------------------------------------------------


@cli.command("package-dataset")
@click.argument("data_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--version", "-v", required=True, help="Dataset version string, e.g. 'v1.0'.")
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Create the archive even if the QA suite reports failures.",
)
def package_dataset(data_dir: Path, output_dir: Path, version: str, *, force: bool) -> None:
    """Package DATA_DIR into a versioned archive in OUTPUT_DIR.

    Steps:
      1. Run QA suite on DATA_DIR.
      2. Abort if QA fails (unless --force).
      3. Generate a HuggingFace dataset card.
      4. Create OUTPUT_DIR/avdp_synth_{version}.tar.gz (dirty files excluded).
      5. Write SHA256SUMS.txt and the archive .sha256 sidecar alongside it.
      6. Write DATASET_CARD.md into OUTPUT_DIR.
    """
    import re

    from synthbanshee.package.archiver import create_archive
    from synthbanshee.package.dataset_card import generate_dataset_card
    from synthbanshee.package.qa import run_qa

    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", version):
        raise click.BadParameter(
            "version must contain only alphanumerics, dots, hyphens, and underscores (e.g. 'v1.0')",
            param_hint="'--version'",
        )

    console.print(f"[cyan]Running QA on {data_dir} …[/cyan]")
    qa_report = run_qa(data_dir)

    status = "[bold green]PASS[/bold green]" if qa_report.passed else "[bold red]FAIL[/bold red]"
    total_attempted = qa_report.stats.total_clips + qa_report.stats.failed_clips
    console.print(
        f"QA: {status} — {total_attempted:,} clips attempted,"
        f" {qa_report.stats.total_clips:,} passed,"
        f" {qa_report.stats.failed_clips} failed"
        f" ({qa_report.failure_rate:.1%} failure rate)"
    )

    if not qa_report.passed and not force:
        console.print("[red]Aborting: QA failed. Use --force to package anyway.[/red]")
        sys.exit(1)

    card_text = generate_dataset_card(qa_report, version)

    archive_name = f"avdp_synth_{version}.tar.gz"
    archive_path = output_dir / archive_name

    console.print(f"[cyan]Creating archive {archive_path} …[/cyan]")
    result = create_archive(data_dir, archive_path, dataset_card_text=card_text)

    card_out = output_dir / "DATASET_CARD.md"
    card_out.write_text(card_text, encoding="utf-8")

    size_mb = result.total_bytes / (1024 * 1024)
    console.print(
        f"[bold green]Archive written:[/bold green] {result.archive_path}\n"
        f"  Files: {result.file_count:,}  Uncompressed: {size_mb:.1f} MB\n"
        f"  SHA-256: {result.checksum}\n"
        f"  Manifest: {result.manifest_path}\n"
        f"  Card: {card_out}"
    )
