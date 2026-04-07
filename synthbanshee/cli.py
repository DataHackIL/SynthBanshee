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


def _run_generate_pipeline(
    config: Path,
    output_dir: Path,
    cache_dir: Path,
    dirty_dir: Path,
) -> tuple[Path | None, list[str]]:
    """Run the single-clip generate pipeline.

    Returns (wav_path, errors). wav_path is None on failure.
    This is the Phase 0/1 stub pipeline; will be replaced with the full
    LLM + multi-speaker pipeline in a later iteration.
    """
    from synthbanshee.augment.preprocessing import preprocess
    from synthbanshee.config.scene_config import SceneConfig
    from synthbanshee.config.speaker_config import SpeakerConfig
    from synthbanshee.labels.generator import LabelGenerator, ScriptEvent
    from synthbanshee.labels.schema import PreprocessingApplied, SpeakerInfo
    from synthbanshee.package.validator import validate_clip
    from synthbanshee.tts.renderer import TTSRenderer

    try:
        scene = SceneConfig.from_yaml(config)
    except Exception as exc:
        return None, [f"Config parse error: {exc}"]

    renderer = TTSRenderer(cache_dir=cache_dir)
    label_gen = LabelGenerator()

    first_speaker_ref = scene.speakers[0]
    speaker_yaml_path = Path("configs/speakers") / f"{first_speaker_ref.speaker_id}.yaml"
    if not speaker_yaml_path.exists():
        speaker_yaml_path = (
            Path("configs/examples") / f"speaker_{first_speaker_ref.speaker_id}.yaml"
        )
    if not speaker_yaml_path.exists():
        return None, [f"Speaker config not found: {first_speaker_ref.speaker_id}"]

    try:
        speaker = SpeakerConfig.from_yaml(speaker_yaml_path)
    except Exception as exc:
        return None, [f"Speaker config parse error: {exc}"]

    stub_template = Path("synthbanshee/script/templates/she_proves/stub_utterance.txt")
    if stub_template.exists():
        utterance_text = stub_template.read_text(encoding="utf-8").strip()
    else:
        utterance_text = "shalom"

    try:
        with tempfile.TemporaryDirectory() as tmp:
            raw_wav = Path(tmp) / "raw.wav"
            renderer.render_utterance_to_file(
                utterance_text, speaker, raw_wav, intensity=scene.intensity_arc[0]
            )

            clip_id = scene.scene_id.lower().replace("-", "_")
            speaker_dir = output_dir / first_speaker_ref.speaker_id.lower()
            clip_wav = speaker_dir / f"{clip_id}_00.wav"
            clip_txt = speaker_dir / f"{clip_id}_00.txt"
            clip_json = speaker_dir / f"{clip_id}_00.json"

            result = preprocess(raw_wav, clip_wav, dirty_dir=dirty_dir)

            clip_txt.parent.mkdir(parents=True, exist_ok=True)
            clip_txt.write_text(
                f"[CLIP_ID: {clip_id}_00]\n"
                f"[SPEAKER: {first_speaker_ref.speaker_id} | ROLE: {first_speaker_ref.role} "
                f"| ONSET: 0.5 | OFFSET: {result.duration_seconds - 0.5:.1f}]\n"
                + utterance_text
                + "\n",
                encoding="utf-8",
            )

            stub_events = [
                ScriptEvent(
                    tier1_category="NONE",
                    tier2_subtype="NONE_AMBIENT",
                    onset=0.5,
                    offset=max(result.duration_seconds - 0.5, 1.0),
                    intensity=scene.intensity_arc[0],
                    speaker_id=first_speaker_ref.speaker_id,
                    speaker_role=first_speaker_ref.role,
                )
            ]
            event_labels = label_gen.generate_event_labels(f"{clip_id}_00", stub_events)

            speakers_meta = [
                SpeakerInfo(
                    speaker_id=speaker.speaker_id,
                    role=speaker.role,
                    gender=speaker.gender,
                    age_range=speaker.age_range,
                    tts_voice_id=speaker.tts_voice_id,
                )
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
            )
            label_gen.write_clip_metadata_json(metadata, clip_json)

    except Exception as exc:
        return None, [f"Pipeline error: {exc}"]

    validation = validate_clip(clip_wav)
    if not validation.is_valid:
        return None, validation.errors

    return clip_wav, list(validation.warnings)


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
    "--dry-run",
    is_flag=True,
    default=False,
    help="Parse and validate config only; do not render.",
)
def generate(
    config: Path,
    output_dir: Path | None,
    cache_dir: Path,
    dirty_dir: Path,
    dry_run: bool,
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
    wav_path, messages = _run_generate_pipeline(config, out_dir, cache_dir, dirty_dir)

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
def generate_batch(
    run_config: Path,
    output_dir: Path | None,
    cache_dir: Path,
    dirty_dir: Path,
    manifest_out: Path | None,
    dry_run: bool,
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
        task = progress.add_task("Rendering clips", total=len(selected))
        for scene_yaml in selected:
            wav_path, messages = _run_generate_pipeline(scene_yaml, out_dir, cache_dir, dirty_dir)
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
            progress.advance(task)

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
    rows = generate_manifest(out_dir, manifest_path, splits=splits)
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
