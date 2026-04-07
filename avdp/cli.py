"""AVDP CLI entry points.

Usage:
    avdp generate --config configs/scenes/test_scene_001.yaml
    avdp validate --clip data/he/agg_m_30-45_001/sp_it_a_0001_00.wav
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

console = Console()


@click.group()
@click.version_option()
def cli() -> None:
    """AVDP Synthetic Dataset Generation Framework."""


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
    from avdp.config.scene_config import SceneConfig

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

    # --- TTS rendering (single-speaker stub for Phase 0) ---
    import tempfile

    from avdp.augment.preprocessing import preprocess
    from avdp.config.speaker_config import SpeakerConfig
    from avdp.labels.generator import LabelGenerator, ScriptEvent
    from avdp.labels.schema import PreprocessingApplied, SpeakerInfo
    from avdp.package.validator import validate_clip
    from avdp.tts.renderer import TTSRenderer

    renderer = TTSRenderer(cache_dir=cache_dir)
    label_gen = LabelGenerator()

    # For Phase 0: generate a minimal stub clip with a placeholder utterance.
    # In Phase 1 the LLM script generator will fill this in.
    first_speaker_ref = scene.speakers[0]
    speaker_yaml_path = (
        Path("configs/speakers") / f"{first_speaker_ref.speaker_id}.yaml"
    )
    if not speaker_yaml_path.exists():
        # Fall back to examples directory
        speaker_yaml_path = (
            Path("configs/examples") / f"speaker_{first_speaker_ref.speaker_id}.yaml"
        )

    if not speaker_yaml_path.exists():
        console.print(
            f"[red]Speaker config not found: {first_speaker_ref.speaker_id}[/red]"
        )
        sys.exit(1)

    speaker = SpeakerConfig.from_yaml(speaker_yaml_path)

    # Placeholder Hebrew utterance text — written in a template file,
    # not hardcoded in Python source. For Phase 0 stub we use a minimal literal.
    # In Phase 1 this comes from the Jinja2 template + LLM generation.
    stub_template = Path("avdp/script/templates/she_proves/stub_utterance.txt")
    if stub_template.exists():
        utterance_text = stub_template.read_text(encoding="utf-8").strip()
    else:
        # Minimal fallback — should not reach here in normal operation
        utterance_text = "shalom"  # ASCII-safe placeholder

    console.print("[cyan]Rendering TTS utterance...[/cyan]")
    with tempfile.TemporaryDirectory() as tmp:
        raw_wav = Path(tmp) / "raw.wav"
        renderer.render_utterance_to_file(
            utterance_text, speaker, raw_wav, intensity=scene.intensity_arc[0]
        )

        # --- Preprocessing ---
        clip_id = scene.scene_id.lower().replace("-", "_")
        speaker_dir = out_dir / first_speaker_ref.speaker_id.lower()
        clip_wav = speaker_dir / f"{clip_id}_00.wav"
        clip_txt = speaker_dir / f"{clip_id}_00.txt"
        clip_json = speaker_dir / f"{clip_id}_00.json"

        console.print("[cyan]Running preprocessing pipeline...[/cyan]")
        result = preprocess(raw_wav, clip_wav, dirty_dir=dirty_dir)

        # --- Transcript ---
        clip_txt.parent.mkdir(parents=True, exist_ok=True)
        clip_txt.write_text(
            f"[CLIP_ID: {clip_id}_00]\n"
            f"[SPEAKER: {first_speaker_ref.speaker_id} | ROLE: {first_speaker_ref.role} "
            f"| ONSET: 0.5 | OFFSET: {result.duration_seconds - 0.5:.1f}]\n"
            + utterance_text + "\n",
            encoding="utf-8",
        )

        # --- Labels ---
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

    # --- Validate ---
    console.print("[cyan]Validating output clip...[/cyan]")
    validation = validate_clip(clip_wav)
    if validation.is_valid:
        console.print(f"[bold green]Clip generated and valid:[/bold green] {clip_wav}")
    else:
        console.print("[bold red]Validation FAILED:[/bold red]")
        for err in validation.errors:
            console.print(f"  [red]• {err}[/red]")
        sys.exit(1)

    if validation.warnings:
        for w in validation.warnings:
            console.print(f"  [yellow]• {w}[/yellow]")


@cli.command()
@click.argument("clip", type=click.Path(exists=True, path_type=Path))
def validate(clip: Path) -> None:
    """Validate an existing clip against the AVDP spec."""
    from avdp.package.validator import validate_clip

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
