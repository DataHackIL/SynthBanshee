"""Clip-level validator: a clip is valid iff validate_clip(clip_path) returns True.

Validates:
  1. All three required files exist (.wav, .txt, .json)
  2. WAV passes validate_audio() checks
  3. JSON parses as ClipMetadata, is_synthetic == True
  4. Filename is ASCII-only, lowercase, no spaces

Spec reference: docs/spec.md §2–§5
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from synthbanshee.augment.preprocessing import validate_audio
from synthbanshee.labels.schema import ClipMetadata

_ASCII_FILENAME_RE = re.compile(r"^[a-z0-9_\-]+$")


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.is_valid


def validate_clip(clip_path: Path | str) -> ValidationResult:
    """Validate a generated clip against the AVDP spec.

    Args:
        clip_path: Path to the .wav file. Companion .txt and .json must exist
                   in the same directory with the same stem.

    Returns:
        ValidationResult with is_valid=True only if ALL checks pass.
    """
    clip_path = Path(clip_path)
    errors: list[str] = []
    warnings: list[str] = []

    stem = clip_path.stem
    parent = clip_path.parent
    txt_path = parent / f"{stem}.txt"
    json_path = parent / f"{stem}.json"
    jsonl_path = parent / f"{stem}.jsonl"

    # ------------------------------------------------------------------
    # 1. Filename constraints (spec §2.5)
    # ------------------------------------------------------------------
    if not _ASCII_FILENAME_RE.match(stem):
        errors.append(
            f"Filename stem {stem!r} contains non-ASCII or disallowed characters. "
            "Only lowercase a-z, 0-9, underscore, and hyphen are allowed."
        )

    # ------------------------------------------------------------------
    # 2. Required files present
    # ------------------------------------------------------------------
    for path, label in [
        (clip_path, "WAV"),
        (txt_path, "transcript (.txt)"),
        (json_path, "metadata (.json)"),
    ]:
        if not path.exists():
            errors.append(f"Missing {label} file: {path}")

    if errors:
        # Cannot proceed with content checks if files are missing
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    # ------------------------------------------------------------------
    # 3. Audio checks (spec §3)
    # ------------------------------------------------------------------
    audio_ok, audio_errors = validate_audio(clip_path)
    errors.extend(audio_errors)

    # ------------------------------------------------------------------
    # 4. Metadata JSON (spec §5.1)
    # ------------------------------------------------------------------
    try:
        raw_json = json_path.read_text(encoding="utf-8")
        metadata = ClipMetadata.model_validate_json(raw_json)
    except (json.JSONDecodeError, ValidationError) as exc:
        errors.append(f"Metadata JSON invalid: {exc}")
        metadata = None

    if metadata is not None:
        if not metadata.is_synthetic:
            errors.append("is_synthetic must be True for all generated clips")

        if metadata.clip_id != stem:
            warnings.append(
                f"clip_id {metadata.clip_id!r} in JSON does not match filename stem {stem!r}"
            )

    # ------------------------------------------------------------------
    # 5. Transcript readable (just existence + UTF-8 readability)
    # ------------------------------------------------------------------
    try:
        txt_path.read_text(encoding="utf-8")
    except Exception as exc:
        errors.append(f"Cannot read transcript: {exc}")

    # ------------------------------------------------------------------
    # 6. Strong labels JSONL (warning only — not a hard spec requirement)
    # ------------------------------------------------------------------
    if not jsonl_path.exists():
        warnings.append(f"Strong labels JSONL missing: {jsonl_path}")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )
