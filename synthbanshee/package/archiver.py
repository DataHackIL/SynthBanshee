"""Versioned dataset archive creation for AVDP (Phase 3.4).

Creates a .tar.gz of a data directory with:
  - a ``SHA256SUMS.txt`` manifest of all included files (alongside the archive)
  - a ``.sha256`` sidecar file containing the archive's own checksum

"Dirty" pre-processing originals (stems containing ``_dirty``) are excluded
from the archive automatically.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ArchiveResult:
    """Result returned by :func:`create_archive`."""

    archive_path: Path
    """Path to the created ``.tar.gz`` file."""

    checksum: str
    """SHA-256 hex digest of the archive file itself (also written to ``.sha256``)."""

    file_count: int
    """Number of data files included (dirty originals excluded)."""

    total_bytes: int
    """Sum of uncompressed file sizes for all included files."""

    manifest_path: Path
    """Path to ``SHA256SUMS.txt`` written alongside the archive."""


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of *path*."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def create_archive(
    data_dir: Path,
    output_path: Path,
    *,
    dataset_card_text: str | None = None,
) -> ArchiveResult:
    """Create a ``.tar.gz`` archive of *data_dir*.

    Args:
        data_dir: Directory to archive (e.g. ``data/``).  Scanned recursively;
            files whose stems contain ``_dirty`` are excluded.
        output_path: Destination path for the archive (e.g.
            ``releases/avdp_synth_v1.0.tar.gz``).  The parent directory is
            created if it does not exist.
        dataset_card_text: Optional text to include as ``DATASET_CARD.md`` at
            the archive root.

    Returns:
        :class:`ArchiveResult` with the archive path, SHA-256 checksum, file
        count, total uncompressed bytes, and path to ``SHA256SUMS.txt``.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p
        for p in data_dir.rglob("*")
        if p.is_file() and not p.is_symlink() and "_dirty" not in p.stem
    )

    file_checksums: list[tuple[str, str]] = []
    total_bytes = 0

    with tarfile.open(output_path, "w:gz") as tar:
        for file_path in files:
            rel_path = file_path.relative_to(data_dir.parent)
            rel = rel_path.as_posix()
            file_checksums.append((rel, _file_sha256(file_path)))
            total_bytes += file_path.stat().st_size
            tar.add(file_path, arcname=rel)

        if dataset_card_text is not None:
            card_bytes = dataset_card_text.encode("utf-8")
            info = tarfile.TarInfo(name="DATASET_CARD.md")
            info.size = len(card_bytes)
            tar.addfile(info, io.BytesIO(card_bytes))
            card_digest = hashlib.sha256(card_bytes).hexdigest()
            file_checksums.append(("DATASET_CARD.md", card_digest))

    # Per-release manifest alongside the archive (versioned to avoid collisions)
    archive_stem = output_path.with_suffix("").with_suffix("").name
    manifest_path = output_path.with_name(f"{archive_stem}_SHA256SUMS.txt")
    manifest_path.write_text(
        "".join(f"{digest}  {rel}\n" for rel, digest in file_checksums),
        encoding="utf-8",
    )

    # .sha256 sidecar for the archive itself
    archive_checksum = _file_sha256(output_path)
    output_path.with_name(output_path.name + ".sha256").write_text(
        f"{archive_checksum}  {output_path.name}\n",
        encoding="utf-8",
    )

    return ArchiveResult(
        archive_path=output_path,
        checksum=archive_checksum,
        file_count=len(files),
        total_bytes=total_bytes,
        manifest_path=manifest_path,
    )
