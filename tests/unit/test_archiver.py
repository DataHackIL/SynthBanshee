"""Unit tests for synthbanshee.package.archiver."""

from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

from synthbanshee.package.archiver import ArchiveResult, create_archive


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_data_dir(base: Path) -> Path:
    """Create a small data/ tree with clean and dirty files."""
    data = base / "data"
    data.mkdir()
    (data / "clip_001.wav").write_bytes(b"wav1")
    (data / "clip_001.txt").write_text("transcript", encoding="utf-8")
    (data / "clip_001.json").write_text("{}", encoding="utf-8")
    (data / "clip_001_dirty.wav").write_bytes(b"dirty")  # must be excluded
    subdir = data / "sub"
    subdir.mkdir()
    (subdir / "clip_002.wav").write_bytes(b"wav2")
    return data


class TestCreateArchive:
    def test_archive_created(self, tmp_path):
        data_dir = _make_data_dir(tmp_path)
        out = tmp_path / "releases" / "dataset.tar.gz"
        result = create_archive(data_dir, out)
        assert out.exists()
        assert isinstance(result, ArchiveResult)
        assert result.archive_path == out

    def test_dirty_files_excluded(self, tmp_path):
        data_dir = _make_data_dir(tmp_path)
        out = tmp_path / "dataset.tar.gz"
        create_archive(data_dir, out)
        with tarfile.open(out, "r:gz") as tar:
            names = tar.getnames()
        assert not any("_dirty" in n for n in names)

    def test_clean_files_included(self, tmp_path):
        data_dir = _make_data_dir(tmp_path)
        out = tmp_path / "dataset.tar.gz"
        result = create_archive(data_dir, out)
        # 4 clean files: clip_001.wav, clip_001.txt, clip_001.json, sub/clip_002.wav
        assert result.file_count == 4

    def test_total_bytes(self, tmp_path):
        data_dir = _make_data_dir(tmp_path)
        out = tmp_path / "dataset.tar.gz"
        result = create_archive(data_dir, out)
        # wav1 (4) + "transcript" (10) + "{}" (2) + wav2 (4) = 20
        assert result.total_bytes == 20

    def test_sha256_sidecar_written(self, tmp_path):
        data_dir = _make_data_dir(tmp_path)
        out = tmp_path / "dataset.tar.gz"
        result = create_archive(data_dir, out)
        sidecar = out.with_name(out.name + ".sha256")
        assert sidecar.exists()
        content = sidecar.read_text()
        assert result.checksum in content
        assert out.name in content

    def test_checksum_matches_archive(self, tmp_path):
        data_dir = _make_data_dir(tmp_path)
        out = tmp_path / "dataset.tar.gz"
        result = create_archive(data_dir, out)
        assert result.checksum == _sha256(out)

    def test_manifest_written(self, tmp_path):
        data_dir = _make_data_dir(tmp_path)
        out = tmp_path / "dataset.tar.gz"
        result = create_archive(data_dir, out)
        assert result.manifest_path.exists()
        lines = result.manifest_path.read_text().splitlines()
        assert len(lines) == 4  # one per clean file; no card
        # Each line: "<hex>  <path>"
        for line in lines:
            digest, _, rel = line.partition("  ")
            assert len(digest) == 64  # SHA-256 hex

    def test_dataset_card_included_in_archive(self, tmp_path):
        data_dir = _make_data_dir(tmp_path)
        out = tmp_path / "dataset.tar.gz"
        card = "# My Dataset Card"
        create_archive(data_dir, out, dataset_card_text=card)
        with tarfile.open(out, "r:gz") as tar:
            assert "DATASET_CARD.md" in tar.getnames()
            member = tar.extractfile("DATASET_CARD.md")
            assert member is not None
            assert member.read().decode() == card

    def test_dataset_card_in_manifest(self, tmp_path):
        """DATASET_CARD.md must appear in SHA256SUMS.txt when card is provided."""
        data_dir = _make_data_dir(tmp_path)
        out = tmp_path / "dataset.tar.gz"
        create_archive(data_dir, out, dataset_card_text="# Card")
        lines = (out.with_name("SHA256SUMS.txt")).read_text().splitlines()
        assert any("DATASET_CARD.md" in line for line in lines)
        # 4 data files + 1 card entry = 5
        assert len(lines) == 5

    def test_no_dataset_card_when_none(self, tmp_path):
        data_dir = _make_data_dir(tmp_path)
        out = tmp_path / "dataset.tar.gz"
        create_archive(data_dir, out, dataset_card_text=None)
        with tarfile.open(out, "r:gz") as tar:
            assert "DATASET_CARD.md" not in tar.getnames()

    def test_output_parent_created(self, tmp_path):
        data_dir = _make_data_dir(tmp_path)
        out = tmp_path / "deep" / "nested" / "dataset.tar.gz"
        create_archive(data_dir, out)
        assert out.exists()

    def test_empty_data_dir(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        out = tmp_path / "dataset.tar.gz"
        result = create_archive(data_dir, out)
        assert result.file_count == 0
        assert result.total_bytes == 0
        assert out.exists()

    def test_symlinks_excluded(self, tmp_path):
        """Symlinks inside data_dir must not be included in the archive."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        real_file = data_dir / "real.wav"
        real_file.write_bytes(b"real")
        link = data_dir / "link.wav"
        link.symlink_to(real_file)
        out = tmp_path / "dataset.tar.gz"
        result = create_archive(data_dir, out)
        assert result.file_count == 1  # only real.wav
        with tarfile.open(out, "r:gz") as tar:
            names = tar.getnames()
        assert not any("link" in n for n in names)
