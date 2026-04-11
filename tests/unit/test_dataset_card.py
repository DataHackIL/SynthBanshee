"""Unit tests for synthbanshee.package.dataset_card."""

from __future__ import annotations

from synthbanshee.package.dataset_card import _size_category, generate_dataset_card
from synthbanshee.package.qa import DatasetStats, QAReport


def _make_report(
    *,
    total_clips: int = 100,
    duration_s: float = 36000.0,
    speaker_count: int = 20,
    failed_clips: int = 1,
    quality_flagged: int = 2,
    typology: dict[str, int] | None = None,
    tier: dict[str, int] | None = None,
    split: dict[str, int] | None = None,
) -> QAReport:
    stats = DatasetStats(
        total_clips=total_clips,
        total_duration_seconds=duration_s,
        speaker_count=speaker_count,
        failed_clips=failed_clips,
        quality_flagged_clips=quality_flagged,
        clips_by_typology=typology or {"IT": 50, "SV": 30, "NEG": 15, "NEU": 5},
        clips_by_tier=tier or {"A": 60, "B": 40},
        clips_by_split=split or {"train": 70, "val": 15, "test": 15},
    )
    return QAReport(data_dir="/data", stats=stats, passed=True)


class TestSizeCategory:
    def test_below_1k(self):
        assert _size_category(0) == "n<1K"
        assert _size_category(999) == "n<1K"

    def test_1k_to_10k(self):
        assert _size_category(1_000) == "1K<n<10K"
        assert _size_category(9_999) == "1K<n<10K"

    def test_10k_to_100k(self):
        assert _size_category(10_000) == "10K<n<100K"
        assert _size_category(99_999) == "10K<n<100K"

    def test_100k_plus(self):
        assert _size_category(100_000) == "100K<n<1M"
        assert _size_category(500_000) == "100K<n<1M"


class TestGenerateDatasetCard:
    def test_yaml_frontmatter_present(self):
        card = generate_dataset_card(_make_report(), "v1.0")
        assert card.startswith("---\n")
        assert "language:\n- he" in card
        assert "license: cc-by-4.0" in card
        assert "pretty_name: AVDP Synthetic Dataset v1.0" in card

    def test_version_in_title(self):
        card = generate_dataset_card(_make_report(), "v2.3")
        assert "v2.3" in card

    def test_total_clips_in_stats_table(self):
        card = generate_dataset_card(_make_report(total_clips=1234), "v1.0")
        assert "1,234" in card

    def test_duration_hours(self):
        card = generate_dataset_card(_make_report(duration_s=7200.0), "v1.0")
        assert "2.0 h" in card

    def test_typology_rows_present(self):
        card = generate_dataset_card(_make_report(), "v1.0")
        assert "| `IT`" in card
        assert "| `SV`" in card
        assert "| `NEG`" in card
        assert "| `NEU`" in card

    def test_tier_rows_present(self):
        card = generate_dataset_card(_make_report(), "v1.0")
        assert "Tier A" in card
        assert "Tier B" in card

    def test_split_rows_present(self):
        card = generate_dataset_card(_make_report(), "v1.0")
        assert "train" in card
        assert "val" in card
        assert "test" in card

    def test_custom_projects(self):
        card = generate_dataset_card(_make_report(), "v1.0", projects=["Foo", "Bar"])
        assert "**Foo**" in card
        assert "**Bar**" in card

    def test_default_projects(self):
        card = generate_dataset_card(_make_report(), "v1.0")
        assert "She-Proves" in card
        assert "Elephant in the Room" in card

    def test_citation_contains_version(self):
        card = generate_dataset_card(_make_report(), "v1.0")
        assert "avdp_synth_v1_0" in card

    def test_known_limitations_section(self):
        card = generate_dataset_card(_make_report(), "v1.0")
        assert "Known Limitations" in card
        assert "Synthetic-to-real gap" in card

    def test_data_format_section(self):
        card = generate_dataset_card(_make_report(), "v1.0")
        assert "Data Format" in card
        assert "manifest.csv" in card

    def test_license_section(self):
        card = generate_dataset_card(_make_report(), "v1.0")
        assert "CC BY 4.0" in card

    def test_size_category_in_frontmatter(self):
        card = generate_dataset_card(_make_report(total_clips=5000), "v1.0")
        assert "1K<n<10K" in card

    def test_empty_stats_no_crash(self):
        """generate_dataset_card must not crash when total_clips is 0."""
        report = QAReport(data_dir="/empty", stats=DatasetStats(), passed=True)
        card = generate_dataset_card(report, "v0.0")
        assert "v0.0" in card
        assert "0" in card
