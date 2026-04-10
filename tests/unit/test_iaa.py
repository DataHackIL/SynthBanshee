"""Unit tests for synthbanshee.labels.iaa."""

from __future__ import annotations

import pytest

from synthbanshee.labels.iaa import (
    MIN_COVERAGE_FRACTION,
    CategoryKappa,
    IAAReport,
    _category_kappa,
    _has_category,
    _intensity_kappa,
    cohen_kappa,
    linear_weighted_kappa,
    run_iaa,
)
from synthbanshee.labels.schema import EventLabel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    tier1: str = "PHYS",
    tier2: str = "PHYS_HARD",
    intensity: int = 3,
    clip_id: str = "clip_001",
    event_id: str = "ev_001",
) -> EventLabel:
    return EventLabel(
        event_id=event_id,
        clip_id=clip_id,
        onset=0.5,
        offset=2.0,
        tier1_category=tier1,
        tier2_subtype=tier2,
        intensity=intensity,
    )


# ---------------------------------------------------------------------------
# cohen_kappa
# ---------------------------------------------------------------------------


class TestCohenKappa:
    def test_perfect_agreement_binary(self):
        labels = [0, 1, 0, 1, 1]
        assert cohen_kappa(labels, labels) == pytest.approx(1.0)

    def test_perfect_agreement_multiclass(self):
        labels = [1, 2, 3, 2, 1]
        assert cohen_kappa(labels, labels) == pytest.approx(1.0)

    def test_complete_disagreement_binary(self):
        a = [0, 0, 1, 1]
        b = [1, 1, 0, 0]
        # All disagreements — kappa should be negative
        assert cohen_kappa(a, b) < 0.0

    def test_all_same_label_returns_zero(self):
        # When all labels are 0, expected agreement = 1, kappa is degenerate.
        assert cohen_kappa([0, 0, 0], [0, 0, 0]) == pytest.approx(0.0)

    def test_empty_returns_zero(self):
        assert cohen_kappa([], []) == pytest.approx(0.0)

    def test_known_value(self):
        # Manually verified: 2×2 confusion matrix
        # A\B  0  1       observed = (4+2)/10 = 0.6
        #   0  4  1       row0_sum=5, row1_sum=5, col0_sum=7, col1_sum=3
        #   1  3  2       expected = (5*7 + 5*3)/100 = 0.5
        #                 kappa = (0.6-0.5)/(1-0.5) = 0.2
        a = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        b = [0, 0, 0, 0, 1, 0, 0, 0, 1, 1]
        assert cohen_kappa(a, b) == pytest.approx(0.2, abs=1e-6)

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            cohen_kappa([0, 1], [0])


# ---------------------------------------------------------------------------
# linear_weighted_kappa
# ---------------------------------------------------------------------------


class TestLinearWeightedKappa:
    def test_perfect_agreement(self):
        labels = [1, 2, 3, 4, 5]
        assert linear_weighted_kappa(labels, labels) == pytest.approx(1.0)

    def test_empty_returns_zero(self):
        assert linear_weighted_kappa([], []) == pytest.approx(0.0)

    def test_scale_zero_returns_one(self):
        # min_val == max_val → degenerate, treated as perfect
        assert linear_weighted_kappa([3], [3], min_val=3, max_val=3) == pytest.approx(1.0)

    def test_partial_agreement_higher_than_chance(self):
        # Close labels (±1) should produce higher kappa than random assignment
        a = [1, 2, 3, 4, 5, 1, 2, 3]
        b = [1, 2, 3, 4, 5, 2, 3, 4]  # off-by-one on last three
        kappa = linear_weighted_kappa(a, b)
        assert kappa > 0.0

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            linear_weighted_kappa([1, 2], [1])


# ---------------------------------------------------------------------------
# _has_category / _category_kappa
# ---------------------------------------------------------------------------


class TestHasCategory:
    def test_present(self):
        events = [_make_event("PHYS", "PHYS_HARD")]
        assert _has_category(events, "PHYS") == 1

    def test_absent(self):
        events = [_make_event("VERB", "VERB_SHOUT")]
        assert _has_category(events, "PHYS") == 0

    def test_empty_list(self):
        assert _has_category([], "PHYS") == 0

    def test_different_prefix(self):
        events = [_make_event("ACOU", "ACOU_SLAM")]
        assert _has_category(events, "ACOU") == 1
        assert _has_category(events, "PHYS") == 0


class TestCategoryKappa:
    def test_perfect_agreement(self):
        events_phys = [_make_event("PHYS", "PHYS_HARD")]
        events_verb = [_make_event("VERB", "VERB_SHOUT")]
        pairs = [
            (events_phys, events_phys),
            (events_verb, events_verb),
        ]
        kappa = _category_kappa(pairs, "PHYS")
        assert kappa == pytest.approx(1.0)

    def test_no_observations_for_category(self):
        # Neither annotator labeled DIST — degenerate (all zeros)
        events_verb = [_make_event("VERB", "VERB_SHOUT")]
        pairs = [(events_verb, events_verb)]
        kappa = _category_kappa(pairs, "DIST")
        assert kappa == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _intensity_kappa
# ---------------------------------------------------------------------------


class TestIntensityKappa:
    def test_perfect_intensity_agreement(self):
        ev_a = _make_event(intensity=3)
        ev_b = _make_event(intensity=3)
        pairs = [([ev_a], [ev_b])]
        kappa = _intensity_kappa(pairs)
        assert kappa == pytest.approx(1.0)

    def test_no_events_returns_zero(self):
        pairs: list[tuple] = [([], [])]
        kappa = _intensity_kappa(pairs)
        assert kappa == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CategoryKappa properties
# ---------------------------------------------------------------------------


class TestCategoryKappaProperties:
    def test_meets_target_true(self):
        ck = CategoryKappa("PHYS", kappa=0.70, n_observations=50, target_kappa=0.65, min_kappa=0.55)
        assert ck.meets_target is True
        assert ck.meets_minimum is True

    def test_meets_minimum_only(self):
        ck = CategoryKappa("PHYS", kappa=0.58, n_observations=50, target_kappa=0.65, min_kappa=0.55)
        assert ck.meets_target is False
        assert ck.meets_minimum is True

    def test_fails_minimum(self):
        ck = CategoryKappa("PHYS", kappa=0.40, n_observations=50, target_kappa=0.65, min_kappa=0.55)
        assert ck.meets_target is False
        assert ck.meets_minimum is False


# ---------------------------------------------------------------------------
# IAAReport properties
# ---------------------------------------------------------------------------


class TestIAAReport:
    def _passing_report(self) -> IAAReport:
        results = [
            CategoryKappa("PHYS", 0.70, 100, 0.65, 0.55),
            CategoryKappa("VERB", 0.65, 100, 0.60, 0.50),
            CategoryKappa("DIST", 0.62, 100, 0.60, 0.50),
            CategoryKappa("ACOU", 0.75, 100, 0.70, 0.60),
            CategoryKappa("EMOT", 0.57, 100, 0.55, 0.45),
            CategoryKappa("INTENS", 0.63, 200, 0.60, 0.50),
        ]
        return IAAReport(category_results=results, n_clips_reviewed=120, total_clips=500)

    def test_coverage_fraction(self):
        report = self._passing_report()
        assert report.coverage_fraction == pytest.approx(0.24)

    def test_meets_coverage(self):
        report = self._passing_report()
        assert report.meets_coverage is True

    def test_fails_coverage(self):
        results = [CategoryKappa("PHYS", 0.70, 10, 0.65, 0.55)]
        report = IAAReport(category_results=results, n_clips_reviewed=5, total_clips=500)
        assert report.meets_coverage is False

    def test_all_meet_minimum_true(self):
        report = self._passing_report()
        assert report.all_meet_minimum is True

    def test_all_meet_minimum_false(self):
        results = [CategoryKappa("PHYS", 0.40, 100, 0.65, 0.55)]
        report = IAAReport(category_results=results, n_clips_reviewed=120, total_clips=500)
        assert report.all_meet_minimum is False

    def test_passes(self):
        report = self._passing_report()
        assert report.passes is True

    def test_fails_due_to_kappa(self):
        results = [CategoryKappa("PHYS", 0.40, 100, 0.65, 0.55)]
        report = IAAReport(category_results=results, n_clips_reviewed=120, total_clips=500)
        assert report.passes is False

    def test_fails_due_to_coverage(self):
        results = [CategoryKappa("PHYS", 0.70, 10, 0.65, 0.55)]
        report = IAAReport(category_results=results, n_clips_reviewed=5, total_clips=500)
        assert report.passes is False

    def test_coverage_fraction_zero_total(self):
        results = [CategoryKappa("PHYS", 0.70, 0, 0.65, 0.55)]
        report = IAAReport(category_results=results, n_clips_reviewed=0, total_clips=0)
        assert report.coverage_fraction == pytest.approx(0.0)

    def test_summary_contains_pass(self):
        report = self._passing_report()
        summary = report.summary()
        assert "PASS" in summary

    def test_summary_contains_fail(self):
        results = [CategoryKappa("PHYS", 0.40, 100, 0.65, 0.55)]
        report = IAAReport(category_results=results, n_clips_reviewed=120, total_clips=500)
        assert "FAIL" in report.summary()

    def test_summary_lists_disagreement_clips(self):
        results = [CategoryKappa("PHYS", 0.40, 10, 0.65, 0.55)]
        report = IAAReport(
            category_results=results,
            n_clips_reviewed=5,
            total_clips=20,
            disagreement_clip_ids=["clip_001", "clip_002"],
        )
        summary = report.summary()
        assert "clip_001" in summary
        assert "clip_002" in summary


# ---------------------------------------------------------------------------
# run_iaa
# ---------------------------------------------------------------------------


class TestRunIAA:
    def test_perfect_agreement_passes(self):
        events = [_make_event("PHYS", "PHYS_HARD", intensity=3)]
        pairs = [(events, events)] * 30
        clip_ids = [f"clip_{i:03d}" for i in range(30)]
        report = run_iaa(pairs, clip_ids, total_clips=100)
        assert report.n_clips_reviewed == 30
        assert report.total_clips == 100
        # All categories agree perfectly → kappa=1 or degenerate=0
        for r in report.category_results:
            assert r.kappa >= 0.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            run_iaa([], ["clip_001"], total_clips=10)

    def test_disagreement_clips_identified(self):
        """Clips where annotators disagree on PHYS presence are flagged."""
        phys_events = [_make_event("PHYS", "PHYS_HARD")]
        no_events: list[EventLabel] = []
        # clip_000: annotator A sees PHYS, annotator B does not → disagreement
        pairs = [(phys_events, no_events)]
        clip_ids = ["clip_000"]
        report = run_iaa(pairs, clip_ids, total_clips=10)
        assert "clip_000" in report.disagreement_clip_ids

    def test_agreeing_clips_not_flagged(self):
        events = [_make_event("PHYS", "PHYS_HARD")]
        pairs = [(events, events)]
        clip_ids = ["clip_000"]
        report = run_iaa(pairs, clip_ids, total_clips=10)
        assert "clip_000" not in report.disagreement_clip_ids

    def test_coverage_fraction_computed(self):
        events = [_make_event()]
        pairs = [(events, events)] * 25
        clip_ids = [f"c{i}" for i in range(25)]
        report = run_iaa(pairs, clip_ids, total_clips=100)
        assert report.coverage_fraction == pytest.approx(0.25)

    def test_six_category_results(self):
        events = [_make_event()]
        pairs = [(events, events)]
        clip_ids = ["c0"]
        report = run_iaa(pairs, clip_ids, total_clips=5)
        assert len(report.category_results) == 6  # PHYS VERB DIST ACOU EMOT INTENS

    def test_intensity_kappa_in_results(self):
        events = [_make_event(intensity=3)]
        pairs = [(events, events)]
        clip_ids = ["c0"]
        report = run_iaa(pairs, clip_ids, total_clips=5)
        intens_result = next(r for r in report.category_results if r.category == "INTENS")
        assert intens_result.kappa == pytest.approx(1.0)
        assert intens_result.n_observations == 1


# ---------------------------------------------------------------------------
# MIN_COVERAGE_FRACTION constant
# ---------------------------------------------------------------------------


def test_min_coverage_fraction_matches_spec():
    """spec.md §6.1 requires 20% minimum coverage."""
    assert pytest.approx(0.20) == MIN_COVERAGE_FRACTION
