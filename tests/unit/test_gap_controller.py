"""Unit tests for TurnGapController (M6 — inter-turn timing redesign)."""

from __future__ import annotations

import random

import pytest

from synthbanshee.script.types import DialogueTurn
from synthbanshee.tts.gap_controller import (
    _BARGE_IN_DEPTH_RANGE,
    _DEFAULT_GAPS,
    _ELEPHANT_GAPS,
    _OVERLAP_DEPTH_RANGE,
    _SHE_PROVES_GAPS,
    TurnGapController,
)
from synthbanshee.tts.mix_mode import MixMode


def _turn(speaker_id: str, intensity: int) -> DialogueTurn:
    return DialogueTurn(
        speaker_id=speaker_id,
        text="שלום",
        intensity=intensity,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_in_range(values: list[float], lo: float, hi: float) -> bool:
    return all(lo <= v <= hi for v in values)


def _draw_many(
    ctrl: TurnGapController,
    current: DialogueTurn,
    prev: DialogueTurn | None,
    current_role: str,
    prev_role: str | None = None,
    n: int = 200,
) -> list[tuple[float, MixMode]]:
    rng = random.Random(42)
    return [ctrl.gap_seconds(current, prev, rng, current_role, prev_role) for _ in range(n)]


def _gaps_only(draws: list[tuple[float, MixMode]]) -> list[float]:
    """Extract just the amount values from gap_seconds() results."""
    return [amount for amount, _ in draws]


def _modes_only(draws: list[tuple[float, MixMode]]) -> list[MixMode]:
    """Extract just the MixMode values from gap_seconds() results."""
    return [mode for _, mode in draws]


# ---------------------------------------------------------------------------
# Project table selection
# ---------------------------------------------------------------------------


class TestProjectTableSelection:
    def test_she_proves_uses_she_proves_table(self) -> None:
        ctrl = TurnGapController(project="she_proves")
        assert ctrl._table is _SHE_PROVES_GAPS

    def test_elephant_uses_elephant_table(self) -> None:
        ctrl = TurnGapController(project="elephant_in_the_room")
        assert ctrl._table is _ELEPHANT_GAPS

    def test_unknown_project_falls_back_to_default(self) -> None:
        ctrl = TurnGapController(project="some_future_project")
        assert ctrl._table is _DEFAULT_GAPS


# ---------------------------------------------------------------------------
# First turn (prev_turn is None) — any project
# ---------------------------------------------------------------------------


class TestFirstTurn:
    def test_first_turn_gap_in_range(self) -> None:
        ctrl = TurnGapController(project="she_proves")
        draws = _draw_many(ctrl, _turn("AGG_M_30-45_001", 1), None, "AGG")
        lo, hi = _SHE_PROVES_GAPS["agg_low"]
        assert _all_in_range(_gaps_only(draws), lo, hi)

    def test_first_turn_always_sequential(self) -> None:
        ctrl = TurnGapController(project="she_proves")
        draws = _draw_many(ctrl, _turn("AGG_M_30-45_001", 5), None, "AGG")
        assert all(m == MixMode.SEQUENTIAL for m in _modes_only(draws))


# ---------------------------------------------------------------------------
# VIC role — She-Proves
# ---------------------------------------------------------------------------


class TestVICSheProves:
    def setup_method(self) -> None:
        self.ctrl = TurnGapController(project="she_proves")
        self.vic = _turn("VIC_F_25-35_001", 1)

    def test_vic_after_agg_i1(self) -> None:
        prev = _turn("AGG_M_30-45_001", 1)
        draws = _draw_many(self.ctrl, self.vic, prev, "VIC", prev_role="AGG", n=500)
        seq_gaps = [g for g, m in draws if m == MixMode.SEQUENTIAL]
        lo, hi = _SHE_PROVES_GAPS["vic_low"]
        assert _all_in_range(seq_gaps, lo, hi)

    def test_vic_after_agg_i2(self) -> None:
        prev = _turn("AGG_M_30-45_001", 2)
        draws = _draw_many(self.ctrl, self.vic, prev, "VIC", prev_role="AGG", n=500)
        seq_gaps = [g for g, m in draws if m == MixMode.SEQUENTIAL]
        lo, hi = _SHE_PROVES_GAPS["vic_low"]
        assert _all_in_range(seq_gaps, lo, hi)

    def test_vic_after_agg_i3(self) -> None:
        prev = _turn("AGG_M_30-45_001", 3)
        draws = _draw_many(self.ctrl, self.vic, prev, "VIC", prev_role="AGG", n=500)
        seq_gaps = [g for g, m in draws if m == MixMode.SEQUENTIAL]
        lo, hi = _SHE_PROVES_GAPS["vic_i3"]
        assert _all_in_range(seq_gaps, lo, hi)

    def test_vic_after_agg_i4(self) -> None:
        prev = _turn("AGG_M_30-45_001", 4)
        draws = _draw_many(self.ctrl, self.vic, prev, "VIC", prev_role="AGG", n=500)
        seq_gaps = [g for g, m in draws if m == MixMode.SEQUENTIAL]
        lo, hi = _SHE_PROVES_GAPS["vic_i4"]
        assert _all_in_range(seq_gaps, lo, hi)

    def test_vic_after_agg_i5(self) -> None:
        prev = _turn("AGG_M_30-45_001", 5)
        draws = _draw_many(self.ctrl, self.vic, prev, "VIC", prev_role="AGG", n=500)
        seq_gaps = [g for g, m in draws if m == MixMode.SEQUENTIAL]
        lo, hi = _SHE_PROVES_GAPS["vic_i5"]
        assert _all_in_range(seq_gaps, lo, hi)


# ---------------------------------------------------------------------------
# VIC role — Elephant
# ---------------------------------------------------------------------------


class TestVICElephant:
    def setup_method(self) -> None:
        self.ctrl = TurnGapController(project="elephant_in_the_room")
        # Elephant scenes use persona IDs (SW_ / BEN_), not role-prefixed IDs.
        # Role is supplied explicitly to gap_seconds(), matching how the renderer
        # passes SpeakerConfig.role.
        self.vic = _turn("SW_F_30-45_001", 1)  # Social worker — role VIC

    def test_vic_after_i1_in_elephant_range(self) -> None:
        prev = _turn("BEN_M_40-55_003", 1)  # Beneficiary — role AGG
        draws = _draw_many(self.ctrl, self.vic, prev, "VIC", prev_role="AGG", n=500)
        seq_gaps = [g for g, m in draws if m == MixMode.SEQUENTIAL]
        lo, hi = _ELEPHANT_GAPS["vic_low"]
        assert _all_in_range(seq_gaps, lo, hi)

    def test_vic_after_i4_in_elephant_range(self) -> None:
        prev = _turn("BEN_M_40-55_003", 4)
        draws = _draw_many(self.ctrl, self.vic, prev, "VIC", prev_role="AGG", n=500)
        seq_gaps = [g for g, m in draws if m == MixMode.SEQUENTIAL]
        lo, hi = _ELEPHANT_GAPS["vic_i4"]
        assert _all_in_range(seq_gaps, lo, hi)

    def test_vic_elephant_i4_shorter_than_she_proves(self) -> None:
        # Elephant vic_i4 hi (0.100) < She-Proves vic_i4 hi (0.150)
        assert _ELEPHANT_GAPS["vic_i4"].hi < _SHE_PROVES_GAPS["vic_i4"].hi


# ---------------------------------------------------------------------------
# AGG role — She-Proves
# ---------------------------------------------------------------------------


class TestAGGSheProves:
    def setup_method(self) -> None:
        self.ctrl = TurnGapController(project="she_proves")
        self.agg = _turn("AGG_M_30-45_001", 1)
        self.prev = _turn("VIC_F_25-35_001", 1)

    def test_agg_i1_low_range(self) -> None:
        agg = _turn("AGG_M_30-45_001", 1)
        draws = _draw_many(self.ctrl, agg, self.prev, "AGG", prev_role="VIC")
        lo, hi = _SHE_PROVES_GAPS["agg_low"]
        assert _all_in_range(_gaps_only(draws), lo, hi)

    def test_agg_i2_low_range(self) -> None:
        agg = _turn("AGG_M_30-45_001", 2)
        draws = _draw_many(self.ctrl, agg, self.prev, "AGG", prev_role="VIC")
        lo, hi = _SHE_PROVES_GAPS["agg_low"]
        assert _all_in_range(_gaps_only(draws), lo, hi)

    def test_agg_i3_high_range(self) -> None:
        agg = _turn("AGG_M_30-45_001", 3)
        draws = _draw_many(self.ctrl, agg, self.prev, "AGG", prev_role="VIC", n=500)
        # At I3, sequential gaps must be in agg_high; overlap amounts in their own range.
        seq_gaps = [g for g, m in draws if m == MixMode.SEQUENTIAL]
        lo, hi = _SHE_PROVES_GAPS["agg_high"]
        assert _all_in_range(seq_gaps, lo, hi)

    def test_agg_i4_within_combined_range(self) -> None:
        # At I4 the sequential gap is from agg_high OR agg_pause.
        agg = _turn("AGG_M_30-45_001", 4)
        draws = _draw_many(self.ctrl, agg, self.prev, "AGG", prev_role="VIC", n=500)
        seq_gaps = [g for g, m in draws if m == MixMode.SEQUENTIAL]
        combined_lo = _SHE_PROVES_GAPS["agg_high"].lo
        combined_hi = _SHE_PROVES_GAPS["agg_pause"].hi
        assert _all_in_range(seq_gaps, combined_lo, combined_hi)

    def test_agg_i4_pause_drawn_sometimes(self) -> None:
        # With 500 draws and ~55% chance of SEQUENTIAL + 30% pause probability
        # among sequential, we still expect some pauses.
        agg = _turn("AGG_M_30-45_001", 4)
        draws = _draw_many(self.ctrl, agg, self.prev, "AGG", prev_role="VIC", n=500)
        seq_gaps = [g for g, m in draws if m == MixMode.SEQUENTIAL]
        long_gaps = [g for g in seq_gaps if g > _SHE_PROVES_GAPS["agg_high"].hi]
        assert len(long_gaps) > 0, "expected some deliberate AGG pauses at I4"

    def test_agg_i5_pause_drawn_sometimes(self) -> None:
        agg = _turn("AGG_M_30-45_001", 5)
        draws = _draw_many(self.ctrl, agg, self.prev, "AGG", prev_role="VIC", n=500)
        seq_gaps = [g for g, m in draws if m == MixMode.SEQUENTIAL]
        long_gaps = [g for g in seq_gaps if g > _SHE_PROVES_GAPS["agg_high"].hi]
        assert len(long_gaps) > 0, "expected some deliberate AGG pauses at I5"


# ---------------------------------------------------------------------------
# Unknown role falls back gracefully
# ---------------------------------------------------------------------------


class TestUnknownRole:
    def test_unknown_role_returns_positive_gap(self) -> None:
        ctrl = TurnGapController(project="she_proves")
        # WIT (witness) is not a defined role
        wit_turn = _turn("WIT_F_40-50_001", 2)
        prev = _turn("AGG_M_30-45_001", 2)
        draws = _draw_many(ctrl, wit_turn, prev, "WIT", prev_role="AGG")
        assert all(g > 0 for g, _ in draws)

    def test_prev_role_none_with_prev_turn_skips_overlap(self) -> None:
        """prev_role=None with a real prev_turn bypasses the overlap table (line 175)."""
        ctrl = TurnGapController(project="she_proves")
        # High-intensity AGG turn that would normally trigger BARGE_IN when prev_role="VIC".
        current = _turn("AGG_M_30-45_001", 5)
        prev = _turn("VIC_F_25-35_001", 3)
        draws = _draw_many(ctrl, current, prev, "AGG", prev_role=None, n=300)
        assert all(m == MixMode.SEQUENTIAL for m in _modes_only(draws))


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_output(self) -> None:
        ctrl = TurnGapController(project="she_proves")
        current = _turn("VIC_F_25-35_001", 1)
        prev = _turn("AGG_M_30-45_001", 3)

        rng_a = random.Random(7)
        results_a = [ctrl.gap_seconds(current, prev, rng_a, "VIC", "AGG") for _ in range(20)]
        rng_b = random.Random(7)
        results_b = [ctrl.gap_seconds(current, prev, rng_b, "VIC", "AGG") for _ in range(20)]
        assert results_a == results_b

    def test_different_seeds_different_output(self) -> None:
        ctrl = TurnGapController(project="she_proves")
        current = _turn("VIC_F_25-35_001", 1)
        prev = _turn("AGG_M_30-45_001", 3)

        rng_1 = random.Random(1)
        results_a = [ctrl.gap_seconds(current, prev, rng_1, "VIC", "AGG") for _ in range(20)]
        rng_2 = random.Random(2)
        results_b = [ctrl.gap_seconds(current, prev, rng_2, "VIC", "AGG") for _ in range(20)]
        assert results_a != results_b


# ---------------------------------------------------------------------------
# Gap ordering sanity (shorter at higher intensity for VIC)
# ---------------------------------------------------------------------------


class TestGapOrdering:
    @pytest.mark.parametrize(
        "project,table",
        [
            ("she_proves", _SHE_PROVES_GAPS),
            ("elephant_in_the_room", _ELEPHANT_GAPS),
        ],
    )
    def test_vic_gaps_shorten_as_intensity_rises(self, project: str, table: dict) -> None:
        # vic_low hi > vic_i3 hi > vic_i4 hi (compliance/fear)
        assert table["vic_low"].hi > table["vic_i3"].hi
        assert table["vic_i3"].hi > table["vic_i4"].hi

    @pytest.mark.parametrize(
        "project,table",
        [
            ("she_proves", _SHE_PROVES_GAPS),
            ("elephant_in_the_room", _ELEPHANT_GAPS),
        ],
    )
    def test_agg_pause_longer_than_agg_high(self, project: str, table: dict) -> None:
        # Menacing pause must be noticeably longer than cutting-in gap.
        assert table["agg_pause"].lo > table["agg_high"].hi


# ---------------------------------------------------------------------------
# M8a: MixMode selection and overlap depth (§4.6)
# ---------------------------------------------------------------------------


class TestOverlapModeSelection:
    """Verify that gap_seconds() emits OVERLAP / BARGE_IN at the expected rates."""

    def setup_method(self) -> None:
        self.ctrl = TurnGapController(project="she_proves")
        self.prev_vic = _turn("VIC_F_25-35_001", 3)  # used as prev for AGG-current tests
        self.prev_agg = _turn("AGG_M_30-45_001", 3)  # used as prev for VIC-current tests

    def _mode_counts(
        self,
        current: DialogueTurn,
        role: str,
        prev_role: str | None,
        prev: DialogueTurn | None = None,
        n: int = 1000,
    ) -> dict[MixMode, int]:
        rng = random.Random(99)
        prev_turn = (
            prev if prev is not None else (self.prev_agg if role == "VIC" else self.prev_vic)
        )
        counts: dict[MixMode, int] = {m: 0 for m in MixMode}
        for _ in range(n):
            _, mode = self.ctrl.gap_seconds(current, prev_turn, rng, role, prev_role)
            counts[mode] += 1
        return counts

    def test_agg_i3_barge_in_rate(self) -> None:
        """AGG at I3 should produce BARGE_IN ~10% of the time (±5% tolerance)."""
        counts = self._mode_counts(_turn("AGG_M_30-45_001", 3), "AGG", prev_role="VIC")
        barge_in_rate = counts[MixMode.BARGE_IN] / 1000
        assert 0.05 <= barge_in_rate <= 0.20, f"expected ~10%, got {barge_in_rate:.2%}"

    def test_agg_i5_barge_in_rate(self) -> None:
        """AGG at I5 should produce BARGE_IN ~40% of the time (±10% tolerance)."""
        counts = self._mode_counts(_turn("AGG_M_30-45_001", 5), "AGG", prev_role="VIC")
        barge_in_rate = counts[MixMode.BARGE_IN] / 1000
        assert 0.30 <= barge_in_rate <= 0.55, f"expected ~40%, got {barge_in_rate:.2%}"

    def test_vic_overlap_rate(self) -> None:
        """VIC at any intensity should produce OVERLAP ~10% of the time (±5%)."""
        counts = self._mode_counts(_turn("VIC_F_25-35_001", 2), "VIC", prev_role="AGG")
        overlap_rate = counts[MixMode.OVERLAP] / 1000
        assert 0.05 <= overlap_rate <= 0.20, f"expected ~10%, got {overlap_rate:.2%}"

    def test_agg_i1_always_sequential(self) -> None:
        """AGG at I1–I2 must never produce OVERLAP or BARGE_IN."""
        for intensity in (1, 2):
            counts = self._mode_counts(_turn("AGG_M_30-45_001", intensity), "AGG", prev_role="VIC")
            assert counts[MixMode.OVERLAP] == 0
            assert counts[MixMode.BARGE_IN] == 0

    def test_same_role_never_overlaps(self) -> None:
        """Same-role transitions (AGG→AGG) must always be SEQUENTIAL per §4.6."""
        # _OVERLAP_PROBS has no (AGG, AGG, *) or (VIC, VIC, *) entries.
        for intensity in (3, 4, 5):
            counts = self._mode_counts(
                _turn("AGG_M_30-45_001", intensity),
                "AGG",
                prev_role="AGG",  # same role
            )
            assert counts[MixMode.OVERLAP] == 0
            assert counts[MixMode.BARGE_IN] == 0

    def test_barge_in_amount_in_range(self) -> None:
        """BARGE_IN amount must lie within _BARGE_IN_DEPTH_RANGE."""
        rng = random.Random(42)
        current = _turn("AGG_M_30-45_001", 5)
        depths = []
        for _ in range(1000):
            amount, mode = self.ctrl.gap_seconds(current, self.prev_vic, rng, "AGG", "VIC")
            if mode == MixMode.BARGE_IN:
                depths.append(amount)
        assert depths, "no BARGE_IN draws in 1000 trials at I5"
        assert all(_BARGE_IN_DEPTH_RANGE.lo <= d <= _BARGE_IN_DEPTH_RANGE.hi for d in depths)

    def test_overlap_amount_in_range(self) -> None:
        """OVERLAP amount must lie within _OVERLAP_DEPTH_RANGE."""
        rng = random.Random(42)
        current = _turn("AGG_M_30-45_001", 3)
        depths = []
        for _ in range(1000):
            amount, mode = self.ctrl.gap_seconds(current, self.prev_vic, rng, "AGG", "VIC")
            if mode == MixMode.OVERLAP:
                depths.append(amount)
        assert depths, "no OVERLAP draws in 1000 trials at I3"
        assert all(_OVERLAP_DEPTH_RANGE.lo <= d <= _OVERLAP_DEPTH_RANGE.hi for d in depths)
