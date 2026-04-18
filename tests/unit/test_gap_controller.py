"""Unit tests for TurnGapController (M6 — inter-turn timing redesign)."""

from __future__ import annotations

import random

import pytest

from synthbanshee.script.types import DialogueTurn
from synthbanshee.tts.gap_controller import (
    _DEFAULT_GAPS,
    _ELEPHANT_GAPS,
    _SHE_PROVES_GAPS,
    TurnGapController,
)


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
    n: int = 200,
) -> list[float]:
    rng = random.Random(42)
    return [ctrl.gap_seconds(current, prev, rng, current_role) for _ in range(n)]


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
        gaps = _draw_many(ctrl, _turn("AGG_M_30-45_001", 1), None, "AGG")
        lo, hi = _SHE_PROVES_GAPS["agg_low"]
        assert _all_in_range(gaps, lo, hi)


# ---------------------------------------------------------------------------
# VIC role — She-Proves
# ---------------------------------------------------------------------------


class TestVICSheProves:
    def setup_method(self) -> None:
        self.ctrl = TurnGapController(project="she_proves")
        self.vic = _turn("VIC_F_25-35_001", 1)

    def test_vic_after_agg_i1(self) -> None:
        prev = _turn("AGG_M_30-45_001", 1)
        gaps = _draw_many(self.ctrl, self.vic, prev, "VIC")
        lo, hi = _SHE_PROVES_GAPS["vic_low"]
        assert _all_in_range(gaps, lo, hi)

    def test_vic_after_agg_i2(self) -> None:
        prev = _turn("AGG_M_30-45_001", 2)
        gaps = _draw_many(self.ctrl, self.vic, prev, "VIC")
        lo, hi = _SHE_PROVES_GAPS["vic_low"]
        assert _all_in_range(gaps, lo, hi)

    def test_vic_after_agg_i3(self) -> None:
        prev = _turn("AGG_M_30-45_001", 3)
        gaps = _draw_many(self.ctrl, self.vic, prev, "VIC")
        lo, hi = _SHE_PROVES_GAPS["vic_i3"]
        assert _all_in_range(gaps, lo, hi)

    def test_vic_after_agg_i4(self) -> None:
        prev = _turn("AGG_M_30-45_001", 4)
        gaps = _draw_many(self.ctrl, self.vic, prev, "VIC")
        lo, hi = _SHE_PROVES_GAPS["vic_i4"]
        assert _all_in_range(gaps, lo, hi)

    def test_vic_after_agg_i5(self) -> None:
        prev = _turn("AGG_M_30-45_001", 5)
        gaps = _draw_many(self.ctrl, self.vic, prev, "VIC")
        lo, hi = _SHE_PROVES_GAPS["vic_i5"]
        assert _all_in_range(gaps, lo, hi)


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
        gaps = _draw_many(self.ctrl, self.vic, prev, "VIC")
        lo, hi = _ELEPHANT_GAPS["vic_low"]
        assert _all_in_range(gaps, lo, hi)

    def test_vic_after_i4_in_elephant_range(self) -> None:
        prev = _turn("BEN_M_40-55_003", 4)
        gaps = _draw_many(self.ctrl, self.vic, prev, "VIC")
        lo, hi = _ELEPHANT_GAPS["vic_i4"]
        assert _all_in_range(gaps, lo, hi)

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
        gaps = _draw_many(self.ctrl, agg, self.prev, "AGG")
        lo, hi = _SHE_PROVES_GAPS["agg_low"]
        assert _all_in_range(gaps, lo, hi)

    def test_agg_i2_low_range(self) -> None:
        agg = _turn("AGG_M_30-45_001", 2)
        gaps = _draw_many(self.ctrl, agg, self.prev, "AGG")
        lo, hi = _SHE_PROVES_GAPS["agg_low"]
        assert _all_in_range(gaps, lo, hi)

    def test_agg_i3_high_range(self) -> None:
        agg = _turn("AGG_M_30-45_001", 3)
        # At I3, no pause chance (only I4–I5 allow deliberate pause).
        gaps = _draw_many(self.ctrl, agg, self.prev, "AGG")
        lo, hi = _SHE_PROVES_GAPS["agg_high"]
        assert _all_in_range(gaps, lo, hi)

    def test_agg_i4_within_combined_range(self) -> None:
        # At I4 the gap is from agg_high OR agg_pause — both must fit within
        # the union [agg_high.lo, agg_pause.hi].
        agg = _turn("AGG_M_30-45_001", 4)
        gaps = _draw_many(self.ctrl, agg, self.prev, "AGG", n=500)
        combined_lo = _SHE_PROVES_GAPS["agg_high"].lo
        combined_hi = _SHE_PROVES_GAPS["agg_pause"].hi
        assert _all_in_range(gaps, combined_lo, combined_hi)

    def test_agg_i4_pause_drawn_sometimes(self) -> None:
        # With 500 draws and 30 % probability, the chance of zero pauses is
        # < (0.70)^500 ≈ 10^{-79} — effectively impossible.
        agg = _turn("AGG_M_30-45_001", 4)
        gaps = _draw_many(self.ctrl, agg, self.prev, "AGG", n=500)
        # Pauses must exceed agg_high.hi (0.200 s) to be distinguishable.
        long_gaps = [g for g in gaps if g > _SHE_PROVES_GAPS["agg_high"].hi]
        assert len(long_gaps) > 0, "expected some deliberate AGG pauses at I4"

    def test_agg_i5_pause_drawn_sometimes(self) -> None:
        agg = _turn("AGG_M_30-45_001", 5)
        gaps = _draw_many(self.ctrl, agg, self.prev, "AGG", n=500)
        long_gaps = [g for g in gaps if g > _SHE_PROVES_GAPS["agg_high"].hi]
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
        gaps = _draw_many(ctrl, wit_turn, prev, "WIT")
        assert all(g > 0 for g in gaps)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_output(self) -> None:
        ctrl = TurnGapController(project="she_proves")
        current = _turn("VIC_F_25-35_001", 1)
        prev = _turn("AGG_M_30-45_001", 3)

        rng_a = random.Random(7)
        gaps_a = [ctrl.gap_seconds(current, prev, rng_a, "VIC") for _ in range(20)]
        rng_b = random.Random(7)
        gaps_b = [ctrl.gap_seconds(current, prev, rng_b, "VIC") for _ in range(20)]
        assert gaps_a == gaps_b

    def test_different_seeds_different_output(self) -> None:
        ctrl = TurnGapController(project="she_proves")
        current = _turn("VIC_F_25-35_001", 1)
        prev = _turn("AGG_M_30-45_001", 3)

        rng_1 = random.Random(1)
        gaps_a = [ctrl.gap_seconds(current, prev, rng_1, "VIC") for _ in range(20)]
        rng_2 = random.Random(2)
        gaps_b = [ctrl.gap_seconds(current, prev, rng_2, "VIC") for _ in range(20)]
        assert gaps_a != gaps_b


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
