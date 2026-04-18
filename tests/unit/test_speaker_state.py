"""Unit tests for SpeakerState (M7 — stateful cross-turn emotional controller)."""

from __future__ import annotations

import pytest

from synthbanshee.tts.speaker_state import (
    _DRIFT_RATE_DECAY,
    _DRIFT_RATE_ESCALATE,
    SpeakerState,
    _target_for,
)

# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


class TestInitialState:
    def test_rate_offset_is_neutral(self) -> None:
        assert SpeakerState().rate_offset == 1.0

    def test_pitch_and_volume_are_zero(self) -> None:
        s = SpeakerState()
        assert s.pitch_offset_st == 0.0
        assert s.volume_offset_db == 0.0

    def test_breathiness_is_zero(self) -> None:
        assert SpeakerState().breathiness_level == 0.0

    def test_intensity_history_is_empty(self) -> None:
        assert SpeakerState().intensity_history == []

    def test_metadata_dict_neutral(self) -> None:
        d = SpeakerState().to_metadata_dict()
        assert d["rate_offset"] == 1.0
        assert d["pitch_offset_st"] == 0.0
        assert d["volume_offset_db"] == 0.0
        assert d["breathiness_level"] == 0.0


# ---------------------------------------------------------------------------
# Target table sanity
# ---------------------------------------------------------------------------


class TestIntensityClamping:
    @pytest.mark.parametrize("oob", [0, -1, 6, 99])
    def test_target_for_clamps_oob_to_boundary(self, oob: int) -> None:
        """Out-of-range intensities must not return an extreme or KeyError."""
        clamped = max(1, min(5, oob))
        assert _target_for("AGG", oob) == _target_for("AGG", clamped)
        assert _target_for("VIC", oob) == _target_for("VIC", clamped)

    @pytest.mark.parametrize("oob", [0, -1, 6, 99])
    def test_update_clamps_oob_intensity(self, oob: int) -> None:
        """update() with OOB intensity records the clamped value, not the raw one."""
        s = SpeakerState()
        s.update(oob, "AGG")
        assert s.intensity_history[0] == max(1, min(5, oob))

    def test_update_zero_does_not_hit_i5_target(self) -> None:
        """Before fix: intensity=0 fell through to table[max(table)] → I5 target."""
        s_zero = SpeakerState()
        s_zero.update(0, "AGG")  # should clamp to I1, not I5

        s_five = SpeakerState()
        s_five.update(5, "AGG")

        # I1-clamped state must not equal I5-drifted state
        assert s_zero.rate_offset != pytest.approx(s_five.rate_offset)
        # And it must be much closer to neutral than I5
        assert abs(s_zero.rate_offset - 1.0) < abs(s_five.rate_offset - 1.0)


class TestTargetTable:
    @pytest.mark.parametrize("intensity", [1, 2, 3, 4, 5])
    def test_agg_rate_monotonically_rising(self, intensity: int) -> None:
        if intensity == 1:
            return
        prev_rate, _, _ = _target_for("AGG", intensity - 1)
        curr_rate, _, _ = _target_for("AGG", intensity)
        assert curr_rate >= prev_rate

    @pytest.mark.parametrize("intensity", [2, 3, 4, 5])
    def test_vic_rate_monotonically_falling(self, intensity: int) -> None:
        prev_rate, _, _ = _target_for("VIC", intensity - 1)
        curr_rate, _, _ = _target_for("VIC", intensity)
        assert curr_rate <= prev_rate

    @pytest.mark.parametrize("intensity", [2, 3, 4, 5])
    def test_agg_volume_monotonically_rising(self, intensity: int) -> None:
        _, _, prev_vol = _target_for("AGG", intensity - 1)
        _, _, curr_vol = _target_for("AGG", intensity)
        assert curr_vol >= prev_vol

    @pytest.mark.parametrize("intensity", [2, 3, 4, 5])
    def test_vic_volume_monotonically_falling(self, intensity: int) -> None:
        _, _, prev_vol = _target_for("VIC", intensity - 1)
        _, _, curr_vol = _target_for("VIC", intensity)
        assert curr_vol <= prev_vol

    def test_unknown_role_returns_neutral(self) -> None:
        rate, pitch, vol = _target_for("WIT", 5)
        assert rate == 1.0
        assert pitch == 0.0
        assert vol == 0.0


# ---------------------------------------------------------------------------
# Single update — escalation drift math
# ---------------------------------------------------------------------------


class TestSingleUpdate:
    def test_first_update_treated_as_escalation(self) -> None:
        """With empty history, first update uses ESCALATE drift (prev = new)."""
        s = SpeakerState()
        t_rate, t_pitch, t_vol = _target_for("AGG", 5)
        s.update(5, "AGG")
        expected_rate = 1.0 + _DRIFT_RATE_ESCALATE * (t_rate - 1.0)
        assert s.rate_offset == pytest.approx(expected_rate)

    def test_agg_i5_rate_exceeds_neutral(self) -> None:
        s = SpeakerState()
        s.update(5, "AGG")
        assert s.rate_offset > 1.0

    def test_vic_i5_rate_below_neutral(self) -> None:
        s = SpeakerState()
        s.update(5, "VIC")
        assert s.rate_offset < 1.0

    def test_agg_i5_volume_positive(self) -> None:
        s = SpeakerState()
        s.update(5, "AGG")
        assert s.volume_offset_db > 0.0

    def test_vic_i5_volume_negative(self) -> None:
        s = SpeakerState()
        s.update(5, "VIC")
        assert s.volume_offset_db < 0.0

    def test_i1_update_keeps_state_near_neutral(self) -> None:
        """I1 target ≈ neutral, so a single I1 update barely moves state."""
        s = SpeakerState()
        s.update(1, "AGG")
        assert s.rate_offset == pytest.approx(1.0)
        assert s.pitch_offset_st == pytest.approx(0.0)
        assert s.volume_offset_db == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Escalation accumulation
# ---------------------------------------------------------------------------


class TestEscalationAccumulation:
    def _run_sequence(self, role: str, intensities: list[int]) -> SpeakerState:
        s = SpeakerState()
        for i in intensities:
            s.update(i, role)
        return s

    def test_agg_rate_grows_with_sustained_high_intensity(self) -> None:
        s1 = self._run_sequence("AGG", [3])
        s5 = self._run_sequence("AGG", [3, 3, 3, 3, 3])
        assert s5.rate_offset > s1.rate_offset

    def test_vic_rate_falls_with_sustained_high_intensity(self) -> None:
        s1 = self._run_sequence("VIC", [5])
        s5 = self._run_sequence("VIC", [5, 5, 5, 5, 5])
        assert s5.rate_offset < s1.rate_offset

    def test_agg_escalating_arc_rate_monotonically_rises(self) -> None:
        s = SpeakerState()
        prev_rate = s.rate_offset
        for intensity in [1, 2, 3, 4, 5]:
            s.update(intensity, "AGG")
            assert s.rate_offset >= prev_rate
            prev_rate = s.rate_offset

    def test_vic_escalating_arc_rate_monotonically_falls(self) -> None:
        s = SpeakerState()
        prev_rate = s.rate_offset
        for intensity in [1, 2, 3, 4, 5]:
            s.update(intensity, "VIC")
            assert s.rate_offset <= prev_rate
            prev_rate = s.rate_offset


# ---------------------------------------------------------------------------
# De-escalation uses slower decay rate
# ---------------------------------------------------------------------------


class TestDeescalation:
    def test_decay_drift_smaller_than_escalation_drift(self) -> None:
        """Stepping from I5 back to I1 moves state less than stepping I1→I5."""
        # Escalate to I5 first so there's somewhere to decay from.
        s_esc = SpeakerState()
        s_esc.update(5, "AGG")
        rate_after_escalation = s_esc.rate_offset

        s_dec = SpeakerState()
        s_dec.update(5, "AGG")  # same starting point
        rate_before_decay = s_dec.rate_offset

        t_rate_i1, _, _ = _target_for("AGG", 1)
        expected_decay = rate_before_decay + _DRIFT_RATE_DECAY * (t_rate_i1 - rate_before_decay)
        s_dec.update(1, "AGG")
        assert s_dec.rate_offset == pytest.approx(expected_decay)
        # De-escalation step smaller than escalation step from neutral
        assert abs(s_dec.rate_offset - rate_after_escalation) < abs(rate_after_escalation - 1.0)

    def test_intensity_history_records_full_arc(self) -> None:
        s = SpeakerState()
        for i in [1, 3, 5, 2]:
            s.update(i, "AGG")
        assert s.intensity_history == [1, 3, 5, 2]


# ---------------------------------------------------------------------------
# Metadata serialization
# ---------------------------------------------------------------------------


class TestMetadataDict:
    def test_keys_present(self) -> None:
        d = SpeakerState().to_metadata_dict()
        assert set(d) == {"rate_offset", "pitch_offset_st", "volume_offset_db", "breathiness_level"}

    def test_values_are_floats(self) -> None:
        s = SpeakerState()
        s.update(4, "AGG")
        d = s.to_metadata_dict()
        assert all(isinstance(v, float) for v in d.values())

    def test_values_rounded_to_4dp(self) -> None:
        s = SpeakerState()
        s.update(3, "VIC")
        d = s.to_metadata_dict()
        for v in d.values():
            assert v == round(v, 4)

    def test_snapshot_reflects_pre_update_state(self) -> None:
        """to_metadata_dict() called before update returns neutral values."""
        s = SpeakerState()
        snapshot = s.to_metadata_dict()
        s.update(5, "AGG")
        assert snapshot["rate_offset"] == 1.0  # was neutral at snapshot time


# ---------------------------------------------------------------------------
# Unknown / neutral role
# ---------------------------------------------------------------------------


class TestNeutralRole:
    def test_unknown_role_state_stays_near_neutral(self) -> None:
        s = SpeakerState()
        for _ in range(10):
            s.update(5, "WIT")
        assert s.rate_offset == pytest.approx(1.0)
        assert s.pitch_offset_st == pytest.approx(0.0)
        assert s.volume_offset_db == pytest.approx(0.0)
