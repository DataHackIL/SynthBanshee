"""Unit tests for ProjectProfile model and profile loading (M13)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from synthbanshee.config.project_profile import (
    AcousticDefaults,
    GapRange,
    GapTimingDefaults,
    LoudnessDefaults,
    ProjectProfile,
    clear_profile_cache,
    load_profile,
)
from synthbanshee.config.run_config import RunConfig
from synthbanshee.tts.gap_controller import TurnGapController
from synthbanshee.tts.mix_mode import MixMode


def _default_gap_timing() -> GapTimingDefaults:
    """Return a GapTimingDefaults with framework-default values."""
    return GapTimingDefaults(
        vic_low=GapRange(lo=0.30, hi=0.60),
        vic_i3=GapRange(lo=0.15, hi=0.35),
        vic_i4=GapRange(lo=0.05, hi=0.15),
        vic_i5=GapRange(lo=0.10, hi=0.30),
        agg_low=GapRange(lo=0.20, hi=0.50),
        agg_high=GapRange(lo=0.05, hi=0.20),
        agg_pause=GapRange(lo=0.60, hi=1.40),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PROFILE_DIR = Path(__file__).parent.parent.parent / "configs" / "run_configs"


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear the profile cache before each test."""
    clear_profile_cache()
    yield
    clear_profile_cache()


# ---------------------------------------------------------------------------
# GapRange
# ---------------------------------------------------------------------------


class TestGapRange:
    def test_valid_range(self):
        r = GapRange(lo=0.1, hi=0.5)
        assert r.lo == 0.1
        assert r.hi == 0.5

    def test_equal_lo_hi_allowed(self):
        r = GapRange(lo=0.3, hi=0.3)
        assert r.lo == r.hi

    def test_negative_rejected(self):
        with pytest.raises(ValidationError):
            GapRange(lo=-0.1, hi=0.5)

    def test_inverted_range_rejected(self):
        with pytest.raises(ValidationError, match="lo.*must be <= hi"):
            GapRange(lo=0.5, hi=0.1)


# ---------------------------------------------------------------------------
# GapTimingDefaults
# ---------------------------------------------------------------------------


class TestGapTimingDefaults:
    def test_all_fields_required(self):
        with pytest.raises(ValidationError):
            GapTimingDefaults()  # type: ignore[call-arg]

    def test_explicit_construction(self):
        g = _default_gap_timing()
        assert g.vic_low.lo == 0.30
        assert g.agg_pause.hi == 1.40
        assert g.vic_i4.lo == 0.05


# ---------------------------------------------------------------------------
# ProjectProfile
# ---------------------------------------------------------------------------


class TestProjectProfile:
    def test_requires_gap_timing(self):
        with pytest.raises(ValidationError):
            ProjectProfile(name="test")  # type: ignore[call-arg]

    def test_with_gap_timing(self):
        p = ProjectProfile(name="test", gap_timing=_default_gap_timing())
        assert p.name == "test"
        assert p.description == ""
        assert p.gap_timing.vic_low.lo == 0.30
        assert p.preprocessing.wiener_denoise is False

    def test_custom_values(self):
        p = ProjectProfile(
            name="custom",
            gap_timing=_default_gap_timing(),
            loudness=LoudnessDefaults(agg_rms_dbfs=-18.0, vic_rms_dbfs=-22.0),
            acoustic=AcousticDefaults(snr_target_db=24.0),
        )
        assert p.loudness.agg_rms_dbfs == -18.0
        assert p.acoustic.snr_target_db == 24.0

    def test_from_yaml_she_proves(self):
        path = _PROFILE_DIR / "profile_she_proves.yaml"
        p = ProjectProfile.from_yaml(path)
        assert p.name == "she_proves"
        assert p.gap_timing.vic_low.lo == 0.30
        assert p.gap_timing.vic_low.hi == 0.60
        assert p.overlap.agg_pause_prob == 0.30
        assert p.loudness.agg_rms_dbfs == -20.0
        assert p.loudness.vic_rms_dbfs == -24.0
        assert p.acoustic.snr_target_db == 18.0
        assert "phone_in_hand" in p.acoustic.preferred_devices

    def test_from_yaml_elephant(self):
        path = _PROFILE_DIR / "profile_elephant.yaml"
        p = ProjectProfile.from_yaml(path)
        assert p.name == "elephant"
        assert p.gap_timing.vic_i4.lo == 0.05
        assert p.gap_timing.vic_i4.hi == 0.10
        assert p.overlap.agg_pause_prob == 0.25
        assert p.loudness.agg_rms_dbfs == -18.0
        assert p.acoustic.snr_target_db == 24.0
        assert "pi_budget_mic" in p.acoustic.preferred_devices


# ---------------------------------------------------------------------------
# load_profile
# ---------------------------------------------------------------------------


class TestLoadProfile:
    def test_generic_returns_defaults(self):
        p = load_profile("generic")
        assert p.name == "generic"

    def test_she_proves_profile(self):
        p = load_profile("she_proves", profile_dir=_PROFILE_DIR)
        assert p.name == "she_proves"
        assert p.acoustic.snr_target_db == 18.0

    def test_elephant_profile(self):
        p = load_profile("elephant", profile_dir=_PROFILE_DIR)
        assert p.name == "elephant"
        assert p.acoustic.snr_target_db == 24.0

    def test_missing_profile_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="no_such_profile"):
            load_profile("no_such_profile", profile_dir=tmp_path)

    def test_caching(self):
        p1 = load_profile("she_proves", profile_dir=_PROFILE_DIR)
        p2 = load_profile("she_proves", profile_dir=_PROFILE_DIR)
        assert p1 is p2

    def test_custom_profile_from_tmp(self, tmp_path: Path):
        """A new profile YAML can be added without code changes."""
        profile_data = {
            "name": "custom_project",
            "description": "A custom project profile",
            "gap_timing": {
                "vic_low": {"lo": 0.40, "hi": 0.70},
                "vic_i3": {"lo": 0.20, "hi": 0.40},
                "vic_i4": {"lo": 0.10, "hi": 0.20},
                "vic_i5": {"lo": 0.15, "hi": 0.35},
                "agg_low": {"lo": 0.25, "hi": 0.55},
                "agg_high": {"lo": 0.08, "hi": 0.25},
                "agg_pause": {"lo": 0.70, "hi": 1.50},
            },
            "loudness": {"agg_rms_dbfs": -16.0, "vic_rms_dbfs": -20.0},
            "acoustic": {"snr_target_db": 15.0},
        }
        yaml_path = tmp_path / "profile_custom_project.yaml"
        yaml_path.write_text(yaml.dump(profile_data), encoding="utf-8")

        p = load_profile("custom_project", profile_dir=tmp_path)
        assert p.name == "custom_project"
        assert p.gap_timing.vic_low.lo == 0.40
        assert p.loudness.agg_rms_dbfs == -16.0
        assert p.acoustic.snr_target_db == 15.0


# ---------------------------------------------------------------------------
# RunConfig integration
# ---------------------------------------------------------------------------

_RUN_YAML_WITH_PROFILE = textwrap.dedent("""\
    run_id: test_with_profile
    project: she_proves
    tier: A
    project_profile: she_proves
    targets:
      - violence_typology: IT
        count: 100
    splits:
      train: 0.70
      val: 0.15
      test: 0.15
""")


class TestRunConfigProfile:
    def test_default_profile_is_generic(self):
        data = yaml.safe_load(
            textwrap.dedent("""\
                run_id: test
                project: she_proves
                targets:
                  - violence_typology: IT
                    count: 10
                splits:
                  train: 0.70
                  val: 0.15
                  test: 0.15
            """)
        )
        cfg = RunConfig.model_validate(data)
        assert cfg.project_profile == "generic"

    def test_profile_field_parsed(self):
        cfg = RunConfig.model_validate(yaml.safe_load(_RUN_YAML_WITH_PROFILE))
        assert cfg.project_profile == "she_proves"

    def test_resolved_profile_returns_profile(self):
        cfg = RunConfig.model_validate(yaml.safe_load(_RUN_YAML_WITH_PROFILE))
        profile = cfg.resolved_profile()
        assert profile.name == "she_proves"
        assert profile.acoustic.snr_target_db == 18.0

    def test_resolved_generic_profile(self):
        data = yaml.safe_load(
            textwrap.dedent("""\
                run_id: test
                project: she_proves
                targets:
                  - violence_typology: IT
                    count: 10
                splits:
                  train: 0.70
                  val: 0.15
                  test: 0.15
            """)
        )
        cfg = RunConfig.model_validate(data)
        profile = cfg.resolved_profile()
        assert profile.name == "generic"


# ---------------------------------------------------------------------------
# Gap controller integration with profiles
# ---------------------------------------------------------------------------


class TestGapControllerWithProfile:
    def test_from_profile_uses_profile_gaps(self):
        import random

        from synthbanshee.script.types import DialogueTurn

        profile = load_profile("elephant", profile_dir=_PROFILE_DIR)
        ctrl = TurnGapController.from_profile("elephant_in_the_room", profile)

        prev = DialogueTurn(speaker_id="agg1", text="test", intensity=1)
        curr = DialogueTurn(speaker_id="vic1", text="test", intensity=1)
        rng = random.Random(42)

        # Draw many gaps and verify they fall within the elephant profile ranges
        for _ in range(100):
            amount, mode = ctrl.gap_seconds(curr, prev, rng, "VIC", "AGG")
            if mode == MixMode.SEQUENTIAL:
                # vic_low range from elephant profile: 0.25–0.50
                assert 0.25 <= amount <= 0.50

    def test_from_profile_uses_profile_agg_pause_prob(self):
        """Elephant profile has agg_pause_prob=0.25, different from default 0.30."""
        profile = load_profile("elephant", profile_dir=_PROFILE_DIR)
        ctrl = TurnGapController.from_profile("elephant_in_the_room", profile)
        assert ctrl._agg_pause_prob == 0.25

    def test_from_profile_she_proves(self):
        profile = load_profile("she_proves", profile_dir=_PROFILE_DIR)
        ctrl = TurnGapController.from_profile("she_proves", profile)
        assert ctrl._agg_pause_prob == 0.30

    def test_without_profile_uses_hardcoded_tables(self):
        from synthbanshee.tts.gap_controller import _SHE_PROVES_GAPS

        ctrl = TurnGapController(project="she_proves")
        assert ctrl._table is _SHE_PROVES_GAPS
