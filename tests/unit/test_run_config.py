"""Unit tests for RunConfig Pydantic model."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from synthbanshee.config.run_config import RunConfig, SplitFractions, TypologyTarget

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_RUN_YAML = textwrap.dedent("""\
    run_id: test_run_001
    project: she_proves
    tier: A
    language: he
    random_seed: 42
    output_dir: data/he
    scene_configs_dir: configs/scenes/she_proves
    targets:
      - violence_typology: IT
        count: 150
      - violence_typology: SV
        count: 150
      - violence_typology: NEG
        count: 150
      - violence_typology: NEU
        count: 50
    splits:
      train: 0.70
      val: 0.15
      test: 0.15
    max_retries: 3
    fail_fast: false
""")


def _load_yaml_dict(text: str) -> dict:
    return yaml.safe_load(text)


# ---------------------------------------------------------------------------
# TypologyTarget
# ---------------------------------------------------------------------------


class TestTypologyTarget:
    def test_valid_typology(self):
        t = TypologyTarget(violence_typology="IT", count=100)
        assert t.violence_typology == "IT"
        assert t.count == 100

    def test_invalid_typology_rejected(self):
        with pytest.raises(ValidationError, match="not in taxonomy"):
            TypologyTarget(violence_typology="BOGUS", count=10)

    def test_count_must_be_positive(self):
        with pytest.raises(ValidationError):
            TypologyTarget(violence_typology="IT", count=0)


# ---------------------------------------------------------------------------
# SplitFractions
# ---------------------------------------------------------------------------


class TestSplitFractions:
    def test_defaults_sum_to_one(self):
        s = SplitFractions()
        assert abs(s.train + s.val + s.test - 1.0) < 1e-9

    def test_custom_fractions(self):
        s = SplitFractions(train=0.8, val=0.1, test=0.1)
        assert s.train == 0.8

    def test_fractions_not_summing_to_one_rejected(self):
        with pytest.raises(ValidationError, match="sum to 1.0"):
            SplitFractions(train=0.6, val=0.1, test=0.1)

    def test_zero_fraction_rejected(self):
        with pytest.raises(ValidationError):
            SplitFractions(train=0.0, val=0.5, test=0.5)


# ---------------------------------------------------------------------------
# RunConfig
# ---------------------------------------------------------------------------


class TestRunConfig:
    def test_parse_valid_dict(self):
        cfg = RunConfig.model_validate(_load_yaml_dict(_VALID_RUN_YAML))
        assert cfg.run_id == "test_run_001"
        assert cfg.project == "she_proves"
        assert cfg.tier == "A"
        assert len(cfg.targets) == 4
        assert cfg.total_target == 500

    def test_invalid_project_rejected(self):
        d = _load_yaml_dict(_VALID_RUN_YAML)
        d["project"] = "unknown_project"
        with pytest.raises(ValidationError, match="project"):
            RunConfig.model_validate(d)

    def test_invalid_tier_rejected(self):
        d = _load_yaml_dict(_VALID_RUN_YAML)
        d["tier"] = "D"
        with pytest.raises(ValidationError):
            RunConfig.model_validate(d)

    def test_empty_targets_rejected(self):
        d = _load_yaml_dict(_VALID_RUN_YAML)
        d["targets"] = []
        with pytest.raises(ValidationError):
            RunConfig.model_validate(d)

    def test_total_target(self):
        cfg = RunConfig.model_validate(_load_yaml_dict(_VALID_RUN_YAML))
        assert cfg.total_target == 150 + 150 + 150 + 50

    def test_targets_by_typology(self):
        cfg = RunConfig.model_validate(_load_yaml_dict(_VALID_RUN_YAML))
        by_typ = cfg.targets_by_typology()
        assert by_typ["IT"] == 150
        assert by_typ["NEU"] == 50

    def test_from_yaml_she_proves_example(self):
        """The bundled she_proves example run config must parse without error."""
        yaml_path = (
            Path(__file__).parent.parent.parent
            / "configs"
            / "run_configs"
            / "tier_a_500_she_proves.yaml"
        )
        cfg = RunConfig.from_yaml(yaml_path)
        assert cfg.project == "she_proves"
        assert cfg.total_target == 500

    def test_from_yaml_elephant_example(self):
        """The bundled elephant example run config must parse without error."""
        yaml_path = (
            Path(__file__).parent.parent.parent
            / "configs"
            / "run_configs"
            / "tier_a_500_elephant.yaml"
        )
        cfg = RunConfig.from_yaml(yaml_path)
        assert cfg.project == "elephant_in_the_room"
        assert cfg.total_target == 500

    def test_max_retries_default(self):
        cfg = RunConfig.model_validate(_load_yaml_dict(_VALID_RUN_YAML))
        assert cfg.max_retries == 3

    def test_max_retries_must_be_at_least_one(self):
        d = _load_yaml_dict(_VALID_RUN_YAML)
        d["max_retries"] = 0
        with pytest.raises(ValidationError):
            RunConfig.model_validate(d)

    def test_elephant_project_valid(self):
        d = _load_yaml_dict(_VALID_RUN_YAML)
        d["project"] = "elephant_in_the_room"
        cfg = RunConfig.model_validate(d)
        assert cfg.project == "elephant_in_the_room"
