"""Unit tests for Jinja2 prompt templates.

Verifies that every template in the template library:
  - Renders without error using a representative context
  - Produces non-empty output containing the JSON schema block
  - Does not leave any un-rendered Jinja2 variable references
"""

from __future__ import annotations

from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader, StrictUndefined

# ---------------------------------------------------------------------------
# Template discovery
# ---------------------------------------------------------------------------

TEMPLATES_ROOT = Path(__file__).parent.parent.parent / "synthbanshee" / "script" / "templates"


def _discover_templates(project: str) -> list[Path]:
    """Return all non-partial .j2 templates for a project."""
    project_dir = TEMPLATES_ROOT / project
    return sorted(p for p in project_dir.glob("*.j2") if not p.name.startswith("_"))


SHE_PROVES_TEMPLATES = _discover_templates("she_proves")
ELEPHANT_TEMPLATES = _discover_templates("elephant")
ALL_TEMPLATES = SHE_PROVES_TEMPLATES + ELEPHANT_TEMPLATES


# ---------------------------------------------------------------------------
# Shared render context
# ---------------------------------------------------------------------------

_SHE_PROVES_CONTEXT = dict(
    scene_id="SP_IT_A_TEST",
    project="she_proves",
    violence_typology="IT",
    script_slots={
        "relationship": "spouse",
        "setting": "apartment_kitchen",
        "grievance": "going_out_without_permission",
        "surveillance_method": "phone_inspection",
        "financial_trigger": "credit_card_purchase",
        "isolation_target": "victim_mother",
        "prior_incident_type": "physical_altercation",
        "argument_trigger": "return_home_late",
        "degradation_theme": "intelligence_and_competence",
        "argument_topic": "household_chores",
        "third_party_role": "neighbour",
        "activity": "preparing_dinner",
        "coordination_topic": "school_pickup_schedule",
        "assault_trigger": "dinner_dispute",
        "object_type": "kitchen_knife",
        "children_ages": "toddler_and_school_age",
    },
    intensity_arc=[1, 2, 3, 4, 5],
    target_duration_minutes=3.0,
    speakers=[
        {"speaker_id": "AGG_M_30-45_001", "role": "AGG", "gender": "male"},
        {"speaker_id": "VIC_F_25-40_002", "role": "VIC", "gender": "female"},
    ],
)

_ELEPHANT_CONTEXT = dict(
    scene_id="EL_SV_A_TEST",
    project="elephant_in_the_room",
    violence_typology="SV",
    script_slots={
        "scene_type": "welfare_office",
        "client_issue": "welfare_claim_denied",
        "escalation_trigger": "second_denial",
        "object_type": "heavy_folder",
        "dispute_subject": "housing_assistance",
        "benefit_type": "disability_allowance",
        "denial_reason": "missing_documentation",
        "custody_context": "removal_risk_assessment",
        "aggression_direction": "toward_social_worker",
        "deescalation_technique": "validation_and_alternative",
        "intake_topic": "new_benefits_application",
        "wait_duration": "forty_five_minutes",
        "case_context": "ongoing_support_monitoring",
        "aggression_style": "personalised_insults",
    },
    intensity_arc=[1, 2, 3, 4, 5],
    target_duration_minutes=2.0,
    speakers=[
        {"speaker_id": "BEN_M_40-55_003", "role": "AGG", "gender": "male"},
        {"speaker_id": "SW_F_30-45_001", "role": "VIC", "gender": "female"},
    ],
)


def _context_for(template_path: Path) -> dict:
    if "she_proves" in template_path.parts:
        return _SHE_PROVES_CONTEXT
    return _ELEPHANT_CONTEXT


# ---------------------------------------------------------------------------
# Render helper
# ---------------------------------------------------------------------------


def _render(template_path: Path) -> str:
    """Render a template with the appropriate test context."""
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    template = env.get_template(template_path.name)
    return template.render(**_context_for(template_path))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("template_path", ALL_TEMPLATES, ids=lambda p: p.name)
class TestTemplateRendering:
    def test_renders_without_error(self, template_path: Path):
        """Every template must render cleanly with a representative context."""
        rendered = _render(template_path)
        assert len(rendered) > 0

    def test_no_undefined_variables(self, template_path: Path):
        """StrictUndefined ensures no unresolved {{ variable }} references remain."""
        # If any variable is missing, UndefinedError is raised during render.
        # This test passes if _render() does not raise.
        rendered = _render(template_path)
        # Double check: no raw Jinja2 delimiters should remain in the output
        assert "{{" not in rendered, "Un-rendered Jinja2 variable in output"
        assert "{%" not in rendered, "Un-rendered Jinja2 block tag in output"

    def test_output_format_present(self, template_path: Path):
        """Every template must include the JSON turns schema so the LLM knows the format."""
        rendered = _render(template_path)
        assert '"turns"' in rendered, "JSON output format schema missing"
        assert '"speaker_id"' in rendered, "speaker_id field missing from schema"
        assert '"intensity"' in rendered, "intensity field missing from schema"

    def test_speaker_ids_in_output(self, template_path: Path):
        """At least one speaker ID from the context should appear in the rendered prompt."""
        rendered = _render(template_path)
        ctx = _context_for(template_path)
        speaker_ids = [s["speaker_id"] for s in ctx["speakers"]]
        assert any(sid in rendered for sid in speaker_ids), (
            f"No speaker ID from context found in rendered template. Expected one of: {speaker_ids}"
        )

    def test_scene_id_in_output(self, template_path: Path):
        """The scene_id should be interpolated into the rendered prompt."""
        rendered = _render(template_path)
        ctx = _context_for(template_path)
        assert ctx["scene_id"] in rendered

    def test_intensity_arc_in_output(self, template_path: Path):
        """The intensity_arc should appear in the output."""
        rendered = _render(template_path)
        # Arc is rendered as a list like [1, 2, 3, 4, 5]
        assert "1" in rendered and "5" in rendered


# ---------------------------------------------------------------------------
# Coverage: partials are not rendered standalone
# ---------------------------------------------------------------------------


class TestTemplateDiscovery:
    def test_she_proves_templates_found(self):
        assert len(SHE_PROVES_TEMPLATES) >= 5, (
            f"Expected ≥5 she_proves templates, found {len(SHE_PROVES_TEMPLATES)}"
        )

    def test_elephant_templates_found(self):
        assert len(ELEPHANT_TEMPLATES) >= 5, (
            f"Expected ≥5 elephant templates, found {len(ELEPHANT_TEMPLATES)}"
        )

    def test_no_partials_in_list(self):
        """Partials (prefixed with _) must not be in the render list."""
        for tpl in ALL_TEMPLATES:
            assert not tpl.name.startswith("_"), (
                f"Partial file {tpl.name} should not appear in template list"
            )

    def test_all_templates_are_j2(self):
        for tpl in ALL_TEMPLATES:
            assert tpl.suffix == ".j2", f"{tpl.name} is not a .j2 file"

    def test_expected_she_proves_templates_exist(self):
        names = {p.name for p in SHE_PROVES_TEMPLATES}
        required = {
            "intimate_terror_coercive_control.j2",
            "intimate_terror_jealousy_surveillance.j2",
            "intimate_terror_financial_control.j2",
            "intimate_terror_isolation_tactics.j2",
            "intimate_terror_post_incident_honeymoon.j2",
            "sexual_violence_coercive_pressure.j2",
            "sexual_violence_threat_escalation.j2",
            "physical_assault_escalation.j2",
            "physical_threat_object_use.j2",
            "physical_threat_children_present.j2",
            "emotional_abuse_degradation.j2",
            "negative_argument_deescalation.j2",
            "negative_third_party_intervention.j2",
            "neutral_domestic_routine.j2",
            "neutral_family_coordination.j2",
        }
        missing = required - names
        assert not missing, f"Missing she_proves templates: {sorted(missing)}"

    def test_expected_elephant_templates_exist(self):
        names = {p.name for p in ELEPHANT_TEMPLATES}
        required = {
            "sv_physical_threat_to_worker.j2",
            "sv_object_brandished.j2",
            "sv_colleague_called_in.j2",
            "it_verbal_aggression_persistent.j2",
            "it_benefits_denial_escalation.j2",
            "it_child_custody_confrontation.j2",
            "neg_successful_deescalation.j2",
            "neg_supervisor_intervention.j2",
            "neu_routine_intake_session.j2",
            "neu_frustrated_waiting_client.j2",
            "neu_follow_up_appointment.j2",
        }
        missing = required - names
        assert not missing, f"Missing elephant templates: {sorted(missing)}"
