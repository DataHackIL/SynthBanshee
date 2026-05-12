"""Unit-test-specific pytest configuration.

Scoped to ``tests/unit/`` so the integration suite (which calls the
SpeakerConfig / SceneConfig Python APIs directly, never via CliRunner)
is unaffected. See #107 for the scope guard in the original issue.
"""

from __future__ import annotations

import pytest

# Env vars consumed by `synthbanshee/cli.py` Click options as defaults.
# If these leak into a ``CliRunner``-backed unit test, generated artifacts
# land in the developer's corpus tree (whatever ``.envrc`` points at)
# instead of ``tmp_path``. See #107 for the delivery-003 leak fingerprint.
_SYNTHBANSHEE_DIR_ENV_VARS = (
    "SYNTHBANSHEE_DATA_DIR",
    "SYNTHBANSHEE_CACHE_DIR",
    "SYNTHBANSHEE_SCRIPT_CACHE_DIR",
)


@pytest.fixture(autouse=True)
def _isolate_synthbanshee_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strip ``SYNTHBANSHEE_*`` dir env vars before every unit test.

    Function-scoped (not session-scoped) on purpose: individual test
    classes can override this fixture with a no-op to demonstrate the
    leak vector — see ``test_cli.py::TestSynthbansheeEnvVarLeakWithoutFixture``.
    A session-scoped strip would defeat that override pattern.
    """
    for name in _SYNTHBANSHEE_DIR_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
