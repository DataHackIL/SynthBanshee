"""Shared pytest configuration for the SynthBanshee test suite.

Adds a ``--run-live-tts`` flag that gates tests marked
``@pytest.mark.live_tts`` (real TTS API calls — they cost real money).
Default ``pytest`` runs skip them; opt-in is explicit:

    pytest --run-live-tts                           # run live tests
    pytest tests/integration/test_google_tts.py --run-live-tts

This protects CI and contributors who happen to have ADC credentials
and the optional TTS SDKs installed from being billed on every test
run.  The existing ``-m "not live_tts"`` filter still works for the
inverse use case.
"""

from __future__ import annotations

import pytest

# Env vars consumed by `synthbanshee/cli.py` Click options as defaults.
# If these leak into a `CliRunner`-backed test, generated artifacts land
# in the developer's corpus tree instead of `tmp_path`. See #107.
_SYNTHBANSHEE_DIR_ENV_VARS = (
    "SYNTHBANSHEE_DATA_DIR",
    "SYNTHBANSHEE_CACHE_DIR",
    "SYNTHBANSHEE_SCRIPT_CACHE_DIR",
)


@pytest.fixture(autouse=True)
def _isolate_synthbanshee_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strip ``SYNTHBANSHEE_*`` dir env vars for the duration of every test.

    Click options in ``synthbanshee/cli.py`` read these as ``envvar=``
    defaults. Without this fixture, sourcing the repo's ``.envrc`` before
    running ``pytest`` causes ``CliRunner``-backed tests to write
    generated clips into the developer's corpus tree.
    """
    for name in _SYNTHBANSHEE_DIR_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-live-tts",
        action="store_true",
        default=False,
        help=(
            "Run tests marked @pytest.mark.live_tts (real TTS API calls; "
            "bills real money on services like Google Cloud TTS / Azure)."
        ),
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if config.getoption("--run-live-tts"):
        return
    skip_live = pytest.mark.skip(
        reason="live_tts test — pass --run-live-tts to run (real API call)"
    )
    for item in items:
        if "live_tts" in item.keywords:
            item.add_marker(skip_live)
