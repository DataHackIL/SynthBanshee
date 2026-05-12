"""Shared helper for repo-relative path rendering in clip metadata
and manifest CSV. #108 — keep cli.py and manifest.py in sync.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def relative_to_data_root(path: Path, data_root: Path | None) -> str:
    """Render *path* as a string relative to *data_root* when possible.

    Returns ``str(path.resolve().relative_to(data_root.resolve()))`` when
    *path* is genuinely under *data_root* (symlinks resolved on both
    sides), falling back to ``str(path)`` otherwise. *data_root* of
    ``None`` short-circuits to the absolute form. Emits a
    ``logger.warning`` on the out-of-root fallback so a misconfigured
    ``--data-root`` does not silently revert to the pre-#108 absolute-
    path behaviour.
    """
    if data_root is None:
        return str(path)
    try:
        return str(Path(path).resolve().relative_to(Path(data_root).resolve()))
    except ValueError:
        logger.warning(
            "Path %s is outside data_root %s; recording absolute path. "
            "Configure --data-root / SYNTHBANSHEE_DATA_ROOT to fix.",
            path,
            data_root,
        )
        return str(path)
