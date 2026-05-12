"""Shared helper for repo-relative path rendering in clip metadata
and manifest CSV. #108 — keep cli.py and manifest.py in sync.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def relative_to_data_root(path: Path, data_root: Path | None) -> str:
    """Render *path* as a string anchored at *data_root* when possible.

    Returns a POSIX-style relative string (``"data/he/clip.wav"``) when
    *path* resolves to a location under *data_root* (symlinks resolved
    on both sides). Falls back to a POSIX-style **absolute** string in
    every other case:

    - *data_root* is ``None``: nothing to anchor against.
    - *path* resolves outside *data_root*: also emits a
      ``logger.warning`` so a misconfigured ``--data-root`` is loud
      rather than silent.

    Both branches always call ``Path.resolve()`` and ``Path.as_posix()``
    so the JSON / CSV path shape is **stable** regardless of:

    - whether the caller passed a relative or absolute ``path``
      (#108 review: a relative input used to produce a relative-string
      fallback, contradicting the docstring and making metadata depend
      on the working directory),
    - the host OS (#108 review: Windows would otherwise emit
      backslashes; POSIX separators keep the corpus portable).
    """
    resolved = Path(path).resolve()
    if data_root is None:
        return resolved.as_posix()
    try:
        return resolved.relative_to(Path(data_root).resolve()).as_posix()
    except ValueError:
        logger.warning(
            "Path %s is outside data_root %s; recording absolute path. "
            "Configure --data-root / SYNTHBANSHEE_DATA_ROOT to fix.",
            path,
            data_root,
        )
        return resolved.as_posix()
