---
name: SynthBanshee local uv venv
description: All Python commands for this project must use the local uv venv at .venv — activate with `source .venv/bin/activate` from repo root, or prefix commands with `source .venv/bin/activate &&`
type: project
---

Use the local uv venv at `.venv` for all Python operations in this project.

Activate: `source .venv/bin/activate` (from repo root)
Install packages: `uv pip install <pkg>` (not plain `pip install` which may be slow)
Run tests: `source .venv/bin/activate && python -m pytest tests/ -v`
Run linting: `source .venv/bin/activate && ruff check .`

**Why:** User explicitly requested this. The system Python has incompatible torch/torchaudio versions. The venv uses uv and cpython-3.12.11.

**How to apply:** Always prefix python/pytest/ruff commands with `source .venv/bin/activate &&` when working in this repo.
