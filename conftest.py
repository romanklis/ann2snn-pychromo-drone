"""Repo-root test bootstrap: make ``src``, ``tools`` and the repo root importable."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for _path in (ROOT / "src", ROOT / "tools", ROOT):
    _p = str(_path)
    if _p not in sys.path:
        sys.path.insert(0, _p)
