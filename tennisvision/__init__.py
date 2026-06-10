from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC_PACKAGE = _ROOT.parent / "src" / "tennisvision"

if _SRC_PACKAGE.is_dir():
    __path__ = [str(_SRC_PACKAGE)]
