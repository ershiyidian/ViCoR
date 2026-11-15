"""ViCoR: Visual Conditioned Reflexes for Real-Time VLM Agents."""

from __future__ import annotations

import sys
from pathlib import Path

__version__ = "0.1.0"

_VISTA_RUNTIME = str(Path(__file__).resolve().parent / "vista_runtime")
if sys.path[:1] != [_VISTA_RUNTIME]:
    if _VISTA_RUNTIME in sys.path:
        sys.path.remove(_VISTA_RUNTIME)
    sys.path.insert(0, _VISTA_RUNTIME)
