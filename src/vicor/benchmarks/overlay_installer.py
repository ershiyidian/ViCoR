"""Install ViCoR multi-benchmark adapters and task catalogs onto a GameWorld checkout."""

from __future__ import annotations

import shutil
from pathlib import Path


OVERLAYS_ROOT = Path(__file__).resolve().parent / "overlays"


def install_overlays(gameworld_root: Path) -> list[Path]:
    gameworld_root = Path(gameworld_root).resolve()
    if not (gameworld_root / "catalog").is_dir() or not (gameworld_root / "env").is_dir():
        raise FileNotFoundError(
            f"{gameworld_root} does not look like a GameWorld checkout (missing catalog/ or env/)."
        )
    copied: list[Path] = []
    for src in sorted(OVERLAYS_ROOT.rglob("*")):
        if src.is_dir() or "__pycache__" in src.parts:
            continue
        rel = src.relative_to(OVERLAYS_ROOT)
        dst = gameworld_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst)
    # Also place robodojo_sim_server.py next to env/robodojo_manager.py for standalone fallback
    sim_server = Path(__file__).resolve().parent / "robodojo/robodojo_sim_server.py"
    if sim_server.is_file():
        dst_srv = gameworld_root / "env/robodojo_sim_server.py"
        shutil.copy2(sim_server, dst_srv)
        copied.append(dst_srv)
    return copied
