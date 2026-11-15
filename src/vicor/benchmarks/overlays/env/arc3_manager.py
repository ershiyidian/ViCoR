"""ARC-AGI-3 GameManager adapter exposing ARC-AGI-3 games as a standard GameWorld BrowserGameManager."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

_EXTRA_SITE = os.environ.get("ARC3_SITE_PACKAGES", "")
if _EXTRA_SITE and _EXTRA_SITE not in sys.path:
    sys.path.append(_EXTRA_SITE)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/arc3_mplconfig")
os.environ.setdefault("ONLY_RESET_LEVELS", "true")

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

ARC3_ENV_DIR = os.environ.get("ARC3_ENVIRONMENTS_DIR", str(Path.cwd() / "local/arc3/environments_dir"))
GRID_SIZE = 64
RENDER_SCALE = 8
DISPLAY_SIZE = GRID_SIZE * RENDER_SCALE  # 512x512

ARC3_PALETTE = np.array([
    [255, 255, 255],
    [204, 204, 204],
    [153, 153, 153],
    [102, 102, 102],
    [51, 51, 51],
    [0, 0, 0],
    [229, 58, 163],
    [255, 123, 204],
    [249, 60, 49],
    [30, 147, 255],
    [136, 216, 241],
    [255, 220, 0],
    [255, 133, 27],
    [146, 18, 49],
    [79, 204, 48],
    [163, 86, 214],
], dtype=np.uint8)


class Arc3CDPSession:
    def __init__(self, manager: "Arc3GameManager") -> None:
        self.manager = manager

    async def send(self, method: str, params: Optional[dict] = None) -> dict:
        if method == "Page.captureScreenshot":
            png_bytes = await self.manager.capture_png_bytes()
            return {"data": base64.b64encode(png_bytes).decode("ascii")}
        return {}

    async def detach(self) -> None:
        pass


class Arc3BrowserContext:
    def __init__(self, manager: "Arc3GameManager") -> None:
        self.manager = manager

    async def new_cdp_session(self, page: Any = None) -> Arc3CDPSession:
        return Arc3CDPSession(self.manager)


class Arc3Keyboard:
    def __init__(self, manager: "Arc3GameManager") -> None:
        self.manager = manager

    async def down(self, key: str) -> None:
        await asyncio.to_thread(self.manager._key_event, key, True)

    async def up(self, key: str) -> None:
        pass

    async def press(self, key: str) -> None:
        await self.down(key)

    async def type(self, text: str) -> None:
        for ch in str(text or ""):
            await self.down(ch)


class Arc3Mouse:
    def __init__(self, manager: "Arc3GameManager") -> None:
        self.manager = manager
        self.last_x = 256.0
        self.last_y = 256.0

    async def move(self, x: float, y: float) -> None:
        self.last_x, self.last_y = float(x), float(y)

    async def down(self, button: str = "left") -> None:
        await asyncio.to_thread(self.manager._click_event, self.last_x, self.last_y)

    async def up(self, button: str = "left") -> None:
        pass

    async def click(self, x: float, y: float, button: str = "left") -> None:
        await self.move(x, y)
        await self.down(button=button)

    async def wheel(self, delta_x: float = 0.0, delta_y: float = 0.0) -> None:
        pass


class Arc3VirtualPage:
    def __init__(self, manager: "Arc3GameManager") -> None:
        self.manager = manager
        self.context = Arc3BrowserContext(manager)
        self.keyboard = Arc3Keyboard(manager)
        self.mouse = Arc3Mouse(manager)
        self.main_frame = self
        self.frames = [self]
        self.viewport_size = {"width": DISPLAY_SIZE, "height": DISPLAY_SIZE}

    async def evaluate(self, script: str, arg: Any = None) -> Any:
        s = str(script or "")
        if "capabilities" in s:
            return {
                "supports_seed": False,
                "supports_reload_reset": True,
                "supports_inplace_reset": True,
            }
        if "window.gameAPI.reset" in s or "api.reset" in s:
            ok = await self.manager.reset_game()
            return {"ok": ok, "applied": {"seed": arg}}
        if "window.gameAPI.init" in s or "api.init" in s:
            return {"ok": True, "applied": {"seed": arg}}
        if "__getGameSpeedState" in s:
            return await self.manager.get_gameplay_clock()
        if "getState" in s:
            return await self.manager.get_game_state()
        return None


class Arc3GameManager:
    """Drop-in replacement for BrowserGameManager backed by ARC-AGI-3 offline toolkit."""

    def __init__(self, runtime_config: Any) -> None:
        self.runtime_config = runtime_config
        self.game_id = str(runtime_config.game_id)
        self.raw_game_id = self.game_id.removeprefix("arc3_")
        self.arcade: Optional[Arcade] = None
        self.scorecard_id: Optional[str] = None
        self.env: Any = None
        self.obs: Any = None
        self.page: Optional[Arc3VirtualPage] = None
        self.is_paused = False
        self.started_monotonic = time.monotonic()
        self.paused_accum_s = 0.0
        self.pause_started_at: Optional[float] = None
        self.actions_taken: int = 0
        self.resets_count: int = 0
        self.max_levels_seen: int = 0
        self.max_rhae_seen: float = 0.0
        self.screenshot_dir = Path(".screenshots_temp") / f"arc3_{uuid.uuid4().hex[:8]}"
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._last_png: Optional[bytes] = None
        self._last_png_key: Optional[tuple[int, int]] = None

    def _active_ms(self) -> float:
        now = time.monotonic()
        paused_extra = (now - self.pause_started_at) if self.pause_started_at is not None else 0.0
        active_s = max(0.0, (now - self.started_monotonic) - (self.paused_accum_s + paused_extra))
        return active_s * 1000.0

    def _init_env_sync(self) -> None:
        self.arcade = Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=ARC3_ENV_DIR,
        )
        self.scorecard_id = str(self.arcade.open_scorecard(tags=[]))
        self.env = self.arcade.make(self.raw_game_id, scorecard_id=self.scorecard_id)
        self.obs = getattr(self.env, "observation_space", None)
        if self.obs is None:
            self.obs = self.env.reset()
        self.actions_taken = 0
        self.resets_count = 0
        self.max_levels_seen = int(getattr(self.obs, "levels_completed", 0) or 0)
        self.max_rhae_seen = 0.0
        self._last_png = None
        self._last_png_key = None
        self.is_paused = False
        self.started_monotonic = time.monotonic()
        self.paused_accum_s = 0.0
        self.pause_started_at = None

    async def start(self) -> None:
        await asyncio.to_thread(self._init_env_sync)
        self.page = Arc3VirtualPage(self)

    def _step_env_sync(self, action: GameAction, data: Optional[dict[str, int]] = None) -> None:
        if self.env is None:
            return
        if self.obs is not None and getattr(self.obs, "state", None) == GameState.WIN:
            return
        if action == GameAction.RESET:
            nxt = self.env.reset()
            if nxt is not None:
                self.obs = nxt
            self.resets_count += 1
            self._last_png_key = None
            return
        if self.obs is not None and getattr(self.obs, "state", None) == GameState.GAME_OVER:
            # In ARC-AGI-3, GAME_OVER requires a level reset before new actions take effect.
            nxt = self.env.reset()
            if nxt is not None:
                self.obs = nxt
            self.resets_count += 1
            self._last_png_key = None
            return
        avail = set(getattr(self.env, "action_space", []) or [])
        if action not in avail:
            return
        nxt = self.env.step(action, data=data)
        if nxt is not None:
            self.obs = nxt
        self.actions_taken += 1
        self._last_png_key = None

    def _key_event(self, key: str, pressed: bool) -> None:
        if not pressed:
            return
        k = str(key or "").strip().lower()
        key_map = {
            "arrowup": GameAction.ACTION1,
            "up": GameAction.ACTION1,
            "w": GameAction.ACTION1,
            "1": GameAction.ACTION1,
            "arrowdown": GameAction.ACTION2,
            "down": GameAction.ACTION2,
            "s": GameAction.ACTION2,
            "2": GameAction.ACTION2,
            "arrowleft": GameAction.ACTION3,
            "left": GameAction.ACTION3,
            "a": GameAction.ACTION3,
            "3": GameAction.ACTION3,
            "arrowright": GameAction.ACTION4,
            "right": GameAction.ACTION4,
            "d": GameAction.ACTION4,
            "4": GameAction.ACTION4,
            "space": GameAction.ACTION5,
            "spacebar": GameAction.ACTION5,
            "enter": GameAction.ACTION5,
            "f": GameAction.ACTION5,
            "e": GameAction.ACTION5,
            "5": GameAction.ACTION5,
            "z": GameAction.ACTION7,
            "backspace": GameAction.ACTION7,
            "u": GameAction.ACTION7,
            "7": GameAction.ACTION7,
            "r": GameAction.RESET,
            "escape": GameAction.RESET,
            "esc": GameAction.RESET,
        }
        act = key_map.get(k)
        if act is not None:
            self._step_env_sync(act)

    def _click_event(self, x: float, y: float) -> None:
        gx = min(GRID_SIZE - 1, max(0, int(float(x)) * GRID_SIZE // DISPLAY_SIZE))
        gy = min(GRID_SIZE - 1, max(0, int(float(y)) * GRID_SIZE // DISPLAY_SIZE))
        self._step_env_sync(GameAction.ACTION6, data={"x": gx, "y": gy})

    def _render_png_sync(self) -> bytes:
        key = (self.actions_taken, self.resets_count)
        if self._last_png is not None and self._last_png_key == key:
            return self._last_png
        frames = getattr(self.obs, "frame", None)
        if frames and len(frames) > 0:
            grid = np.asarray(frames[-1], dtype=np.uint8) & 0xF
        else:
            grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        rgb = ARC3_PALETTE[grid]
        image = Image.fromarray(rgb, mode="RGB").resize(
            (DISPLAY_SIZE, DISPLAY_SIZE), Image.Resampling.NEAREST
        )
        # Draw subtle 8px cell grid lines matching VISTA render_frame_image(grid_lines=True)
        pixels = np.asarray(image, dtype=np.uint16).copy()
        mask = np.zeros(pixels.shape[:2], dtype=bool)
        mask[::RENDER_SCALE, :] = True
        mask[:, ::RENDER_SCALE] = True
        pixels[mask] = (pixels[mask] * 3) // 4
        image = Image.fromarray(pixels.astype(np.uint8), mode="RGB")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        self._last_png = buf.getvalue()
        self._last_png_key = key
        return self._last_png

    async def capture_png_bytes(self) -> bytes:
        return await asyncio.to_thread(self._render_png_sync)

    async def capture_screenshot(self, name: str) -> Path:
        png_bytes = await self.capture_png_bytes()
        target = self.screenshot_dir / name
        target.write_bytes(png_bytes)
        return target

    async def pause_game(self) -> dict:
        if not self.is_paused:
            self.is_paused = True
            self.pause_started_at = time.monotonic()
        ms = self._active_ms()
        return {"ok": True, "isPaused": True, "activeRealTimeMs": ms, "active_real_time_ms": ms}

    async def resume_game(self) -> dict:
        if self.is_paused:
            self.is_paused = False
            if self.pause_started_at is not None:
                self.paused_accum_s += time.monotonic() - self.pause_started_at
                self.pause_started_at = None
        ms = self._active_ms()
        return {"ok": True, "isPaused": False, "activeRealTimeMs": ms, "active_real_time_ms": ms}

    async def get_gameplay_clock(self) -> dict:
        ms = self._active_ms()
        return {
            "isPaused": self.is_paused,
            "speedMultiplier": 1.0,
            "activeRealTimeMs": ms,
            "active_real_time_ms": ms,
            "gameplayTimeMs": ms,
        }

    def _poll_state_sync(self) -> dict:
        levels_completed = int(getattr(self.obs, "levels_completed", 0) or 0)
        win_levels = int(getattr(self.obs, "win_levels", 1) or 1)
        if levels_completed > self.max_levels_seen:
            self.max_levels_seen = levels_completed
        rhae_score = self.max_rhae_seen
        try:
            if self.arcade and self.scorecard_id:
                sc = self.arcade.get_scorecard(self.scorecard_id)
                sc_val = float(getattr(sc, "score", 0.0) or 0.0)
                if sc_val > self.max_rhae_seen:
                    self.max_rhae_seen = sc_val
                rhae_score = self.max_rhae_seen
        except Exception:
            pass
        obs_state = getattr(self.obs, "state", None)
        is_win = (obs_state == GameState.WIN) or (self.max_levels_seen >= win_levels)
        return {
            "status": "win" if is_win else "playing",
            "is_actionable": not is_win,
            "seed": self.runtime_config.random_seed,
            "score": float(self.max_levels_seen),
            "game_state": {
                "score": float(self.max_levels_seen),
                "levels_completed": int(self.max_levels_seen),
                "win_levels": int(win_levels),
                "rhae_score": round(float(rhae_score), 2),
                "actions_taken": int(self.actions_taken),
                "resets": int(self.resets_count),
                "arc_state": str(obs_state),
            },
            "terminal": {
                "isTerminal": bool(is_win),
                "outcome": "success" if is_win else None,
            },
        }

    async def get_game_state(self) -> dict:
        return await asyncio.to_thread(self._poll_state_sync)

    async def wait_until_actionable(self, stage: str = "startup", timeout_s: float = 30.0, **kwargs: Any) -> bool:
        return True

    async def reset_game(self) -> bool:
        if self.env is not None and getattr(self.obs, "state", None) != GameState.WIN:
            await asyncio.to_thread(self._step_env_sync, GameAction.RESET)
            return True
        await asyncio.to_thread(self._init_env_sync)
        return True

    async def close(self) -> None:
        if self.arcade and self.scorecard_id:
            try:
                await asyncio.to_thread(self.arcade.close_scorecard, self.scorecard_id)
            except Exception:
                pass
        self.env = None
        self.arcade = None
