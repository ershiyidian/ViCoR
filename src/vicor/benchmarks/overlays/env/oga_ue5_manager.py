"""UE5 GameManager adapter that exposes OmniGameArena UE5 scenes as a standard GameWorld BrowserGameManager."""

from __future__ import annotations
import fcntl

import asyncio
import base64
import io
import os
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from PIL import Image

OGA_ROOT = os.environ.get("OMNIGAMEARENA_ROOT", str(Path.cwd() / "local/omnigamearena/OmniGameArena"))
if OGA_ROOT not in sys.path:
    sys.path.insert(0, OGA_ROOT)

from omni_game_arena.env.client_ue5 import UE5Client
from omni_game_arena.utils.mouse_calibration import degrees_to_mouse_units


OGA_3D_CAMERA_GAMES = {
    "oga_last_stand",
    "oga_monster_shoot",
    "oga_cue_chase",
    "oga_scene_escape",
    "oga_obstacle_run_3d",
}


class UE5CDPSession:
    def __init__(self, manager: "UE5GameManager") -> None:
        self.manager = manager

    async def send(self, method: str, params: Optional[dict] = None) -> dict:
        if method == "Page.captureScreenshot":
            png_bytes = await self.manager.capture_png_bytes()
            return {"data": base64.b64encode(png_bytes).decode("ascii")}
        return {}

    async def detach(self) -> None:
        pass


class UE5BrowserContext:
    def __init__(self, manager: "UE5GameManager") -> None:
        self.manager = manager

    async def new_cdp_session(self, page: Any = None) -> UE5CDPSession:
        return UE5CDPSession(self.manager)


class UE5Keyboard:
    def __init__(self, manager: "UE5GameManager") -> None:
        self.manager = manager

    async def down(self, key: str) -> None:
        await asyncio.to_thread(self.manager._key_event, key, True)

    async def up(self, key: str) -> None:
        await asyncio.to_thread(self.manager._key_event, key, False)

    async def press(self, key: str) -> None:
        await self.down(key)
        await asyncio.sleep(0.08)
        await self.up(key)


class UE5Mouse:
    def __init__(self, manager: "UE5GameManager") -> None:
        self.manager = manager
        self.last_x = 320.0
        self.last_y = 180.0

    async def move(self, x: float, y: float) -> None:
        dx_deg = ((float(x) - 320.0) / 320.0) * 30.0
        dy_deg = ((180.0 - float(y)) / 180.0) * 18.0
        ux = int(round(degrees_to_mouse_units(dx_deg)))
        uy = int(round(degrees_to_mouse_units(dy_deg)))
        if ux != 0 or uy != 0:
            await asyncio.to_thread(self.manager.client.send_mouse_move, ux, uy)
        self.last_x, self.last_y = float(x), float(y)

    async def down(self, button: str = "left") -> None:
        await asyncio.to_thread(self.manager.client.send_mouse_button, button, True)

    async def up(self, button: str = "left") -> None:
        await asyncio.to_thread(self.manager.client.send_mouse_button, button, False)

    async def click(self, x: float, y: float, button: str = "left") -> None:
        await self.move(x, y)
        await self.down(button=button)
        await asyncio.sleep(0.08)
        await self.up(button=button)

    async def wheel(self, delta_x: float = 0.0, delta_y: float = 0.0) -> None:
        pass


class UE5VirtualPage:
    def __init__(self, manager: "UE5GameManager") -> None:
        self.manager = manager
        self.context = UE5BrowserContext(manager)
        self.keyboard = UE5Keyboard(manager)
        self.mouse = UE5Mouse(manager)
        self.main_frame = self
        self.frames = [self]
        self.viewport_size = {"width": 640, "height": 360}

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



def _ensure_local_ue5_running(host: str, port: int) -> None:
    import socket, subprocess, time
    try:
        with socket.create_connection((host, port), timeout=1.5):
            return
    except OSError:
        pass
    ue5_sh = os.environ.get("OGA_UE5_SH", str(Path.cwd() / "local/omnigamearena/Linux/OmniGameArena.sh"))
    if not Path(ue5_sh).is_file():
        raise FileNotFoundError(f"OmniGameArena UE5 binary launcher not found at {ue5_sh}. Set OGA_UE5_SH.")
    subprocess.Popen(
        [ue5_sh, "-RenderOffScreen", "-graphicsadapter=1", "-ResX=640", "-ResY=360", f"-RemoteInputPort={port}"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    for _ in range(45):
        try:
            with socket.create_connection((host, port), timeout=1.5):
                time.sleep(2.5)
                return
        except OSError:
            time.sleep(1.0)


class UE5GameManager:
    """Drop-in replacement for BrowserGameManager backed by UE5Client (port 12345)."""

    def __init__(self, runtime_config: Any, host: str = "127.0.0.1", port: int = 12345) -> None:
        self.runtime_config = runtime_config
        self.game_id = str(runtime_config.game_id)
        raw_map = self.game_id.removeprefix("oga_")
        maps_yaml = Path(OGA_ROOT) / "configs/maps.yaml"
        map_lookup = {}
        if maps_yaml.exists():
            import yaml as _yaml
            map_lookup = (_yaml.safe_load(maps_yaml.read_text()) or {}).get("maps", {})
        self.map_name = map_lookup.get(raw_map, raw_map)
        self.host = host
        self.port = port
        self.client: Optional[UE5Client] = None
        self.page: Optional[UE5VirtualPage] = None
        self.browser = SimpleNamespace(version="ue5-offscreen")
        self.is_paused = False
        self.started_monotonic = time.monotonic()
        self.paused_accum_s = 0.0
        self.pause_started_at: Optional[float] = None
        self.max_score_seen: float = 0.0
        self.held_keys: set[str] = set()
        self.screenshot_dir = Path(".screenshots_temp") / f"oga_{uuid.uuid4().hex[:8]}"
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._last_png: Optional[bytes] = None
        self._last_state_poll: float = 0.0
        self._cached_state: Optional[dict] = None

    def _active_ms(self) -> float:
        now = time.monotonic()
        paused_extra = (now - self.pause_started_at) if self.pause_started_at is not None else 0.0
        active_s = max(0.0, (now - self.started_monotonic) - (self.paused_accum_s + paused_extra))
        return active_s * 1000.0

    def _connect_and_open_map(self) -> None:
        if self.client:
            try:
                self.client.disconnect()
            except Exception:
                pass
        self.client = UE5Client(host=self.host, port=self.port, screenshot_quality=85)
        self.client.strict_response_matching = True
        _ensure_local_ue5_running(self.host, self.port)
        if not self.client.connect():
            _ensure_local_ue5_running(self.host, self.port)
            if not self.client.connect():
                raise RuntimeError(f"Failed to connect to UE5 server at {self.host}:{self.port}")
        self.client.resume_world()
        self.client.open_map(self.map_name)
        time.sleep(3.0)
        if not self.client.connected:
            _ensure_local_ue5_running(self.host, self.port)
            if not self.client.reconnect():
                raise RuntimeError("Failed to reconnect to UE5 after map switch")
            self.client.strict_response_matching = True
            time.sleep(0.5)
        self.client.reset_level()
        time.sleep(2.0)
        if not self.client.connected:
            if not self.client.reconnect():
                raise RuntimeError("Failed to reconnect to UE5 after reset_level")
            self.client.strict_response_matching = True
            time.sleep(0.5)
        self.client.game_over = False
        self.client.resume_world()
        self.is_paused = False
        self.started_monotonic = time.monotonic()
        self.paused_accum_s = 0.0
        self.pause_started_at = None
        self.max_score_seen = 0.0
        self.held_keys.clear()
        self._cached_state = None

    async def start(self) -> None:
        if getattr(self, '_lock_fd', None) is None:
            self._lock_fd = open(f'/tmp/oga_ue5_{self.port}.lock', 'w')
            await asyncio.to_thread(fcntl.flock, self._lock_fd, fcntl.LOCK_EX)
        await asyncio.to_thread(self._connect_and_open_map)
        self.page = UE5VirtualPage(self)

    def _release_all_sync(self) -> None:
        if not self.client or not self.client.connected:
            return
        for k in list(self.held_keys):
            try:
                if k == "LMB":
                    self.client.send_mouse_button("left", False)
                else:
                    self.client.send_key(k, False)
            except Exception:
                pass
        self.held_keys.clear()

    def _key_event(self, key: str, pressed: bool) -> None:
        if not self.client or not self.client.connected:
            return
        k_low = key.lower()
        if self.game_id in OGA_3D_CAMERA_GAMES and k_low in ("arrowleft", "arrowright", "arrowup", "arrowdown"):
            if pressed:
                deg_x = -25.0 if k_low == "arrowleft" else (25.0 if k_low == "arrowright" else 0.0)
                deg_y = 15.0 if k_low == "arrowup" else (-15.0 if k_low == "arrowdown" else 0.0)
                ux = int(round(degrees_to_mouse_units(deg_x))) if deg_x else 0
                uy = int(round(degrees_to_mouse_units(deg_y))) if deg_y else 0
                self.client.send_mouse_move(ux, uy)
            return
        if k_low == "x":
            self.client.send_mouse_button("left", pressed)
            if pressed:
                self.held_keys.add("LMB")
            else:
                self.held_keys.discard("LMB")
            return
        ue_map = {
            "w": "W",
            "a": "A",
            "s": "S",
            "d": "D",
            "f": "F",
            "q": "Q",
            "e": "E",
            "space": "SpaceBar",
            "shift": "LeftShift",
            "ctrl": "LeftControl",
            "control": "LeftControl",
            "alt": "LeftAlt",
            "enter": "Enter",
            "escape": "Escape",
            "esc": "Escape",
            "arrowleft": "A",
            "arrowright": "D",
            "arrowup": "W",
            "arrowdown": "S",
        }
        ue_key = ue_map.get(k_low, key)
        self.client.send_key(ue_key, pressed)
        if pressed:
            self.held_keys.add(ue_key)
        else:
            self.held_keys.discard(ue_key)

    def _capture_png_sync(self) -> bytes:
        res = self.client.screenshot()
        im = Image.open(io.BytesIO(res["data"])).convert("RGB")
        if im.size != (640, 360):
            im = im.resize((640, 360), Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        self._last_png = buf.getvalue()
        return self._last_png

    async def capture_png_bytes(self) -> bytes:
        try:
            return await asyncio.to_thread(self._capture_png_sync)
        except Exception:
            if self._last_png is not None:
                return self._last_png
            buf = io.BytesIO()
            Image.new("RGB", (640, 360), (20, 20, 20)).save(buf, format="PNG")
            return buf.getvalue()

    async def capture_screenshot(self, name: str) -> Path:
        png_bytes = await self.capture_png_bytes()
        target = self.screenshot_dir / name
        target.write_bytes(png_bytes)
        return target

    async def pause_game(self) -> dict:
        if self.client and self.client.connected and not self.is_paused:
            await asyncio.to_thread(self._release_all_sync)
            await asyncio.to_thread(self.client.pause_world)
            self.is_paused = True
            self.pause_started_at = time.monotonic()
        ms = self._active_ms()
        return {"ok": True, "isPaused": True, "is_paused": True, "speedMultiplier": 1.0, "speed_multiplier": 1.0, "activeRealTimeMs": ms, "active_real_time_ms": ms, "gameplayTimeMs": ms, "gameplay_time_ms": ms}

    async def resume_game(self) -> dict:
        if self.client and self.client.connected and self.is_paused:
            await asyncio.to_thread(self.client.resume_world)
            self.is_paused = False
            if self.pause_started_at is not None:
                self.paused_accum_s += time.monotonic() - self.pause_started_at
                self.pause_started_at = None
        ms = self._active_ms()
        return {"ok": True, "isPaused": False, "is_paused": False, "speedMultiplier": 1.0, "speed_multiplier": 1.0, "activeRealTimeMs": ms, "active_real_time_ms": ms, "gameplayTimeMs": ms, "gameplay_time_ms": ms}

    async def get_gameplay_clock(self) -> dict:
        ms = self._active_ms()
        return {
            "isPaused": self.is_paused,
            "is_paused": self.is_paused,
            "speedMultiplier": 1.0,
            "speed_multiplier": 1.0,
            "activeRealTimeMs": ms,
            "active_real_time_ms": ms,
            "gameplayTimeMs": ms,
            "gameplay_time_ms": ms,
        }

    def _poll_state_sync(self) -> dict:
        now = time.monotonic()
        if self._cached_state is not None and (now - self._last_state_poll) < 0.25:
            return self._cached_state
        self._last_state_poll = now
        try:
            self.client.get_score()
        except Exception:
            pass
        try:
            if not self.client.game_over:
                self.client.check_game_over(timeout=0.3)
        except Exception:
            pass
        sc = float(self.client.score or 0.0)
        if sc > self.max_score_seen:
            self.max_score_seen = sc
        target = 10.0 if "solocraft" in self.map_name.lower() or "solo_craft" in self.game_id else 1.0
        is_win = self.max_score_seen >= target - 1e-4
        is_lose = bool(self.client.game_over) and not is_win
        is_term = is_win or is_lose
        state = {
            "status": "win" if is_win else ("gameover" if is_lose else "playing"),
            "is_actionable": not is_term,
            "seed": self.runtime_config.random_seed,
            "score": self.max_score_seen,
            "game_state": {
                "score": self.max_score_seen,
                "max_score_seen": self.max_score_seen,
                "survival_time": float(self.client.survival_time or 0.0),
            },
            "terminal": {
                "isTerminal": is_term,
                "outcome": "success" if is_win else ("fail" if is_lose else None),
            },
        }
        self._cached_state = state
        return state

    async def get_game_state(self) -> dict:
        return await asyncio.to_thread(self._poll_state_sync)

    async def wait_until_actionable(self, stage: str = "startup", timeout_s: float = 30.0, **kwargs: Any) -> bool:
        return True

    async def reset_game(self) -> bool:
        await asyncio.to_thread(self._connect_and_open_map)
        return True

    async def close(self) -> None:
        if self.client:
            try:
                await asyncio.to_thread(self._release_all_sync)
                await asyncio.to_thread(self.client.resume_world)
                await asyncio.to_thread(self.client.disconnect)
            except Exception:
                pass
            self.client = None
        if getattr(self, '_lock_fd', None) is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                self._lock_fd.close()
            except Exception:
                pass
            self._lock_fd = None
