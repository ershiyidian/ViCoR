"""RoboDojo GameManager adapter exposing RoboDojo 3D dual-arm simulation via a 6-DoF Virtual Gamepad."""

from __future__ import annotations

import asyncio
import atexit
import base64
import json
import os
import socket
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Optional

DISPLAY_W = 640
DISPLAY_H = 720

_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

_PERSISTENT_SERVERS: dict[str, tuple[str, subprocess.Popen, Any]] = {}

_INSTANT_KEYS = frozenset({"space", "enter", "return", "1", "2", "3", "4", "f1", "f2", "f3", "f4", "h", "backspace", "v", "x"})


def _cleanup_persistent_servers() -> None:
    for task_name, (sim_url, proc, log_fp) in list(_PERSISTENT_SERVERS.items()):
        try:
            req = urllib.request.Request(
                f"{sim_url}/shutdown",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            _NO_PROXY_OPENER.open(req, timeout=3.0).read()
        except Exception:
            pass
        try:
            proc.wait(timeout=5.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        if log_fp is not None:
            try:
                log_fp.close()
            except Exception:
                pass
    _PERSISTENT_SERVERS.clear()


atexit.register(_cleanup_persistent_servers)


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class RoboDojoCDPSession:
    def __init__(self, manager: "RoboDojoGameManager") -> None:
        self.manager = manager

    async def send(self, method: str, params: Optional[dict] = None) -> dict:
        if method == "Page.captureScreenshot":
            png_bytes = await self.manager.capture_png_bytes()
            return {"data": base64.b64encode(png_bytes).decode("ascii")}
        return {}

    async def detach(self, *args: Any, **kwargs: Any) -> None:
        pass


class RoboDojoBrowserContext:
    def __init__(self, manager: "RoboDojoGameManager") -> None:
        self.manager = manager

    async def new_cdp_session(self, page: Any = None) -> RoboDojoCDPSession:
        return RoboDojoCDPSession(self.manager)


class RoboDojoKeyboard:
    def __init__(self, manager: "RoboDojoGameManager") -> None:
        self.manager = manager
        self._pressed: set[str] = set()
        self._chord: list[str] = []
        self._down_active_s: Optional[float] = None
        self._sent_sim_s: float = 0.0
        self._stream_task: Optional[asyncio.Task] = None
        self._io_lock = asyncio.Lock()
        self._hold_seq: int = 0
        self._active_hold_seq: int = 0

    async def _stream_loop(self) -> None:
        try:
            while self._pressed:
                await asyncio.sleep(0.26)
                if not self._pressed or self.manager.is_paused:
                    continue
                async with self._io_lock:
                    if not self._pressed or self.manager.is_paused or self._down_active_s is None:
                        continue
                    now_s = self.manager._active_ms() / 1000.0
                    elapsed_s = max(0.0, min(1.50, now_s - self._down_active_s))
                    unsent_s = elapsed_s - self._sent_sim_s
                    if unsent_s < 0.16:
                        continue
                    chunk = min(0.26, unsent_s)
                    keys_now = list(self._chord) if self._chord else list(self._pressed)
                    seq_now = self._active_hold_seq
                    self._sent_sim_s += chunk
                    try:
                        await asyncio.to_thread(self.manager._send_gamepad_sync, keys_now, chunk, seq_now)
                    except Exception as exc:
                        print(f"[RoboDojoKeyboard] warning: stream step {keys_now} failed: {exc}", flush=True)
        except asyncio.CancelledError:
            return

    async def down(self, key: str) -> None:
        k = str(key or "").strip()
        if not k:
            return
        if k.lower() in _INSTANT_KEYS:
            self._hold_seq += 1
            seq = self._hold_seq
            async with self._io_lock:
                try:
                    await asyncio.to_thread(self.manager._send_gamepad_sync, [k], 0.10, seq)
                except Exception as exc:
                    print(f"[RoboDojoKeyboard] warning: instant key {k} failed: {exc}", flush=True)
            return

        if not self._pressed:
            self._hold_seq += 1
            self._active_hold_seq = self._hold_seq
            self._down_active_s = self.manager._active_ms() / 1000.0
            self._sent_sim_s = 0.0
        self._pressed.add(k)
        if k not in self._chord:
            self._chord.append(k)
        if self._stream_task is None or self._stream_task.done():
            self._stream_task = asyncio.create_task(self._stream_loop())

    async def up(self, key: str) -> None:
        k = str(key or "").strip()
        if not k or k.lower() in _INSTANT_KEYS:
            return
        # Sample release timestamp BEFORE awaiting in-flight sim HTTP calls so HTTP latency never inflates hold duration!
        up_active_s = self.manager._active_ms() / 1000.0
        self._pressed.discard(k)
        if not self._pressed:
            task = self._stream_task
            self._stream_task = None
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            async with self._io_lock:
                keys = list(self._chord)
                self._chord.clear()
                down_s = self._down_active_s
                sent_s = self._sent_sim_s
                seq_now = self._active_hold_seq
                self._down_active_s = None
                self._sent_sim_s = 0.0
                if keys and down_s is not None:
                    total_intended = max(0.04, min(1.50, up_active_s - down_s))
                    rem = total_intended - sent_s
                    if sent_s == 0.0:
                        send_dt = total_intended
                    elif rem >= 0.02:
                        send_dt = rem
                    else:
                        send_dt = 0.0
                    if send_dt > 0.0:
                        try:
                            await asyncio.to_thread(self.manager._send_gamepad_sync, keys, send_dt, seq_now)
                        except Exception as exc:
                            print(f"[RoboDojoKeyboard] warning: flush step {keys} failed: {exc}", flush=True)

    async def press(self, key: str) -> None:
        await self.down(key)
        if str(key or "").strip().lower() not in _INSTANT_KEYS:
            await asyncio.sleep(0.08)
        await self.up(key)

    async def type(self, text: str) -> None:
        for ch in str(text or ""):
            await self.press(ch)


class RoboDojoMouse:
    def __init__(self, manager: "RoboDojoGameManager") -> None:
        self.manager = manager

    async def move(self, x: float, y: float) -> None:
        pass

    async def down(self, button: str = "left") -> None:
        pass

    async def up(self, button: str = "left") -> None:
        pass

    async def click(self, x: float, y: float, button: str = "left") -> None:
        pass

    async def wheel(self, delta_x: float = 0.0, delta_y: float = 0.0) -> None:
        pass


class RoboDojoVirtualPage:
    def __init__(self, manager: "RoboDojoGameManager") -> None:
        self.manager = manager
        self.context = RoboDojoBrowserContext(manager)
        self.keyboard = RoboDojoKeyboard(manager)
        self.mouse = RoboDojoMouse(manager)
        self.main_frame = self
        self.frames = [self]
        self.viewport_size = {"width": DISPLAY_W, "height": DISPLAY_H}

    async def evaluate(self, script: str, arg: Any = None) -> Any:
        s = str(script or "")
        if "capabilities" in s:
            return {
                "supports_seed": True,
                "supports_reload_reset": False,
                "supports_inplace_reset": True,
            }
        if "window.gameAPI.reset" in s or "api.reset" in s:
            seed_val = arg.get("seed") if isinstance(arg, dict) else arg
            ok = await self.manager.reset_game(seed=seed_val)
            return {"ok": ok, "applied": {"seed": seed_val}}
        if "window.gameAPI.init" in s or "api.init" in s:
            seed_val = arg.get("seed") if isinstance(arg, dict) else arg
            return {"ok": True, "applied": {"seed": seed_val}}
        if "__getGameSpeedState" in s:
            return await self.manager.get_gameplay_clock()
        if "getState" in s:
            return await self.manager.get_game_state()
        return None


class RoboDojoGameManager:
    def __init__(self, runtime_config: Any) -> None:
        self.runtime_config = runtime_config
        self.game_id = str(runtime_config.game_id)
        self.task_name = self.game_id.removeprefix("robodojo_")
        self.page: Optional[RoboDojoVirtualPage] = None
        self.sim_url: str = os.environ.get("ROBODOJO_SIM_URL", "").strip().rstrip("/")

        self.is_paused = False
        self.started_monotonic = time.monotonic()
        self.paused_accum_s = 0.0
        self.pause_started_at: Optional[float] = None
        self.screenshot_dir = Path(".screenshots_temp") / f"robodojo_{uuid.uuid4().hex[:8]}"
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    def _active_ms(self) -> float:
        now = time.monotonic()
        paused_extra = (now - self.pause_started_at) if self.pause_started_at is not None else 0.0
        active_s = max(0.0, (now - self.started_monotonic) - (self.paused_accum_s + paused_extra))
        return active_s * 1000.0

    def _http_get_json(self, path: str, timeout: float = 30.0) -> dict[str, Any]:
        req = urllib.request.Request(f"{self.sim_url}{path}", method="GET")
        with _NO_PROXY_OPENER.open(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _http_get_bytes(self, path: str, timeout: float = 30.0) -> bytes:
        req = urllib.request.Request(f"{self.sim_url}{path}", method="GET")
        with _NO_PROXY_OPENER.open(req, timeout=timeout) as resp:
            return resp.read()

    def _http_post_json(self, path: str, payload: dict[str, Any], timeout: float = 90.0) -> dict[str, Any]:
        raw = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.sim_url}{path}",
            data=raw,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _NO_PROXY_OPENER.open(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _ensure_server_sync(self) -> None:
        seed = getattr(self.runtime_config, "random_seed", None)
        if self.sim_url:
            info = self._http_get_json("/health", timeout=10.0)
            if info.get("ok"):
                self._http_post_json("/reset", {"advance_seed": False, "seed": seed}, timeout=120.0)
                return

        cached = _PERSISTENT_SERVERS.get(self.task_name)
        if cached is not None:
            cached_url, cached_proc, cached_log = cached
            if cached_proc.poll() is None:
                self.sim_url = cached_url
                try:
                    info = self._http_get_json("/health", timeout=30.0)
                    if info.get("ok"):
                        self._http_post_json("/reset", {"advance_seed": False, "seed": seed}, timeout=120.0)
                        return
                except Exception as exc:
                    print(f"[RoboDojoGameManager] cached server reset failed ({exc}); killing old PID {cached_proc.pid}", flush=True)
            try:
                cached_proc.kill()
                cached_proc.wait(timeout=5.0)
            except Exception:
                pass
            if cached_log is not None:
                try:
                    cached_log.close()
                except Exception:
                    pass
            _PERSISTENT_SERVERS.pop(self.task_name, None)
            time.sleep(2.0)

        ws_dir = os.environ.get("ROBODOJO_WS_DIR", str(Path.cwd() / "local/robodojo"))
        conda_env = os.environ.get("ROBODOJO_CONDA_ENV", "")
        py_bin = os.environ.get("ROBODOJO_PYTHON") or (
            os.path.join(conda_env, "bin", "python") if conda_env else sys.executable
        )
        bundled_server = Path(__file__).resolve().parent / "robodojo_sim_server.py"
        try:
            import vicor
            pkg_server = Path(vicor.__file__).resolve().parent / "benchmarks/robodojo/robodojo_sim_server.py"
            if pkg_server.is_file():
                bundled_server = pkg_server
        except Exception:
            pass
        ws_server = Path(ws_dir) / "robodojo_sim_server.py"
        server_script = os.environ.get(
            "ROBODOJO_SIM_SERVER",
            str(ws_server if ws_server.is_file() else bundled_server),
        )
        device_id = int(os.environ.get("ROBODOJO_DEVICE_ID", "0"))
        init_seed = int(seed if seed is not None else 0)
        port = _pick_free_port()
        self.sim_url = f"http://127.0.0.1:{port}"

        cache_root = Path(os.environ.get("VICOR_CACHE_DIR", str(Path.home() / ".cache/vicor")))
        (cache_root / "tmp").mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["OMNI_KIT_ACCEPT_EULA"] = "YES"
        env["CUDA_VISIBLE_DEVICES"] = str(device_id)
        env["ROBODOJO_PHYS_GPU"] = str(device_id)
        env["PYTHONNOUSERSITE"] = "1"
        if conda_env:
            env["PATH"] = f"{os.path.join(conda_env, 'bin')}:{env.get('PATH', '')}"
        env.setdefault("TMPDIR", str(cache_root / "tmp"))
        env.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg_cache"))
        env.setdefault("XDG_DATA_HOME", str(cache_root / "xdg_data"))
        env.setdefault("WARP_CACHE_PATH", str(cache_root / "warp"))
        for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
            env.pop(k, None)

        log_dir = cache_root / "robodojo_sim_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"sim_{self.task_name}_{port}.log"
        log_fp = open(log_path, "w", encoding="utf-8")

        cmd = [
            py_bin,
            server_script,
            "--task_name",
            self.task_name,
            "--env_cfg_type",
            os.environ.get("ROBODOJO_ENV_CFG", "arx_x5"),
            "--device_id",
            str(device_id),
            "--seed",
            str(init_seed),
            "--http_host",
            "127.0.0.1",
            "--http_port",
            str(port),
            "--headless",
            "--enable_cameras",
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            env=env,
        )

        deadline = time.monotonic() + 300.0
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                tail = ""
                try:
                    tail = Path(log_path).read_text(encoding="utf-8", errors="ignore")[-2000:]
                except Exception:
                    pass
                raise RuntimeError(
                    f"RoboDojo sim server exited prematurely (rc={proc.returncode}). Log tail:\n{tail}"
                )
            try:
                info = self._http_get_json("/health", timeout=2.0)
                if info.get("ok"):
                    _PERSISTENT_SERVERS[self.task_name] = (self.sim_url, proc, log_fp)
                    return
            except Exception:
                pass
            time.sleep(1.0)

        raise TimeoutError(f"Timed out waiting for RoboDojo sim server on {self.sim_url} (log: {log_path})")

    async def start(self) -> None:
        await asyncio.to_thread(self._ensure_server_sync)
        self.page = RoboDojoVirtualPage(self)
        self.started_monotonic = time.monotonic()
        self.paused_accum_s = 0.0
        self.pause_started_at = None
        self.is_paused = False

    def _send_gamepad_sync(self, keys: list[str], duration_s: float, hold_seq: int = 0) -> dict[str, Any]:
        return self._http_post_json(
            "/gamepad",
            {"keys": keys, "duration_s": float(duration_s), "hold_seq": int(hold_seq)},
            timeout=90.0,
        )

    async def capture_png_bytes(self) -> bytes:
        return await asyncio.to_thread(self._http_get_bytes, "/screenshot.png", 30.0)

    async def capture_screenshot(self, name: str) -> Path:
        png_bytes = await self.capture_png_bytes()
        target = self.screenshot_dir / name
        target.write_bytes(png_bytes)
        return target

    async def pause_game(self) -> dict:
        if not self.is_paused:
            self.is_paused = True
            self.pause_started_at = time.monotonic()
        if self.sim_url:
            try:
                await asyncio.to_thread(self._http_post_json, "/pause", {}, 10.0)
            except Exception:
                pass
        ms = self._active_ms()
        return {"ok": True, "isPaused": True, "activeRealTimeMs": ms, "active_real_time_ms": ms}

    async def resume_game(self) -> dict:
        if self.is_paused:
            self.is_paused = False
            if self.pause_started_at is not None:
                self.paused_accum_s += time.monotonic() - self.pause_started_at
                self.pause_started_at = None
        if self.sim_url:
            try:
                await asyncio.to_thread(self._http_post_json, "/resume", {}, 10.0)
            except Exception:
                pass
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

    async def get_game_state(self) -> dict:
        return await asyncio.to_thread(self._http_get_json, "/state", 30.0)

    async def wait_until_actionable(self, stage: str = "startup", timeout_s: float = 30.0, **kwargs: Any) -> bool:
        return True

    async def reset_game(self, seed: Optional[int] = None) -> bool:
        eff_seed = seed if seed is not None else getattr(self.runtime_config, "random_seed", None)
        payload: dict[str, Any] = {"advance_seed": eff_seed is None, "paused": self.is_paused}
        if eff_seed is not None:
            payload["seed"] = int(eff_seed)
        await asyncio.to_thread(self._http_post_json, "/reset", payload, 120.0)
        return True

    async def close(self) -> None:
        self.page = None
