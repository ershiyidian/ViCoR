import asyncio
import base64
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def sample_frame_timeline(frames: list[dict], count: int) -> list[dict]:
    ordered = sorted(frames, key=lambda row: row["timestamp"])
    if not ordered or count < 1:
        raise ValueError("Timeline window and sample count must be positive")
    if len(ordered) <= count:
        return ordered
    if count == 1:
        return [ordered[len(ordered) // 2]]
    start, end = ordered[0]["timestamp"], ordered[-1]["timestamp"]
    chosen = {}
    for index in range(count):
        timestamp = start + (end - start) * index / (count - 1)
        frame = min(ordered, key=lambda row: abs(row["timestamp"] - timestamp))
        chosen[frame["frame_id"]] = frame
    return sorted(chosen.values(), key=lambda row: row["timestamp"])


class Events:
    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.file = (directory / "events.jsonl").open("a", encoding="utf-8", buffering=1)
        self.rows: list[dict[str, Any]] = []

    def add(self, kind: str, **values: Any) -> dict[str, Any]:
        row = {"id": len(self.rows) + 1, "event": kind, "time": time.time(),
               "monotonic": time.monotonic(), **values}
        self.rows.append(row)
        self.file.write(json.dumps(row, ensure_ascii=False) + "\n")
        if kind in {"job_started", "rule_activated", "rule_failed", "job_completed",
                    "job_failed", "incident", "runtime_started"}:
            print(json.dumps(row, ensure_ascii=False), flush=True)
        return row

    def since(self, event_id: int, limit: int = 25) -> list[dict[str, Any]]:
        important = {"job_started", "job_completed", "job_failed", "rule_activated", "rule_restored",
                     "rule_restored_version", "rule_review_result", "rule_maintenance_action",
                     "rule_review_scheduled", "rule_action_result", "rule_action_followup",
                     "rule_notification", "skill_emergency_handoff",
                     "learning_pause_resumed", "main_learning_resumed",
                     "subagent_message",
                     "experience_remembered", "episode_result",
                     "rule_failed", "rule_disabled", "action_rejected", "incident",
                     "episode_reset", "episode_intermission_started"}
        pending = [row for row in self.rows if row["id"] > event_id]
        now = time.monotonic()
        milestones = [row for row in pending if row["event"] in important]
        interventions = [row for row in pending if row["event"] == "action_started"
                         and row["source"].startswith("rule:")]
        selected = milestones[-limit:]
        urgent = [row for row in milestones if row["event"] == "rule_notification"
                  and row["urgency"] == "high"][-5:]
        selected = list({row["id"]: row for row in [*selected, *urgent]}.values())
        if interventions:
            selected.extend(interventions[-3:])
        selected.sort(key=lambda row: row["id"])
        results = []
        for row in selected:
            if row["event"] == "episode_result":
                results.append({"id": row["id"], "event": row["event"],
                                "episode_id": row["episode_id"], "status": row["status"],
                                "actual_objectives": row["metrics"]["objective_target_results"],
                                "stop_reason": row["stop_reason"]})
            else:
                delivered = {key: value for key, value in row.items() if key != "frames"}
                if row["event"] == "rule_notification":
                    # Freshness metadata does not suppress unread event delivery.
                    delivered["age_seconds"] = round(now - row["monotonic"], 3)
                    delivered["currently_fresh"] = row.get("expires_monotonic", now) >= now
                results.append(delivered)
        if interventions:
            results.append({
                "id": interventions[-1]["id"],
                "event": "rule_intervention_summary",
                "count": len(interventions),
                "last_actions": [
                    {"source": row["source"], "action_id": row["action_id"],
                     "action": row["action"], "frame_id": row["frame_id"]}
                    for row in interventions[-3:]
                ],
            })
        return results

    def close(self) -> None:
        self.file.close()


@dataclass(frozen=True)
class Frame:
    frame_id: int
    timestamp: float
    received_at: float
    episode_id: str
    image: bytes
    phase: str = "unknown"
    scene_time_s: float | None = None


class FrameStream:
    def __init__(self, env, events: Events, fps: float = 15,
                 screenshot_every: int = 5):
        self.env = env
        self.events = events
        self.fps = fps
        self.screenshot_every = screenshot_every
        self.latest: Frame | None = None
        self.ring: deque[Frame] = deque(maxlen=max(16, int(fps * 14)))
        self.condition = asyncio.Condition()
        self.capture_lock = asyncio.Lock()
        self.session = None
        self.closed = False
        self.episode_id = env._episode_id
        self.frame_index = 0
        self.phase = "unknown"

    async def capture(self, source: str = "stream") -> Frame:
        async with self.capture_lock:
            started = time.monotonic()
            async with self.env._browser_lock:
                # Read scene clock and screenshot atomically without toggling pause state.
                clock = await self.env._require_game_manager().get_gameplay_clock()
                result = await self.session.send("Page.captureScreenshot", {
                    "format": "png", "fromSurface": True, "captureBeyondViewport": False,
                    "optimizeForSpeed": True,
                })
                episode_id = self.episode_id
            self.frame_index += 1
            frame = Frame(self.frame_index, started, time.monotonic(), episode_id,
                          base64.b64decode(result["data"]), self.phase,
                          scene_time_s=clock["active_real_time_ms"] / 1000.0)
            async with self.condition:
                self.latest = frame
                if (self.ring and self.ring[-1].scene_time_s == frame.scene_time_s
                        and self.ring[-1].image == frame.image):
                    self.ring[-1] = frame
                else:
                    self.ring.append(frame)
                self.condition.notify_all()
            path = (self.save(frame, reason="periodic")
                    if source == "stream" and frame.frame_id % self.screenshot_every == 0
                    else None)
            self.events.add("frame", frame_id=frame.frame_id,
                            captured_at=frame.timestamp, scene_time_s=frame.scene_time_s,
                            episode_id=frame.episode_id, capture_s=frame.received_at-started,
                            source=source, phase=frame.phase, saved_path=str(path) if path else None)
            return frame

    async def bind_page(self, page) -> None:
        self.session = await page.context.new_cdp_session(page)

    async def detach_page(self) -> None:
        session, self.session = self.session, None
        if session is not None:
            await session.detach()

    async def run(self) -> None:
        await self.bind_page(self.env._require_game_manager().page)
        try:
            while not self.closed:
                if self.phase in {"learning_paused", "outcome_review"}:
                    # When the game clock is frozen, main owns the next review
                    # turn. A new screenshot is explicitly captured per decision,
                    # rather than streaming thousands of identical paused frames.
                    await asyncio.sleep(.1)
                    continue
                started = time.monotonic()
                await self.capture()
                await asyncio.sleep(max(0, 1 / self.fps - (time.monotonic() - started)))
        finally:
            await self.detach_page()

    async def next(self, after: int = 0, *,
                   captured_after: float | None = None) -> Frame:
        async with self.condition:
            await self.condition.wait_for(
                lambda: self.closed or (
                    self.latest is not None and self.latest.frame_id > after
                    and (captured_after is None or self.latest.timestamp > captured_after)
                )
            )
            if self.closed:
                raise asyncio.CancelledError()
            return self.latest

    def save(self, frame: Frame, *, reason: str = "evidence") -> Path:
        directory = self.events.directory / "frames"
        directory.mkdir(exist_ok=True)
        path = directory / f"{frame.frame_id:07d}.png"
        if not path.exists():
            path.write_bytes(frame.image)
            self.events.add("frame_saved", frame_id=frame.frame_id,
                            episode_id=frame.episode_id, reason=reason,
                            path=str(path))
        return path

    async def suspend_for_reset(self) -> None:
        # Share capture lock so post-reset frames are never tagged with the prior episode.
        await self.capture_lock.acquire()

    def resume_after_reset(self) -> None:
        self.capture_lock.release()

    def set_episode(self, episode_id: str) -> None:
        # Never serve cached frames from a prior episode after reset.
        self.episode_id = episode_id
        self.latest = None
        self.ring.clear()
        self.phase = "unknown"

    def set_phase(self, phase: str) -> None:
        if phase != self.phase:
            self.phase = phase
            self.events.add("game_phase_changed", episode_id=self.episode_id, phase=phase)

    def incident(self, description: str, source_frame: Frame | None = None) -> dict[str, Any]:
        if self.latest is None:
            raise RuntimeError("No game frame captured yet")
        source = source_frame or self.latest
        recent = [f for f in self.ring if f.episode_id == source.episode_id
                  and source.timestamp - 6 <= f.timestamp <= source.timestamp]
        stride = max(1, len(recent) // 12)
        selected = recent[::stride]
        if not selected or selected[-1].frame_id != source.frame_id:
            selected.append(source)
        frames = [{"frame_id": f.frame_id, "timestamp": f.timestamp,
                   "path": str(self.save(f)), "episode_id": f.episode_id, "phase": f.phase}
                  for f in selected]
        return self.events.add("incident", description=description,
                               episode_id=source.episode_id, source_frame_id=source.frame_id, frames=frames)

    async def terminal_incident(self, description: str, *, is_terminal: bool = True) -> dict[str, Any]:
        # Share monotonic sequence numbering across terminal frames.
        if self.latest is None or self.session is None:
            raise RuntimeError("Browser page is not ready for terminal screenshot capture")
        started = time.monotonic()
        terminal = await self.capture(source="terminal")
        captured_at = terminal.received_at
        terminal_dir = self.events.directory / ("terminal" if is_terminal else "attempt_boundary")
        terminal_dir.mkdir(parents=True, exist_ok=True)
        terminal_path = self.save(terminal, reason="terminal")
        terminal_alias = terminal_dir / f"{terminal.episode_id}_{terminal.frame_id:07d}.png"
        terminal_alias.write_bytes(terminal.image)

        # Bound saved terminal frames while preserving earlier interaction history.
        candidates = [frame for frame in self.ring
                      if frame.episode_id == terminal.episode_id
                      and captured_at - 12 <= frame.timestamp <= captured_at]
        earlier = [frame for frame in candidates if frame.frame_id != terminal.frame_id
                   and frame.timestamp < captured_at - 2]
        dense = [frame for frame in candidates if frame.frame_id != terminal.frame_id
                 and frame.timestamp >= captured_at - 2]
        stride = max(1, (len(earlier) + 7) // 8)
        selected = earlier[::stride] + dense
        selected.append(terminal)
        selected.sort(key=lambda frame: frame.timestamp)
        frames = [
            {"frame_id": frame.frame_id, "timestamp": frame.timestamp,
             "path": str(self.save(frame, reason="incident")), "episode_id": frame.episode_id,
             "phase": frame.phase,
                "terminal": bool(is_terminal and frame.frame_id == terminal.frame_id),
                "attempt_boundary": bool(not is_terminal and frame.frame_id == terminal.frame_id)}
            for frame in selected
        ]
        return self.events.add(
            "incident", description=description,
            episode_id=terminal.episode_id, source_frame_id=terminal.frame_id,
            terminal_frame_id=terminal.frame_id if is_terminal else None,
            attempt_boundary_frame_id=terminal.frame_id if not is_terminal else None,
            terminal_screenshot=str(terminal_path),
            terminal_alias=str(terminal_alias),
            terminal_capture_s=captured_at - started,
            preterminal_dense_frames=len(dense), frames=frames,
        )

    async def stop(self) -> None:
        self.closed = True
        async with self.condition:
            self.condition.notify_all()
