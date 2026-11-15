import asyncio
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any

from env.action_executor import ActionExecutor

MAIN_ACTION_PRIORITY = 101  # Configurable reflex priorities span 1..100


class ActionArbiter:
    rule_phases = frozenset({'playing'})

    def __init__(self, page, controls, events, *, game_env=None, max_steps=None):
        self.bind_page(page, controls)
        self.events = events
        self.lock = asyncio.Lock()
        self.epoch = 0
        self.lifecycle_epoch = 0
        self.sequence = 0
        self.episode_steps = 0
        self.episode_completed_steps = 0
        self.max_steps = max_steps
        self.game_env = game_env
        self.clock_env = (game_env if game_env is not None
                          and game_env.config.pause_during_inference else None)
        self.clock_actions: set[int] = set()
        self._inference_count = 0
        self._inference_started_at: float | None = None
        self._inference_paused_seconds = 0.0
        self._inference_changed = asyncio.Condition(self.lock)
        self.episode_id: str | None = None
        self.intermission_active = False
        self.intermission_until = 0.0
        self.playing = asyncio.Event()
        self.playing.set()
        self.current: dict[str, Any] | None = None
        self.timer: asyncio.Task | None = None
        self.passive_waits: dict[int, dict[str, Any]] = {}
        self.history: deque[dict[str, Any]] = deque(maxlen=256)
        self.scene_phase = "unknown"
        self.awaiting_start = False
        self.learning_paused = False
        self.main_busy = False
        self.completed: dict[int, dict[str, Any]] = {}
        self.completion_changed = asyncio.Condition()

    def bind_page(self, page, controls) -> None:
        self.page = page
        self.validator = ActionExecutor(page, controls)

    @property
    def can_run_reflexes(self) -> bool:
        return (self.scene_phase in self.rule_phases
                and not self.intermission_active and not self.awaiting_start)

    def main_input_after(self, frame) -> bool:
        return any(row['event'] == 'action_started' and row['source'] == 'main'
                   and not row.get('passive') and row['episode_id'] == frame.episode_id
                   and row['execution_started_at'] > frame.timestamp
                   for row in reversed(self.history))

    def _action_time(self) -> float:
        now = time.monotonic() if self._inference_started_at is None else self._inference_started_at
        return now - self._inference_paused_seconds

    async def _finish_inference(self) -> None:
        async with self.lock:
            self._inference_count -= 1
            last = self._inference_count == 0
            try:
                if (last and self.clock_actions and not self.awaiting_start
                        and not self.intermission_active and not self.learning_paused):
                    await self.clock_env.resume_game()
            finally:
                if last and self._inference_started_at is not None:
                    self._inference_paused_seconds += time.monotonic() - self._inference_started_at
                    self._inference_started_at = None
                self._inference_changed.notify_all()

    @asynccontextmanager
    async def inference(self):
        if self.clock_env is None:
            yield
            return
        entered = False
        try:
            async with self.lock:
                if self._inference_count == 0:
                    self._inference_started_at = time.monotonic()
                self._inference_count += 1
                entered = True
                if self._inference_count == 1:
                    pause = asyncio.create_task(self.clock_env.pause_game())
                    try:
                        await asyncio.shield(pause)
                    except asyncio.CancelledError:
                        await pause
                        raise
            yield
        finally:
            if entered:
                cleanup = asyncio.create_task(self._finish_inference())
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    await cleanup
                    raise

    async def _begin_clock_action(self, action_id: int, *, starts_episode: bool = False) -> None:
        if self.awaiting_start and starts_episode:
            if self.clock_env is None or not self._inference_count:
                await self.game_env.resume_game()
            self.awaiting_start = False
        elif (self.clock_env is not None and not self.clock_actions
              and not self.awaiting_start and not self._inference_count):
            await self.clock_env.resume_game()
        self.clock_actions.add(action_id)

    async def _end_clock_action(self, action_id: int) -> None:
        self.clock_actions.discard(action_id)
        if self.clock_env is not None and not self.clock_actions:
            if self.scene_phase == "ready":
                for _ in range(24):
                    if self.scene_phase == "playing" or self.clock_actions:
                        break
                    await asyncio.sleep(0.05)
            if not self.clock_actions:
                await self.clock_env.pause_game()

    async def _stop_clock_actions(self) -> None:
        self.clock_actions.clear()
        if self.clock_env is not None:
            await self.clock_env.pause_game()
        self._inference_changed.notify_all()

    async def capture_reflex_aftermath(self, frames, *, episode_id: str,
                                       duration_s: float = 0.25):
        """Capture the resulting scene without advancing the game clock during observation."""
        if not 0 < duration_s <= 1:
            raise ValueError('Effect window must be positive and at most 1 second')
        if episode_id != self.episode_id or self.intermission_active:
            return None
        if self.clock_env is None:
            await asyncio.sleep(duration_s)
        if episode_id != frames.episode_id or self.intermission_active:
            return None
        return await frames.capture(source='reflex_effect_after')

    def _budget_rejection(self, source: str, frame) -> dict | None:
        if self.max_steps is not None and self.episode_steps >= self.max_steps:
            return self.events.add(
                "action_rejected", source=source, reason="episode_action_budget",
                frame_id=frame.frame_id, episode_id=frame.episode_id,
                episode_steps=self.episode_steps, max_steps=self.max_steps)
        return None

    async def _start_passive_wait(self, action: dict[str, Any], *, source: str, frame) -> dict:
        # Caller holds the action lock; waiting advances the clock without holding inputs.
        duration = self.validator.resolve_action_duration(action)
        self.sequence += 1
        if source == "main":
            self.episode_steps += 1
        action_id = self.sequence
        await self._begin_clock_action(action_id)
        began = time.monotonic()
        record = self.events.add(
            "action_started", action_id=action_id, source=source,
            frame_id=frame.frame_id, episode_id=frame.episode_id,
            action=action, passive=True, episode_step=self.episode_steps,
            execution_started_at=began, control_epoch=self.epoch,
            observation_age_s=began-frame.timestamp)
        self.history.append(record)
        self.passive_waits[action_id] = {
            "action_id": action_id, "source": source, "keys": [],
            "started_at": began, "started_action_time": self._action_time(), "deadline": self._action_time() + duration,
            "episode_id": frame.episode_id,
            "frame_id": frame.frame_id, "action": action,
            "timer": asyncio.create_task(self._expire(action_id, self._action_time()+duration)),
        }
        return {"status": "executing", "action_id": action_id, "source": source,
                "control_epoch": self.epoch, "duration_s": duration,
                "execution_started_at": began}

    async def _stop_passive_waits(self, reason: str, sources: set[str] | None = None) -> list:
        timers = []
        for action_id, wait in list(self.passive_waits.items()):
            if sources is None or wait['source'] in sources:
                timers.append(wait['timer'])
                await self._release(reason, action_id=action_id)
        return timers

    async def _release(self, reason: str, *, action_id: int | None = None) -> None:
        current = self.current if action_id is None else self.passive_waits.pop(action_id, None)
        if current is None:
            return
        if action_id is not None and current['timer'] is not asyncio.current_task():
            current['timer'].cancel()
        for key in reversed(current["keys"]):
            await self.page.keyboard.up(key)
        for button in reversed(current.get("buttons", [])):
            await self.page.mouse.up(button=button)
        if action_id is None:
            self.current = None
        await self._end_clock_action(current["action_id"])
        if current["episode_id"] == self.episode_id:
            self.episode_completed_steps += 1
        record = self.events.add(
            "action_finished", action_id=current["action_id"],
            source=current["source"], reason=reason,
            episode_id=current["episode_id"], frame_id=current["frame_id"],
            action=current["action"], execution_started_at=current["started_at"],
            held_s=round(max(0.0, self._action_time() - current.get("started_action_time", current["started_at"])), 4),
            **({'passive': True} if action_id is not None else {}),
        )
        self.history.append(record)
        async with self.completion_changed:
            self.completed[current["action_id"]] = record
            if len(self.completed) > 256:
                self.completed.pop(next(iter(self.completed)))
            self.completion_changed.notify_all()

    async def wait_for_completion(self, action_id: int) -> dict[str, Any]:
        """Wait until the active action finishes by expiration, preemption, or reset."""
        async with self.completion_changed:
            await self.completion_changed.wait_for(lambda: action_id in self.completed)
            return self.completed[action_id]

    async def run_sequence(self, steps: list[dict[str, Any]], *, source: str,
                           frame, expected_epoch: int | None = None) -> dict[str, Any]:
        """Execute validated key presses in order, stopping on any interruption.

        Each step goes through the usual priority arbitration. Main-agent
        commands override reflexes; only a world transition can invalidate the
        sequence. No LLM request is made between steps.
        """
        if not isinstance(steps, list) or not steps:
            raise ValueError("press_sequence requires a nonempty list of steps")
        prepared = []
        for step in steps:
            if not isinstance(step, dict) or not isinstance(step.get("key"), str):
                raise ValueError("Each sequence step requires a permitted key")
            duration_ms = step.get("duration_ms", 150)
            if isinstance(duration_ms, bool) or not isinstance(duration_ms, (int, float)):
                raise ValueError("Sequence duration_ms must be numeric")
            if not 0 < duration_ms <= 2000:
                raise ValueError("Each sequence step must last 1–2000 milliseconds")
            action = {"action": "press_key", "key": step["key"],
                      "duration": duration_ms / 1000}
            inspected = self.validator.inspect_action(action)
            if not inspected["is_valid"]:
                raise ValueError(f"Illegal sequence step: {inspected['reason']}")
            prepared.append(action)

        outcomes = []
        sequence_epoch = self.lifecycle_epoch
        for action in prepared:
            if self.lifecycle_epoch != sequence_epoch:
                return {"status": "interrupted", "completed_steps": len(outcomes),
                        "steps": outcomes, "reason": "episode_changed",
                        "last_finished_at": outcomes[-1]["finished_at"] if outcomes else None}
            started = await self.submit(action, source=source, frame=frame,
                                        expected_epoch=expected_epoch)
            if started.get("status") != "executing":
                return {"status": "interrupted", "completed_steps": len(outcomes),
                        "steps": outcomes, "reason": started.get("reason", started.get("status")),
                        "last_finished_at": (outcomes[-1]["finished_at"] if outcomes else None)}
            ended = await self.wait_for_completion(started["action_id"])
            outcomes.append({"key": action["key"], "action_id": started["action_id"],
                             "reason": ended["reason"], "finished_at": ended["monotonic"]})
            if ended["reason"] != "duration_completed":
                return {"status": "interrupted", "completed_steps": len(outcomes),
                        "steps": outcomes, "reason": ended["reason"],
                        "last_finished_at": ended["monotonic"]}
        return {"status": "completed", "completed_steps": len(outcomes),
                "steps": outcomes, "last_finished_at": outcomes[-1]["finished_at"]}

    async def _expire(self, sequence: int, deadline: float) -> None:
        while True:
            await asyncio.sleep(max(0.0, deadline - self._action_time()))
            async with self.lock:
                await self._inference_changed.wait_for(
                    lambda: not self._inference_count
                    or (sequence not in self.passive_waits
                        and (self.current is None or self.current["action_id"] != sequence)))
                if (sequence not in self.passive_waits
                        and (self.current is None or self.current["action_id"] != sequence)):
                    return
                if self._action_time() < deadline:
                    continue
                if sequence in self.passive_waits:
                    await self._release("duration_completed", action_id=sequence)
                elif self.current is not None and self.current["action_id"] == sequence:
                    await self._release("duration_completed")
                return

    def _submission_block_reason(self, source: str) -> str | None:
        return None

    async def submit(self, action: dict[str, Any] | None, *, source: str, frame,
                     expected_epoch: int | None = None, priority: int = 0,
                     evidence: list | None = None,
                     expected_lifecycle_epoch: int | None = None) -> dict[str, Any]:
        normalized = None
        if action is not None:
            inspection = self.validator.inspect_action(action)
            if not inspection["is_valid"]:
                raise ValueError(f"Invalid game action: {inspection['reason']} {action}")
            normalized = inspection["normalized_action"]
            if normalized["action"] not in {"press_key", "press_keys", "wait", "click", "click_hold", "mouse_move", "drag", "scroll", "type"}:
                raise ValueError("Unsupported native GameWorld computer-use action")
        effective_priority = MAIN_ACTION_PRIORITY if source == "main" else priority
        async with self.lock:
            if (expected_lifecycle_epoch is not None
                    and expected_lifecycle_epoch != self.lifecycle_epoch):
                return self.events.add("action_rejected", source=source,
                                       reason="lifecycle_changed_while_waiting",
                                       frame_id=frame.frame_id, episode_id=frame.episode_id)
            if self.learning_paused:
                return self.events.add("action_rejected", source=source,
                                       reason="learning_paused", frame_id=frame.frame_id,
                                       episode_id=frame.episode_id)
            if source.startswith("rule:") and not self.can_run_reflexes:
                return self.events.add("action_rejected", source=source,
                                       reason=("awaiting_main_input" if self.awaiting_start
                                               else "not_in_gameplay"), phase=self.scene_phase,
                                       frame_id=frame.frame_id)
            if self.episode_id is not None and frame.episode_id != self.episode_id:
                return self.events.add("action_rejected", source=source,
                                       reason="episode_changed", frame_id=frame.frame_id,
                                       source_episode_id=frame.episode_id,
                                       current_episode_id=self.episode_id)
            if self.intermission_active:
                return self.events.add("action_rejected", source=source,
                                       reason="episode_intermission", frame_id=frame.frame_id,
                                       episode_id=frame.episode_id,
                                       remaining_s=max(0.0, self.intermission_until-time.monotonic()))
            rejected = self._budget_rejection(source, frame)
            if rejected is not None:
                return rejected
            if action is None:
                self.episode_steps += 1
                if self.clock_env is not None and not self.clock_actions and not self.awaiting_start:
                    try:
                        await self.clock_env.resume_game()
                    finally:
                        await self.clock_env.pause_game()
                self.episode_completed_steps += 1
                return self.events.add(
                    "decision_completed", status="no_action", reason="no_function_call",
                    source=source, action=None, frame_id=frame.frame_id,
                    episode_id=frame.episode_id, episode_step=self.episode_steps,
                    finished_at=time.monotonic())
            block_reason = self._submission_block_reason(source)
            if block_reason is not None:
                return self.events.add("action_rejected", source=source,
                                       reason=block_reason, frame_id=frame.frame_id,
                                       episode_id=frame.episode_id)
            if source.startswith('rule:') and self.main_input_after(frame):
                return self.events.add('action_rejected', source=source,
                                       reason='main_input_after_observation',
                                       frame_id=frame.frame_id, episode_id=frame.episode_id)
            if normalized['action'] == 'wait':
                return await self._start_passive_wait(normalized, source=source, frame=frame)
            # Record concurrent reflex actions while preserving main-agent override priority.
            intervened_during_reasoning = (expected_epoch is not None
                                            and expected_epoch != self.epoch)
            if (self.current and source != "main"
                    and self.current["priority"] > effective_priority):
                return self.events.add("action_rejected", source=source,
                                       reason="higher_priority_action_active", frame_id=frame.frame_id,
                                       blocking_source=self.current["source"])
            if (self.current and self.current["priority"] == effective_priority
                    and source != "main"):
                return {"status": "already_handling", "source": self.current["source"]}
            if self.timer:
                self.timer.cancel()
            await self._release("preempted_by_main" if source == "main" else "preempted")
            if source.startswith("rule:"):
                self.epoch += 1
            self.sequence += 1
            self.episode_steps += 1
            action_id = self.sequence
            # Validated for the current episode; main-agent dispatch starts manual-start games.
            await self._begin_clock_action(action_id, starts_episode=source == "main")
            keys = []
            buttons = []
            if normalized["action"] == "press_key":
                keys = [normalized["key"]]
            elif normalized["action"] == "press_keys":
                keys = normalized["keys"]
            began = time.monotonic()
            for key in keys:
                await self.page.keyboard.down(key)
            if normalized["action"] == "click":
                await self.page.mouse.click(normalized["x"], normalized["y"],
                                            button=normalized.get("button", "left"))
            elif normalized["action"] == "click_hold":
                button = normalized.get("button", "left")
                await self.page.mouse.move(normalized["x"], normalized["y"])
                await self.page.mouse.down(button=button)
                buttons = [button]
            elif normalized["action"] == "mouse_move":
                if "from_x" in normalized and "from_y" in normalized:
                    await self.page.mouse.move(normalized["from_x"], normalized["from_y"])
                await self.page.mouse.move(normalized["x"], normalized["y"])
            elif normalized["action"] == "scroll":
                await self.page.mouse.wheel(normalized.get("delta_x", 0),
                                            normalized.get("delta_y", 0))
            elif normalized["action"] in {"type", "drag"}:
                # Native timed operations complete under the input lock and
                # already include their duration; no further hold is needed.
                execute = (self.validator._execute_drag if normalized["action"] == "drag"
                           else self.validator._execute_type)
                try:
                    await execute(normalized)
                except (Exception, asyncio.CancelledError):
                    if normalized["action"] == "drag":
                        await self.page.mouse.up(button=normalized.get("button", "left"))
                    raise
            duration = (0.0 if normalized["action"] in {"type", "drag"}
                        else self.validator.resolve_action_duration(normalized))
            deadline = self._action_time() + duration
            self.current = {
                "action_id": action_id, "source": source, "keys": keys, "buttons": buttons,
                "priority": effective_priority, "started_at": began, "started_action_time": self._action_time(),
                "episode_id": frame.episode_id, "frame_id": frame.frame_id,
                "action": normalized, "duration_s": duration,
            }
            record = self.events.add(
                "action_started", action_id=action_id, source=source,
                frame_id=frame.frame_id, episode_id=frame.episode_id,
                control_epoch=self.epoch, effective_priority=effective_priority,
                episode_step=self.episode_steps,
                action=normalized,
                execution_started_at=began,
                intervened_during_reasoning=intervened_during_reasoning,
                observation_age_s=time.monotonic() - frame.timestamp,
                frame_to_keydown_s=time.monotonic() - frame.timestamp,
                dispatch_s=time.monotonic() - began, evidence=evidence or [],
            )
            self.history.append(record)
            self.timer = asyncio.create_task(self._expire(action_id, deadline))
            return {"status": "executing", "action_id": action_id, "source": source,
                    "control_epoch": self.epoch, "duration_s": duration,
                    "execution_started_at": began}

    async def reset(self, reason: str = "episode_reset") -> None:
        async with self.lock:
            self.epoch += 1
            self.lifecycle_epoch += 1
            if self.timer:
                self.timer.cancel()
                self.timer = None
            await self._release(reason)
            timers = await self._stop_passive_waits(reason)
            await self._stop_clock_actions()
        await asyncio.gather(*timers, return_exceptions=True)

    async def begin_intermission(self, seconds: float) -> float:
        self.intermission_active = True
        async with self.lock:
            self.epoch += 1
            self.lifecycle_epoch += 1
            if self.timer:
                self.timer.cancel()
                self.timer = None
            await self._release("episode_intermission")
            timers = await self._stop_passive_waits("episode_intermission")
            await self._stop_clock_actions()
            self.intermission_until = time.monotonic() + seconds
            self.playing.clear()
            until = self.intermission_until
        await asyncio.gather(*timers, return_exceptions=True)
        return until

    async def end_intermission(self) -> None:
        async with self.lock:
            self.epoch += 1
            self.lifecycle_epoch += 1
            self.intermission_active = False
            self.intermission_until = 0.0
            if not self.learning_paused:
                self.playing.set()

    async def wait_until_playing(self) -> None:
        await self.playing.wait()

    async def pause_for_learning(self) -> None:
        """Freeze normal input without changing the official game episode."""
        async with self.lock:
            self.learning_paused = True
            self.playing.clear()
            if self.timer:
                self.timer.cancel()
                self.timer = None
            await self._release("learning_paused")
            timers = await self._stop_passive_waits("learning_paused")
            await self._stop_clock_actions()
            self.events.add("action_control_paused_for_learning",
                            episode_id=self.episode_id)
        await asyncio.gather(*timers, return_exceptions=True)

    async def resume_from_learning(self) -> None:
        async with self.lock:
            self.learning_paused = False
            if self.clock_env is not None and not self.clock_actions:
                await self.clock_env.pause_game()
            if not self.intermission_active:
                self.playing.set()
            self.events.add("action_control_resumed_from_learning",
                            episode_id=self.episode_id)

    def set_episode(self, episode_id: str) -> None:
        if self.episode_id != episode_id:
            self.episode_steps = 0
            self.episode_completed_steps = 0
            self.awaiting_start = bool(self.game_env is not None
                                       and self.game_env.config.manual_start)
        self.episode_id = episode_id
        self.scene_phase = "unknown"

    async def set_scene_phase(self, phase: str) -> None:
        timers = []
        async with self.lock:
            self.scene_phase = phase
            if (phase not in self.rule_phases and self.current
                    and self.current["source"].startswith("rule:")):
                if self.timer:
                    self.timer.cancel()
                await self._release("left_gameplay")
            if phase not in self.rule_phases:
                sources = {wait['source'] for wait in self.passive_waits.values()
                           if wait['source'].startswith('rule:')}
                timers = await self._stop_passive_waits("left_gameplay", sources)
        await asyncio.gather(*timers, return_exceptions=True)

    async def disable_source(self, source: str, *, keep_waits: bool = False) -> None:
        async with self.lock:
            if self.current and self.current["source"] == source:
                if self.timer:
                    self.timer.cancel()
                await self._release("rule_disabled")
            timers = [] if keep_waits else await self._stop_passive_waits("rule_disabled", {source})
        await asyncio.gather(*timers, return_exceptions=True)


class LocalActionArbiter(ActionArbiter):
    """Wait briefly for an in-flight short reflex input to finish before dispatching."""

    def __init__(self, page, controls, events, **kwargs):
        super().__init__(page, controls, events, **kwargs)
        self.pending_main_commands = 0

    def _submission_block_reason(self, source: str) -> str | None:
        if source.startswith("rule:") and self.pending_main_commands:
            return "main_command_waiting"
        return None

    async def submit(self, action: dict[str, Any] | None, *, source: str, frame,
                     expected_epoch: int | None = None, priority: int = 0,
                     evidence: list | None = None,
                     expected_lifecycle_epoch: int | None = None) -> dict[str, Any]:
        waiting = False
        try:
            if source == "main" and action is not None:
                async with self.lock:
                    current = self.current
                    if (current is not None and current["source"].startswith("rule:")
                            and 0 < current["duration_s"] <= .5):
                        action_id = current["action_id"]
                        if expected_lifecycle_epoch is None:
                            expected_lifecycle_epoch = self.lifecycle_epoch
                        self.pending_main_commands += 1
                        waiting = True
                        began = time.monotonic()
                        self.events.add("main_action_waiting_for_local",
                                        action_id=action_id, source=source,
                                        blocking_source=current["source"],
                                        frame_id=frame.frame_id, episode_id=frame.episode_id)
                if waiting:
                    finished = await self.wait_for_completion(action_id)
                    self.events.add("main_action_local_wait_finished", action_id=action_id,
                                    episode_id=frame.episode_id, frame_id=frame.frame_id,
                                    reason=finished["reason"], held_s=finished["held_s"],
                                    waited_s=time.monotonic() - began)
                    if finished["reason"] in {"left_gameplay", "learning_paused"}:
                        return self.events.add(
                            "action_rejected", source=source,
                            reason="gameplay_ended_while_waiting",
                            frame_id=frame.frame_id, episode_id=frame.episode_id)
            return await super().submit(
                action, source=source, frame=frame, expected_epoch=expected_epoch,
                priority=priority, evidence=evidence,
                expected_lifecycle_epoch=expected_lifecycle_epoch)
        finally:
            if waiting:
                # Decrement without awaiting the lock so cancellation restores reflex dispatch immediately.
                self.pending_main_commands -= 1

