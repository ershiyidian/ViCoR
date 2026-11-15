"""Deterministic episode seeding wrapper around GameWorld's GameEnv."""
from __future__ import annotations

from runtime import GameEnv


class SeededGameEnv(GameEnv):
    async def start(self):
        self.base_seed = self.config.random_seed
        self.episode_index = 0
        await self._start_episode()

    async def _start_episode(self):
        await super().start()
        await self._apply_episode_seed()
        if self.config.pause_during_inference:
            await self.pause_game()

    async def _apply_episode_seed(self):
        if self.config.random_seed is None:
            return
        page = self._require_game_manager().page
        capabilities = await page.evaluate(
            "() => window.gameAPI.capabilities"
        )
        if not isinstance(capabilities, dict):
            raise RuntimeError("GameWorld did not publish its control capabilities")
        self.seed_supported = bool(capabilities["supports_seed"])
        snapshot = await self.capture_state()
        if snapshot is None or snapshot.state is None:
            raise RuntimeError("The seeded game has no actual state")
        self.initial_game_seed = snapshot.state.get("seed")
        if not self.seed_supported:
            self.seed_applied = None
            return
        if self.initial_game_seed == self.config.random_seed:
            self.seed_applied = self.config.random_seed
            return
        if self.config.manual_start:
            # Initialize seed in manual-start mode while preserving the game's pre-start menu state.
            await page.evaluate(
                "async (seed) => await window.gameAPI.init({seed})",
                self.config.random_seed,
            )
            self.seed_applied = self.config.random_seed
        elif capabilities.get("supports_reload_reset") and not capabilities.get(
            "supports_inplace_reset"
        ):
            # Wait for page reload navigation when gameAPI.reset triggers a full reload.
            async with page.expect_navigation(wait_until="domcontentloaded"):
                await page.evaluate(
                    """(seed) => {
                        setTimeout(() => {
                            window.gameAPI.reset({seed});
                        }, 0);
                    }""",
                    self.config.random_seed,
                )
            await self._require_game_manager()._maybe_init_game_api()
            self.seed_applied = self.config.random_seed
        else:
            result = await page.evaluate(
                "async (seed) => await window.gameAPI.reset({seed: seed})",
                self.config.random_seed,
            )
            if not isinstance(result, dict) or result.get("ok") is not True:
                raise RuntimeError(f"Official game reset failed: {result}")
            self.seed_applied = (result.get("applied") or {}).get("seed")
        if not await self.wait_until_ready(
            stage="official_seed_reset", timeout_s=60.0
        ):
            raise RuntimeError("GameWorld did not become ready after seeded reset")
        # Wait for rendered canvas frames after reload-based reset.
        await self.wait_until_rendered()
        snapshot = await self.capture_state()
        if snapshot is None or snapshot.state is None:
            raise RuntimeError("The seeded game has no actual state")
        self.initial_game_seed = snapshot.state.get("seed")
        if self.seed_supported and self.initial_game_seed != self.config.random_seed:
            print(f"[WARN] GameWorld seed mismatch: applied {self.initial_game_seed} instead of {self.config.random_seed}; continuing.")

    async def reset_game(self) -> bool:
        self.episode_index += 1
        self.config.random_seed = (None if self.base_seed is None
                                   else self.base_seed + self.episode_index)
        # Each native episode uses the same fresh-browser startup as the first.
        await super().close_game()
        self._executors.clear()
        self._episode_id = self._next_episode_id()
        await self._start_episode()
        return True
