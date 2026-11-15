# Reflex rules

Complete real-time games and tasks faster with fewer turns by using `set_reflex` to bind tight, unobstructed target crops in `images` (never cropping your own player/arm; or use `"mode": "motion"` for generic moving objects) and declare reflex rules with `await api.run_rules(rules)`.
When the environment keeps running while you think, or once moving obstacles, enemies, balls, platforms, or target objects are present, do not poll or chase moving targets manually with `act`/`wait` (turn latency will miss them); instead position yourself at an intercept point, call `set_reflex` so the reflex reacts in real time when the target enters range, and call `list_reflexes(seconds)` to let it run and refine thresholds as needed.
On static puzzles, board games, or stationary turn-based tasks where nothing moves on its own while you think, play directly with `act` without setting a reflex.
