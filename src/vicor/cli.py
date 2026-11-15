"""Command-line interface for ViCoR (`vicor run`, `vicor list-tasks`, `vicor install-overlays`)."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path

from vicor.benchmarks.overlay_installer import install_overlays
from vicor.core.session_context import DEFAULT_LONG_CONTEXT_TRIGGER_TOKENS


BENCHMARK_PREFIXES = {
    "gameworld": ("0", "1", "2"),
    "aigamestore": ("aigamestore_",),
    "omnigamearena": ("oga_",),
    "arc3": ("arc3_",),
    "robodojo": ("robodojo_",),
}


def resolve_gameworld_root(explicit: str | None = None) -> Path:
    candidates = [
        explicit,
        os.environ.get("VICOR_GAMEWORLD_ROOT"),
        os.environ.get("GAMEWORLD_ROOT"),
        str(Path.cwd() / "local/gameworld/GameWorld"),
        str(Path.cwd() / "vendor/GameWorld"),
    ]
    for cand in candidates:
        if cand and (Path(cand) / "catalog").is_dir() and (Path(cand) / "runtime").is_dir():
            return Path(cand).resolve()
    raise FileNotFoundError(
        "Could not locate GameWorld runtime directory. Run `bash scripts/env_setup/gameworld.sh` "
        "or set `VICOR_GAMEWORLD_ROOT=/path/to/GameWorld`."
    )


def activate_gameworld_path(gw_root: Path) -> None:
    if str(gw_root) not in sys.path:
        sys.path.insert(0, str(gw_root))


async def _run_agent(args: argparse.Namespace) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    gw_root = resolve_gameworld_root(args.gameworld_root)
    activate_gameworld_path(gw_root)

    from catalog import build_runtime_config
    from utils import setup_logging
    from vicor.benchmarks.seeded_env import SeededGameEnv
    from vicor.core.vista_reflex import VISTAReflexCoordinator

    setup_logging()
    logging.getLogger().setLevel(logging.WARNING)
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=args.resume)

    reflex_endpoint = args.reflex_endpoint or os.environ.get("VICOR_REFLEX_ENDPOINT") or args.endpoint
    reflex_model = args.reflex_model or os.environ.get("VICOR_REFLEX_MODEL") or args.model

    VISTAReflexCoordinator.reflex_mode = args.mode
    VISTAReflexCoordinator.reflex_endpoint = reflex_endpoint
    VISTAReflexCoordinator.reflex_model_name = reflex_model
    VISTAReflexCoordinator.baseline_style = "vista"

    config = build_runtime_config(f"{args.game}+{args.task}+qwen3.5-35b-a3b-cua")
    config.pause_during_inference = args.pause_during_inference
    config.manual_start = args.manual_start
    config.max_steps = args.max_steps
    config.random_seed = args.seed
    config.experiment_episode_limit = args.episodes
    config.log_root = str(output)
    config.model_ids = [args.model]

    run_cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    run_cfg["resolved_gameworld_root"] = str(gw_root)
    run_cfg["resolved_reflex_endpoint"] = reflex_endpoint
    run_cfg["resolved_reflex_model"] = reflex_model
    (output / "config.json").write_text(json.dumps(run_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    env = SeededGameEnv(config, headless=not args.headed, port=args.port)
    coordinator = VISTAReflexCoordinator(
        env,
        config,
        endpoint=args.endpoint,
        model=args.model,
        output=output,
        fps=args.fps,
        instruction=args.instruction,
        main_effort=args.effort,
        keep_playing=args.keep_playing,
        tokenizer_file=Path(args.tokenizer_file).resolve() if args.tokenizer_file else None,
        intermission_seconds=args.intermission_seconds,
        screenshot_every=args.screenshot_every,
        main_compaction_threshold=args.compaction_threshold,
        single_attempt=args.single_attempt,
        api_backend=args.backend,
        codex_config=args.codex_config,
        codex_auth=args.codex_auth,
        api_slot_dir=args.api_slot_dir,
        api_slot_limit=args.api_slot_limit,
    )
    summary = await coordinator.run()
    print("VICOR_RUN_COMPLETED", json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def _cmd_list_tasks(args: argparse.Namespace) -> None:
    gw_root = resolve_gameworld_root(args.gameworld_root)
    tasks_root = gw_root / "catalog/tasks"
    rows: list[tuple[str, str, str]] = []
    for td in sorted(tasks_root.iterdir()):
        if not td.is_dir() or td.name.startswith("_"):
            continue
        for yf in sorted(td.glob("*.yaml")):
            text = yf.read_text(encoding="utf-8")
            game_id = td.name
            task_id = yf.stem
            for line in text.splitlines():
                if line.startswith("game_id:"):
                    game_id = line.split(":", 1)[1].strip().strip("'\"")
                elif line.startswith("task_id:"):
                    task_id = line.split(":", 1)[1].strip().strip("'\"")
            bench = "gameworld"
            for bname, prefixes in BENCHMARK_PREFIXES.items():
                if bname != "gameworld" and game_id.startswith(prefixes):
                    bench = bname
                    break
            if args.benchmark == "all" or args.benchmark == bench:
                rows.append((bench, game_id, task_id))
    for bench, game_id, task_id in rows:
        print(f"{bench:15s}  {game_id:38s}  {task_id}")
    print(f"Total: {len(rows)} tasks")


def _cmd_install_overlays(args: argparse.Namespace) -> None:
    copied = install_overlays(Path(args.gameworld_root))
    print(f"Installed {len(copied)} ViCoR multi-benchmark overlay files into {Path(args.gameworld_root).resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vicor",
        description="ViCoR: Visual Conditioned Reflexes for Real-Time VLM Agents",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    run_p = sub.add_parser("run", help="Run ViCoR on a benchmark task")
    run_p.add_argument("--game", required=True, help="Game identifier (e.g. 10_doodle-jump, oga_obstacle_run_2d, arc3_ls20)")
    run_p.add_argument("--task", required=True, help="Task identifier (e.g. 10_01)")
    run_p.add_argument(
        "--mode",
        choices=["full", "pure-python", "pure-vlm", "no-bank"],
        default="full",
        help="Agent paradigm: full (ViCoR), pure-python (programmatic script), pure-vlm (direct VLM)",
    )
    run_p.add_argument(
        "--endpoint",
        default=os.environ.get("VICOR_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions"),
        help="OpenAI-compatible endpoint URL for the main VLM",
    )
    run_p.add_argument(
        "--reflex-endpoint",
        default=None,
        help="OpenAI-compatible endpoint URL for the fast Reflex VLM (defaults to --endpoint or $VICOR_REFLEX_ENDPOINT)",
    )
    run_p.add_argument(
        "--model",
        default=os.environ.get("VICOR_MODEL", "Qwen3-VL-30B-A3B"),
        help="Model identifier served at --endpoint",
    )
    run_p.add_argument(
        "--reflex-model",
        default=None,
        help="Model identifier for the fast Reflex VLM (defaults to --model)",
    )
    run_p.add_argument("--effort", choices=["none", "low", "medium", "high"], default="low", help="Main VLM reasoning effort")
    run_p.add_argument("--episodes", type=int, default=1, help="Number of native episodes to run")
    run_p.add_argument("--max-steps", type=int, default=400, help="Maximum environment steps per episode (None for unbounded)")
    run_p.add_argument("--seed", type=int, default=1800, help="Base deterministic episode seed")
    run_p.add_argument("--port", type=int, default=29310, help="Local HTTP server port for browser-based environments")
    run_p.add_argument("--fps", type=float, default=15.0, help="Target observation stream frame rate (1-60 Hz)")
    run_p.add_argument("--screenshot-every", type=int, default=5)
    run_p.add_argument("--intermission-seconds", type=float, default=0.0)
    run_p.add_argument("--pause-during-inference", action=argparse.BooleanOptionalAction, default=False)
    run_p.add_argument("--manual-start", action="store_true")
    run_p.add_argument("--keep-playing", action="store_true")
    run_p.add_argument("--single-attempt", action="store_true")
    run_p.add_argument("--headed", action="store_true", help="Launch visible browser window instead of headless Chromium")
    run_p.add_argument("--resume", action="store_true", help="Allow writing into an existing output directory")
    run_p.add_argument("--backend", choices=["openai", "ninfer", "codex-responses"], default="openai")
    run_p.add_argument("--codex-config", default=None)
    run_p.add_argument("--codex-auth", default=None)
    run_p.add_argument("--api-slot-dir", default=None)
    run_p.add_argument("--api-slot-limit", type=int, default=2)
    run_p.add_argument("--tokenizer-file", default=None)
    run_p.add_argument("--compaction-threshold", type=int, default=DEFAULT_LONG_CONTEXT_TRIGGER_TOKENS)
    run_p.add_argument("--gameworld-root", default=None, help="Path to GameWorld checkout (or set $VICOR_GAMEWORLD_ROOT)")
    run_p.add_argument(
        "--instruction",
        default=(
            "Pursue the official goal. Learn useful short visual reflexes from your own "
            "observations and keep improving them across native rounds."
        ),
    )
    run_p.add_argument("--output", required=True, help="Directory for run logs, episode summaries, and reflex artifacts")

    list_p = sub.add_parser("list-tasks", help="List available benchmark tasks")
    list_p.add_argument(
        "--benchmark",
        choices=["all", "gameworld", "aigamestore", "omnigamearena", "arc3", "robodojo"],
        default="all",
    )
    list_p.add_argument("--gameworld-root", default=None)

    ov_p = sub.add_parser("install-overlays", help="Install ViCoR multi-benchmark adapters into a GameWorld checkout")
    ov_p.add_argument("--gameworld-root", required=True, help="Target GameWorld repository directory")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.subcommand == "run":
        if not 1 <= args.fps <= 60:
            raise ValueError("--fps must be between 1 and 60")
        if args.episodes is not None and args.episodes < 1:
            raise ValueError("--episodes must be positive")
        asyncio.run(_run_agent(args))
    elif args.subcommand == "list-tasks":
        _cmd_list_tasks(args)
    elif args.subcommand == "install-overlays":
        _cmd_install_overlays(args)


if __name__ == "__main__":
    main()
