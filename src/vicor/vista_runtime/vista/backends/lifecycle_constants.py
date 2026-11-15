"""Compact-checkpoint constants and context-window threshold calculation."""

from __future__ import annotations

from dataclasses import dataclass

CHECKPOINT_PROMPT = (
    '<codex_internal_context source="compact_checkpoint">\n'
    "Call save_compact_checkpoint now as your first and only action, then end "
    "this turn.\n"
    "</codex_internal_context>"
)

MAX_CHECKPOINT_ATTEMPTS = 2

CHECKPOINT_HARD_CONTEXT_RESERVE = 32_000
CHECKPOINT_TURN_HEADROOM = 40_000
MIN_NORMAL_AUTO_COMPACT_LIMIT = 16_000


@dataclass(frozen=True)
class CompactThresholds:
    model_context_window: int
    normal_auto_compact_limit: int
    checkpoint_auto_compact_limit: int
    source: str


def compact_thresholds_for_context(
    context_window: int,
    *,
    source: str,
) -> CompactThresholds:
    checkpoint_limit = context_window - CHECKPOINT_HARD_CONTEXT_RESERVE
    normal_limit = checkpoint_limit - CHECKPOINT_TURN_HEADROOM
    if normal_limit < MIN_NORMAL_AUTO_COMPACT_LIMIT:
        raise ValueError(
            f"Model context window {context_window} is too small for compact checkpointing"
        )
    return CompactThresholds(
        model_context_window=context_window,
        normal_auto_compact_limit=normal_limit,
        checkpoint_auto_compact_limit=checkpoint_limit,
        source=source,
    )
