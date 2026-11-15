#!/usr/bin/env bash
set -euo pipefail

GAME="${1:?Usage: run_benchmark.sh <game_id> <task_id> [mode] [output_dir]}"
TASK="${2:?Usage: run_benchmark.sh <game_id> <task_id> [mode] [output_dir]}"
MODE="${3:-full}"
OUTPUT_DIR="${4:-runs/${GAME}_${TASK}_${MODE}}"

vicor run \
  --game "${GAME}" \
  --task "${TASK}" \
  --mode "${MODE}" \
  --endpoint "${VICOR_ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}" \
  --reflex-endpoint "${VICOR_REFLEX_ENDPOINT:-${VICOR_ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}}" \
  --model "${VICOR_MODEL:-Qwen3-VL-30B-A3B}" \
  --effort "${VICOR_EFFORT:-low}" \
  --episodes "${VICOR_EPISODES:-1}" \
  --max-steps "${VICOR_MAX_STEPS:-400}" \
  --seed "${VICOR_SEED:-1800}" \
  --output "${OUTPUT_DIR}"
