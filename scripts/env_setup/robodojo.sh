#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RD_DIR="${LOCAL_DIR}/robodojo"
mkdir -p "${RD_DIR}"
echo "After installing RoboDojo / Isaac Sim in a dedicated Python environment, export:"
echo "  export ROBODOJO_WS_DIR=${RD_DIR}"
echo "  export ROBODOJO_CONDA_ENV=/path/to/conda/envs/RoboDojo"
echo "  export ROBODOJO_DEVICE_ID=0"
