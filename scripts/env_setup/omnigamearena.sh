#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OGA_BASE="${LOCAL_DIR}/omnigamearena"
mkdir -p "${OGA_BASE}"

if [[ ! -d "${OGA_BASE}/OmniGameArena/.git" ]]; then
  git clone https://github.com/mxlin043/OmniGameArena.git "${OGA_BASE}/OmniGameArena"
fi
pip install -r "${OGA_BASE}/OmniGameArena/requirements.txt"

echo "Place the Linux UE5 build from https://huggingface.co/datasets/mxlin043/OmniGameArena under:"
echo "  ${OGA_BASE}/Linux/OmniGameArena.sh"
echo "Then export:"
echo "  export OMNIGAMEARENA_ROOT=${OGA_BASE}/OmniGameArena"
echo "  export OGA_UE5_SH=${OGA_BASE}/Linux/OmniGameArena.sh"
