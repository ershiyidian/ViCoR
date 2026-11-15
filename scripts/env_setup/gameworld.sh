#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

GW_DIR="${LOCAL_DIR}/gameworld/GameWorld"
mkdir -p "$(dirname "${GW_DIR}")"

if [[ ! -d "${GW_DIR}/.git" ]]; then
  git clone https://github.com/gameworld-project/GameWorld.git "${GW_DIR}"
fi

if [[ ! -d "${GW_DIR}/games/gameworld-games" ]]; then
  mkdir -p "${GW_DIR}/games"
  git clone https://github.com/gameworld-project/GameWorld-Games.git "${GW_DIR}/games/gameworld-games"
fi

pip install -r "${GW_DIR}/requirements.txt"
playwright install chromium

vicor install-overlays --gameworld-root "${GW_DIR}"
echo "GameWorld + ViCoR multi-benchmark overlays ready at ${GW_DIR}"
