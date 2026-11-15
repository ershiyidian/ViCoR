#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

AIGS_DIR="${LOCAL_DIR}/aigamestore"
if [[ ! -d "${AIGS_DIR}/.git" ]]; then
  git clone https://github.com/lance-ying/aigamestore_harness.git "${AIGS_DIR}"
fi

echo "AI GameStore assets ready. Export:"
echo "  export AIGAMESTORE_GAMES_ROOT=${AIGS_DIR}/gym/aigamestore"
