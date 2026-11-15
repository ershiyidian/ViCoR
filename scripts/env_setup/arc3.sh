#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ARC3_DIR="${LOCAL_DIR}/arc3/environments_dir"
mkdir -p "${ARC3_DIR}"
pip install arc-agi arcengine

echo "Place offline ARC-AGI-3 environment packages in ${ARC3_DIR} and export:"
echo "  export ARC3_ENVIRONMENTS_DIR=${ARC3_DIR}"
