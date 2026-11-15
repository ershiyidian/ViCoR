#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_DIR="${VICOR_LOCAL_DIR:-${REPO_ROOT}/local}"
mkdir -p "${LOCAL_DIR}"
