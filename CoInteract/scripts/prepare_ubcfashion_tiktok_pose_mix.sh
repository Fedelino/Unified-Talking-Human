#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

python tools/merge_pose_datasets.py \
  --ubcfashion-root ./data/ubcfashion_pose_full \
  --tiktok-root ./data/tiktok_pose_full \
  --output-root ./data/ubcfashion_tiktok_pose_full \
  --ubcfashion-repeat "${UBCFASHION_REPEAT:-1}" \
  --tiktok-repeat "${TIKTOK_REPEAT:-2}" \
  --seed "${MERGE_SEED:-42}"
