#!/usr/bin/env bash
set -euo pipefail

cd /data1/workspace/linxinliang/CoInteract

echo "[watch] waiting for NPU7 at $(date)"
while npu-smi info | grep -qE '^\| 7[[:space:]]+0[[:space:]]+\|[[:space:]]+[0-9]+'; do
  npu-smi info | grep -E '^\| 7[[:space:]]+0[[:space:]]+\|' || true
  sleep 60
done

echo "[watch] NPU7 free at $(date), launching midstep experiment"
exec bash scripts/run_midstep24283236_postseqall_s0005_full40_20260731.sh
