#!/usr/bin/env bash
set -euo pipefail

cd /data1/workspace/linxinliang/CoInteract

echo "[watch] waiting for any non-7 free NPU at $(date)"
while true; do
  info="$(npu-smi info)"
  for npu_id in 0 1 2 3 4 5 6; do
    if ! printf "%s\n" "$info" | grep -qE "^\\|[[:space:]]*${npu_id}[[:space:]]+0[[:space:]]+\\|[[:space:]]+[0-9]+"; then
      echo "[watch] selected NPU ${npu_id} at $(date)"
      exec bash scripts/run_midstep24283236_postseqall_s0005_full40_on_npu_20260731.sh "${npu_id}"
    fi
  done
  echo "[watch] no non-7 NPU free at $(date)"
  sleep 60
done
