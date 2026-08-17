#!/usr/bin/env bash
set -euo pipefail
# Wait until the half-body phase has actually started.
while [ ! -f /data1/workspace/linxinliang/CoInteract/logs/qilin_og_m2v_halfbody_720p_20260622.log ]; do
  sleep 30
done

# Then wait until the half-body script and its batch process are both finished.
while pgrep -f "run_qilin_og_m2v_halfbody_720p.sh" >/dev/null || pgrep -f "cointeract_qilin_og_m2v_halfbody.csv" >/dev/null; do
  sleep 30
done
cd /data1/workspace/linxinliang/CoInteract
bash /data1/workspace/linxinliang/CoInteract/run_qilin_og_m2v_fullbody_720p.sh > /data1/workspace/linxinliang/CoInteract/logs/qilin_og_m2v_fullbody_720p_20260622.log 2>&1
