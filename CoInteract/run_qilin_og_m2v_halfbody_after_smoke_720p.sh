#!/usr/bin/env bash
set -euo pipefail
while pgrep -f "run_qilin_og_m2v_smoke_fullframe_720p.sh" >/dev/null || pgrep -f "qilin_og_m2v_smoke_fullframe_720p_20260622.log" >/dev/null; do
  sleep 30
done
cd /data1/workspace/linxinliang/CoInteract
bash /data1/workspace/linxinliang/CoInteract/run_qilin_og_m2v_halfbody_720p.sh > /data1/workspace/linxinliang/CoInteract/logs/qilin_og_m2v_halfbody_720p_20260622.log 2>&1
bash /data1/workspace/linxinliang/CoInteract/run_qilin_og_m2v_fullbody_720p.sh > /data1/workspace/linxinliang/CoInteract/logs/qilin_og_m2v_fullbody_720p_20260622.log 2>&1
