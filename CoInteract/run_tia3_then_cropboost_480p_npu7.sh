#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/workspace/linxinliang/CoInteract"
MASTER_LOG="$ROOT/logs/tia3_then_cropboost_480p_npu7_20260703.log"

mkdir -p "$(dirname "$MASTER_LOG")"

{
  echo "===== Master chain start $(date '+%F %T') ====="
  bash "$ROOT/run_cointeract_tia2v_3cases_30s_npu7.sh"
  bash "$ROOT/run_cointeract_fullbody_case1_motion2_cropboost_480p_npu7.sh"
  echo "===== Master chain end $(date '+%F %T') ====="
} >> "$MASTER_LOG" 2>&1
