#!/usr/bin/env bash
set -euo pipefail

TARGET_MP4="/data1/workspace/linxinliang/CoInteract/output_videos/side_by_side_fullbody4_cointeract_og_betterpose_720p_20260626/th_fullbody_001_motion1_bp720.mp4"
SCRIPT_720="/data1/workspace/linxinliang/CoInteract/run_cointeract_fullbody4_halfbody4_betterpose_720p_cycle_npu6.sh"
SCRIPT_480="/data1/workspace/linxinliang/CoInteract/run_cointeract_halfbody4_fullbody4_betterpose_480p_cycle_npu6.sh"
HANDOFF_LOG="/data1/workspace/linxinliang/CoInteract/logs/handoff_after_first_720p_to_480_betterpose_npu6_20260626.log"
LAUNCH_LOG="/data1/workspace/linxinliang/CoInteract/logs/launch_side_by_side_halfbody4_fullbody4_cointeract_og_betterpose_480p_cycle_20260626.log"

mkdir -p "$(dirname "$HANDOFF_LOG")"

echo "[$(date '+%F %T')] handoff watcher start" >>"$HANDOFF_LOG"
echo "[$(date '+%F %T')] waiting for $TARGET_MP4" >>"$HANDOFF_LOG"

while [ ! -f "$TARGET_MP4" ]; do
  sleep 60
done

echo "[$(date '+%F %T')] detected first 720p video; waiting 30s for mux settle" >>"$HANDOFF_LOG"
sleep 30

pkill -f "$SCRIPT_720" || true
pkill -f "batch_infer_qilin_og_m2v.py.*side_by_side_fullbody4_cointeract_og_betterpose_720p_20260626" || true

echo "[$(date '+%F %T')] stopped 720p betterpose cycle" >>"$HANDOFF_LOG"
sleep 20

nohup bash "$SCRIPT_480" >"$LAUNCH_LOG" 2>&1 &
echo "[$(date '+%F %T')] launched 480p betterpose cycle" >>"$HANDOFF_LOG"
