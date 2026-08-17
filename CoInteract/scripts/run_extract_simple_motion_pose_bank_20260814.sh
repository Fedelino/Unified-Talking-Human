#!/usr/bin/env bash
set -euo pipefail

cd /data1/workspace/linxinliang/CoInteract
source /data1/miniconda3/etc/profile.d/conda.sh
conda activate interact_avatar

export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/data1/miniconda3/envs/interact_avatar/lib:${LD_LIBRARY_PATH:-}"

python scripts/extract_simple_motion_pose_bank_20260814.py
