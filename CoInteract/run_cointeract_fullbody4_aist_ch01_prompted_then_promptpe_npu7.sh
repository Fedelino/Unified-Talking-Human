#!/usr/bin/env bash
set -euo pipefail

source /data1/miniconda3/etc/profile.d/conda.sh
conda activate interact_avatar

cd /data1/workspace/linxinliang/CoInteract

export TOKENIZERS_PARALLELISM=false
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-7}"
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128

echo "===== CoInteract AIST prompted then promptpe chain start $(date '+%F %T') ====="
bash /data1/workspace/linxinliang/CoInteract/run_cointeract_fullbody4_aist_ch01_prompted_cycle_npu7.sh
bash /data1/workspace/linxinliang/CoInteract/run_cointeract_fullbody4_aist_ch01_promptpe_cycle_npu7.sh
echo "===== CoInteract AIST prompted then promptpe chain end $(date '+%F %T') ====="
