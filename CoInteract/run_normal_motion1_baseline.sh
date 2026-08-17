#!/bin/bash
source /data1/miniconda3/etc/profile.d/conda.sh
conda activate cointeract
cd /data1/workspace/linxinliang/CoInteract
export ASCEND_RT_VISIBLE_DEVICES=${DEV:-5}
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
python batch_infer.py   --base_model_path ./models/Wan2.2-S2V-14B   --audio_encoder_path ./models/chinese-wav2vec2-large   --lora_path ./models/CoInteract/checkpoint.safetensors   --use_moe --expert_hidden_dim 256   --csv_path examples/cointeract_fullbody4_motion1_m2v_prompted_20260628.csv   --output_dir output_videos/normal_motion1_baseline_20260716   --height 480 --width 320 --num_frames 81 --num_clips 1 --num_inference_steps 30
echo "EXIT_CODE=$?"
