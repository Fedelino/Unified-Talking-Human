# CoInteract 全身 Pose 微调说明

本文记录当前仓库中 CoInteract S2V Pose-Driven 全身姿态微调的本地流程。

## 接入方式

我们把每条训练样本整理成 CoInteract 训练格式：

- `motion_video`：目标片段前 73 帧，作为运动上下文。
- `input_video`：目标 81 帧。
- `person_image`：目标片段第一帧，作为人物身份/外观参考。
- `pose_video`：与 `input_video` 对齐的 81 帧 DWPose 骨架视频。
- `audio`：16 kHz 单声道音频；TikTok 数据使用静音音频。

训练时，`pose_video` 会作为 `s2v_pose_video` 读入，经 VAE 编码成 `s2v_pose_latents`，再通过 pose condition encoder 注入 DiT。

## 数据集

当前已准备的数据：

| 数据集 | 路径 | 行数 |
| --- | --- | ---: |
| UBCFashion | `data/ubcfashion_pose_full/data.csv` | 500 |
| TikTok | `data/tiktok_pose_full/data.csv` | 250 |
| 混合数据 | `data/ubcfashion_tiktok_pose_full/data.csv` | 1000 |

混合数据不复制视频文件，只重写 CSV 路径：

```bash
bash scripts/prepare_ubcfashion_tiktok_pose_mix.sh
```

默认混合比例：

- UBCFashion repeat：`1`，得到 500 行
- TikTok repeat：`2`，得到 500 行

Pose 提取参数：

- `--pose-thickness 8`
- `--draw-face`

## 训练

混合训练命令：

```bash
ASCEND_RT_VISIBLE_DEVICES=6 bash scripts/train_ubcfashion_tiktok_pose_full_npu6.sh
```

训练脚本：

- `scripts/train_ubcfashion_tiktok_pose_full_npu6.sh`

训练入口：

- `examples/wanvideo/model_training/train.py`

关键超参：

| 参数 | 当前值 |
| --- | --- |
| 基座模型 | `models/Wan2.2-S2V-14B` |
| Audio encoder | `models/chinese-wav2vec2-large` |
| 初始 LoRA | `models/CoInteract/checkpoint_pose.safetensors` |
| 训练模块 | `dit` |
| LoRA 注入位置 | `q,k,v,o` |
| LoRA rank | `128` |
| 分辨率 | `480x320` |
| 目标帧数 | `81` |
| motion 上下文 | `73` 帧 |
| 学习率 | `5e-6` |
| epoch | `5` |
| save_steps | `25` |
| gradient accumulation | `1` |
| pose dropout | `0.1` |
| train shift | `5.0` |
| NPU | 6 号卡，单卡 |
| 显存策略 | `--use_gradient_checkpointing_offload` |

LoRA checkpoint 输出位置：

```text
output/ubcfashion_tiktok_pose_full/version_*/step-*.safetensors
```

查看最新 ckpt：

```bash
find output/ubcfashion_tiktok_pose_full -path '*/step-*.safetensors' | sort -V | tail -n 1
```

## 推理

推理脚本：

```bash
scripts/infer_ubcfashion_pose_npu6.sh
```

注意：该脚本的默认 ckpt fallback 可能仍指向旧的 UBCFashion ckpt。使用混合训练结果时，建议显式指定 `LORA_PATH`：

```bash
LORA_PATH="$(find output/ubcfashion_tiktok_pose_full -path '*/step-*.safetensors' | sort -V | tail -n 1)" \
ASCEND_RT_VISIBLE_DEVICES=6 \
OUTPUT_DIR=./output_videos/ubcfashion_tiktok_pose_eval \
bash scripts/infer_ubcfashion_pose_npu6.sh
```

默认推理 CSV：

```text
examples/demos/ubcfashion_pose_infer.csv
```

也可以指定自己的推理 CSV：

```bash
CSV_PATH=./path/to/infer.csv \
LORA_PATH=./output/ubcfashion_tiktok_pose_full/version_0/step-XXXX.safetensors \
ASCEND_RT_VISIBLE_DEVICES=6 \
bash scripts/infer_ubcfashion_pose_npu6.sh
```

推理使用 `--no_use_moe`，与当前 attention-only LoRA 微调方式一致。
