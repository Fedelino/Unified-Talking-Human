import os
import shlex
import subprocess
import time
from pathlib import Path


ROOT = Path("/data1/workspace/linxinliang/CoInteract")
CSV = "examples/th_fullbody_001_custom_motion2_noaudio_baseline_20260716.csv"
DATE = "20260813"
NPUS = ["6", "7"]

COMMON_ARGS = [
    "--base_model_path", "/data1/Wan-AI/wan22_s2v",
    "--audio_encoder_path", "/data1/workspace/linxinliang/CoInteract/models/chinese-wav2vec2-large",
    "--lora_path", "/data1/workspace/linxinliang/CoInteract/models/CoInteract/checkpoint_pose.safetensors",
    "--csv_path", CSV,
    "--data_base_path", "/data1/workspace/linxinliang/CoInteract",
    "--height", "832",
    "--width", "480",
    "--num_frames", "80",
    "--num_clips", "1",
    "--num_inference_steps", "40",
    "--cfg_scale", "7.0",
    "--sigma_shift", "7.0",
    "--reference_compose_mode", "stretch",
    "--no_resize_output_to_reference",
    "--arcface_guidance_scale", "0.0",
    "--face_reference_guidance_counterfactual", "latent_blur",
    "--face_reference_guidance_blur_kernel", "5",
    "--face_reference_guidance_samg",
    "--face_reference_guidance_samg_min_mult", "0.5",
    "--face_reference_guidance_samg_max_mult", "1.5",
]

VARIANTS = []
for scale in ["1.5", "2.0", "3.0", "5.0"]:
    scale_label = scale.replace(".", "p")
    VARIANTS.append(
        {
            "name": f"latblur_k5_samg_scale{scale_label}",
            "scale": scale,
            "extra": [],
        }
    )
for scale in ["1.5", "2.0", "3.0", "5.0"]:
    scale_label = scale.replace(".", "p")
    VARIANTS.append(
        {
            "name": f"latblur_k5_samg_apg025_scale{scale_label}",
            "scale": scale,
            "extra": ["--face_reference_guidance_apg_eta", "0.25"],
        }
    )


def build_command(variant, out_dir: Path) -> list[str]:
    args = (
        COMMON_ARGS
        + ["--face_reference_guidance_scale", variant["scale"]]
        + ["--output_dir", str(out_dir)]
        + variant["extra"]
    )
    quoted_args = " ".join(shlex.quote(arg) for arg in args)
    script = (
        "source /data1/miniconda3/etc/profile.d/conda.sh && "
        "conda activate cointeract && "
        f"cd {shlex.quote(str(ROOT))} && "
        "export TOKENIZERS_PARALLELISM=false && "
        "export PYTHONUNBUFFERED=1 && "
        "export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256,garbage_collection_threshold:0.8 && "
        "python batch_infer.py "
        + quoted_args
    )
    return ["bash", "-lc", script]


def launch(variant, npu: str):
    out_dir = ROOT / "output_videos" / f"p2v_idguide_strong_{variant['name']}_motion2_{DATE}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "logs" / f"p2v_idguide_strong_{variant['name']}_motion2_{DATE}.log"
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = npu
    with log_path.open("ab") as log_file:
        log_file.write(f"\n===== launch {variant['name']} on NPU {npu} =====\n".encode())
        process = subprocess.Popen(
            build_command(variant, out_dir),
            cwd=str(ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    print(f"START\t{variant['name']}\tnpu={npu}\tpid={process.pid}\tlog={log_path}\tout={out_dir}", flush=True)
    return process, variant, npu


def main() -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    queue = list(VARIANTS)
    running = []

    while queue or running:
        while queue and len(running) < len(NPUS):
            npu = next(n for n in NPUS if n not in [r[2] for r in running])
            running.append(launch(queue.pop(0), npu))

        time.sleep(20)
        still_running = []
        for process, variant, npu in running:
            code = process.poll()
            if code is None:
                still_running.append((process, variant, npu))
                continue
            print(f"DONE\t{variant['name']}\tnpu={npu}\treturncode={code}", flush=True)
        running = still_running


if __name__ == "__main__":
    main()
