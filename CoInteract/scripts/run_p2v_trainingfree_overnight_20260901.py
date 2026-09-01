#!/usr/bin/env python3
"""Orchestrate overnight training-free CoInteract P2V identity experiments.

The script prepares artifacts once, then runs a partition of generation variants.
Each variant invokes batch_infer.py as a fresh Python process.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path("/data1/workspace/linxinliang/CoInteract")
IA_ROOT = Path("/data1/workspace/linxinliang/InteractAvatar")
REF = IA_ROOT / "InterDemo/TalkingHumanDemo_fullbody/test_data/ref_img/wholebody/fullImage1a3b5329d192f028c14bc03d20c24afa629a3123.jpg"
PROMPT = "A full-body person follows the provided motion with stable facial identity, stable eyes, stable nose, stable lips, stable jawline, and stable whole-body proportions."
BASE_OUT = ROOT / "output_videos/p2v_trainingfree_overnight_20260901"
EVAL_OUT = ROOT / "output_videos/p2v_trainingfree_overnight_20260901_eval"
LOG_DIR = ROOT / "logs/p2v_trainingfree_overnight_20260901"
ASSET_DIR = ROOT / "output_videos/p2v_trainingfree_overnight_20260901_assets"
CSV_DIR = ROOT / "examples/p2v_trainingfree_overnight_20260901"
ARC = ROOT / "models/arcface/w600k_r50.onnx"

POSES = {
    "m1": {
        "pose": IA_ROOT / "InterDemo/custom_motion/dwpose/motion1_pose.mp4",
        "raw": IA_ROOT / "InterDemo/custom_motion/raw/motion1.mp4",
    },
    "m2": {
        "pose": IA_ROOT / "InterDemo/custom_motion/dwpose/motion2_pose.mp4",
        "raw": IA_ROOT / "InterDemo/custom_motion/raw/motion2.mp4",
    },
    "handwave": {
        "pose": IA_ROOT / "InterDemo/simple_motion_pose_bank_20260814/dwpose_qilin_aligned/th_motion1_handwave_qilin_aligned_pose.mp4",
        "raw": IA_ROOT / "InterDemo/simple_motion_pose_bank_20260814/raw_25fps_max4s/th_motion1_handwave.mp4",
    },
}


@dataclass
class Variant:
    name: str
    motion: str
    person_image: Path
    pose_video: Path
    kind: str
    args: dict[str, str | float | int | bool] = field(default_factory=dict)


def run(cmd: list[str], log_path: Path | None = None, env: dict[str, str] | None = None, check: bool = True):
    print("[cmd]", " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    if log_path is None:
        return subprocess.run(cmd, cwd=ROOT, env=env, check=check)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("# " + " ".join(shlex.quote(str(x)) for x in cmd) + "\n\n")
        log.flush()
        return subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True, check=check)


def write_csv(path: Path, sample_id: str, person_image: Path, pose_video: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "prompt", "audio", "person_image", "product_image", "pose_video"])
        writer.writeheader()
        writer.writerow({
            "sample_id": sample_id,
            "prompt": PROMPT,
            "audio": "",
            "person_image": str(person_image),
            "product_image": "",
            "pose_video": str(pose_video),
        })


def prepare_assets() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable,
        "scripts/prepare_reference_face_variants_20260901.py",
        "--reference", str(REF),
        "--out_dir", str(ASSET_DIR / "reference_variants"),
    ], LOG_DIR / "prepare_reference_face_variants.log")

    for motion in ("m1", "m2", "handwave"):
        raw = POSES[motion]["raw"]
        if not raw.exists():
            print(f"[skip prepare] missing raw motion for {motion}: {raw}", flush=True)
            continue
        for blend in (0.15, 0.25, 0.35):
            tag = f"{motion}_idface_blend{int(blend * 100):02d}_pose.mp4"
            run([
                sys.executable,
                "scripts/id_face_retarget.py",
                "--driving_video", str(raw),
                "--reference_image", str(REF),
                "--out_video", str(ASSET_DIR / "pose_variants" / tag),
                "--height", "832",
                "--width", "480",
                "--fps", "25",
                "--max_frames", "80",
                "--blend", str(blend),
            ], LOG_DIR / f"prepare_{tag}.log", check=False)


def build_variants() -> list[Variant]:
    ref_dir = ASSET_DIR / "reference_variants"
    variants: list[Variant] = []

    for motion in ("m1", "m2"):
        variants.append(Variant(f"00_baseline_{motion}", motion, REF, POSES[motion]["pose"], "baseline"))

    for ref_name in ("ref_face_clahe035.jpg", "ref_face_sharp040.jpg", "ref_face_clahe_sharp045.jpg"):
        for motion in ("m1", "m2"):
            variants.append(Variant(f"a_refprep_{Path(ref_name).stem}_{motion}", motion, ref_dir / ref_name, POSES[motion]["pose"], "refprep"))

    for scale in (0.025, 0.05, 0.075):
        for motion in ("m1", "m2"):
            variants.append(Variant(
                f"b_facekv_s{str(scale).replace('.', 'p')}_{motion}",
                motion,
                REF,
                POSES[motion]["pose"],
                "facekv",
                {
                    "reference_kv_guidance_scale": scale,
                    "reference_kv_guidance_mode": "head_attn",
                    "reference_kv_guidance_blocks": "10:22",
                    "reference_kv_guidance_start_t": 0.05,
                    "reference_kv_guidance_end_t": 0.75,
                },
            ))

    for blend in (0.15, 0.25, 0.35):
        for motion in ("m1", "m2", "handwave"):
            pose = ASSET_DIR / "pose_variants" / f"{motion}_idface_blend{int(blend * 100):02d}_pose.mp4"
            if pose.exists():
                variants.append(Variant(f"c_idface_blend{int(blend * 100):02d}_{motion}", motion, REF, pose, "idface_light"))

    return variants


def batch_command(csv_path: Path, out_dir: Path, extra: dict[str, str | float | int | bool]) -> list[str]:
    cmd = [
        sys.executable,
        "batch_infer.py",
        "--base_model_path", "./models/Wan2.2-S2V-14B",
        "--audio_encoder_path", "./models/chinese-wav2vec2-large",
        "--csv_path", str(csv_path),
        "--height", "832",
        "--width", "480",
        "--num_frames", "80",
        "--num_clips", "1",
        "--num_inference_steps", "40",
        "--cfg_scale", "7.0",
        "--sigma_shift", "7.0",
        "--lora_path", "./models/CoInteract/checkpoint_pose.safetensors",
        "--reference_compose_mode", "stretch",
        "--output_dir", str(out_dir),
        "--no_resize_output_to_reference",
    ]
    for key, value in extra.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    return cmd


def selected_variants(variants: list[Variant], partition: int, partitions: int) -> list[Variant]:
    if partitions <= 1:
        return variants
    return [v for i, v in enumerate(variants) if i % partitions == partition]


def run_generation(args) -> None:
    variants = selected_variants(build_variants(), args.partition, args.partitions)
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(args.npu)
    env["ASCEND_VISIBLE_DEVICES"] = str(args.npu)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_NPU_ALLOC_CONF", "max_split_size_mb:256,garbage_collection_threshold:0.8")
    for variant in variants:
        csv_path = CSV_DIR / f"{variant.name}.csv"
        sample_id = variant.name
        out_dir = BASE_OUT / variant.name
        write_csv(csv_path, sample_id, variant.person_image, variant.pose_video)
        if (out_dir / f"{sample_id}.mp4").exists():
            print(f"[skip exists] {variant.name}", flush=True)
            continue
        run(batch_command(csv_path, out_dir, variant.args), LOG_DIR / f"{variant.name}.log", env=env, check=False)


def eval_one(video: Path, reference: Path, name: str, summary_rows: list[dict[str, str]]) -> None:
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    txt_path = EVAL_OUT / f"{name}.txt"
    csv_path = EVAL_OUT / f"{name}.csv"
    sheet_path = EVAL_OUT / f"{name}_sheet.jpg"
    cmd = [
        sys.executable,
        "eval/id_drift_metric.py",
        "--video", str(video),
        "--reference", str(reference),
        "--arcface_onnx", str(ARC),
        "--detector", "insightface",
        "--every", "5",
        "--out_csv", str(csv_path),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    txt_path.write_text(proc.stdout, encoding="utf-8")
    subprocess.run([
        sys.executable,
        "scripts/make_frame_sheet.py",
        "--video", str(video),
        "--out", str(sheet_path),
        "--frames", "0,25,50,75",
    ], cwd=ROOT, check=False)
    row = {"name": name, "video": str(video)}
    for pattern, key in [
        (r"cos_ref\s+mean/min:\s+([0-9.+-]+)\s*/\s*([0-9.+-]+)", "cos"),
        (r"cos_ref\s+first/last:\s+([0-9.+-]+)\s*/\s*([0-9.+-]+)", "firstlast"),
        (r"drift slope \(cos/s\):\s+([0-9.+-]+)", "slope"),
    ]:
        m = re.search(pattern, proc.stdout)
        if key == "cos" and m:
            row["cos_mean"], row["cos_min"] = m.group(1), m.group(2)
        elif key == "firstlast" and m:
            row["cos_first"], row["cos_last"] = m.group(1), m.group(2)
        elif key == "slope" and m:
            row["slope"] = m.group(1)
    summary_rows.append(row)


def run_postprocess_and_eval() -> None:
    rows: list[dict[str, str]] = []
    for video in sorted(BASE_OUT.glob("*/*.mp4")):
        name = video.parent.name
        eval_one(video, REF, name, rows)
        if name.startswith("00_baseline_"):
            for strength in (0.25, 0.40, 0.55):
                post_name = f"d_postface_s{str(strength).replace('.', 'p')}_{name.replace('00_baseline_', '')}"
                out_video = BASE_OUT / post_name / f"{post_name}.mp4"
                if not out_video.exists():
                    run([
                        sys.executable,
                        "scripts/postprocess_face_region_20260901.py",
                        "--video", str(video),
                        "--out", str(out_video),
                        "--strength", str(strength),
                    ], LOG_DIR / f"{post_name}.log", check=False)
                if out_video.exists():
                    eval_one(out_video, REF, post_name, rows)

    summary_path = EVAL_OUT / "summary.csv"
    fields = ["name", "cos_mean", "cos_min", "cos_first", "cos_last", "slope", "video"]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] wrote {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--npu", default="7")
    parser.add_argument("--partition", type=int, default=0)
    parser.add_argument("--partitions", type=int, default=1)
    args = parser.parse_args()
    os.chdir(ROOT)
    for path in (BASE_OUT, EVAL_OUT, LOG_DIR, ASSET_DIR, CSV_DIR):
        path.mkdir(parents=True, exist_ok=True)
    if args.eval_only:
        run_postprocess_and_eval()
        return
    if not args.skip_prepare:
        prepare_assets()
    if args.prepare_only:
        return
    run_generation(args)


if __name__ == "__main__":
    main()
