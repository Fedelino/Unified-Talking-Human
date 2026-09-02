#!/usr/bin/env python3
"""Round4 training-free CoInteract P2V identity experiments.

Round4 is focused on methods suggested by 3D/keyframe identity papers while
keeping CoInteract frozen:

* schedule variants for safer face-reference velocity guidance,
* pseudo-3D yaw-gated face-pose control plus guidance,
* deterministic keyframe/recurrent face-anchor postprocess,
* dataset/motion inventory for follow-up experiments.
"""

from __future__ import annotations

import argparse
import csv
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
ROUND4_OUT = BASE_OUT / "round4"
EVAL_OUT = ROOT / "output_videos/p2v_trainingfree_overnight_20260901_eval_round4"
LOG_DIR = ROOT / "logs/p2v_trainingfree_overnight_20260901/round4"
ASSET_DIR = ROOT / "output_videos/p2v_trainingfree_overnight_20260901_assets/round4"
CSV_DIR = ROOT / "examples/p2v_trainingfree_overnight_20260901_round4"
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
    "motion5": {
        "pose": ROOT / "output_videos/qilin_keypoint_aligned_pose_fb01_talkinghuman_motions_20260721/pose/fb01_motion5_qilin_aligned_25fps.mp4",
        "raw": ROOT / "output_videos/qilin_keypoint_aligned_pose_fb01_talkinghuman_motions_20260721/raw_4s/motion5_first4s.mp4",
    },
    "sayhi": {
        "pose": ROOT / "output_videos/qilin_keypoint_aligned_pose_fb01_talkinghuman_motions_20260721/pose/fb01_zsy_say_hi_1_0306_qilin_aligned_25fps.mp4",
        "raw": ROOT / "output_videos/qilin_keypoint_aligned_pose_fb01_talkinghuman_motions_20260721/raw_4s/zsy_say_hi_1_0306_first4s.mp4",
    },
}


@dataclass
class Variant:
    name: str
    motion: str
    person_image: Path
    pose_video: Path
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


def candidate_dataset_roots() -> list[Path]:
    roots = [
        Path("/data1/workspace/leijunwei/CoInteract/data/tiktok_pose_full"),
        Path("/data1/workspace/leijunwei/CoInteract/data/ubcfashion_pose_full"),
        Path("/data1/workspace/leijunwei/CoInteract/data/ubcfashion_tiktok_pose_full"),
        IA_ROOT / "InterDemo/custom_motion",
        IA_ROOT / "InterDemo/simple_motion_pose_bank_20260814",
        IA_ROOT / "InterDemo/TalkingHumanDemo_fullbody",
        ROOT / "output_videos/qilin_keypoint_aligned_pose_fb01_talkinghuman_motions_20260721",
    ]
    return roots


def write_dataset_inventory() -> None:
    lines = [
        "Round4 dataset/motion inventory",
        "",
        "Local usable data:",
    ]
    for root in candidate_dataset_roots():
        status = "FOUND" if root.exists() else "missing"
        mp4_count = 0
        image_count = 0
        if root.exists():
            mp4_count = sum(1 for _ in root.rglob("*.mp4"))
            image_count = sum(1 for ext in ("*.jpg", "*.jpeg", "*.png") for _ in root.rglob(ext))
        lines.append(f"- {root}: {status}, mp4={mp4_count}, images={image_count}")
    lines.extend([
        "",
        "External datasets to consider next:",
        "- BEAT/BEAT2: speech gesture and upper/full-body motions; useful for non-dance conversational movement.",
        "- Motion-X / HumanML3D / BABEL-AMASS: broad 3D human motion labels; useful for simple daily action pose synthesis.",
        "- UBCFashion + TikTok mixed pose-full: already available locally for CoInteract finetune, but dance/runway biased.",
        "- TED Gesture / SHOW / Trinity Speech Gesture: conversational body/hand movement candidates.",
        "- ActorsHQ / RenderPeople / THuman-style scans: useful person appearance sources, not direct P2V driving motions.",
    ])
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "round4_dataset_inventory.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def probe_external_weights() -> None:
    names = ["DECA", "FLAME", "CodeFormer", "GFPGAN", "RestoreFormer", "Stand-In", "MagicMirror"]
    search_roots = [Path("/data1/workspace/linxinliang"), Path("/data1/workspace/leijunwei"), Path("/data1/workspace/wangqilin")]
    lines = []
    for name in names:
        found = []
        low = name.lower().replace("-", "")
        for root in search_roots:
            if not root.exists():
                continue
            for p in root.glob("*"):
                compact = p.name.lower().replace("-", "")
                if low in compact:
                    found.append(str(p))
        lines.append(f"{name}: {'FOUND ' + '; '.join(found[:8]) if found else 'unavailable - skipped'}")
    (LOG_DIR / "round4_optional_weights_probe.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_assets() -> None:
    for path in (ROUND4_OUT, EVAL_OUT, LOG_DIR, ASSET_DIR, CSV_DIR):
        path.mkdir(parents=True, exist_ok=True)
    write_dataset_inventory()
    probe_external_weights()
    for motion in ("m1", "m2", "handwave", "motion5", "sayhi"):
        raw = POSES[motion]["raw"]
        if not raw.exists():
            print(f"[skip prepare] missing raw motion for {motion}: {raw}", flush=True)
            continue
        for blend, yaw in ((0.20, 25.0), (0.30, 35.0), (0.40, 45.0)):
            tag = f"{motion}_yawgated_blend{int(blend * 100):02d}_yaw{int(yaw):02d}_pose.mp4"
            out = ASSET_DIR / "pose_variants" / tag
            if out.exists():
                continue
            run([
                sys.executable,
                "scripts/id_face_retarget_yawgated_20260901.py",
                "--driving_video", str(raw),
                "--reference_image", str(REF),
                "--out_video", str(out),
                "--height", "832",
                "--width", "480",
                "--fps", "25",
                "--max_frames", "80",
                "--blend", str(blend),
                "--yaw_threshold", str(yaw),
                "--yaw_softness", "8",
            ], LOG_DIR / f"prepare_{tag}.log", check=False)


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


def champion_guidance(scale: float = 2.0, start: float = 0.05, end: float = 0.75) -> dict[str, str | float | int | bool]:
    return {
        "face_reference_guidance_scale": scale,
        "face_reference_guidance_counterfactual": "latent_blur",
        "face_reference_guidance_blur_kernel": 7,
        "face_reference_guidance_samg": True,
        "face_reference_guidance_samg_min_mult": 0.5,
        "face_reference_guidance_samg_max_mult": 1.5,
        "face_reference_guidance_apg_eta": 0.25,
        "face_reference_guidance_start_t": start,
        "face_reference_guidance_end_t": end,
    }


def build_variants() -> list[Variant]:
    variants: list[Variant] = []
    motions = [m for m in ("m1", "m2", "handwave", "motion5", "sayhi") if POSES[m]["pose"].exists()]
    core = [m for m in ("m1", "m2", "handwave") if m in motions]
    extra_simple = [m for m in ("motion5", "sayhi") if m in motions]

    for motion in core + extra_simple:
        variants.append(Variant(f"r4_baseline_{motion}", motion, REF, POSES[motion]["pose"]))

    schedule_variants = [
        ("r4_dv_full_scale2p5", champion_guidance(2.5, 0.05, 0.75)),
        ("r4_dv_detail_scale3p0", champion_guidance(3.0, 0.00, 0.35)),
        ("r4_dv_broad_scale3p0", champion_guidance(3.0, 0.00, 0.90)),
        ("r4_dv_noapg_scale2p0", {**champion_guidance(2.0, 0.05, 0.75), "face_reference_guidance_apg_eta": 1.0}),
        ("r4_dv_coarsefine_scale2p0", {
            "face_reference_guidance_scale": 2.0,
            "face_reference_guidance_counterfactual": "coarse_fine",
            "face_reference_guidance_samg": True,
            "face_reference_guidance_apg_eta": 0.25,
            "face_reference_guidance_start_t": 0.05,
            "face_reference_guidance_end_t": 0.75,
        }),
    ]
    for motion in core:
        for name, extra in schedule_variants:
            variants.append(Variant(f"{name}_{motion}", motion, REF, POSES[motion]["pose"], extra))

    for motion in core + extra_simple:
        for blend, yaw in ((0.20, 25), (0.30, 35), (0.40, 45)):
            pose = ASSET_DIR / "pose_variants" / f"{motion}_yawgated_blend{int(blend * 100):02d}_yaw{int(yaw):02d}_pose.mp4"
            if pose.exists():
                variants.append(Variant(f"r4_geom_blend{int(blend * 100):02d}_yaw{int(yaw):02d}_{motion}", motion, REF, pose))
                variants.append(Variant(f"r4_geom_blend{int(blend * 100):02d}_yaw{int(yaw):02d}_dv2_{motion}", motion, REF, pose, champion_guidance(2.0)))

    for motion in extra_simple:
        variants.append(Variant(f"r4_dv_champion_{motion}", motion, REF, POSES[motion]["pose"], champion_guidance(2.0)))

    return variants


def selected_variants(variants: list[Variant], partition: int, partitions: int) -> list[Variant]:
    if partitions <= 1:
        return variants
    return [variant for idx, variant in enumerate(variants) if idx % partitions == partition]


def run_generation(args) -> None:
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(args.npu)
    env["ASCEND_VISIBLE_DEVICES"] = str(args.npu)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_NPU_ALLOC_CONF", "max_split_size_mb:256,garbage_collection_threshold:0.8")
    variants = selected_variants(build_variants(), args.partition, args.partitions)
    for variant in variants:
        csv_path = CSV_DIR / f"{variant.name}.csv"
        out_dir = ROUND4_OUT / variant.name
        mp4 = out_dir / f"{variant.name}.mp4"
        write_csv(csv_path, variant.name, variant.person_image, variant.pose_video)
        if mp4.exists():
            print(f"[skip exists] {mp4}", flush=True)
            continue
        run(batch_command(csv_path, out_dir, variant.args), LOG_DIR / f"{variant.name}.log", env=env, check=False)


def baseline_video_for_motion(motion: str) -> Path | None:
    candidates = [
        ROUND4_OUT / f"r4_baseline_{motion}" / f"r4_baseline_{motion}.mp4",
        BASE_OUT / f"00_baseline_{motion}" / f"00_baseline_{motion}.mp4",
        BASE_OUT / f"round3_baseline_{motion}" / f"round3_baseline_{motion}.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def run_keyframe_postprocess() -> None:
    for motion in ("m1", "m2", "handwave", "motion5", "sayhi"):
        base = baseline_video_for_motion(motion)
        if base is None:
            continue
        for source, strength in (("video", 0.20), ("reference", 0.12), ("reference", 0.20)):
            tag = f"r4_kfanchor_{source}_s{str(strength).replace('.', 'p')}_{motion}"
            out = ROUND4_OUT / tag / f"{tag}.mp4"
            if out.exists():
                continue
            run([
                sys.executable,
                "scripts/keyframe_face_anchor_20260901.py",
                "--video", str(base),
                "--reference", str(REF),
                "--out", str(out),
                "--anchor_frames", "0,25,50,75",
                "--anchor_source", source,
                "--strength", str(strength),
                "--detail_strength", "0.55",
            ], LOG_DIR / f"{tag}.log", check=False)


def eval_one(video: Path, reference: Path, name: str, rows: list[dict[str, str]]) -> None:
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    txt_path = EVAL_OUT / f"{name}.txt"
    csv_path = EVAL_OUT / f"{name}.csv"
    sheet_path = EVAL_OUT / f"{name}_sheet.jpg"
    proc = subprocess.run([
        sys.executable,
        "eval/id_drift_metric.py",
        "--video", str(video),
        "--reference", str(reference),
        "--arcface_onnx", str(ARC),
        "--detector", "insightface",
        "--every", "5",
        "--out_csv", str(csv_path),
    ], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    txt_path.write_text(proc.stdout, encoding="utf-8")
    subprocess.run([
        sys.executable,
        "scripts/make_frame_sheet.py",
        "--video", str(video),
        "--out", str(sheet_path),
        "--frames", "0,25,50,75",
    ], cwd=ROOT, check=False)
    row = {"name": name, "video": str(video)}
    for pattern, keys in [
        (r"cos_ref\s+mean/min:\s+([0-9.+-]+)\s*/\s*([0-9.+-]+)", ("cos_mean", "cos_min")),
        (r"cos_ref\s+first/last:\s+([0-9.+-]+)\s*/\s*([0-9.+-]+)", ("cos_first", "cos_last")),
        (r"drift slope \(cos/s\):\s+([0-9.+-]+)", ("slope",)),
        (r"face detect(?:ion)? rate:\s+([0-9.+-]+)", ("detect_rate",)),
    ]:
        match = re.search(pattern, proc.stdout)
        if match:
            for idx, key in enumerate(keys, start=1):
                row[key] = match.group(idx)
    rows.append(row)


def run_eval() -> None:
    run_keyframe_postprocess()
    rows: list[dict[str, str]] = []
    for video in sorted(ROUND4_OUT.glob("*/*.mp4")):
        eval_one(video, REF, video.parent.name, rows)
    summary_path = EVAL_OUT / "summary.csv"
    fields = ["name", "cos_mean", "cos_min", "cos_first", "cos_last", "slope", "detect_rate", "video"]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] wrote {summary_path}", flush=True)


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
    for path in (ROUND4_OUT, EVAL_OUT, LOG_DIR, ASSET_DIR, CSV_DIR):
        path.mkdir(parents=True, exist_ok=True)
    if args.eval_only:
        run_eval()
        return
    if not args.skip_prepare:
        prepare_assets()
    if args.prepare_only:
        return
    run_generation(args)


if __name__ == "__main__":
    main()
