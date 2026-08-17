#!/usr/bin/env python3
"""
ArcFace seed/trajectory selector for CoInteract.

This is training-free and gradient-free:
1. Generate short preview videos for several seeds.
2. Score each preview against the reference face with eval/id_drift_metric.py.
3. Re-run the same case once at full quality using the best seed.

It keeps ArcFace as a selector/metric instead of using ArcFace gradients to edit
the video latent.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd


MEAN_MIN_RE = re.compile(r"cos_ref\s+mean/min:\s+([-+0-9.]+)\s*/\s*([-+0-9.]+)")
FIRST_LAST_RE = re.compile(r"cos_ref\s+first/last:\s+([-+0-9.]+)\s*/\s*([-+0-9.]+)")
SLOPE_RE = re.compile(r"drift slope \(cos/s\):\s+([-+0-9.]+)")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--row_index", type=int, default=0)
    ap.add_argument("--data_base_path", default=".")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--batch_infer", default="batch_infer.py")
    ap.add_argument("--metric_script", default="eval/id_drift_metric.py")
    ap.add_argument("--arcface_onnx", default="models/arcface/w600k_r50.onnx")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--preview_steps", type=int, default=12)
    ap.add_argument("--final_steps", type=int, default=40)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--detector", default="insightface", choices=["insightface", "haar"])
    ap.add_argument("--score", default="mean", choices=["mean", "min", "last", "mean_min"],
                    help="Primary selector score.")
    ap.add_argument("--extra_args", default="",
                    help="Additional args passed to batch_infer.py, shell-style.")
    return ap.parse_args()


def resolve_path(path_value: str, base: str) -> str:
    path_value = str(path_value).strip()
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(base, path_value))


def sample_name_from_row(row) -> str:
    if "sample_id" in row and pd.notna(row["sample_id"]) and str(row["sample_id"]).strip():
        return str(row["sample_id"]).strip()
    return Path(str(row["person_image"])).stem


def write_one_row_csv(row, out_csv: str, sample_id: str):
    out_row = dict(row)
    out_row["sample_id"] = sample_id
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_row.keys()))
        writer.writeheader()
        writer.writerow(out_row)


def run_cmd(cmd, log_path: str):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}); see {log_path}")


def metric_for_video(args, video_path: str, reference_path: str, csv_path: str):
    out_csv = f"{csv_path}.metrics.csv"
    cmd = [
        sys.executable,
        args.metric_script,
        "--video", video_path,
        "--reference", reference_path,
        "--arcface_onnx", args.arcface_onnx,
        "--every", str(args.every),
        "--detector", args.detector,
        "--out_csv", out_csv,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    text = proc.stdout
    if proc.returncode != 0:
        raise RuntimeError(f"metric failed for {video_path}\n{text}")

    mean_min = MEAN_MIN_RE.search(text)
    first_last = FIRST_LAST_RE.search(text)
    slope = SLOPE_RE.search(text)
    if not mean_min or not first_last:
        raise RuntimeError(f"could not parse metric output for {video_path}\n{text}")
    mean = float(mean_min.group(1))
    min_v = float(mean_min.group(2))
    first = float(first_last.group(1))
    last = float(first_last.group(2))
    slope_v = float(slope.group(1)) if slope else 0.0
    return {
        "mean": mean,
        "min": min_v,
        "first": first,
        "last": last,
        "slope": slope_v,
        "metric_csv": out_csv,
        "metric_stdout": text,
    }


def selector_score(metrics, mode: str) -> float:
    if mode == "min":
        return metrics["min"]
    if mode == "last":
        return metrics["last"]
    if mode == "mean_min":
        return 0.7 * metrics["mean"] + 0.3 * metrics["min"]
    return metrics["mean"]


def run_generation(args, one_csv: str, out_dir: str, steps: int, seed: int, log_name: str):
    cmd = [
        sys.executable,
        args.batch_infer,
        "--csv_path", one_csv,
        "--data_base_path", args.data_base_path,
        "--output_dir", out_dir,
        "--num_inference_steps", str(steps),
        "--seed_base", str(seed),
    ]
    cmd.extend(shlex.split(args.extra_args))
    run_cmd(cmd, os.path.join(args.output_dir, "logs", log_name))


def main():
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv_path)
    if len(df) == 0:
        raise ValueError("csv_path has no rows")
    row = df.iloc[int(args.row_index)].to_dict()
    base_name = sample_name_from_row(row)
    reference_path = resolve_path(row["person_image"], args.data_base_path)
    seeds = [int(x.strip()) for x in args.seeds.replace(";", ",").split(",") if x.strip()]
    if not seeds:
        raise ValueError("no seeds provided")

    rows = []
    best = None
    for seed in seeds:
        sample_id = f"{base_name}_seed{seed:03d}_preview"
        one_csv = str(output_root / "csv" / f"{sample_id}.csv")
        preview_dir = str(output_root / "previews" / f"seed{seed:03d}")
        write_one_row_csv(row, one_csv, sample_id)
        print(f"[preview] seed={seed} steps={args.preview_steps}")
        run_generation(args, one_csv, preview_dir, args.preview_steps, seed, f"{sample_id}.log")
        video_path = os.path.join(preview_dir, f"{sample_id}.mp4")
        metrics = metric_for_video(args, video_path, reference_path, one_csv)
        score = selector_score(metrics, args.score)
        result = {
            "seed": seed,
            "score": score,
            "video": video_path,
            **metrics,
        }
        rows.append(result)
        if best is None or score > best["score"]:
            best = result
        print(
            f"[score] seed={seed} score={score:.4f} "
            f"mean={metrics['mean']:.4f} min={metrics['min']:.4f} last={metrics['last']:.4f}"
        )

    assert best is not None
    summary_path = output_root / "seed_selection.tsv"
    with open(summary_path, "w", newline="") as f:
        fieldnames = ["seed", "score", "mean", "min", "first", "last", "slope", "video", "metric_csv"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row_out in rows:
            writer.writerow({k: row_out.get(k) for k in fieldnames})

    final_sample_id = f"{base_name}_seed{best['seed']:03d}_arcselect"
    final_csv = str(output_root / "csv" / f"{final_sample_id}.csv")
    final_dir = str(output_root / "final")
    write_one_row_csv(row, final_csv, final_sample_id)
    print(f"[final] best_seed={best['seed']} steps={args.final_steps}")
    run_generation(args, final_csv, final_dir, args.final_steps, int(best["seed"]), f"{final_sample_id}.log")
    final_video = os.path.join(final_dir, f"{final_sample_id}.mp4")
    final_metrics = metric_for_video(args, final_video, reference_path, final_csv)
    print(
        f"[done] final={final_video} "
        f"mean={final_metrics['mean']:.4f} min={final_metrics['min']:.4f} last={final_metrics['last']:.4f}"
    )
    print(f"[summary] {summary_path}")


if __name__ == "__main__":
    main()
