#!/usr/bin/env bash
set -euo pipefail

cd /data1/workspace/linxinliang/CoInteract
source /data1/miniconda3/etc/profile.d/conda.sh
conda activate cointeract

DATE="20260801"
REF="/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/ref_img/wholebody/fullImage1a3b5329d192f028c14bc03d20c24afa629a3123.jpg"
ARC="models/arcface/w600k_r50.onnx"
OUTROOT="output_videos/trfree_cfg_motion5_ranked_${DATE}"
mkdir -p "$OUTROOT"

python - <<'PY'
import csv
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw

ref = "/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/ref_img/wholebody/fullImage1a3b5329d192f028c14bc03d20c24afa629a3123.jpg"
arc = "models/arcface/w600k_r50.onnx"
outroot = Path("output_videos/trfree_cfg_motion5_ranked_20260801")
outroot.mkdir(parents=True, exist_ok=True)
items = [
    ("3p5", "output_videos/trfree_cfg_motion5_cfg3p5_20260801/qalign_fb01_motion5.mp4"),
    ("5p0", "output_videos/trfree_cfg_motion5_cfg5p0_20260801/qalign_fb01_motion5.mp4"),
    ("6p0", "output_videos/trfree_cfg_motion5_cfg6p0_20260801/qalign_fb01_motion5.mp4"),
    ("7p0", "output_videos/trfree_seedrank_motion5_seed0_20260801/qalign_fb01_motion5.mp4"),
    ("8p0", "output_videos/trfree_cfg_motion5_cfg8p0_20260801/qalign_fb01_motion5.mp4"),
]

for label, video in items:
    csv_path = outroot / f"cfg{label}_arcface.csv"
    log_path = outroot / f"cfg{label}_arcface.log"
    if not Path(video).exists():
        log_path.write_text(f"missing {video}\n")
        continue
    proc = subprocess.run([
        "python", "eval/id_drift_metric.py",
        "--video", video,
        "--reference", ref,
        "--arcface_onnx", arc,
        "--detector", "insightface",
        "--every", "5",
        "--out_csv", str(csv_path),
    ], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(proc.stdout)
    print(proc.stdout)

rows = []
for label, video in items:
    csv_path = outroot / f"cfg{label}_arcface.csv"
    vals = []
    ts = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("detected", "1")) in {"0", "False", "false"}:
                    continue
                raw = row.get("cos_ref") or row.get("cosine_to_reference") or row.get("cosine")
                if raw in (None, "", "nan"):
                    continue
                vals.append(float(raw))
                ts.append(float(row.get("time_s") or row.get("time_sec") or len(ts)))
    if vals:
        n = len(vals)
        mean = sum(vals) / n
        mn = min(vals)
        first = vals[0]
        last = vals[-1]
        xmean = sum(ts) / n
        ymean = mean
        denom = sum((x - xmean) ** 2 for x in ts)
        slope = 0.0 if denom == 0 else sum((x - xmean) * (y - ymean) for x, y in zip(ts, vals)) / denom
        rows.append({"cfg": label, "mean": mean, "min": mn, "first": first, "last": last, "drift_slope": slope, "faces": n, "video": video})
    else:
        rows.append({"cfg": label, "mean": float("nan"), "min": float("nan"), "first": float("nan"), "last": float("nan"), "drift_slope": float("nan"), "faces": 0, "video": video})

rows.sort(key=lambda r: (r["min"] if r["min"] == r["min"] else -999, r["mean"] if r["mean"] == r["mean"] else -999), reverse=True)
with open(outroot / "ranking.tsv", "w") as f:
    f.write("rank\tcfg\tmean\tmin\tfirst\tlast\tdrift_slope\tfaces\tvideo\n")
    for i, row in enumerate(rows, 1):
        f.write("\t".join([
            str(i), row["cfg"], f"{row['mean']:.6f}", f"{row['min']:.6f}",
            f"{row['first']:.6f}", f"{row['last']:.6f}", f"{row['drift_slope']:.6f}",
            str(row["faces"]), row["video"],
        ]) + "\n")
print((outroot / "ranking.tsv").read_text())

frames = [0, 25, 50, 75]
extract_dir = outroot / "frames"
extract_dir.mkdir(exist_ok=True)
for row in rows:
    for frame in frames:
        img = extract_dir / f"cfg{row['cfg']}_f{frame:03d}.jpg"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", row["video"],
            "-vf", f"select=eq(n\\,{frame})", "-frames:v", "1", str(img)
        ], check=False)

thumbs = []
for row in rows:
    imgs = []
    for frame in frames:
        p = extract_dir / f"cfg{row['cfg']}_f{frame:03d}.jpg"
        if not p.exists():
            continue
        im = Image.open(p).convert("RGB")
        im.thumbnail((240, 320))
        canvas = Image.new("RGB", (240, 340), "white")
        canvas.paste(im, ((240 - im.width) // 2, 20))
        d = ImageDraw.Draw(canvas)
        d.text((8, 4), f"CFG {row['cfg']} frame {frame}", fill=(0, 0, 0))
        imgs.append(canvas)
    if imgs:
        row_canvas = Image.new("RGB", (240 * len(imgs), 340), "white")
        for i, im in enumerate(imgs):
            row_canvas.paste(im, (240 * i, 0))
        thumbs.append(row_canvas)

if thumbs:
    sheet = Image.new("RGB", (max(t.width for t in thumbs), sum(t.height for t in thumbs)), "white")
    y = 0
    for im in thumbs:
        sheet.paste(im, (0, y))
        y += im.height
    sheet.save(outroot / "cfg_frame_sheet.jpg", quality=95)
PY
