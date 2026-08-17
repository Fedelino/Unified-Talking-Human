import csv
import subprocess
from pathlib import Path


ROOT = Path("/data1/workspace/linxinliang/CoInteract")
PACKAGE = ROOT / "output_videos" / "p2v_idguide_sweep_motion2_20260813_package"
REF = Path("/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/ref_img/wholebody/fullImage1a3b5329d192f028c14bc03d20c24afa629a3123.jpg")
ARC = ROOT / "models" / "arcface" / "w600k_r50.onnx"


def summarize_csv(csv_path: Path):
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = row.get("cosine_to_reference") or row.get("cosine")
            if not value or value.lower() == "nan":
                continue
            try:
                time_value = float(row.get("time_sec", len(rows)))
                rows.append((time_value, float(value)))
            except ValueError:
                continue
    if not rows:
        return None
    vals = [v for _, v in rows]
    mean = sum(vals) / len(vals)
    min_v = min(vals)
    first = vals[0]
    last = vals[-1]
    xs = [x for x, _ in rows]
    xmean = sum(xs) / len(xs)
    denom = sum((x - xmean) ** 2 for x in xs)
    slope = 0.0 if denom == 0 else sum((x - xmean) * (y - mean) for x, y in rows) / denom
    return {
        "faces": len(vals),
        "mean": mean,
        "min": min_v,
        "first": first,
        "last": last,
        "drift_slope": slope,
    }


def main() -> None:
    rows = []
    for video_path in sorted(PACKAGE.glob("*.mp4")):
        label = video_path.stem
        csv_path = PACKAGE / f"{label}_arcface.csv"
        log_path = PACKAGE / f"{label}_arcface.log"
        cmd = [
            "python",
            "eval/id_drift_metric.py",
            "--video",
            str(video_path),
            "--reference",
            str(REF),
            "--arcface_onnx",
            str(ARC),
            "--detector",
            "insightface",
            "--every",
            "5",
            "--out_csv",
            str(csv_path),
        ]
        result = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log_path.write_text(result.stdout, encoding="utf-8")
        summary = summarize_csv(csv_path)
        if summary is None:
            rows.append([label, "nan", "nan", "nan", "nan", "nan", "0"])
        else:
            rows.append([
                label,
                f"{summary['mean']:.6f}",
                f"{summary['min']:.6f}",
                f"{summary['first']:.6f}",
                f"{summary['last']:.6f}",
                f"{summary['drift_slope']:.6f}",
                str(summary["faces"]),
            ])

    summary_path = PACKAGE / "arcface_summary.tsv"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("variant\tmean\tmin\tfirst\tlast\tdrift_slope\tfaces\n")
        for row in rows:
            handle.write("\t".join(row) + "\n")
    print(summary_path)
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
