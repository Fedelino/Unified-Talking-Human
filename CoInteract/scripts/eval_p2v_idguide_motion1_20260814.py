import re
import subprocess
from pathlib import Path


ROOT = Path("/data1/workspace/linxinliang/CoInteract")
PKG = ROOT / "output_videos" / "p2v_idguide_isolated_ablation_motion1_20260814_package"
REF = Path("/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/ref_img/wholebody/fullImage1a3b5329d192f028c14bc03d20c24afa629a3123.jpg")
ARC = ROOT / "models" / "arcface" / "w600k_r50.onnx"


def parse_summary(text: str):
    mean_min = re.search(r"cos_ref  mean/min:\s*([-0-9.]+) / ([-0-9.]+)", text)
    first_last = re.search(r"cos_ref  first/last:\s*([-0-9.]+) / ([-0-9.]+)", text)
    slope = re.search(r"drift slope \(cos/s\):\s*([+-]?[0-9.]+)", text)
    if not (mean_min and first_last and slope):
        return None
    return {
        "mean": mean_min.group(1),
        "min": mean_min.group(2),
        "first": first_last.group(1),
        "last": first_last.group(2),
        "slope": slope.group(1),
    }


def main() -> None:
    rows = []
    for video in sorted(PKG.glob("*.mp4")):
        name = video.stem
        csv_path = PKG / f"{name}_arcface.csv"
        log_path = PKG / f"{name}_arcface.log"
        cmd = [
            "python",
            "eval/id_drift_metric.py",
            "--video", str(video),
            "--reference", str(REF),
            "--arcface_onnx", str(ARC),
            "--detector", "insightface",
            "--every", "5",
            "--out_csv", str(csv_path),
        ]
        print(f"SCORING {name}", flush=True)
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path.write_text(proc.stdout)
        if proc.returncode != 0:
            raise RuntimeError(f"ArcFace scoring failed for {name}; see {log_path}")
        parsed = parse_summary(proc.stdout)
        if parsed is None:
            raise RuntimeError(f"Could not parse ArcFace summary for {name}; see {log_path}")
        rows.append((name, parsed))

    summary = PKG / "arcface_summary.tsv"
    with summary.open("w", encoding="utf-8") as fh:
        fh.write("variant\tmean\tmin\tfirst\tlast\tdrift_slope\n")
        for name, metrics in rows:
            fh.write(
                f"{name}\t{metrics['mean']}\t{metrics['min']}\t"
                f"{metrics['first']}\t{metrics['last']}\t{metrics['slope']}\n"
            )
    print(summary)
    print(summary.read_text())


if __name__ == "__main__":
    main()
