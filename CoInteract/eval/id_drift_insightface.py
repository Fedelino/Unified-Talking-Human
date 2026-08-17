#!/usr/bin/env python3
"""
ArcFace ID-drift metric using insightface FaceAnalysis (buffalo_l: det_10g + w600k_r50).
Properly aligned 5-point face crops -> reliable cosine. Run in the `facetools` env.

Reports, over sampled frames of a generated video:
  - cos_ref   : cosine(face_t, reference image face)   -> absolute identity preservation
  - cos_first : cosine(face_t, first valid frame face)  -> self-consistency / drift
  - drift slope of cos_ref over time (cos/sec); negative = identity drifting away
"""
import argparse, os, sys, csv
import numpy as np
import cv2

def cos(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    def biggest_emb(img_bgr):
        faces = app.get(img_bgr)
        if not faces:
            return None
        f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        return f.normed_embedding

    ref = cv2.imread(args.reference)
    if ref is None:
        print(f"[fatal] cannot read reference {args.reference}", file=sys.stderr); sys.exit(1)
    ref_emb = biggest_emb(ref)
    if ref_emb is None:
        print("[fatal] no face detected in reference image", file=sys.stderr); sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    rows, first_emb = [], None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.every == 0:
            emb = biggest_emb(frame)
            if emb is not None:
                if first_emb is None:
                    first_emb = emb
                rows.append((idx, idx/args.fps, cos(emb, ref_emb), cos(emb, first_emb)))
        idx += 1
    cap.release()

    if not rows:
        print(f"[{args.tag}] NO FACES DETECTED in {total} frames"); return
    t = np.array([r[1] for r in rows]); cr = np.array([r[2] for r in rows])
    slope = float(np.polyfit(t, cr, 1)[0]) if len(rows) > 1 else 0.0
    det_rate = len(rows) / max(1, (total + args.every - 1)//args.every)
    print(f"[{args.tag}] frames_with_face={len(rows)}/{(total+args.every-1)//args.every} (det_rate={det_rate:.2f})")
    print(f"[{args.tag}] cos_ref: mean={cr.mean():.4f} min={cr.min():.4f} first={cr[0]:.4f} last={cr[-1]:.4f}")
    print(f"[{args.tag}] drift slope (cos_ref/sec) = {slope:+.5f}   (negative = drifting away)")
    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["frame","t_sec","cos_ref","cos_first"]); w.writerows(rows)
        print(f"[{args.tag}] wrote {args.out_csv}")

if __name__ == "__main__":
    main()
