#!/usr/bin/env python3
"""Identity-specific 3D-face retargeting of a DWPose driving video (training-free, Exp B).

Replaces the generic per-frame DWPose *face* landmarks with the REFERENCE person's own
3D face geometry (insightface landmark_3d_68), re-posed to each frame's head orientation and
aligned to the driving face box. Body + hands keep the driving motion. Re-renders a new
DWPose video that CoInteract can consume unchanged.

Hypothesis: at large head yaw the generic 68 landmarks encode a *generic* face; the reference
person's 3D face projected at that yaw encodes *this person's* nose/jaw/cheek geometry -> less
identity drift at profile. (Frontal motions will barely change -- diagnostic prints yaw range.)

Env: needs onnxruntime + cv2 + insightface + the StableAnimator DWPose detector on sys.path.
"""
import os, sys, argparse
import numpy as np
import cv2

STABLEANIMATOR_ROOT = "/data1/workspace/linxinliang/StableAnimator"
DWPOSE_ROOT = os.path.join(STABLEANIMATOR_ROOT, "DWPose")
DW_CKPT = os.path.join(STABLEANIMATOR_ROOT, "checkpoints/DWPose")
sys.path.insert(0, DWPOSE_ROOT)
os.environ.setdefault("DWPOSE_DET", os.path.join(DW_CKPT, "yolox_l.onnx"))
os.environ.setdefault("DWPOSE_POSE", os.path.join(DW_CKPT, "dw-ll_ucoco_384.onnx"))
# StableAnimator's DWPose helper loads checkpoint paths relative to its repo root.
os.chdir(STABLEANIMATOR_ROOT)

from dwpose_utils.dwpose_detector import dwpose_detector_aligned  # noqa: E402
from skeleton_extraction import draw_pose  # noqa: E402
from insightface.app import FaceAnalysis  # noqa: E402


def euler_R(yaw, pitch, roll):
    """insightface pose (deg): yaw about Y, pitch about X, roll about Z. Return 3x3."""
    y, p, r = np.deg2rad([yaw, pitch, roll])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def similarity_align(src, dst):
    """2D similarity (scale+rot+trans) mapping src(N,2)->dst(N,2), applied to src. Umeyama."""
    src = np.asarray(src, np.float64); dst = np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    var_s = (s0 ** 2).sum() / len(src)
    cov = (d0.T @ s0) / len(src)
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1; R = U @ Vt
    scale = np.trace(np.diag(S)) / (var_s + 1e-9)
    return (scale * (src @ R.T)) + (mu_d - scale * (mu_s @ R.T))


FACE_MESH_EDGES = [
    *[(i, i + 1) for i in range(0, 16)],
    *[(i, i + 1) for i in range(17, 21)],
    *[(i, i + 1) for i in range(22, 26)],
    *[(i, i + 1) for i in range(27, 30)],
    *[(i, i + 1) for i in range(31, 35)],
    (30, 33),
    *[(i, i + 1) for i in range(36, 41)],
    (41, 36),
    *[(i, i + 1) for i in range(42, 47)],
    (47, 42),
    *[(i, i + 1) for i in range(48, 59)],
    (59, 48),
    *[(i, i + 1) for i in range(60, 67)],
    (67, 60),
]


def overlay_face_mesh(pose_img_hwc_rgb: np.ndarray, face_norm: np.ndarray | None, alpha: float, radius: int):
    """Overlay a DWPose-compatible face wireframe from the retargeted identity landmarks."""
    if face_norm is None or len(face_norm) < 68 or alpha <= 0:
        return pose_img_hwc_rgb
    h, w = pose_img_hwc_rgb.shape[:2]
    pts = np.asarray(face_norm[:68], dtype=np.float32) * np.array([w, h], dtype=np.float32)
    overlay = pose_img_hwc_rgb.copy()
    color = (80, 220, 255)
    thickness = max(1, int(radius))
    for a, b in FACE_MESH_EDGES:
        p0 = tuple(np.round(pts[a]).astype(int))
        p1 = tuple(np.round(pts[b]).astype(int))
        if 0 <= p0[0] < w and 0 <= p0[1] < h and 0 <= p1[0] < w and 0 <= p1[1] < h:
            cv2.line(overlay, p0, p1, color, thickness, cv2.LINE_AA)
    for p in pts:
        x, y = np.round(p).astype(int)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(overlay, (int(x), int(y)), thickness, color, -1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, float(alpha), pose_img_hwc_rgb, 1.0 - float(alpha), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driving_video", required=True)  # raw RGB motion video
    ap.add_argument("--reference_image", required=True)
    ap.add_argument("--out_video", required=True)
    ap.add_argument("--height", type=int, default=832)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--max_frames", type=int, default=0, help="Optional cap before DWPose detection; 0 keeps all frames.")
    ap.add_argument("--blend", type=float, default=1.0, help="1=full identity landmarks, 0=driving")
    ap.add_argument("--mesh_overlay_alpha", type=float, default=0.0, help="Overlay identity face wireframe after DWPose rendering.")
    ap.add_argument("--mesh_overlay_radius", type=int, default=1, help="Line/point radius for identity face wireframe.")
    args = ap.parse_args()
    print(
        f"[method] blend={args.blend:.3f} mesh_alpha={args.mesh_overlay_alpha:.3f} "
        f"mesh_radius={args.mesh_overlay_radius}",
        flush=True,
    )

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    # reference person's canonical 3D face (un-rotate by its own head pose)
    ref_bgr = cv2.imread(args.reference_image)
    rf = max(app.get(ref_bgr), key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    ref_l3d = rf.landmark_3d_68.astype(np.float64)
    ref_c = ref_l3d - ref_l3d.mean(0)
    R_ref = euler_R(*rf.pose)
    canonical = ref_c @ R_ref  # ~frontalized reference 3D face
    print(f"[ref] head pose yaw,pitch,roll={rf.pose}", flush=True)

    cap = cv2.VideoCapture(args.driving_video)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
        if args.max_frames > 0 and len(frames) >= args.max_frames:
            break
    cap.release()
    print(f"[driving] {len(frames)} frames", flush=True)

    H, W = args.height, args.width
    out_frames = []
    yaws = []
    n_id, n_fallback = 0, 0
    for i, bgr in enumerate(frames):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pose = dwpose_detector_aligned(rgb)
        faces = pose["faces"]  # (P,68,2) normalized
        # estimate target head pose from the driving RGB
        det = app.get(bgr)
        if faces is not None and len(faces) > 0 and len(det) > 0:
            tf = max(det, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            yaws.append(float(tf.pose[0]))
            # reference face re-posed to this frame's orientation, projected to 2D
            posed = canonical @ euler_R(*tf.pose).T
            proj = posed[:, :2]
            drv_px = faces[0] * np.array([W, H])           # driving face in pixels
            aligned = similarity_align(proj, drv_px)        # overlay driving position/scale/roll
            new_face = aligned / np.array([W, H])           # back to normalized
            faces[0] = args.blend * new_face + (1 - args.blend) * faces[0]
            identity_face = faces[0].copy()
            pose["faces"] = faces
            n_id += 1
        else:
            identity_face = None
            n_fallback += 1
        canvas = draw_pose(pose, H, W)  # (3,H,W) rgb
        frame = canvas.transpose(1, 2, 0)
        frame = overlay_face_mesh(frame, identity_face, args.mesh_overlay_alpha, args.mesh_overlay_radius)
        out_frames.append(frame)

    if yaws:
        ya = np.array(yaws)
        print(f"[yaw] min={ya.min():.1f} max={ya.max():.1f} mean={ya.mean():.1f} |max|={np.abs(ya).max():.1f}  id_frames={n_id} fallback={n_fallback}", flush=True)

    os.makedirs(os.path.dirname(args.out_video), exist_ok=True)
    vw = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    for f in out_frames:
        vw.write(cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"[done] wrote {args.out_video}  ({len(out_frames)} frames)", flush=True)


if __name__ == "__main__":
    main()
