"""Precompute ALIGNED ArcFace embeddings for every person_image (facetools env).
insightface buffalo_l = SCRFD det_10g + 5-point norm_crop align + w600k_r50 recog.
Key = raw CSV person_image path (what train_consisid.py reads as data['person_image_path']).
Saves models/arcface/personimg_arcface_cache.npz : keys[str], embs[N,512] f32, detected[bool].
"""
import os, csv, numpy as np
from insightface.app import FaceAnalysis

BASE = 'data/ubcfashion_tiktok_pose_full'
CSV  = BASE + '/data.csv'
OUT  = 'models/arcface/personimg_arcface_cache.npz'

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))

import cv2
rows = list(csv.DictReader(open(CSV)))
keys, embs, det = [], [], []
n_ok = n_miss = 0
for i, r in enumerate(rows):
    rp = r['person_image']
    fp = os.path.normpath(os.path.join(BASE, rp))
    bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
    emb = None; found = False
    if bgr is not None:
        faces = app.get(bgr)  # SCRFD detect + align + w600k recog
        if faces:
            f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            emb = f.normed_embedding.astype(np.float32); found = True
    if emb is None:
        emb = np.zeros(512, np.float32)
    keys.append(rp); embs.append(emb); det.append(found)
    n_ok += found; n_miss += (not found)
    if (i+1) % 100 == 0:
        print(f'{i+1}/{len(rows)}  detected={n_ok} missed={n_miss}', flush=True)

np.savez(OUT, keys=np.array(keys), embs=np.stack(embs), detected=np.array(det))
print(f'DONE -> {OUT}  total={len(rows)} detected={n_ok} missed={n_miss}', flush=True)
