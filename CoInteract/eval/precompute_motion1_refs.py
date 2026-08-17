import csv, numpy as np, cv2
from insightface.app import FaceAnalysis
CSV='examples/cointeract_fullbody4_motion1_m2v_prompted_20260628.csv'
OUT='models/arcface/motion1_refs_arcface.npz'
app=FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); app.prepare(ctx_id=-1, det_size=(640,640))
rows=list(csv.DictReader(open(CSV)))
keys,embs,det=[],[],[]
for r in rows:
    p=r['person_image']  # absolute path, used verbatim as cache key
    bgr=cv2.imread(p, cv2.IMREAD_COLOR); emb=None; found=False
    if bgr is not None:
        fs=app.get(bgr)
        if fs:
            f=max(fs, key=lambda x:(x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            emb=f.normed_embedding.astype(np.float32); found=True
    if emb is None: emb=np.zeros(512,np.float32)
    keys.append(p); embs.append(emb); det.append(found)
    print(p.split('/')[-1], 'detected=', found, flush=True)
np.savez(OUT, keys=np.array(keys), embs=np.stack(embs), detected=np.array(det))
print('DONE ->', OUT, 'detected', sum(det), '/', len(rows), flush=True)
