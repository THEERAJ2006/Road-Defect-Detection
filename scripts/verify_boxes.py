"""Verify every predicted box has gradient/CAM support (proposed model).

For each of the 10 selected images: predict all boxes, TP/FP-match vs GT,
then for EACH box compute a per-box LayerCAM at the best layer (L22 neck
output) and measure heat energy inside its own box + hottest-pixel hit.
A box with ~zero energy has no gradient path supporting it.
"""
import os
os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import json, pathlib, sys
ROOT = pathlib.Path("/home/sandy/Projects/PEP-26")
sys.path.insert(0, str(ROOT))
from src.model import install
install()
from ultralytics import YOLO
import cv2, torch, numpy as np
from src.explain import deteval as D
from src.explain.gradcam import (letterbox, overlay_heatmap, predict_boxes,
                                 box_target_cam, unpad_cam, threshold_cam)

CKPT = ROOT / "weights" / "F_lsk_proposed.pt"
OUT = pathlib.Path("/tmp/opencode/verify_boxes")
(OUT / "tiles").mkdir(parents=True, exist_ok=True)
IMGDIR = ROOT / "data" / "rdd2022-cn" / "images" / "test"
LBLDIR = ROOT / "data" / "rdd2022-cn" / "labels" / "test"
selected = json.loads((ROOT / "runs" / "selected" / "selected.json").read_text())
sel_images = sorted({p["image"] for picks in selected.values() for p in picks})

yolo = YOLO(str(CKPT))
names = yolo.model.names
det = yolo.model.eval()
adev = next(det.parameters()).device
L22 = 22  # best single layer from all-layers benchmark
print(f"target layer L{L22} {type(det.model[L22]).__name__}", flush=True)

def box_energy(cam_lb, ratio, pad, hw, box):
    cam = unpad_cam(cam_lb, ratio, pad, hw)
    tot = float(cam.sum())
    if tot <= 0:
        return 0.0, 0, float(cam.max()) if cam.size else 0.0
    x1, y1, x2, y2 = box[:4]
    h, w = cam.shape
    ys, xs = np.mgrid[0:h, 0:w]
    inside = (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)
    my, mx = np.unravel_index(int(cam.argmax()), cam.shape)
    return float(cam[inside].sum()) / tot, int(inside[my, mx]), float(cam.max())

rows = []
n_unsupported = 0
for img_name in sel_images:
    stem = pathlib.Path(img_name).stem
    imgp = IMGDIR / img_name
    orig = cv2.imread(str(imgp))
    h, w = orig.shape[:2]
    gts = D.load_gt(LBLDIR, stem, w, h)
    preds = sorted(predict_boxes(yolo, imgp, 0.25, 640), key=lambda r: -r[5])
    tp, fp, fn = D.match(preds, gts, 0.5)
    tp, fp = set(tp), set(fp)
    padded, ratio, pad = letterbox(orig, 640)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    t = (torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0).to(adev)
    dw, dh = pad
    for bi, (x1, y1, x2, y2, c, cf) in enumerate(preds):
        lb = (x1 * ratio + dw, y1 * ratio + dh, x2 * ratio + dw, y2 * ratio + dh)
        cam = box_target_cam(det, t, L22, lb, method="layercam")
        e, hit, mx = box_energy(cam, ratio, pad, (h, w), (x1, y1, x2, y2))
        status = "TP" if bi in tp else "FP"
        flag = "" if e >= 0.10 else "  <-- WEAK/NO SUPPORT"
        if e < 0.10:
            n_unsupported += 1
        rows.append(dict(img=img_name, box=bi, status=status, cls=names[c],
                         conf=round(cf, 3), energy=round(e, 3), hit=hit))
        print(f"{img_name} b{bi} {status} {names[c]}@{cf:.2f} energy={e:.3f} hit={hit}{flag}", flush=True)
        overlay, _ = overlay_heatmap(orig, threshold_cam(cam), ratio, pad)
        D.draw_marks(overlay, preds, gts, sorted(tp), sorted(fp), fn, names)
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 3)
        cv2.putText(overlay, f"target b{bi} e={e:.2f}", (int(x1), max(0, int(y1) - 24)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(OUT / "tiles" / f"{stem}_b{bi}_{status}_e{e:.2f}.jpg"),
                    D.draw_legend(overlay))

(OUT / "verify_boxes.json").write_text(json.dumps(rows, indent=1))
n = len(rows)
weak = sum(1 for r in rows if r["energy"] < 0.10)
zero = sum(1 for r in rows if r["energy"] <= 0.001)
print(f"\nTOTAL boxes={n} TP={sum(1 for r in rows if r['status']=='TP')} "
      f"FP={sum(1 for r in rows if r['status']=='FP')} | weak(<0.10)={weak} zero={zero}", flush=True)
print(f"mean energy TP={np.mean([r['energy'] for r in rows if r['status']=='TP']):.3f} "
      f"FP={np.mean([r['energy'] for r in rows if r['status']=='FP']):.3f}", flush=True)
