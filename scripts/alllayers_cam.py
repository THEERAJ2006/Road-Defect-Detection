"""All-layers CAM for proposed model: per-layer + fused-max-accuracy output."""
import os
os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import pathlib, sys
ROOT = pathlib.Path("/home/sandy/Projects/PEP-26")
sys.path.insert(0, str(ROOT))
from src.model import install
install()
from ultralytics import YOLO
import cv2, torch, numpy as np
from src.explain import deteval as D
from src.explain.gradcam import (all_cam_layers, cam_heatmap, letterbox,
                                 overlay_heatmap, predict_boxes, unpad_cam)

CKPT = ROOT / "weights" / "F_lsk_proposed.pt"
OUT = pathlib.Path("/tmp/opencode/alllayers_cam")
(OUT / "tiles").mkdir(parents=True, exist_ok=True)
IMGS = ["China_MotorBike_000664.jpg", "China_Drone_000966.jpg"]
IMGDIR = ROOT / "data" / "rdd2022-cn" / "images" / "test"
LBLDIR = ROOT / "data" / "rdd2022-cn" / "labels" / "test"

yolo = YOLO(str(CKPT))
det = yolo.model.eval()
adev = next(det.parameters()).device
layers = all_cam_layers(det)
print(f"layers({len(layers)}): " + ",".join(
    f"{i}{type(det.model[i]).__name__}" for i in layers), flush=True)

for img_name in IMGS:
    stem = pathlib.Path(img_name).stem
    img_path = IMGDIR / img_name
    orig = cv2.imread(str(img_path))
    h, w = orig.shape[:2]
    gts = D.load_gt(LBLDIR, stem, w, h)
    padded, ratio, pad = letterbox(orig, 640)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = (torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0).to(adev)
    per, energies = {}, {}
    for li in layers:
        try:
            cam = cam_heatmap(det, tensor, li, conf=0.25, method="layercam", topk=8)
        except Exception as e:
            print(f"[{stem} L{li}] SKIP {e}", flush=True)
            continue
        per[li] = cam
        cu = unpad_cam(cam, ratio, pad, (h, w))
        tot = float(cu.sum())
        ys, xs = np.mgrid[0:h, 0:w]
        inside = np.zeros_like(cu, dtype=bool)
        for x1, y1, x2, y2, _c in gts:
            inside |= (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)
        energies[li] = float(cu[inside].sum()) / tot if tot > 0 else 0.0
    fused = sum(per.values()) / len(per)
    fused -= fused.min()
    if fused.max() > 0:
        fused /= fused.max()
    fu = unpad_cam(fused.astype(np.float32), ratio, pad, (h, w))
    tot = float(fu.sum())
    ys, xs = np.mgrid[0:h, 0:w]
    inside = np.zeros_like(fu, dtype=bool)
    for x1, y1, x2, y2, _c in gts:
        inside |= (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)
    e_fused = float(fu[inside].sum()) / tot if tot > 0 else 0.0
    rank = sorted(energies.items(), key=lambda kv: -kv[1])
    print(f"[{stem}] fused-energy={e_fused:.3f} | " + " ".join(
        f"L{i}{type(det.model[i]).__name__}={e:.2f}" for i, e in rank), flush=True)
    boxes = predict_boxes(yolo, img_path, 0.25, 640)
    # save key tiles: top-3 layers + SPPF + LSK + Add/SSFF fusion + FUSED
    lsk = next(i for i in layers if type(det.model[i]).__name__ == "LSKBlock")
    sppf = max(i for i in layers if type(det.model[i]).__name__ == "SPPF")
    add = next((i for i in layers if type(det.model[i]).__name__ == "Add"), None)
    ssff = next((i for i in layers if type(det.model[i]).__name__ == "ScaleSequenceFusion"), None)
    picks = [("LSK", lsk), ("SPPF", sppf)]
    if ssff is not None: picks.append(("SSFF", ssff))
    if add is not None: picks.append(("Add", add))
    picks += [(f"TOP{k+1}", i) for k, (i, _) in enumerate(rank[:2])]
    picks.append(("FUSED-ALL", None))
    tiles = []
    for name, li in picks:
        cam = fused if li is None else per[li]
        overlay, _ = overlay_heatmap(orig, cam.astype(np.float32), ratio, pad)
        for x1, y1, x2, y2, c, cf in boxes:
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cap = np.zeros((34, overlay.shape[1], 3), dtype=np.uint8)
        tag = f"{stem} {name}" + (f" L{li} e={energies[li]:.2f}" if li is not None else f" e={e_fused:.2f}")
        cv2.putText(cap, tag, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tile = np.vstack([cap, overlay])
        cv2.imwrite(str(OUT / "tiles" / f"{stem}_{name}.jpg"), tile)
        tiles.append(tile)
    hh = max(t.shape[0] for t in tiles)
    normed = []
    for t in tiles:
        if t.shape[0] != hh:
            t = cv2.resize(t, (round(t.shape[1] * hh / t.shape[0]), hh))
        normed.append(t)
    # 2 rows for readability
    r1 = np.hstack(normed[:4])
    r2 = np.hstack(normed[4:])
    W = max(r1.shape[1], r2.shape[1])
    def padrow(r):
        return cv2.copyMakeBorder(r, 0, 0, 0, W - r.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.imwrite(str(OUT / f"{stem}_ALLLAYERS.jpg"), np.vstack([padrow(r1), padrow(r2)]))
    print(f"wrote {OUT / f'{stem}_ALLLAYERS.jpg'}", flush=True)
