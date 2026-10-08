"""Multilayer CAM for proposed model: P3/P4/P5 + fused, layercam/top8."""
import os
os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import pathlib, sys
ROOT = pathlib.Path("/home/sandy/Projects/PEP-26")
sys.path.insert(0, str(ROOT))
from src.model import install
install()
from ultralytics import YOLO
import cv2, torch
from src.explain.gradcam import backbone_pyramid, cam_heatmap, multiscale_cam, letterbox, overlay_heatmap, predict_boxes

CKPT = ROOT / "weights" / "F_lsk_proposed.pt"
OUT = pathlib.Path("/tmp/opencode/multilayer_cam")
OUT.mkdir(parents=True, exist_ok=True)
IMGS = ["China_MotorBike_000664.jpg", "China_Drone_000966.jpg"]  # thin cracks + hard case
IMGDIR = ROOT / "data" / "rdd2022-cn" / "images" / "test"

yolo = YOLO(str(CKPT))
det = yolo.model.eval()
adev = next(det.parameters()).device
p3, p4, p5 = backbone_pyramid(det)
print(f"pyramid P3={p3} ({type(det.model[p3]).__name__}) P4={p4} ({type(det.model[p4]).__name__}) P5={p5} ({type(det.model[p5]).__name__})", flush=True)
layers = {"P3": p3, "P4": p4, "P5-SPPF": p5}

for img_name in IMGS:
    stem = pathlib.Path(img_name).stem
    img_path = IMGDIR / img_name
    orig = cv2.imread(str(img_path))
    padded, ratio, pad = letterbox(orig, 640)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = (torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0).to(adev)
    cams = {}
    for lname, li in layers.items():
        cams[lname] = cam_heatmap(det, tensor, li, conf=0.25, method="layercam", topk=8)
    cams["FUSED-P3P4P5"] = multiscale_cam(det, tensor, [p3, p4, p5], method="layercam", topk=8, conf=0.25)
    boxes = predict_boxes(yolo, img_path, 0.25, 640)
    tiles = []
    for lname, cam in cams.items():
        overlay, heat = overlay_heatmap(orig, cam, ratio, pad)
        for x1, y1, x2, y2, c, cf in boxes:
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        bar = cv2.copyMakeBorder(overlay, 0, 0, 0, 0, cv2.BORDER_CONSTANT)
        import numpy as np
        cap = np.zeros((34, overlay.shape[1], 3), dtype=np.uint8)
        cv2.putText(cap, f"{stem} {lname} lyr={layers.get(lname, 'pyr')}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tile = np.vstack([cap, overlay])
        cv2.imwrite(str(OUT / f"{stem}_{lname}.jpg"), tile)
        tiles.append(tile)
        e = float(cam.sum())
        print(f"[{stem} {lname}] cam_sum={e:.1f} max={float(cam.max()):.3f} boxes={len(boxes)}", flush=True)
    # side-by-side compare of P3|P4|P5|FUSED
    h = max(t.shape[0] for t in tiles)
    normed = []
    for t in tiles:
        if t.shape[0] != h:
            w = round(t.shape[1] * h / t.shape[0])
            t = cv2.resize(t, (w, h))
        normed.append(t)
    import numpy as np
    cv2.imwrite(str(OUT / f"{stem}_COMPARE.jpg"), np.hstack(normed))
    print(f"wrote {OUT / f'{stem}_COMPARE.jpg'}", flush=True)
