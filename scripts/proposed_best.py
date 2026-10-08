"""Proposed-model (paper + LSK, F_lsk) testчув report: best-per-class selection.

1. Full test-split eval of the proposed checkpoint (P/R/mAP overall +
   per-class) -> metrics_proposed_test.json.
2. Predict every test image with the proposed model, greedily match
   predictions to ground truth (same class, IoU >= 0.5) and keep, per class,
   the 2 highest-confidence TRUE POSITIVES on distinct images ("best"
   = strongest correct detections) -> selected.json.
3. Predict the selected images with all 10 ablation checkpoints, match each
   arm against GT, and draw TP (green) / FP (red) / FN-missed-GT (magenta).
4. Grad-CAM (auto SPPF target) for every arm on the selected images with the
   same TP/FP/FN marks + legend, plus per-image 10-arm compare grids for
   detections and heatmaps.

Terminology note: "true negatives" (background correctly ignored) cannot be
drawn as boxes in detection; the explicit marks are TP / FP / FN (missed GT).

Outputs under runs/selected/:
    metrics_proposed_test.json, selected.json,
    det/<arm>/<stem>.jpg, cam/<arm>/<stem>.jpg,
    compare/<stem>_det.jpg, compare/<stem>_cam.jpg, SELECTED.md

Usage:
    .venv/bin/python scripts/proposed_best.py [--conf 0.25] [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROPOSED_CKPT = ROOT / "weights" / "F_lsk_proposed.pt"
ARMS = ["baseline", "A_spd", "B_neck", "C_faster", "D_wiou", "E_all",
        "F_lsk", "G_lsk", "H_eiou", "I_eiou", "proposed"]
# Canonical checkpoint per arm label. F_lsk here = stock+LSK (other scheme
# name); the PROPOSED model (paper arch + LSK) is `proposed`.
CKPTS = {
    "baseline": "runs/ablation/baseline/weights/best.pt",
    "A_spd": "runs/ablation/A_spd/weights/best.pt",
    "B_neck": "runs/ablation/B_neck/weights/best.pt",
    "C_faster": "runs/ablation/C_faster/weights/best.pt",
    "D_wiou": "runs/ablation/D_wiou/weights/best.pt",
    "E_all": "runs/ablation/E_all/weights/best.pt",
    "F_lsk": "runs/ablation/F_lsk/weights/best.pt",
    "G_lsk": "runs/ablation/G_lsk_stale_F_arch/weights/best.pt",
    "H_eiou": "runs/ablation/H_eiou/weights/best.pt",
    "I_eiou": "runs/ablation/I_eiou/weights/best.pt",
    "proposed": "weights/F_lsk_proposed.pt",
}
DATA = "data/rdd2022-cn.yaml"
TEST_IMG = ROOT / "data" / "rdd2022-cn" / "images" / "test"
TEST_LBL = ROOT / "data" / "rdd2022-cn" / "labels" / "test"

TP, FP, FN = (0, 255, 0), (0, 0, 255), (255, 0, 255)  # green / red / magenta


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iou", type=float, default=0.5,
                    help="IoU threshold for TP matching")
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "selected"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def box_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter)


def load_gt(stem: str, img_w: int, img_h: int):
    """Ground-truth boxes (x1,y1,x2,y2,cls) in pixels from the YOLO label file."""
    gts = []
    p = TEST_LBL / f"{stem}.txt"
    if p.exists():
        for line in p.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            c, xc, yc, w, h = int(parts[0]), *(float(v) for v in parts[1:5])
            gts.append(((xc - w / 2) * img_w, (yc - h / 2) * img_h,
                        (xc + w / 2) * img_w, (yc + h / 2) * img_h, c))
    return gts


def match(preds, gts, iou_thr: float):
    """Greedy match, preds sorted desc by conf. Returns (tp, fp, fn) index lists.

    Duplicate fragments of an already-matched GT count as FP (COCO
    convention); see match_detailed() for the dup/near/cls/bg split.
    """
    from src.explain.deteval import match_detailed as _md
    d = _md(preds, gts, iou_thr=iou_thr)
    return d["tp"], d["fp"], d["fn"]


def match_detailed(preds, gts, iou_thr: float = 0.5, near_thr: float = 0.3):
    from src.explain.deteval import match_detailed as _md
    return _md(preds, gts, iou_thr=iou_thr, near_thr=near_thr)


def predict_boxes(yolo, img_path, conf, imgsz):
    res = yolo.predict(source=str(img_path), conf=conf, imgsz=imgsz, verbose=False)[0]
    out = []
    if res.boxes is not None and len(res.boxes):
        for (x1, y1, x2, y2), c, p in zip(res.boxes.xyxy.cpu().numpy(),
                                          res.boxes.cls.cpu().numpy().astype(int),
                                          res.boxes.conf.cpu().numpy()):
            out.append((float(x1), float(y1), float(x2), float(y2), int(c), float(p)))
    return out


def draw_marks(img_bgr, preds, gts, tp, fp, fn, names):
    import cv2

    for pi in tp:
        x1, y1, x2, y2, c, cf = preds[pi]
        cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), TP, 2)
        cv2.putText(img_bgr, f"{names[c]} {cf:.2f} TP", (int(x1), max(0, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TP, 2)
    for pi in fp:
        x1, y1, x2, y2, c, cf = preds[pi]
        cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), FP, 2)
        cv2.putText(img_bgr, f"{names[c]} {cf:.2f} FP", (int(x1), max(0, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, FP, 2)
    for gi in fn:
        x1, y1, x2, y2, c = gts[gi]
        cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), FN, 3)
        cv2.putText(img_bgr, f"missed {names[c]} FN", (int(x1), max(0, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, FN, 2)


def draw_legend(img_bgr):
    import cv2
    import numpy as np

    items = [((0, 255, 0), "TP"), ((0, 0, 255), "FP"), ((255, 0, 255), "FN=missed GT")]
    w = 40 + max(len(t) for _, t in items) * 11 + 24 * len(items)
    legend = np.zeros((30, w, 3), dtype=np.uint8)
    x = 6
    for color, text in items:
        cv2.rectangle(legend, (x, 8), (x + 14, 22), color, -1)
        cv2.putText(legend, text, (x + 18, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        x += 18 + len(text) * 11 + 16
    h, ww = img_bgr.shape[:2]
    legend = legend[:, : min(w, ww)]
    img_bgr[h - 30:h, 0:legend.shape[1]] = legend
    return img_bgr


def caption_tile(img_bgr, text: str):
    import cv2
    import numpy as np

    bar = np.zeros((34, img_bgr.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return np.vstack([bar, img_bgr])


def grid(tiles, per_row=5, tile_h=420):
    """Tiles (already captioned) into a rows-of-`per_row` grid, aspect kept."""
    import cv2
    import numpy as np

    normed = []
    for t in tiles:
        w = round(t.shape[1] * tile_h / t.shape[0])
        normed.append(cv2.resize(t, (w, tile_h)))
    rows = []
    for i in range(0, len(normed), per_row):
        chunk = normed[i:i + per_row]
        wmax = max(t.shape[1] for t in chunk)
        padded = [cv2.copyMakeBorder(t, 0, 0, 0, wmax - t.shape[1],
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0)) for t in chunk]
        rows.append(np.hstack(padded))
    wmax = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, wmax - r.shape[1],
                               cv2.BORDER_CONSTANT, value=(0, 0, 0)) for r in rows]
    return np.vstack(rows)


def main():
    import cv2
    import numpy as np
    import torch

    from src.explain.gradcam import gradcam_heatmap, letterbox, overlay_heatmap
    from src.model import install

    install()
    from ultralytics import YOLO

    args = parse_args()
    out = pathlib.Path(args.out_dir)
    (out / "compare").mkdir(parents=True, exist_ok=True)
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")

    for label, ckpt in CKPTS.items():
        assert (ROOT / ckpt).exists(), f"missing checkpoint for {label}: {ckpt}"

    # ---- 1. full test eval of the proposed model ----
    print("[proposed] full test eval", flush=True)
    prop = YOLO(str(PROPOSED_CKPT))
    names = prop.model.names
    res = prop.val(data=DATA, split="test", imgsz=args.imgsz, batch=4,
                   device=device, plots=False, verbose=False)
    b = res.box
    per_class = {}
    for i in range(len(names)):
        p_i, r_i, ap50_i, ap_i = b.class_result(i)
        per_class[names[i]] = {"precision": float(p_i), "recall": float(r_i),
                               "ap50": float(ap50_i), "ap50_95": float(ap_i)}
    metrics = {"precision": float(b.mp), "recall": float(b.mr),
               "map50": float(b.map50), "map50_95": float(b.map),
               "per_class": per_class}
    (out / "metrics_proposed_test.json").write_text(json.dumps(metrics, indent=1))
    print(f"[proposed] test mAP50={metrics['map50']:.4f} "
          f"mAP50-95={metrics['map50_95']:.4f}", flush=True)

    # ---- 2. predict all test images, select top-2 TP per class ----
    sel_path = out / "selected.json"
    if sel_path.exists() and not args.force:
        selected = json.loads(sel_path.read_text())
        print("[select] reusing", sel_path, flush=True)
    else:
        tp_by_class: dict[int, list] = {i: [] for i in range(len(names))}
        test_imgs = sorted(TEST_IMG.glob("*.jpg"))
        for img_path in test_imgs:
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            gts = load_gt(img_path.stem, w, h)
            preds = sorted(predict_boxes(prop, img_path, args.conf, args.imgsz),
                           key=lambda r: -r[5])
            tp, _, _ = match(preds, gts, args.iou)
            for pi in tp:
                x1, y1, x2, y2, c, cf = preds[pi]
                tp_by_class[c].append({"image": img_path.name, "conf": cf,
                                       "box": [x1, y1, x2, y2]})
        selected = {}
        for c in range(len(names)):
            cands = sorted(tp_by_class[c], key=lambda r: -r["conf"])
            picks, seen = [], set()
            for r in cands:
                if r["image"] not in seen:
                    picks.append(r)
                    seen.add(r["image"])
                if len(picks) == 2:
                    break
            selected[names[c]] = picks
            print(f"[select] {names[c]}: {len(tp_by_class[c])} TPs -> "
                  + ", ".join(f"{p['image']}@{p['conf']:.2f}" for p in picks), flush=True)
        sel_path.write_text(json.dumps(selected, indent=1))

    sel_images = sorted({p["image"] for picks in selected.values() for p in picks})
    print(f"[select] {len(sel_images)} distinct images", flush=True)

    # ---- 3+4. cross-arm predictions + grad-cam with TP/FP/FN marks ----
    det = torch.device(device if device != "0" else "cuda:0")
    summary: dict[str, dict] = {}
    for arm in ARMS:
        ckpt = ROOT / CKPTS[arm]
        print(f"[{arm}] loading {ckpt}", flush=True)
        yolo = YOLO(str(ckpt))
        dm = yolo.model.eval()
        sppfs = [i for i, m in enumerate(dm.model) if type(m).__name__ == "SPPF"]
        layer = sppfs[-1]
        adev = next(dm.parameters()).device
        arm_det = out / "det" / arm
        arm_cam = out / "cam" / arm
        arm_det.mkdir(parents=True, exist_ok=True)
        arm_cam.mkdir(parents=True, exist_ok=True)
        summary[arm] = {}
        for img_name in sel_images:
            stem = pathlib.Path(img_name).stem
            dpath, cpath = arm_det / f"{stem}.jpg", arm_cam / f"{stem}.jpg"
            if dpath.exists() and cpath.exists() and not args.force:
                continue
            img_path = TEST_IMG / img_name
            orig = cv2.imread(str(img_path))
            h, w = orig.shape[:2]
            gts = load_gt(stem, w, h)
            preds = sorted(predict_boxes(yolo, img_path, args.conf, args.imgsz),
                           key=lambda r: -r[5])
            det = match_detailed(preds, gts, args.iou)
            tp, fp, fn = det["tp"], det["fp"], det["fn"]
            summary[arm][stem] = {"tp": len(tp), "fp": len(fp), "fn": len(fn),
                                  "fp_dup": len(det["fp_dup"]),
                                  "fp_near": len(det["fp_near"]),
                                  "fp_cls": len(det["fp_cls"]),
                                  "fp_bg": len(det["fp_bg"]),
                                  "fp_info": {str(k): v for k, v in det["fp_info"].items()}}

            det_img = orig.copy()
            draw_marks(det_img, preds, gts, tp, fp, fn, names)
            cv2.imwrite(str(dpath), draw_legend(det_img))

            padded, ratio, pad = letterbox(orig, args.imgsz)
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            tensor = (torch.from_numpy(rgb).permute(2, 0, 1)
                      .unsqueeze(0).float() / 255.0).to(adev)
            cam = gradcam_heatmap(dm, tensor, layer, conf=args.conf)
            overlay, _ = overlay_heatmap(orig, cam, ratio, pad)
            draw_marks(overlay, preds, gts, tp, fp, fn, names)
            cv2.imwrite(str(cpath), draw_legend(overlay))
            print(f"[{arm}] {stem} tp={len(tp)} fp={len(fp)} "
                  f"(dup={len(det['fp_dup'])} near={len(det['fp_near'])} "
                  f"cls={len(det['fp_cls'])} bg={len(det['fp_bg'])}) "
                  f"fn={len(fn)}", flush=True)
        del yolo, dm
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- 5. compare grids + report ----
    for img_name in sel_images:
        stem = pathlib.Path(img_name).stem
        cv2.imwrite(str(out / "compare" / f"{stem}_det.jpg"),
                    grid([caption_tile(cv2.imread(str(out / "det" / a / f"{stem}.jpg")), a)
                          for a in ARMS]))
        cv2.imwrite(str(out / "compare" / f"{stem}_cam.jpg"),
                    grid([caption_tile(cv2.imread(str(out / "cam" / a / f"{stem}.jpg")), a)
                          for a in ARMS]))

    lines = ["# Proposed-model best-per-class report (F_lsk = paper + LSK)\n",
             f"Proposed checkpoint: `weights/F_lsk_proposed.pt` "
             f"(paper arch + LSKBlock, SPPF@10, {PROPOSED_CKPT.stat().st_size / 2**20:.1f} MB).\n",
             "## Test-split metrics (proposed, all 437 images)\n",
             f"mAP50={metrics['map50']:.4f} mAP50-95={metrics['map50_95']:.4f} "
             f"P={metrics['precision']:.4f} R={metrics['recall']:.4f}\n"]
    for cls, vals in metrics["per_class"].items():
        lines.append(f"- {cls}: P={vals['precision']:.3f} R={vals['recall']:.3f} "
                     f"AP50={vals['ap50']:.4f} AP50-95={vals['ap50_95']:.4f}")
    lines.append("\n## Selected images (top-2 TP by confidence, distinct images)\n")
    for cls, picks in selected.items():
        for p in picks:
            s = pathlib.Path(p["image"]).stem
            row = " | ".join(f"{a}:{summary[a][s]['tp']}TP/"
                             f"{summary[a][s]['fp']}FP/{summary[a][s]['fn']}FN"
                             for a in ARMS)
            lines.append(f"- {cls} `{p['image']}` conf={p['conf']:.2f} :: {row}")
    lines += ["\nMarks: green box = TP, red = FP, magenta = FN (missed GT). "
              "True negatives (background correctly ignored) have no box to draw.\n",
              "`compare/<stem>_det.jpg` = detections grid, "
              "`compare/<stem>_cam.jpg` = Grad-CAM grid (11 arms incl. proposed).\n"]
    (out / "SELECTED.md").write_text("\n".join(lines))
    (out / "summary.json").write_text(json.dumps(
        {"metrics": metrics, "selected": selected, "per_arm": summary}, indent=1))
    print(f"wrote {out}/SELECTED.md + compare grids for {len(sel_images)} images", flush=True)


if __name__ == "__main__":
    main()
