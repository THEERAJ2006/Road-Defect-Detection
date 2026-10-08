"""Best target-specific Grad-CAM example for the FINAL Proposed model (RSD-LSK).

Architecture (read off the real checkpoint ``weights/F_lsk_proposed.pt``):
  24 layers; LSKBlock@4 (after the P3 SPDConv, feeding the P3 backbone);
  SPPF@10; Detect@23 with ``.f = [16, 19, 22]``. The highest-resolution
  feature before the detection head is layer 16 (``Add``, 128ch @ P3/8),
  fused from the LSK-enhanced P3 path -- the preferred Grad-CAM target.
  Because each Detect head owns one anchor scale, the CAM is evaluated at
  all three head inputs [16, 19, 22] and the max box-energy head is kept
  (P3 preferred on ties); the used layer is printed and logged.

Honesty rules (enforced):
  * No retraining, no weight/architecture/dataset changes.
  * Detection verified FIRST: only genuine TPs (same class, IoU >= --iou).
  * Target-specific Grad-CAM: gradient target = the selected detection's own
    class score over its own anchors (``box_target_cam``, vanilla Grad-CAM).
  * The heatmap is NEVER masked/painted/scaled by GT or predicted boxes.
  * Grad-CAM is computed on the FULL image; the predicted-box crop (15%
    padding) is ONLY for display.

Pipeline: full image -> RSD-LSK -> selected TP detection ->
target-specific Grad-CAM (full image) -> crop for figure.

Outputs (default --out-dir ``outputs/``):
  proposed_rsd_lsk_gradcam_best.png    (crop | overlay | heat)
  proposed_rsd_lsk_gradcam_candidates.csv
  proposed_rsd_lsk_best_input.jpg      (full original test image)

Usage:
  python scripts/proposed_gradcam_best.py
  python scripts/proposed_gradcam_best.py --conf 0.25 --iou 0.5 --max-candidates 30
"""

from __future__ import annotations

import argparse
import csv
import datetime
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
DATA_YAML = "data/rdd2022-cn.yaml"
TEST_IMG_DIR = ROOT / "data" / "rdd2022-cn" / "images" / "test"
TEST_LBL_DIR = ROOT / "data" / "rdd2022-cn" / "labels" / "test"

PREFER_LAYER = 16  # Add: LSK-enhanced P3-level feature @ /8, P3 head input
CROP_PAD = 0.15
DPI = 300


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5,
                    help="IoU threshold for TP matching")
    ap.add_argument("--iou-want", type=float, default=0.70)
    ap.add_argument("--conf-want", type=float, default=0.80)
    ap.add_argument("--inside-want", type=float, default=0.60)
    ap.add_argument("--max-candidates", type=int, default=30,
                    help="TPs (ranked by IoU, conf) for which CAM is computed")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out-dir", default=str(ROOT / "outputs"))
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def box_energy_in(cam_orig, box):
    """Fraction of heat mass inside box (original-pixel coords). Genuine
    measurement on the unmodified heatmap -- never used to alter it."""
    import numpy as np
    total = float(cam_orig.sum())
    if total <= 0:
        return 0.0
    x1, y1, x2, y2 = box[:4]
    h, w = cam_orig.shape
    ys, xs = np.mgrid[0:h, 0:w]
    inside = (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)
    return float(cam_orig[inside].sum()) / total


def main():
    import cv2
    import numpy as np
    import torch

    from src.explain.deteval import box_iou, load_gt
    from src.explain.gradcam import (box_target_cam, letterbox, predict_boxes,
                                     unpad_cam)
    from src.model import install
    install()
    from ultralytics import YOLO

    args = parse_args()
    assert PROPOSED_CKPT.exists(), f"missing proposed checkpoint {PROPOSED_CKPT}"
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")

    yolo = YOLO(str(PROPOSED_CKPT))
    names = [yolo.model.names[i] for i in range(len(yolo.model.names))]
    dm = yolo.model.eval()
    dev = torch.device("cuda:0" if (torch.cuda.is_available()
                                    and device != "cpu") else "cpu")
    dm.to(dev)  # explicit: raw forward for CAM must match tensor device
    adev = next(dm.parameters()).device
    assert adev == dev, f"model device {adev} != {dev}"
    types = [type(m).__name__ for m in dm.model]
    from src.explain.gradcam import detect_head_layers
    heads = detect_head_layers(dm)  # [P3-in, P4-in, P5-in]
    assert PREFER_LAYER in heads, f"layer {PREFER_LAYER} not a Detect input"
    print(f"Grad-CAM target layer: {PREFER_LAYER} "
          f"({types[PREFER_LAYER]}, LSK-enhanced P3 @ /8, P3 head input); "
          f"head inputs evaluated: {heads} on {adev}", flush=True)

    # ---- 1. genuine TP detections over the whole TEST split -----------------
    test_imgs = sorted(p for p in TEST_IMG_DIR.glob("*")
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    tps = []  # dicts: image/stem/box/cls/conf/iou
    for img_path in test_imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = load_gt(TEST_LBL_DIR, img_path.stem, w, h)
        if not gts:
            continue
        preds = sorted(predict_boxes(yolo, img_path, args.conf, args.imgsz),
                       key=lambda r: -r[5])
        used = [False] * len(gts)
        for (x1, y1, x2, y2, c, cf) in preds:
            best, bj = args.iou, -1
            for gi, (gx1, gy1, gx2, gy2, gc) in enumerate(gts):
                if used[gi] or gc != c:
                    continue
                v = box_iou((x1, y1, x2, y2), (gx1, gy1, gx2, gy2))
                if v >= best:
                    best, bj = v, gi
            if bj >= 0:
                used[bj] = True
                tps.append({"image": img_path.name, "stem": img_path.stem,
                            "box": (x1, y1, x2, y2), "cls": c, "conf": cf,
                            "iou": best})
    print(f"[detect] {len(tps)} genuine TPs on {len(test_imgs)} test images",
          flush=True)
    pre = sorted(tps, key=lambda r: (-r["iou"], -r["conf"]))[:args.max_candidates]
    dm.to(dev)  # re-sync: YOLO.predict may have relocated weights
    dm.eval()
    # Anchors/strides cached by Detect during yolo.predict are inference
    # tensors (no backward possible). Clone them into normal tensors --
    # values bit-identical, only the tensor type changes (the workaround
    # PyTorch itself suggests). No model behavior is altered.
    det_head = dm.model[-1]
    det_head.anchors = det_head.anchors.clone()
    det_head.strides = det_head.strides.clone()
    with torch.enable_grad():
        dm(torch.zeros(1, 3, args.imgsz, args.imgsz, device=dev))

    # ---- 2. target-specific CAM per candidate (full image, unmodified) ------
    rows = []
    for r in pre:
        img_path = TEST_IMG_DIR / r["image"]
        orig = cv2.imread(str(img_path))
        padded, ratio, pad = letterbox(orig, args.imgsz)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = (torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
                  .float() / 255.0).to(adev)
        x1, y1, x2, y2 = r["box"]
        dw, dh = pad
        lb = (x1 * ratio + dw, y1 * ratio + dh,
              x2 * ratio + dw, y2 * ratio + dh)
        cams, energies = {}, {}
        for li in heads:
            cam = box_target_cam(dm, tensor, li, lb, target_class=r["cls"],
                                 method="gradcam", iou_thr=0.5, topk=3)
            co = unpad_cam(cam, ratio, pad, orig.shape[:2])
            cams[li], energies[li] = cam, box_energy_in(co, r["box"])
        # prefer P3 (highest resolution, LSK-enhanced); keep max-energy head
        best_li = max(heads, key=lambda li: (round(energies[li], 6),
                                             li == PREFER_LAYER))
        inside = energies[best_li]
        rows.append({**r, "target_layer": best_li,
                     "target_type": types[best_li],
                     "inside": inside, "outside": 1.0 - inside,
                     "cam": cams[best_li], "ratio": ratio, "pad": pad})
        print(f"[cam] {r['image']} {names[r['cls']]} conf={r['conf']:.2f} "
              f"IoU={r['iou']:.3f} layer={best_li}({types[best_li]}) "
              f"inside={inside:.3f}", flush=True)

    # ---- 3. select best: thresholds met, then IoU, conf, inside ------------
    # A huge (whole-image) box makes inside-ratio trivially high, so among
    # threshold-satisfying candidates prefer a tight, localized defect
    # (box area <= 50% of the image): the crop then truly focuses on it.
    img_areas = {}
    for r in rows:
        img = cv2.imread(str(TEST_IMG_DIR / r["image"]))
        h, w = img.shape[:2]
        x1, y1, x2, y2 = r["box"]
        r["area_frac"] = float((x2 - x1) * (y2 - y1) / (w * h))

    def tier(r):
        return ((r["iou"] >= args.iou_want) + (r["conf"] >= args.conf_want)
                + (r["inside"] >= args.inside_want))
    rows.sort(key=lambda r: (-tier(r), r["area_frac"] > 0.5,
                             -r["iou"], -r["conf"], -r["inside"],
                             r["area_frac"]))
    best = rows[0]
    print(f"[select] {best['image']} tier={tier(best)}/3 IoU={best['iou']:.3f} "
          f"conf={best['conf']:.3f} inside={best['inside']:.3f}", flush=True)

    # ---- 4. figure: crop ONLY for display (CAM already computed) -----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    orig = cv2.imread(str(TEST_IMG_DIR / best["image"]))
    H, W = orig.shape[:2]
    x1, y1, x2, y2 = best["box"]
    bw, bh = x2 - x1, y2 - y1
    cx1 = max(0, int(x1 - CROP_PAD * bw))
    cy1 = max(0, int(y1 - CROP_PAD * bh))
    cx2 = min(W, int(x2 + CROP_PAD * bw))
    cy2 = min(H, int(y2 + CROP_PAD * bh))
    crop = orig[cy1:cy2, cx1:cx2].copy()

    from src.explain.gradcam import overlay_heatmap
    overlay_full, heat_full = overlay_heatmap(orig, best["cam"],
                                             best["ratio"], best["pad"],
                                             alpha=0.5)
    overlay = overlay_full[cy1:cy2, cx1:cx2]
    heat = heat_full[cy1:cy2, cx1:cx2]

    def draw_box(im):
        im = im.copy()
        cv2.rectangle(im, (int(x1 - cx1), int(y1 - cy1)),
                      (int(x2 - cx1), int(y2 - cy1)), (0, 255, 0), 2)
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), dpi=DPI,
                             gridspec_kw={"wspace": 0.08})
    for ax, im, t in zip(axes, (draw_box(crop), draw_box(overlay),
                                 cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)),
                         ("Original crop", "Grad-CAM overlay",
                          "Grad-CAM heat")):
        ax.imshow(im)
        ax.set_title(t, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("RSD-LSK: Target-Specific Grad-CAM", fontsize=14,
                 fontweight="bold", y=0.99)
    fig.text(0.5, 0.02,
             f"Class: {names[best['cls']]}   Confidence: {best['conf']:.2f}   "
             f"IoU: {best['iou']:.3f}   Target layer: {best['target_layer']} "
             f"({best['target_type']})   Inside: {best['inside']:.3f}",
             ha="center", fontsize=10)
    best_path = out_dir / "proposed_rsd_lsk_gradcam_best.png"
    fig.savefig(best_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # ---- 5. CSV + input image + meta ---------------------------------------
    csv_path = out_dir / "proposed_rsd_lsk_gradcam_candidates.csv"
    with csv_path.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["image", "class", "confidence", "IoU", "TP_FP_FN",
                     "target_layer", "inside_activation_ratio",
                     "outside_activation_ratio"])
        for r in rows:
            wr.writerow([r["image"], names[r["cls"]],
                         round(float(r["conf"]), 4), round(float(r["iou"]), 4),
                         "TP", f"{r['target_layer']}({r['target_type']})",
                         round(float(r["inside"]), 4),
                         round(float(r["outside"]), 4)])
    inp_path = out_dir / "proposed_rsd_lsk_best_input.jpg"
    cv2.imwrite(str(inp_path), orig)
    (out_dir / "proposed_rsd_lsk_gradcam_meta.json").write_text(
        json.dumps({
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "model": str(PROPOSED_CKPT),
            "data_yaml": DATA_YAML,
            "class_names": names,
            "conf_threshold": args.conf, "iou_threshold": args.iou,
            "imgsz": args.imgsz,
            "head_inputs": heads, "preferred_layer": PREFER_LAYER,
            "method": "vanilla Grad-CAM, per-box target "
                      "(anchors matching the selected detection, "
                      "predicted-class score)",
            "guarantee": "heatmap never masked/painted; crop display-only",
            "selected": best["image"]}, indent=1))

    print(f"\nSelected image: {best['image']}")
    print(f"Predicted class: {names[best['cls']]}")
    print(f"Confidence: {best['conf']:.4f}")
    print(f"IoU: {best['iou']:.4f}")
    print("TP/FP/FN: TP")
    print(f"Grad-CAM target layer: {best['target_layer']} "
          f"({best['target_type']})")
    print(f"Inside activation ratio: {best['inside']:.4f}")
    print(f"Outside activation ratio: {best['outside']:.4f}")
    print(f"\nOutputs:\n  {best_path}\n  {csv_path}\n  {inp_path}")


if __name__ == "__main__":
    main()
