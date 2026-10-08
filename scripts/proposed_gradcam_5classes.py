"""FINAL 5-class target-specific Grad-CAM examples for RSD-LSK (proposed only).

One genuine-TP example per class: D00, D10, D20, D40, Repair.

Selection priority per class:
  visual clarity (tight, reasonably sized box) > hotspot-on-defect >
  tight box > low outside activation > high inside activation > IoU > conf.
Preferred (relaxed gradually, never across classes): conf>=0.85, IoU>=0.80,
inside>=0.60, outside<=0.30, box area in [0.003, 0.6] of the image.

Genuineness: target-specific ``box_target_cam`` (vanilla Grad-CAM on the
selected detection's own predicted-class score) computed on the FULL image
at Detect head inputs [16,19,22] (16 = LSK-enhanced P3 preferred, max-energy
head kept); heatmap never masked/painted; crop display-only.

Outputs (default --out-dir ``outputs/``):
  gradcam_D00.png, gradcam_D10.png, gradcam_D20.png, gradcam_D40.png,
  gradcam_Repair.png, rsd_lsk_gradcam_5classes.png, rsd_lsk_gradcam_5classes.csv

Usage:
  python scripts/proposed_gradcam_5classes.py
  python scripts/proposed_gradcam_5classes.py --per-class 10 --conf 0.25 --iou 0.5
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
TEST_IMG_DIR = ROOT / "data" / "rdd2022-cn" / "images" / "test"
TEST_LBL_DIR = ROOT / "data" / "rdd2022-cn" / "labels" / "test"

CLASSES = ["D00", "D10", "D20", "D40", "Repair"]
PREFER_LAYER = 16  # Add: LSK-enhanced P3 @ /8, P3 head input
CROP_PAD = 0.15
DPI = 300


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf-want", type=float, default=0.85)
    ap.add_argument("--iou-want", type=float, default=0.80)
    ap.add_argument("--inside-want", type=float, default=0.60)
    ap.add_argument("--outside-want", type=float, default=0.30)
    ap.add_argument("--per-class", type=int, default=10,
                    help="TPs per class (ranked by IoU, conf) evaluated with CAM")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out-dir", default=str(ROOT / "outputs"))
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def heat_ratios(cam_orig, box):
    """Inside/outside/hotspot ratios measured on the UNMODIFIED heatmap."""
    import numpy as np
    total = float(cam_orig.sum())
    if total <= 0:
        return 0.0, 1.0, 0.0
    x1, y1, x2, y2 = box[:4]
    h, w = cam_orig.shape
    ys, xs = np.mgrid[0:h, 0:w]
    inside_m = (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)
    inside = float(cam_orig[inside_m].sum()) / total
    thr = float(np.percentile(cam_orig, 80))
    hot = cam_orig >= thr
    hot_mass = float(cam_orig[hot].sum())
    hotspot = float(cam_orig[hot & inside_m].sum()) / hot_mass if hot_mass else 0.0
    return inside, 1.0 - inside, hotspot


def main():
    import cv2
    import torch

    from src.explain.deteval import box_iou, load_gt
    from src.explain.gradcam import (box_target_cam, detect_head_layers,
                                     letterbox, overlay_heatmap,
                                     predict_boxes, unpad_cam)
    from src.model import install
    install()
    from ultralytics import YOLO

    args = parse_args()
    assert PROPOSED_CKPT.exists(), f"missing {PROPOSED_CKPT}"
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")

    yolo = YOLO(str(PROPOSED_CKPT))
    names = [yolo.model.names[i] for i in range(len(yolo.model.names))]
    assert names == CLASSES, f"class mismatch: {names}"
    dm = yolo.model.eval()
    dev = torch.device("cuda:0" if (torch.cuda.is_available()
                                    and device != "cpu") else "cpu")
    dm.to(dev)
    types = [type(m).__name__ for m in dm.model]
    heads = detect_head_layers(dm)
    print(f"Grad-CAM target layer: {PREFER_LAYER} ({types[PREFER_LAYER]}); "
          f"heads evaluated: {heads} on {next(dm.parameters()).device}",
          flush=True)
    adev = next(dm.parameters()).device

    # ---- 1. genuine TPs per class over the whole TEST split -----------------
    test_imgs = sorted(p for p in TEST_IMG_DIR.glob("*")
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    per_class = {c: [] for c in range(len(names))}
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
                area = (x2 - x1) * (y2 - y1) / (w * h)
                per_class[c].append({"image": img_path.name,
                                     "box": (x1, y1, x2, y2), "conf": cf,
                                     "iou": best, "area": area,
                                     "img_hw": (h, w)})
    for c in range(len(names)):
        print(f"[detect] {names[c]}: {len(per_class[c])} genuine TPs",
              flush=True)
        assert per_class[c], f"NO genuine TP for class {names[c]}"

    # anchors/strides cached by predict are inference tensors: clone to
    # normal tensors (values identical) so backward works.
    det_head = dm.model[-1]
    det_head.anchors = det_head.anchors.clone()
    det_head.strides = det_head.strides.clone()
    with torch.enable_grad():
        dm(torch.zeros(1, 3, args.imgsz, args.imgsz, device=dev))

    # ---- 2. CAM per candidate, select best per class ------------------------
    selected = {}
    for c in range(len(names)):
        cands = sorted(per_class[c], key=lambda r: (-r["iou"], -r["conf"]))
        # Hard gates (relaxed only if the class pool empties): tight,
        # visible box + minimally reliable detection. Focus itself is a
        # RANKING criterion, never a gate (thin cracks / small potholes
        # diffuse vanilla CAM; gating on it would empty the pools).
        gated, conf_floor, bounds = None, 0.6, (0.003, 0.5)
        for conf_floor, bounds in ((0.6, (0.003, 0.5)), (0.5, (0.001, 0.8)),
                                   (0.0, (0.0, 1.0))):
            gated = [r for r in cands
                     if bounds[0] <= r["area"] <= bounds[1]
                     and r["conf"] >= conf_floor]
            if gated:
                break
        short = gated[:args.per_class]
        rows = []
        for r in short:
            orig = cv2.imread(str(TEST_IMG_DIR / r["image"]))
            padded, ratio, pad = letterbox(orig, args.imgsz)
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            tensor = (torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
                      .float() / 255.0).to(adev)
            x1, y1, x2, y2 = r["box"]
            dw, dh = pad
            lb = (x1 * ratio + dw, y1 * ratio + dh,
                  x2 * ratio + dw, y2 * ratio + dh)
            cams, stats = {}, {}
            for li in heads:
                cam = box_target_cam(dm, tensor, li, lb, target_class=c,
                                     method="gradcam", iou_thr=0.5, topk=3)
                co = unpad_cam(cam, ratio, pad, orig.shape[:2])
                cams[li] = (cam, co)
                stats[li] = heat_ratios(co, r["box"])
            best_li = max(heads, key=lambda li: (round(stats[li][0], 6),
                                                 li == PREFER_LAYER))
            inside, outside, hot = stats[best_li]
            rows.append({**r, "target_layer": best_li,
                         "target_type": types[best_li], "inside": inside,
                         "outside": outside, "hotspot": hot,
                         "cam": cams[best_li][0], "ratio": ratio, "pad": pad})
            print(f"[cam] {names[c]} {r['image']} conf={r['conf']:.2f} "
                  f"IoU={r['iou']:.3f} area={r['area']:.3f} "
                  f"L{best_li} in={inside:.3f} out={outside:.3f} "
                  f"hot={hot:.3f}", flush=True)

        # Ranking follows the required priority: hotspot-on-defect first,
        # then whole-package thresholds, outside, inside, IoU, confidence.
        def focus_count(r):
            return ((r["hotspot"] >= 0.5) + (r["inside"] >= 0.5)
                    + (r["outside"] <= 0.5))

        def tier(r):
            return ((r["conf"] >= args.conf_want)
                    + (r["iou"] >= args.iou_want)
                    + (r["inside"] >= args.inside_want)
                    + (r["outside"] <= args.outside_want))
        rows.sort(key=lambda r: (-focus_count(r), -r["hotspot"], -tier(r),
                                 r["outside"], -r["inside"],
                                 -r["iou"], -r["conf"]))
        selected[c] = rows[0]
        b = rows[0]
        print(f"[select] {names[c]}: {b['image']} tier={tier(b)}/4 "
              f"IoU={b['iou']:.3f} conf={b['conf']:.3f} in={b['inside']:.3f} "
              f"out={b['outside']:.3f} hot={b['hotspot']:.3f}", flush=True)
    assert len(selected) == 5, "all five classes must have an example"

    # ---- 3. figures ----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def crop_triplet(cls_id):
        b = selected[cls_id]
        orig = cv2.imread(str(TEST_IMG_DIR / b["image"]))
        H, W = orig.shape[:2]
        x1, y1, x2, y2 = b["box"]
        bw, bh = x2 - x1, y2 - y1
        # Display crop: 15% pad on each side, then widen the short side
        # toward a square-ish view (thin cracks would otherwise render as
        # an unreadable sliver). Display-only; CAM untouched.
        px, py = CROP_PAD * bw, CROP_PAD * bh
        cx1, cy1 = x1 - px, y1 - py
        cx2, cy2 = x2 + px, y2 + py
        cw, ch = cx2 - cx1, cy2 - cy1
        if cw < 0.6 * ch:
            extra = (0.6 * ch - cw) / 2
            cx1, cx2 = cx1 - extra, cx2 + extra
        elif ch < 0.6 * cw:
            extra = (0.6 * cw - ch) / 2
            cy1, cy2 = cy1 - extra, cy2 + extra
        cx1, cy1 = max(0, int(cx1)), max(0, int(cy1))
        cx2, cy2 = min(W, int(cx2)), min(H, int(cy2))
        overlay_full, heat_full = overlay_heatmap(orig, b["cam"], b["ratio"],
                                                 b["pad"], alpha=0.5)

        def frame(im):
            im = im[cy1:cy2, cx1:cx2].copy()
            cv2.rectangle(im, (int(x1 - cx1), int(y1 - cy1)),
                          (int(x2 - cx1), int(y2 - cy1)), (0, 255, 0), 2)
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        heat = cv2.cvtColor(heat_full[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2RGB)
        return frame(orig), frame(overlay_full), heat

    def bottom_line(cls_id):
        b = selected[cls_id]
        return (f"Class: {names[cls_id]}   Confidence: {b['conf']:.2f}   "
                f"IoU: {b['iou']:.3f}   Target layer: {b['target_layer']} "
                f"({b['target_type']})   Inside activation: {b['inside']:.3f}   "
                f"Outside activation: {b['outside']:.3f}")

    triplet = {}
    for c in range(len(names)):
        crop, over, heat = crop_triplet(c)
        triplet[c] = (crop, over, heat)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), dpi=DPI,
                                 gridspec_kw={"wspace": 0.08})
        for ax, im, t in zip(axes, (crop, over, heat),
                             ("Original crop", "Grad-CAM overlay",
                              "Grad-CAM heat")):
            ax.imshow(im)
            ax.set_title(t, fontsize=11, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle("RSD-LSK: Target-specific Grad-CAM", fontsize=14,
                     fontweight="bold", y=0.99)
        fig.text(0.5, 0.02, bottom_line(c), ha="center", fontsize=10)
        p = out_dir / f"gradcam_{names[c]}.png"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[out] {p}", flush=True)

    fig, axes = plt.subplots(5, 3, figsize=(12, 2.4 * 5 + 1.2), dpi=DPI,
                             gridspec_kw={"hspace": 0.3, "wspace": 0.06})
    for r, c in enumerate(range(len(names))):
        for k, im in enumerate(triplet[c]):
            ax = axes[r][k]
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(["Original", "Grad-CAM Overlay",
                              "Grad-CAM Heat"][k],
                             fontsize=11, fontweight="bold")
        axes[r][0].set_ylabel(names[c], fontsize=12, fontweight="bold",
                              rotation=90, labelpad=10)
    fig.suptitle("RSD-LSK: Target-specific Grad-CAM -- 5 classes",
                 fontsize=14, fontweight="bold", y=0.995)
    combo = out_dir / "rsd_lsk_gradcam_5classes.png"
    fig.savefig(combo, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[out] {combo}", flush=True)

    # ---- 4. CSV + summary ----------------------------------------------------
    csv_path = out_dir / "rsd_lsk_gradcam_5classes.csv"
    with csv_path.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["class", "image", "confidence", "IoU", "TP_FP_FN",
                     "target_layer", "inside_activation_ratio",
                     "outside_activation_ratio"])
        for c in range(len(names)):
            b = selected[c]
            wr.writerow([names[c], b["image"], round(float(b["conf"]), 4),
                         round(float(b["iou"]), 4), "TP",
                         f"{b['target_layer']}({b['target_type']})",
                         round(float(b["inside"]), 4),
                         round(float(b["outside"]), 4)])
    print(f"[out] {csv_path}", flush=True)
    (out_dir / "rsd_lsk_gradcam_5classes_meta.json").write_text(json.dumps({
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": str(PROPOSED_CKPT), "imgsz": args.imgsz,
        "conf_threshold": args.conf, "iou_threshold": args.iou,
        "preferred": {"conf": args.conf_want, "iou": args.iou_want,
                      "inside": args.inside_want,
                      "outside": args.outside_want},
        "method": "vanilla Grad-CAM, per-box target (selected detection's "
                  "own predicted-class score); heatmap never masked",
        "selected": {names[c]: {"image": selected[c]["image"],
                                "area_frac": round(selected[c]["area"], 4),
                                "hotspot": round(selected[c]["hotspot"], 4)}
                     for c in range(len(names))}}, indent=1))

    print("\nclass   | image                        | conf  | IoU   | layer | inside | outside | hotspot")
    print("--------+------------------------------+-------+-------+-------+--------+---------+--------")
    for c in range(len(names)):
        b = selected[c]
        print(f"{names[c]:<7} | {b['image']:<28} | {b['conf']:.3f} | "
              f"{b['iou']:.3f} | {b['target_layer']:>3}({b['target_type'][:9]:<9}) | "
              f"{b['inside']:.3f}  | {b['outside']:.3f}   | {b['hotspot']:.3f}")


if __name__ == "__main__":
    main()
