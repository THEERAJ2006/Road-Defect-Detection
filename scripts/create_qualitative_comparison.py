"""Final qualitative comparison: detection + Grad-CAM grids across all arms.

Scientific-honesty rules (enforced, not optional):
  * No retraining, no architecture/weight/dataset changes.
  * Every panel is a REAL inference result (``YOLO.predict``) on the SAME
    held-out TEST image, with the SAME conf/IoU thresholds for every model.
  * TP/FP/FN come from greedy IoU matching against the real GT labels
    (same class + IoU >= --iou  -> TP, else FP; unmatched GT -> FN).
  * Nothing is hidden: FP/FN/low-confidence boxes are drawn as-is.
  * Grad-CAM heatmaps come from each model's own activations (last SPPF).

Image selection is automatic and evidence-based: the whole test split is
scored with the PROPOSED (RSD-LSK) model, ranked by
  1. proposed FP ascending, 2. proposed FN ascending,
  3. proposed mean IoU descending, 4. proposed mean confidence descending,
then diversity-greedily picked so defect categories/scenes vary.

Outputs (default --out-dir ``outputs/``):
  qualitative_detection_comparison.png   (boxes only)
  qualitative_gradcam_comparison.png    (Grad-CAM + boxes)
  qualitative_selected_images.csv       (per-image TP/FP/FN for every model)
  qualitative_comparison_meta.json      (paths, thresholds, layers, timestamp)

Usage:
  python scripts/create_qualitative_comparison.py
  python scripts/create_qualitative_comparison.py --conf 0.25 --iou 0.50 --num-images 8
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

# ---------------------------------------------------------------------------
# Model configuration. Paths are relative to the repo root and were verified
# against the trained checkpoints present in this repository:
#   baseline/H_eiou: stock YOLOv8s scale (SPPF@9), D_wiou/H_eiou differ by loss
#   A_spd/B_neck/C_faster/D_wiou/E_all: Sun et al. reproduction arms
#   F_lsk: stock + LSK only | G_lsk: stale paper+LSK arch (kept for honesty)
#   I_eiou: paper + LSK + EIoU | proposed (RSD-LSK): paper arch + LSKBlock
# ---------------------------------------------------------------------------
MODELS = {
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

# Display order / titles. The proposed column gets the RSD-LSK name + visual
# emphasis (border/header only -- predictions are untouched).
ORDER = ["baseline", "A_spd", "B_neck", "C_faster", "D_wiou", "E_all",
         "F_lsk", "G_lsk", "H_eiou", "I_eiou", "proposed"]
TITLES = {"proposed": "Proposed (RSD-LSK)"}

DATA_YAML = "data/rdd2022-cn.yaml"
TEST_IMG_DIR = ROOT / "data" / "rdd2022-cn" / "images" / "test"
TEST_LBL_DIR = ROOT / "data" / "rdd2022-cn" / "labels" / "test"

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50
DPI = 300
TILE_H = 360  # per-tile render height (px) before figure assembly


# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", type=float, default=CONF_THRESHOLD,
                    help="confidence threshold, shared by ALL models")
    ap.add_argument("--iou", type=float, default=IOU_THRESHOLD,
                    help="IoU threshold for TP matching, shared by ALL models")
    ap.add_argument("--num-images", type=int, default=8,
                    help="representative test images to select (6-10 suggested)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out-dir", default=str(ROOT / "outputs"))
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def check_models():
    missing = [f"{k}: {v}" for k, v in MODELS.items()
               if not (ROOT / v).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing trained weight(s) -- train first, do NOT rename silently:\n"
            + "\n".join(f"  [{k}] {v}" for k, v in MODELS.items()
                        if not (ROOT / v).exists()))


def read_names(yolo):
    names = yolo.model.names
    return [names[i] for i in range(len(names))]


def load_gt(stem, img_w, img_h):
    from src.explain.deteval import load_gt as _load
    return _load(TEST_LBL_DIR, stem, img_w, img_h)


def predict_boxes(yolo, img_path, conf, imgsz):
    from src.explain.gradcam import predict_boxes as _pb
    return _pb(yolo, img_path, conf=conf, imgsz=imgsz)


def box_iou(a, b):
    from src.explain.deteval import box_iou as _iou
    return _iou(a, b)


def match_with_stats(preds, gts, iou_thr):
    """Greedy conf-sorted match (same class, IoU>=thr -> TP).

    Returns dict(tp, fp, fn, matched_ious, mean_iou). Duplicate fragments of
    an already-matched GT count as FP (COCO convention, same as deteval).
    """
    order = sorted(range(len(preds)), key=lambda i: -preds[i][5])
    used = [False] * len(gts)
    tp, fp, ious = [], [], []
    for pi in order:
        x1, y1, x2, y2, c, _cf = preds[pi]
        best, bj = iou_thr, -1
        for gi, (gx1, gy1, gx2, gy2, gc) in enumerate(gts):
            if used[gi] or gc != c:
                continue
            v = box_iou((x1, y1, x2, y2), (gx1, gy1, gx2, gy2))
            if v >= best:
                best, bj = v, gi
        if bj >= 0:
            used[bj] = True
            tp.append(pi)
            ious.append(best)
        else:
            fp.append(pi)
    fn = [gi for gi, u in enumerate(used) if not u]
    return {"tp": tp, "fp": fp, "fn": fn, "ious": ious,
            "mean_iou": float(sum(ious) / len(ious)) if ious else 0.0}


def image_stats(preds, gts, iou_thr):
    m = match_with_stats(preds, gts, iou_thr)
    tp, fp, fn = len(m["tp"]), len(m["fp"]), len(m["fn"])
    confs = [p[5] for p in preds]
    return {"TP": tp, "FP": fp, "FN": fn,
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) else 0.0,
            "mean_conf": float(sum(confs) / len(confs)) if confs else 0.0,
            "mean_iou": m["mean_iou"], "match": m}


# ---------------------------------------------------------------------------
def draw_tile(img_bgr, preds, gts, match, names):
    """Boxes: GREEN TP / RED FP / MAGENTA missed-GT, with class+conf+flag."""
    import cv2

    GREEN, RED, MAG = (0, 255, 0), (0, 0, 255), (255, 0, 255)
    out = img_bgr.copy()
    h, w = out.shape[:2]
    scale = max(h, w) / 640.0
    fs = max(0.45, min(0.7, 0.55 * scale + 0.25))
    th = 2
    for pi in match["tp"]:
        x1, y1, x2, y2, c, cf = preds[pi]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), GREEN, th)
        cv2.putText(out, f"{names[c]} {cf:.2f} TP",
                    (int(x1), max(0, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, GREEN, th)
    for pi in match["fp"]:
        x1, y1, x2, y2, c, cf = preds[pi]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), RED, th)
        cv2.putText(out, f"{names[c]} {cf:.2f} FP",
                    (int(x1), max(0, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, RED, th)
    for gi in match["fn"]:
        x1, y1, x2, y2, c = gts[gi]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), MAG, th + 1)
        cv2.putText(out, f"missed {names[c]} FN",
                    (int(x1), max(0, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, MAG, th)
    # in-tile legend (bottom strip)
    import numpy as np
    items = [((0, 255, 0), "TP"), ((0, 0, 255), "FP"),
             ((255, 0, 255), "FN")]
    lw = 30 + sum(len(t) * 11 + 34 for _, t in items)
    legend = np.zeros((26, min(lw, w), 3), dtype=np.uint8)
    x = 6
    for color, text in items:
        cv2.rectangle(legend, (x, 6), (x + 12, 20), color, -1)
        cv2.putText(legend, text, (x + 15, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        x += 15 + len(text) * 11 + 14
    legend = legend[:, :min(lw, w)]
    out[h - 26:h, 0:legend.shape[1]] = legend
    return out


def norm_height(tile, h=TILE_H):
    import cv2
    w = round(tile.shape[1] * h / tile.shape[0])
    return cv2.resize(tile, (w, h), interpolation=cv2.INTER_AREA)


def pad_row(tiles):
    import cv2
    import numpy as np
    wmax = max(t.shape[1] for t in tiles)
    return [cv2.copyMakeBorder(t, 0, 0, 0, wmax - t.shape[1],
                               cv2.BORDER_CONSTANT, value=(0, 0, 0))
            for t in tiles]


def auto_sppf(det_model):
    sppfs = [i for i, m in enumerate(det_model.model)
             if type(m).__name__ == "SPPF"]
    assert sppfs, "no SPPF layer found"
    return sppfs[-1]


def gradcam_overlay(det_model, layer, orig_bgr, imgsz, conf):
    """Real activation heatmap (vanilla Grad-CAM, SPPF target) + overlay."""
    import cv2
    import torch
    from src.explain.gradcam import (gradcam_heatmap, letterbox,
                                     overlay_heatmap)
    padded, ratio, pad = letterbox(orig_bgr, imgsz)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    dev = next(det_model.parameters()).device
    tensor = (torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
              .float() / 255.0).to(dev)
    cam = gradcam_heatmap(det_model, tensor, layer, conf=conf)
    overlay, _ = overlay_heatmap(orig_bgr, cam, ratio, pad, alpha=0.5)
    return overlay


def build_figure(rows_tiles, col_titles, row_labels, suptitle, out_path,
                 emphasize_last=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2

    n_rows, n_cols = len(rows_tiles), len(col_titles)
    fig_w = 2.6 * n_cols + 1.2
    fig_h = 2.1 * n_rows + 1.6
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(fig_w, fig_h), dpi=DPI,
                             gridspec_kw={"hspace": 0.25, "wspace": 0.06})
    if n_rows == 1:
        axes = np.array([axes])
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r][c]
            rgb = cv2.cvtColor(rows_tiles[r][c], cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_linewidth(1.0)
                s.set_edgecolor("#444444")
            if r == 0:
                is_prop = emphasize_last and c == n_cols - 1
                ax.set_title(col_titles[c], fontsize=11,
                             fontweight="bold",
                             color="white" if is_prop else "black",
                             backgroundcolor="#1f3a8a" if is_prop else "#e5e7eb",
                             pad=4)
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=8, fontweight="bold",
                              rotation=90, labelpad=8)
            if emphasize_last and c == n_cols - 1:
                for s in ax.spines.values():
                    s.set_linewidth(3.0)
                    s.set_edgecolor("#1f3a8a")
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.99)
    fig.text(0.5, 0.015,
             "Green = TP (same class, IoU>=thr)   Red = FP   Magenta = FN (missed GT)  |  "
             "Same image, same conf/IoU thresholds for every model; nothing hidden.",
             ha="center", fontsize=9)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    import cv2
    import torch

    from src.model import install
    install()
    from ultralytics import YOLO

    args = parse_args()
    check_models()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_imgs = sorted(p for p in TEST_IMG_DIR.glob("*")
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    assert test_imgs, f"no test images in {TEST_IMG_DIR}"
    print(f"[info] {len(test_imgs)} test images | conf={args.conf} "
          f"iou={args.iou}", flush=True)

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")

    # ---- pass 1: proposed model over the whole test split -------------------
    prop = YOLO(str(ROOT / MODELS["proposed"]))
    names = read_names(prop)
    print(f"[proposed] classes={names}", flush=True)
    scored = []  # (stem, stats, n_gt, gt_classes)
    for img_path in test_imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = load_gt(img_path.stem, w, h)
        if not gts:  # selection must show actual road defects
            continue
        preds = sorted(predict_boxes(prop, img_path, args.conf, args.imgsz),
                       key=lambda r: -r[5])
        st = image_stats(preds, gts, args.iou)
        gt_cls = sorted({g[4] for g in gts})
        scored.append({"stem": img_path.stem, "name": img_path.name,
                       "stats": st, "n_gt": len(gts), "gt_cls": gt_cls})
    del prop
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    n_cand = len(scored)
    n_fp0 = sum(1 for s in scored if s["stats"]["FP"] == 0)
    n_perfect = sum(1 for s in scored
                    if s["stats"]["FP"] == 0 and s["stats"]["FN"] == 0)
    print(f"[select] candidates(with GT)={n_cand} | FP=0: {n_fp0} | "
          f"FP=0&FN=0: {n_perfect}", flush=True)

    ranked = sorted(scored, key=lambda s: (
        s["stats"]["FP"], s["stats"]["FN"],
        -s["stats"]["mean_iou"], -s["stats"]["mean_conf"]))

    # diversity-greedy pick: first cover unseen dominant classes in rank
    # order, then fill the rest strictly by rank. Never forces a class with
    # no suitable image -- rank always wins ties.
    want = max(1, args.num_images)
    selected, seen_cls = [], set()
    for s in ranked:
        if len(selected) >= want:
            break
        key = tuple(s["gt_cls"])
        if key not in seen_cls:
            selected.append(s)
            seen_cls.add(key)
    for s in ranked:
        if len(selected) >= want:
            break
        if s not in selected:
            selected.append(s)
    selected = sorted(selected, key=lambda s: (
        s["stats"]["FP"], s["stats"]["FN"],
        -s["stats"]["mean_iou"], -s["stats"]["mean_conf"]))
    print("[select] picked:", ", ".join(s["name"] for s in selected),
          flush=True)

    # ---- pass 2: all models on the selected images -------------------------
    per_model_preds, per_model_match, layers_used = {}, {}, {}
    for arm in ORDER:
        ckpt = ROOT / MODELS[arm]
        yolo = YOLO(str(ckpt))
        dm = yolo.model.eval()
        layer = auto_sppf(dm)
        layers_used[arm] = {"layer": layer,
                            "type": type(dm.model[layer]).__name__}
        adev = next(dm.parameters()).device
        per_model_preds[arm], per_model_match[arm] = {}, {}
        for s in selected:
            img_path = TEST_IMG_DIR / s["name"]
            orig = cv2.imread(str(img_path))
            h, w = orig.shape[:2]
            gts = load_gt(s["stem"], w, h)
            preds = sorted(predict_boxes(yolo, img_path, args.conf,
                                         args.imgsz),
                           key=lambda r: -r[5])
            per_model_preds[arm][s["stem"]] = preds
            per_model_match[arm][s["stem"]] = match_with_stats(
                preds, gts, args.iou)
            m = per_model_match[arm][s["stem"]]
            print(f"[{arm}] {s['stem']} TP={len(m['tp'])} FP={len(m['fp'])} "
                  f"FN={len(m['fn'])}", flush=True)
        # cache what the figure pass needs without keeping weights resident
        if arm == "proposed":
            pass
        del yolo, dm
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- tiles: detection-only and grad-cam versions ------------------------
    det_rows, cam_rows, row_labels = [], [], []
    for s in selected:
        img_path = TEST_IMG_DIR / s["name"]
        orig = cv2.imread(str(img_path))
        h, w = orig.shape[:2]
        gts = load_gt(s["stem"], w, h)
        gt_tag = "+".join(names[c] for c in s["gt_cls"])
        ps = image_stats(per_model_preds["proposed"][s["stem"]], gts,
                         args.iou)
        row_labels.append(f"{s['stem']}\nGT:{s['n_gt']}({gt_tag}) "
                          f"P:{ps['TP']} R/FP/FN:{ps['TP']}/{ps['FP']}/{ps['FN']}")
        det_tiles, cam_tiles = [], []
        for arm in ORDER:
            preds = per_model_preds[arm][s["stem"]]
            m = per_model_match[arm][s["stem"]]
            det_tiles.append(norm_height(
                draw_tile(orig, preds, gts, m, names)))
            # real activation heatmap from THIS model, same layer rule,
            # same scale/alpha for every model
            yolo = YOLO(str(ROOT / MODELS[arm]))
            dm = yolo.model.eval()
            heat = gradcam_overlay(dm, layers_used[arm]["layer"],
                                   orig, args.imgsz, args.conf)
            draw = draw_tile(heat, preds, gts, m, names)
            del yolo, dm
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cam_tiles.append(norm_height(draw))
        det_rows.append(pad_row(det_tiles))
        cam_rows.append(pad_row(cam_tiles))

    col_titles = [TITLES.get(a, a) for a in ORDER]
    det_path = out_dir / "qualitative_detection_comparison.png"
    cam_path = out_dir / "qualitative_gradcam_comparison.png"
    build_figure(det_rows, col_titles, row_labels,
                 f"Road defect detection -- qualitative comparison "
                 f"(conf={args.conf}, IoU={args.iou}, {args.imgsz}px)",
                 det_path)
    build_figure(cam_rows, col_titles, row_labels,
                 f"Road defect explainability -- Grad-CAM (SPPF) + boxes "
                 f"(conf={args.conf})",
                 cam_path)
    print(f"[out] {det_path}\n[out] {cam_path}", flush=True)

    # ---- CSV + JSON ---------------------------------------------------------
    csv_path = out_dir / "qualitative_selected_images.csv"
    header = ["image", "category", "GT_count"]
    for a in ORDER:
        header += [f"{a}_TP", f"{a}_FP", f"{a}_FN"]
    header += ["proposed_precision", "proposed_recall",
               "proposed_mean_confidence", "proposed_mean_IoU"]
    with csv_path.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        for s in selected:
            img_path = TEST_IMG_DIR / s["name"]
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            gts = load_gt(s["stem"], w, h)
            row = [s["name"], "+".join(names[c] for c in s["gt_cls"]),
                   len(gts)]
            for a in ORDER:
                m = per_model_match[a][s["stem"]]
                row += [len(m["tp"]), len(m["fp"]), len(m["fn"])]
            ps = image_stats(per_model_preds["proposed"][s["stem"]], gts,
                             args.iou)
            row += [round(ps["precision"], 4), round(ps["recall"], 4),
                    round(ps["mean_conf"], 4), round(ps["mean_iou"], 4)]
            wr.writerow(row)
    meta = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "models": {k: str(ROOT / v) for k, v in MODELS.items()},
        "data_yaml": DATA_YAML,
        "test_images_dir": str(TEST_IMG_DIR),
        "test_labels_dir": str(TEST_LBL_DIR),
        "class_names": names,
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "imgsz": args.imgsz,
        "num_images": len(selected),
        "selected_images": [str(TEST_IMG_DIR / s["name"]) for s in selected],
        "gradcam": {"method": "vanilla Grad-CAM (sum of class scores > conf)",
                    "target_layers": layers_used,
                    "overlay_alpha": 0.5,
                    "note": "same SPPF rule, scale and transparency for all models"},
        "outputs": {"detection_figure": str(det_path),
                    "gradcam_figure": str(cam_path),
                    "csv": str(csv_path)},
    }
    meta_path = out_dir / "qualitative_comparison_meta.json"
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"[out] {csv_path}\n[out] {meta_path}", flush=True)

    # ---- final report -------------------------------------------------------
    print("\nSelected qualitative images:\n")
    for i, s in enumerate(selected, 1):
        img = cv2.imread(str(TEST_IMG_DIR / s["name"]))
        h, w = img.shape[:2]
        ps = image_stats(per_model_preds["proposed"][s["stem"]],
                         load_gt(s["stem"], w, h), args.iou)
        print(f"{i}. {s['name']}\n"
              f"   Proposed: TP={ps['TP']} FP={ps['FP']} FN={ps['FN']}\n"
              f"   Precision={ps['precision']:.3f}\n"
              f"   Recall={ps['recall']:.3f}")
    print(f"\nNumber of candidate test images: {n_cand}")
    print(f"Number of images with Proposed FP = 0: {n_fp0}")
    print(f"Number of images with Proposed FP = 0 and FN = 0: {n_perfect}")
    print(f"Number finally selected: {len(selected)}")
    print(f"\nOutputs:\n  {det_path}\n  {cam_path}\n  {csv_path}\n  {meta_path}")


if __name__ == "__main__":
    main()
