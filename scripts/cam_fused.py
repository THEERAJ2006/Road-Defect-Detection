"""Fused per-box CAM rollout (proposed arm): every box gets a gradient.

For each selected image: predict all boxes, explain each box at its own
Detect-head scale (auto-selected, see fused_perbox_cam), fuse with a
pixel-wise max + faint global context, and draw TP/FP/FN marks.

Outputs under runs/selected_fused/:
    fused/<stem>.jpg, compare/<stem>_vs_legacy.jpg, FUSED.md,
    report_fused.json

Usage:
    .venv/bin/python scripts/cam_fused.py
    .venv/bin/python scripts/cam_fused.py --no-context   # boxes only
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

CKPT = ROOT / "weights" / "F_lsk_proposed.pt"
TEST_IMG = ROOT / "data" / "rdd2022-cn" / "images" / "test"
TEST_LBL = ROOT / "data" / "rdd2022-cn" / "labels" / "test"


def main():
    import cv2
    import gc
    import numpy as np
    import torch

    from src.explain import deteval as D
    from src.explain.gradcam import (detect_head_layers, fused_perbox_cam,
                                     gradcam_heatmap, letterbox,
                                     overlay_heatmap, predict_boxes,
                                     threshold_cam)
    from src.model import install

    install()
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--no-context", action="store_true",
                    help="boxes only, no global-context blend")
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "selected_fused"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    selected = json.loads((ROOT / "runs" / "selected" / "selected.json").read_text())
    sel_images = sorted({p["image"] for picks in selected.values() for p in picks})
    out = pathlib.Path(args.out_dir)
    (out / "fused").mkdir(parents=True, exist_ok=True)
    (out / "compare").mkdir(parents=True, exist_ok=True)

    yolo = YOLO(str(CKPT))
    names = yolo.model.names
    det = yolo.model.eval()
    adev = next(det.parameters()).device
    heads = detect_head_layers(det)
    sppf = max(i for i, m in enumerate(det.model) if type(m).__name__ == "SPPF")
    print(f"heads={heads} sppf={sppf} images={len(sel_images)}", flush=True)

    report: dict[str, dict] = {}
    n_boxes = n_ok = 0
    for img_name in sel_images:
        stem = pathlib.Path(img_name).stem
        img_path = TEST_IMG / img_name
        orig = cv2.imread(str(img_path))
        h, w = orig.shape[:2]
        gts = D.load_gt(TEST_LBL, stem, w, h)
        preds = sorted(predict_boxes(yolo, img_path, args.conf, args.imgsz),
                       key=lambda r: -r[5])
        tp, fp, fn = D.match(preds, gts, args.iou)
        padded, ratio, pad = letterbox(orig, args.imgsz)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = (torch.from_numpy(rgb).permute(2, 0, 1)
                  .unsqueeze(0).float() / 255.0).to(adev)

        fused, rows = fused_perbox_cam(
            det, tensor, [p[:4] for p in preds], ratio, pad, (h, w),
            context_weight=0.0 if args.no_context else 0.25)
        for r, p in zip(rows, preds):
            r.update(status="TP" if preds.index(p) in set(tp) else "FP",
                     cls=names[p[4]], conf=round(p[5], 3))
            n_boxes += 1
            n_ok += r["supported"]
        report[img_name] = {"boxes": rows,
                            "tp": len(tp), "fp": len(fp), "fn": len(fn)}
        print(f"[{stem}] boxes={len(preds)} "
              + " ".join(f"b{r['box']}@{heads.index(r['head'])}"
                         f"e={r['energy']:.2f}{'OK' if r['supported'] else 'WEAK'}"
                         for r in rows), flush=True)

        overlay, _ = overlay_heatmap(orig, threshold_cam(fused), ratio, pad)
        D.draw_marks(overlay, preds, gts, tp, fp, fn, names)
        cv2.imwrite(str(out / "fused" / f"{stem}.jpg"), D.draw_legend(overlay))

        # legacy single-map baseline for the compare strip
        legacy = gradcam_heatmap(det, tensor, sppf, conf=args.conf)
        leg_ov, _ = overlay_heatmap(orig, legacy, ratio, pad)
        D.draw_marks(leg_ov, preds, gts, tp, fp, fn, names)
        cv2.imwrite(str(out / "compare" / f"{stem}_vs_legacy.jpg"), D.grid([
            D.caption_tile(D.draw_legend(leg_ov), "legacy SPPF-only"),
            D.caption_tile(D.draw_legend(overlay), "fused per-box (new)"),
        ]))

    (out / "report_fused.json").write_text(json.dumps(report, indent=1))
    gate = f"{n_ok}/{n_boxes} boxes supported"
    print(f"GATE: {gate}", flush=True)
    assert n_ok == n_boxes, f"regression gate failed: {gate}"
    (out / "FUSED.md").write_text(
        "# Fused per-box CAM (proposed arm)\n\n"
        "Each predicted box is explained at its own Detect-head scale "
        f"(heads={heads}, auto-selected by max own-box energy), fused with "
        "a pixel-wise max + 0.25 global-context blend, rendered "
        "mass-thresholded (top 40%).\n\n"
        f"Regression gate: {gate} (min own-box energy 0.10).\n\n"
        f"Images ({len(sel_images)}): " + ", ".join(sel_images) + "\n\n"
        "Outputs: `fused/<stem>.jpg`, `compare/<stem>_vs_legacy.jpg`.\n\n"
        "Re-run:\n\n```bash\n.venv/bin/python scripts/cam_fused.py"
        f" --conf {args.conf} --imgsz {args.imgsz}"
        + (" --no-context" if args.no_context else "") + "\n```\n")
    print(f"wrote {out}/FUSED.md", flush=True)
    del yolo, det
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
