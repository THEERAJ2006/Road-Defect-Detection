"""Single-image explainability pipeline for the Proposed RSD-LSK model.

Frozen, inference-only: detect -> per-box target-specific Grad-CAM ->
VLM caption per box. No GT exists for arbitrary images, so only raw
predictions (class + confidence) are reported -- no TP/FP/FN claims.

Usage:
  python scripts/run_single_image_pipeline.py --image images.jpeg
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROPOSED_CKPT = ROOT / "weights" / "F_lsk_proposed.pt"
DPI = 300


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "single"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--topk", type=int, default=5)
    return ap.parse_args()


def main():
    import cv2
    import numpy as np
    import torch

    from src.explain.gradcam import (box_target_cam, detect_head_layers,
                                     letterbox, overlay_heatmap,
                                     predict_boxes, unpad_cam)
    from src.model import install
    install()
    from ultralytics import YOLO

    args = parse_args()
    img_path = pathlib.Path(args.image)
    assert img_path.exists(), f"image not found: {img_path}"
    assert PROPOSED_CKPT.exists(), f"missing {PROPOSED_CKPT}"
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")

    yolo = YOLO(str(PROPOSED_CKPT))
    names = [yolo.model.names[i] for i in range(len(yolo.model.names))]
    dm = yolo.model.eval()
    dev = torch.device("cuda:0" if (torch.cuda.is_available()
                                    and device != "cpu") else "cpu")
    dm.to(dev)
    heads = detect_head_layers(dm)
    types = [type(m).__name__ for m in dm.model]
    print(f"model: {PROPOSED_CKPT.name} classes={names}", flush=True)
    print(f"CAM heads: {heads} "
          f"({', '.join(f'{i}:{types[i]}' for i in heads)})", flush=True)

    # ---- 1. detection -------------------------------------------------------
    orig = cv2.imread(str(img_path))
    assert orig is not None, f"cannot read {img_path}"
    preds = sorted(predict_boxes(yolo, img_path, args.conf, args.imgsz),
                   key=lambda r: -r[5])[:args.topk]
    print(f"[detect] {len(preds)} prediction(s) (conf>={args.conf}):",
          flush=True)
    for (x1, y1, x2, y2, c, cf) in preds:
        print(f"  {names[c]} conf={cf:.3f} box=({x1:.0f},{y1:.0f},"
              f"{x2:.0f},{y2:.0f})", flush=True)
    if not preds:
        print("[detect] no predictions -- nothing to explain", flush=True)
        return

    # anchors cached by predict are inference tensors: clone to normal ones
    det_head = dm.model[-1]
    det_head.anchors = det_head.anchors.clone()
    det_head.strides = det_head.strides.clone()
    with torch.enable_grad():
        dm(torch.zeros(1, 3, args.imgsz, args.imgsz, device=dev))
    adev = next(dm.parameters()).device

    # ---- 2. per-box target-specific Grad-CAM (full image) -------------------
    padded, ratio, pad = letterbox(orig, args.imgsz)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = (torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
              .float() / 255.0).to(adev)
    dw, dh = pad
    fused = None
    box_reports = []
    for bi, (x1, y1, x2, y2, c, cf) in enumerate(preds):
        lb = (x1 * ratio + dw, y1 * ratio + dh,
              x2 * ratio + dw, y2 * ratio + dh)
        cams, energies = {}, {}
        for li in heads:
            cam = box_target_cam(dm, tensor, li, lb, target_class=c,
                                 method="gradcam", iou_thr=0.5, topk=3)
            co = unpad_cam(cam, ratio, pad, orig.shape[:2])
            tot = float(co.sum())
            ys, xs = np.mgrid[0:co.shape[0], 0:co.shape[1]]
            m = (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)
            energies[li] = float(co[m].sum()) / tot if tot > 0 else 0.0
            cams[li] = cam
        li_best = max(heads, key=lambda li: (round(energies[li], 6),
                                             li == 16))
        fused = cams[li_best] if fused is None else np.maximum(fused,
                                                              cams[li_best])
        box_reports.append({"box": bi, "class": names[c], "conf": round(cf, 4),
                            "layer": li_best, "type": types[li_best],
                            "inside": round(energies[li_best], 4)})
        print(f"[cam] box{bi} {names[c]} conf={cf:.3f} -> "
              f"layer {li_best}({types[li_best]}) "
              f"inside={energies[li_best]:.3f}", flush=True)

    overlay, heat = overlay_heatmap(orig, fused, ratio, pad, alpha=0.5)

    def draw(im):
        im = im.copy()
        for (x1, y1, x2, y2, c, cf) in preds:
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 2)
            cv2.putText(im, f"{names[c]} {cf:.2f}",
                        (int(x1), max(0, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), dpi=DPI,
                             gridspec_kw={"wspace": 0.06})
    for ax, im, t in zip(axes,
                         (draw(orig), draw(overlay),
                          cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)),
                         ("Detection (RSD-LSK)", "Grad-CAM overlay",
                          "Grad-CAM heat")):
        ax.imshow(im)
        ax.set_title(t, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
    info = "  |  ".join(f"{names[c]} {cf:.2f} (L{r['layer']})"
                        for (x1, y1, x2, y2, c, cf), r
                        in zip(preds, box_reports))
    fig.suptitle(f"RSD-LSK single-image pipeline: {img_path.name}",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.02, info, ha="center", fontsize=9)
    stem = img_path.stem
    fig_path = out_dir / f"{stem}_pipeline.png"
    fig.savefig(fig_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[out] {fig_path}", flush=True)

    # ---- 3. VLM captions per box --------------------------------------------
    from PIL import Image as PILImage
    from src.vlm import (caption_crop, crop_box, load_florence,
                         second_opinion)
    try:
        vlm = load_florence("microsoft/Florence-2-base",
                            "cuda" if dev.type == "cuda" else "cpu")
        mode = "florence2" if vlm is not None else "mock"
    except Exception as e:
        print(f"[vlm] load failed ({e}) -> mock", flush=True)
        vlm, mode = None, "mock"
    if vlm is None:
        print("[vlm] weights unavailable offline -> mock captions",
              flush=True)
    image = PILImage.open(img_path).convert("RGB")
    records = []
    for (x1, y1, x2, y2, c, cf) in preds:
        if vlm is None:
            cap = {"caption": f"mock: {names[c]} defect region "
                              f"(conf {cf:.2f})", "mode": "mock"}
        else:
            cap = caption_crop(vlm, crop_box(image, (x1, y1, x2, y2, c, cf),
                                             1.0, 224), str(dev))
        verdict = second_opinion(cf, cap["caption"], args.conf)
        print(f"[vlm] {names[c]}: {cap['caption'][:100]} "
              f"keep={verdict['keep']}", flush=True)
        records.append({"box": [x1, y1, x2, y2], "cls": names[c],
                        "conf": cf, **cap, **verdict})
    js = out_dir / f"{stem}_vlm.json"
    js.write_text(json.dumps({"image": img_path.name, "vlm_mode": mode,
                              "boxes": records,
                              "cam": box_reports}, indent=1))
    print(f"[out] {js}", flush=True)


if __name__ == "__main__":
    main()
