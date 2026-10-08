"""CAM per-box: one heatmap per detection, then roll out the winner.

Benchmark (default): on the proposed arm, over the 20 selected boxes
(2 TPs x 5 classes from runs/selected/selected.json), compare the global
top-8 target against per-box anchor targets at P5/SPPF, P4 and P3.
Metric: heat energy inside the target box + hottest-pixel hit.

Rollout (--full): per-box CAMs for every arm on the selected boxes ->
runs/selected_pbox/ (box/<arm>/<stem>_b<i>.jpg, compare/<stem>_b<i>.jpg,
PBOX.md). Overlays use mass-thresholded heat (top 40%) so blobs render as
cores; the target box is drawn white, other TP/FP/FN marks as usual.

Usage:
    .venv/bin/python scripts/cam_perbox.py
    .venv/bin/python scripts/cam_perbox.py --full
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

from scripts.cam_v2 import ARMS, CKPTS, TEST_IMG, TEST_LBL  # noqa: E402

TEST_IMG = TEST_IMG
TEST_LBL = TEST_LBL


def to_letterbox(box, ratio, pad):
    dw, dh = pad
    x1, y1, x2, y2 = box
    return (x1 * ratio + dw, y1 * ratio + dh, x2 * ratio + dw, y2 * ratio + dh)


def main():
    import cv2
    import numpy as np
    import torch

    from src.explain import deteval as D
    from src.explain.gradcam import (backbone_pyramid, box_target_cam, letterbox,
                                     overlay_heatmap, threshold_cam, unpad_cam)
    from src.explain.gradcam import predict_boxes
    from src.model import install

    install()
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "selected_pbox"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    selected = json.loads((ROOT / "runs" / "selected" / "selected.json").read_text())
    targets = []  # (image, box_orig, cls_name, conf)
    for cls, picks in selected.items():
        for p in picks:
            targets.append((p["image"], p["box"], cls, p["conf"]))
    out = pathlib.Path(args.out_dir)
    (out / "compare").mkdir(parents=True, exist_ok=True)

    def tensor_for(orig):
        padded, ratio, pad = letterbox(orig, args.imgsz)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        t = (torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0)
        return t, ratio, pad

    # ---- benchmark on proposed ----
    print("[bench] per-box targets, proposed arm, 20 boxes", flush=True)
    yolo = YOLO(str(ROOT / CKPTS["proposed"]))
    names = yolo.model.names
    det = yolo.model.eval()
    adev = next(det.parameters()).device
    p3, p4, p5 = backbone_pyramid(det)
    variants = [("global/top8/P5", None, p5), ("perbox/P5", "box", p5),
                ("perbox/P4", "box", p4), ("perbox/P3", "box", p3)]
    bench: dict[str, dict] = {v[0]: {"en": [], "hit": 0} for v in variants}
    for img_name, box, _cls, _cf in targets:
        orig = cv2.imread(str(TEST_IMG / img_name))
        h, w = orig.shape[:2]
        tensor, ratio, pad = tensor_for(orig)
        tensor = tensor.to(adev)
        lb = to_letterbox(box, ratio, pad)
        for vname, mode, layer in variants:
            if mode is None:
                from src.explain.gradcam import cam_heatmap
                cam = cam_heatmap(det, tensor, layer, conf=args.conf,
                                  method="layercam", topk=8)
            else:
                cam = box_target_cam(det, tensor, layer, lb, method="layercam")
            e = pointing(cam, ratio, pad, (h, w), box)
            bench[vname]["en"].append(e[0])
            bench[vname]["hit"] += e[1]
    for vname, _m, _l in variants:
        s = bench[vname]
        s["energy"], s["maxhit"] = float(np.mean(s["en"])), s["hit"] / len(targets)
        print(f"[bench] {vname:16} energy={s['energy']:.3f} maxhit={s['maxhit']:.2f}",
              flush=True)
    (out / "benchmark_pbox.json").write_text(json.dumps(
        {k: {"energy": v["energy"], "maxhit": v["maxhit"]} for k, v in bench.items()}, indent=1))
    winner = max(variants, key=lambda v: (bench[v[0]]["energy"], bench[v[0]]["maxhit"]))
    print(f"[bench] winner: {winner[0]}", flush=True)
    del yolo, det
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not args.full:
        print("(add --full to roll out)", flush=True)
        return

    # ---- rollout: per selected box, all arms ----
    by_image: dict[str, list] = {}
    for img_name, box, cls, cf in targets:
        by_image.setdefault(img_name, []).append((box, cls, cf))
    for arm in ARMS:
        ckpt = ROOT / CKPTS[arm]
        print(f"[{arm}] per-box {winner[0]}", flush=True)
        yolo = YOLO(str(ckpt))
        dm = yolo.model.eval()
        adev = next(dm.parameters()).device
        p3a, p4a, p5a = backbone_pyramid(dm)
        layer = {"perbox/P5": p5a, "perbox/P4": p4a, "perbox/P3": p3a,
                 "global/top8/P5": p5a}[winner[0]]
        arm_box = out / "box" / arm
        arm_box.mkdir(parents=True, exist_ok=True)
        for img_name, boxes in by_image.items():
            stem = pathlib.Path(img_name).stem
            img_path = TEST_IMG / img_name
            orig = cv2.imread(str(img_path))
            h, w = orig.shape[:2]
            gts = D.load_gt(TEST_LBL, stem, w, h)
            preds = sorted(predict_boxes(yolo, img_path, args.conf, args.imgsz),
                           key=lambda r: -r[5])
            tp, fp, fn = D.match(preds, gts, args.iou)
            tensor, ratio, pad = tensor_for(orig)
            tensor = tensor.to(adev)
            for bi, (box, cls, cf) in enumerate(boxes):
                cpath = arm_box / f"{stem}_b{bi}.jpg"
                if cpath.exists():
                    continue
                lb = to_letterbox(box, ratio, pad)
                if winner[1] is None:
                    from src.explain.gradcam import cam_heatmap
                    cam = cam_heatmap(dm, tensor, layer, conf=args.conf,
                                      method="layercam", topk=8)
                else:
                    cam = box_target_cam(dm, tensor, layer, lb, method="layercam")
                overlay, _ = overlay_heatmap(orig, threshold_cam(cam), ratio, pad)
                D.draw_marks(overlay, preds, gts, tp, fp, fn, names)
                x1, y1, x2, y2 = (int(v) for v in box)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 3)
                cv2.putText(overlay, f"target {cls} {cf:.2f}", (x1, max(0, y1 - 24)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imwrite(str(cpath), D.draw_legend(overlay))
        del yolo, dm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for img_name, boxes in by_image.items():
        stem = pathlib.Path(img_name).stem
        for bi, (box, cls, cf) in enumerate(boxes):
            cv2.imwrite(str(out / "compare" / f"{stem}_b{bi}_pbox.jpg"),
                        D.grid([D.caption_tile(
                            cv2.imread(str(out / "box" / a / f"{stem}_b{bi}.jpg")),
                            f"{a} | {cls} {cf:.2f}") for a in ARMS]))
    lines = ["# Per-box CAM report\n",
             "Global targets blend every object; per-box targets score only the "
             "anchors behind one box (white outline), so each TP/FP is explained "
             "alone. Heat rendered mass-thresholded (top 40%) to cores.\n",
             "## Benchmark (proposed arm, 20 selected boxes)\n",
             "| config | energy-in-target-box | max-pixel-hit-rate |",
             "|---|---|---|"]
    for vname, _m, _l in variants:
        s = bench[vname]
        mark = " <-- winner" if (vname, _m, _l) == winner else ""
        lines.append(f"| {vname} | {s['energy']:.3f} | {s['maxhit']:.2f} |{mark}")
    lines.append(f"\nWinner rolled out to all 11 arms: `box/<arm>/`, "
                 f"`compare/*_pbox.jpg` ({winner[0]} + threshold viz).\n")
    (out / "PBOX.md").write_text("\n".join(lines))
    print(f"wrote {out}/PBOX.md", flush=True)


def pointing(cam_lb: "np.ndarray", ratio: float, pad, orig_hw, box) -> tuple[float, int]:
    """(energy inside box, maxhit) with the cam mapped to original pixels."""
    import numpy as np

    from src.explain.gradcam import unpad_cam

    cam = unpad_cam(cam_lb, ratio, pad, orig_hw)
    total = float(cam.sum())
    if total <= 0:
        return 0.0, 0
    x1, y1, x2, y2 = box
    h, w = cam.shape
    ys, xs = np.mgrid[0:h, 0:w]
    inside = (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)
    my, mx = np.unravel_index(int(cam.argmax()), cam.shape)
    return float(cam[inside].sum()) / total, int(inside[my, mx])


if __name__ == "__main__":
    main()
