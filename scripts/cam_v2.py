"""CAM v2: benchmark tighter localisation methods, roll out the winner.

Stage 1 (default): on the proposed arm and the 10 selected images, score
CAM configs with the pointing game against GT boxes —
  energy = heat mass inside any GT box / total heat (higher better)
  maxhit = 1 if the hottest pixel lands in a GT box (rate reported)
Stage 2 (--full): recompute CAMs for all 11 arms on the selected images with
the winning config -> runs/selected_v2/ (cam/<arm>, compare/*_camv2.jpg,
CAMV2.md). Detection tiles are reused from runs/selected/det/.

Usage:
    .venv/bin/python scripts/cam_v2.py                 # benchmark only
    .venv/bin/python scripts/cam_v2.py --full          # benchmark + rollout
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

ARMS = ["baseline", "A_spd", "B_neck", "C_faster", "D_wiou", "E_all",
        "F_lsk", "G_lsk", "H_eiou", "I_eiou", "proposed"]
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
TEST_IMG = ROOT / "data" / "rdd2022-cn" / "images" / "test"
TEST_LBL = ROOT / "data" / "rdd2022-cn" / "labels" / "test"

# (name, method, topk, multiscale)
CONFIGS = [
    ("gradcam/all (legacy)", "gradcam", None, False),
    ("gradcam/top8", "gradcam", 8, False),
    ("hirescam/all", "hirescam", None, False),
    ("hirescam/top8", "hirescam", 8, False),
    ("layercam/top8", "layercam", 8, False),
    # gplusplus dropped: 2nd-order autograd does not survive the topk/max
    # target selection (sparse, non-differentiable routing), and layercam
    # already saturates the pointing game. Method stays implemented in
    # src/explain/gradcam.py for sum-target use.
    ("eigencam", "eigencam", None, False),
    ("pyr+hirescam/top8", "hirescam", 8, True),
    ("pyr+gradcam/top8", "gradcam", 8, True),
]


def pointing_scores(cam_orig: "np.ndarray", gts) -> tuple[float, int]:
    """(energy inside GT, max-pixel-in-GT flag) for an original-size heatmap."""
    import numpy as np

    total = float(cam_orig.sum())
    if total <= 0:
        return 0.0, 0
    h, w = cam_orig.shape
    ys, xs = np.mgrid[0:h, 0:w]
    inside = np.zeros_like(cam_orig, dtype=bool)
    for (x1, y1, x2, y2, _c) in gts:
        inside |= (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)
    energy = float(cam_orig[inside].sum()) / total
    my, mx = np.unravel_index(int(cam_orig.argmax()), cam_orig.shape)
    hit = int(any(x1 <= mx <= x2 and y1 <= my <= y2 for (x1, y1, x2, y2, _c) in gts))
    return energy, hit


def compute_cam(det, tensor, sppf, cfg, args):
    from src.explain.gradcam import cam_heatmap, multiscale_cam

    _name, method, topk, multi = cfg
    if multi:
        from src.explain.gradcam import backbone_pyramid
        return multiscale_cam(det, tensor, backbone_pyramid(det), method=method,
                              topk=topk, conf=args.conf)
    return cam_heatmap(det, tensor, sppf, conf=args.conf, method=method, topk=topk)


def main():
    import cv2
    import numpy as np
    import torch

    from src.explain import deteval as D
    from src.explain.gradcam import backbone_pyramid, letterbox, overlay_heatmap, unpad_cam
    from src.explain.gradcam import predict_boxes
    from src.model import install

    install()
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "selected_v2"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    selected = json.loads((ROOT / "runs" / "selected" / "selected.json").read_text())
    sel_images = sorted({p["image"] for picks in selected.values() for p in picks})
    out = pathlib.Path(args.out_dir)
    (out / "compare").mkdir(parents=True, exist_ok=True)

    # ---- stage 1: benchmark on proposed ----
    print("[bench] proposed arm, 10 images", flush=True)
    yolo = YOLO(str(ROOT / CKPTS["proposed"]))
    names = yolo.model.names
    det = yolo.model.eval()
    adev = next(det.parameters()).device
    sppf = max(i for i, m in enumerate(det.model) if type(m).__name__ == "SPPF")
    bench: dict[str, dict] = {}
    for cfg in CONFIGS:
        en, hits = [], 0
        for img_name in sel_images:
            img_path = TEST_IMG / img_name
            orig = cv2.imread(str(img_path))
            h, w = orig.shape[:2]
            gts = D.load_gt(TEST_LBL, pathlib.Path(img_name).stem, w, h)
            padded, ratio, pad = letterbox(orig, args.imgsz)
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            tensor = (torch.from_numpy(rgb).permute(2, 0, 1)
                      .unsqueeze(0).float() / 255.0).to(adev)
            cam = compute_cam(det, tensor, sppf, cfg, args)
            e, hit = pointing_scores(unpad_cam(cam, ratio, pad, (h, w)), gts)
            en.append(e)
            hits += hit
        bench[cfg[0]] = {"energy": float(np.mean(en)), "maxhit": hits / len(sel_images)}
        print(f"[bench] {cfg[0]:22} energy={bench[cfg[0]]['energy']:.3f} "
              f"maxhit={bench[cfg[0]]['maxhit']:.2f}", flush=True)
    (out / "benchmark.json").write_text(json.dumps(bench, indent=1))
    winner = max(CONFIGS, key=lambda c: (bench[c[0]]["energy"], bench[c[0]]["maxhit"]))
    print(f"[bench] winner: {winner[0]}", flush=True)
    del yolo, det
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not args.full:
        print("(add --full to roll out the winner to all arms)", flush=True)
        return

    # ---- stage 2: rollout winner to all arms ----
    for arm in ARMS:
        ckpt = ROOT / CKPTS[arm]
        print(f"[{arm}] {winner[0]} from {ckpt}", flush=True)
        yolo = YOLO(str(ckpt))
        dm = yolo.model.eval()
        adev = next(dm.parameters()).device
        sppf = max(i for i, m in enumerate(dm.model) if type(m).__name__ == "SPPF")
        arm_cam = out / "cam" / arm
        arm_cam.mkdir(parents=True, exist_ok=True)
        for img_name in sel_images:
            stem = pathlib.Path(img_name).stem
            cpath = arm_cam / f"{stem}.jpg"
            if cpath.exists():
                continue
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
            cam = compute_cam(dm, tensor, sppf, winner, args)
            overlay, _ = overlay_heatmap(orig, cam, ratio, pad)
            D.draw_marks(overlay, preds, gts, tp, fp, fn, names)
            cv2.imwrite(str(cpath), D.draw_legend(overlay))
        del yolo, dm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for img_name in sel_images:
        stem = pathlib.Path(img_name).stem
        cv2.imwrite(str(out / "compare" / f"{stem}_camv2.jpg"),
                    D.grid([D.caption_tile(cv2.imread(str(out / "cam" / a / f"{stem}.jpg")), a)
                            for a in ARMS]))
    lines = ["# CAM v2 report: tighter localisation\n",
             "Legacy setup (vanilla Grad-CAM, sum-all target, SPPF-only) smeared "
             "because (1) thousands of background anchors diluted the target "
             "score, (2) a single 20x20 map cannot resolve thin cracks, (3) "
             "channel-averaged gradients are noisy on detectors.\n",
             "## Pointing-game benchmark (proposed arm, 10 selected images, GT boxes)\n",
             "| config | energy-in-GT | max-pixel-hit-rate |",
             "|---|---|---|"]
    for cfg in CONFIGS:
        s = bench[cfg[0]]
        mark = " <-- winner" if list(cfg) == list(winner) else ""
        lines.append(f"| {cfg[0]} | {s['energy']:.3f} | {s['maxhit']:.2f} |{mark}")
    lines += ["\n- energy = heat mass inside GT / total heat; maxhit = share of "
              "images whose hottest pixel lands in a GT box.",
              "- `top8` targets only the 8 strongest anchors; `pyr+` fuses "
              "P3/8+P4/16+P5/32 maps; `hirescam` weights each location by its "
              "own gradient instead of channel-averaging.",
              f"\nWinner rolled out to all 11 arms: `cam/<arm>/`, "
              f"`compare/*_camv2.jpg` ({winner[0]}, conf={args.conf}).\n"]
    (out / "CAMV2.md").write_text("\n".join(lines))
    print(f"wrote {out}/CAMV2.md", flush=True)


if __name__ == "__main__":
    main()
