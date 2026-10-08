"""DRISE + LIME benchmark and rollout (black-box XAI vs per-box CAM).

Benchmark (default): DRISE (n=800 masks) and LIME (n=800 samples) on the
proposed arm over the 20 selected boxes, pointing-game vs the per-box CAM
reference (energy 0.799 / maxhit 0.90 from scripts/cam_perbox.py).

Rollout (--full): winner method on all 20 proposed boxes, plus all-11-arm
comparison on 2 hard images (Drone_000966: FPs+FNs everywhere;
MotorBike_000336: Repair with heavy FPs), targeting the same box per arm
so TP/FP/FN reasoning differences show directly.
Outputs: runs/selected_xai/<method>/{proposed/<stem>_b<i>.jpg,
<arm>/<stem>_b<i>.jpg (2 images), compare/, XAIB.md}.

Usage:
    .venv/bin/python scripts/xai_bench.py
    .venv/bin/python scripts/xai_bench.py --full
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

N_DRISE = 800
N_LIME = 800
ALLARM_IMAGES = ["China_Drone_000966.jpg", "China_MotorBike_000336.jpg"]


def main():
    import cv2
    import gc
    import numpy as np
    import torch

    from src.explain import deteval as D
    from src.explain.gradcam import letterbox, overlay_heatmap, threshold_cam, unpad_cam
    from src.explain.gradcam import predict_boxes
    from src.explain.perturb import drise_heatmap, lime_heatmap
    from src.model import install

    install()
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--method", choices=["auto", "drise", "lime"], default="auto",
                    help="rollout method (auto = benchmark winner)")
    ap.add_argument("--skip-bench", action="store_true",
                    help="reuse runs/selected_xai/benchmark_xai.json instead of "
                         "re-running the benchmark (use with --full --method)")
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "selected_xai"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    selected = json.loads((ROOT / "runs" / "selected" / "selected.json").read_text())
    targets = [(p["image"], p["box"], cls, p["conf"])
               for cls, picks in selected.items() for p in picks]
    out = pathlib.Path(args.out_dir)
    (out / "compare").mkdir(parents=True, exist_ok=True)

    def prep(img_name):
        orig = cv2.imread(str(TEST_IMG / img_name))
        h, w = orig.shape[:2]
        padded, ratio, pad = letterbox(orig, args.imgsz)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        t = (torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0)
        return orig, t, ratio, pad, (h, w)

    def lb_box(box, ratio, pad):
        dw, dh = pad
        x1, y1, x2, y2 = box
        return (x1 * ratio + dw, y1 * ratio + dh, x2 * ratio + dw, y2 * ratio + dh)

    def pointing(cam_orig, box):
        total = float(cam_orig.sum())
        if total <= 0:
            return 0.0, 0
        x1, y1, x2, y2 = box
        h, w = cam_orig.shape
        ys, xs = np.mgrid[0:h, 0:w]
        inside = (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)
        my, mx = np.unravel_index(int(cam_orig.argmax()), cam_orig.shape)
        return float(cam_orig[inside].sum()) / total, int(inside[my, mx])

    # ---- benchmark on proposed (or reuse) ----
    if args.skip_bench:
        bench = json.loads((out / "benchmark_xai.json").read_text())
        print("[bench] reused benchmark_xai.json", flush=True)
    else:
        print("[bench] DRISE + LIME, proposed arm, 10 boxes", flush=True)
        yolo = YOLO(str(ROOT / CKPTS["proposed"]))
        names = yolo.model.names
        det = yolo.model.eval()
        adev = next(det.parameters()).device
        bench = {"drise": {"en": [], "hit": 0}, "lime": {"en": [], "hit": 0}}
        for img_name, box, _cls, _cf in targets:
            orig, t, ratio, pad, hw = prep(img_name)
            t = t.to(adev)
            d = drise_heatmap(det, t, lb_box(box, ratio, pad), n_masks=N_DRISE)
            e, hit = pointing(unpad_cam(d, ratio, pad, hw), box)
            bench["drise"]["en"].append(e)
            bench["drise"]["hit"] += hit
            print(f"[bench] drise {img_name} energy={e:.3f} hit={hit}", flush=True)
            li = lime_heatmap(det, orig, ratio, pad, box, n_samples=N_LIME)
            e, hit = pointing(li, box)
            bench["lime"]["en"].append(e)
            bench["lime"]["hit"] += hit
            print(f"[bench] lime  {img_name} energy={e:.3f} hit={hit}", flush=True)
        for k in bench:
            bench[k] = {"energy": float(np.mean(bench[k]["en"])),
                        "maxhit": bench[k]["hit"] / len(targets)}
            print(f"[bench] {k:6} energy={bench[k]['energy']:.3f} "
                  f"maxhit={bench[k]['maxhit']:.2f}", flush=True)
        (out / "benchmark_xai.json").write_text(json.dumps(bench, indent=1))
    winner = args.method if args.method != "auto" else max(
        bench, key=lambda k: (bench[k]["energy"], bench[k]["maxhit"]))
    print(f"[bench] winner: {winner}", flush=True)
    if not args.skip_bench:
        del yolo, det
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not args.full:
        print("(add --full to roll out)", flush=True)
        return

    # model handles for rollout (fresh; benchmark ones were freed)
    import gc as _gc
    _yolo0 = YOLO(str(ROOT / CKPTS["proposed"]))
    names = _yolo0.model.names
    del _yolo0
    _gc.collect()

    # ---- rollout ----
    def winner_cam(dm, t, lb, orig, ratio, pad, box):
        """Winner heatmap in letterbox space (overlays need letterbox input)."""
        if winner == "drise":
            return drise_heatmap(dm, t, lb, n_masks=N_DRISE)
        heat_orig = lime_heatmap(dm, orig, ratio, pad, box, n_samples=N_LIME)
        h, w = heat_orig.shape
        nh, nw = round(h * ratio), round(w * ratio)
        small = cv2.resize(heat_orig, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((args.imgsz, args.imgsz), dtype=np.float32)
        dw, dh = pad
        canvas[dh:dh + nh, dw:dw + nw] = small
        return canvas

    mdir = out / winner
    jobs = [("proposed", t) for t in targets]
    for arm in ARMS:
        if arm == "proposed":
            continue
        for img_name in ALLARM_IMAGES:
            for (ti, b, c, cf) in [t for t in targets if t[0] == img_name]:
                jobs.append((arm, (img_name, b, c, cf)))
    loaded = {}
    for arm, (img_name, box, cls, cf) in jobs:
        stem = pathlib.Path(img_name).stem
        bi = next(i for i, t in enumerate(
            [t for t in targets if t[0] == img_name]) if t[1] == box)
        adir = mdir / arm
        adir.mkdir(parents=True, exist_ok=True)
        cpath = adir / f"{stem}_b{bi}.jpg"
        if cpath.exists():
            continue
        if arm not in loaded:
            print(f"[{arm}] loading {CKPTS[arm]}", flush=True)
            y = YOLO(str(ROOT / CKPTS[arm]))
            loaded[arm] = (y, y.model.eval())
            if len(loaded) > 2:  # keep memory bounded; drop oldest
                old = next(iter(loaded))
                if old != arm:
                    del loaded[old]
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        yolo_a, dm = loaded[arm]
        adev = next(dm.parameters()).device
        orig, t, ratio, pad, hw = prep(img_name)
        t = t.to(adev)
        gts = D.load_gt(TEST_LBL, stem, hw[1], hw[0])
        preds = sorted(predict_boxes(yolo_a, TEST_IMG / img_name, args.conf, args.imgsz),
                       key=lambda r: -r[5])
        tp, fp, fn = D.match(preds, gts, args.iou)
        cam = winner_cam(dm, t, lb_box(box, ratio, pad), orig, ratio, pad, box)
        overlay, _ = overlay_heatmap(orig, threshold_cam(cam), ratio, pad)
        D.draw_marks(overlay, preds, gts, tp, fp, fn, names)
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.putText(overlay, f"target {cls} {cf:.2f}", (x1, max(0, y1 - 24)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(cpath), D.draw_legend(overlay))
        print(f"[{arm}] {stem}_b{bi} done", flush=True)

    for img_name in ALLARM_IMAGES:
        stem = pathlib.Path(img_name).stem
        nbox = len([t for t in targets if t[0] == img_name])
        for bi in range(nbox):
            tiles = []
            for a in ARMS:
                p = mdir / a / f"{stem}_b{bi}.jpg"
                if p.exists():
                    tiles.append(D.caption_tile(cv2.imread(str(p)), a))
            if tiles:
                cv2.imwrite(str(out / "compare" / f"{stem}_b{bi}_{winner}.jpg"),
                            D.grid(tiles))
    lines = ["# Black-box XAI report (DRISE + LIME)\n",
             "Both methods are gradient-free: DRISE weights random occlusion "
             "masks by detection similarity, LIME fits a Ridge surrogate over "
             "quickshift superpixels against the same score. Same pointing "
             "game as the CAM benchmarks (20 selected boxes, proposed arm).\n",
             "| method | energy-in-target-box | max-pixel-hit-rate |",
             "|---|---|---|",
             f"| perbox-LayerCAM/P5 (ref) | 0.799 | 0.90 |"]
    for k, s in bench.items():
        mark = " <-- winner" if k == winner else ""
        lines.append(f"| {k} | {s['energy']:.3f} | {s['maxhit']:.2f} |{mark}")
    lines.append(f"\nWinner: `runs/selected_xai/{winner}/` (proposed: all 20 boxes; "
                 f"all arms: {', '.join(ALLARM_IMAGES)}), "
                 f"`compare/*_{winner}.jpg`.\n")
    (out / "XAIB.md").write_text("\n".join(lines))
    print(f"wrote {out}/XAIB.md", flush=True)


if __name__ == "__main__":
    main()
