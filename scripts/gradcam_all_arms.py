"""Grad-CAM overlays for every trained ablation arm.

Runs the same test images through each arm's ``best.pt``, explains the
shared SPPF backbone-exit layer (index 9 in non-LSK arms) with
:mod:`src.explain.gradcam`, and writes:

- ``runs/gradcam/<arm>/<stem>_overlay.jpg`` — heat + predicted boxes
- ``runs/gradcam/<arm>/<stem>_heat.jpg`` — heat alone, original resolution
- ``runs/gradcam/compare/<stem>_compare.jpg`` — all arms side-by-side
- ``runs/gradcam/GRADCAM.md`` — what was run and how to re-run it

Usage:
    .venv/bin/python scripts/gradcam_all_arms.py
    .venv/bin/python scripts/gradcam_all_arms.py --arms baseline E_all --num-images 4
    .venv/bin/python scripts/gradcam_all_arms.py --images data/rdd2022-cn/images/test/China_Drone_000001.jpg
"""

from __future__ import annotations

import argparse
import gc
import os
import pathlib

# Offline: nothing may phone home.
os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARMS = ["baseline", "A_spd", "B_neck", "C_faster", "D_wiou", "E_all",
        "F_lsk", "G_lsk", "H_eiou", "I_eiou"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="*", default=ARMS,
                    help="subset of arms to explain")
    ap.add_argument("--runs-dir", default=str(ROOT / "runs" / "ablation"),
                    help="project dir holding <arm>/weights/best.pt")
    ap.add_argument("--arm-paths", nargs="*", default=None, metavar="LABEL=DIR",
                    help="override the checkpoint dir for an arm, e.g. "
                         "F_lsk=runs/ablation/F_lsk G_lsk=runs/ablation-lsk/F_lsk "
                         "(DIR must contain weights/best.pt)")
    ap.add_argument("--images", nargs="*", default=None,
                    help="explicit image paths (overrides --num-images sampling)")
    ap.add_argument("--test-dir", default=str(ROOT / "data" / "rdd2022-cn" / "images" / "test"),
                    help="directory sampled when --images is not given")
    ap.add_argument("--num-images", type=int, default=12,
                    help="evenly-spaced samples from --test-dir")
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "gradcam"))
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="model layer indices to explain "
                         "(default: auto = last SPPF layer of each arm)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="score threshold selecting Grad-CAM target anchors")
    ap.add_argument("--target-class", type=int, default=None,
                    help="restrict the target to one class id (default: all classes)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None,
                    help="e.g. '0' or 'cpu' (default: GPU if available)")
    ap.add_argument("--force", action="store_true",
                    help="regenerate overlays that already exist")
    return ap.parse_args()


def pick_images(args) -> list[pathlib.Path]:
    if args.images:
        paths = [pathlib.Path(p) for p in args.images]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"images not found: {missing}")
        return paths
    all_imgs = sorted(pathlib.Path(args.test_dir).glob("*"))
    all_imgs = [p for p in all_imgs if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    assert all_imgs, f"no images in {args.test_dir}"
    n = min(args.num_images, len(all_imgs))
    step = len(all_imgs) / n
    return [all_imgs[int(i * step)] for i in range(n)]


def draw_boxes(img_bgr, boxes, names: dict):
    import cv2

    for x1, y1, x2, y2, cls_id, cf in boxes:
        label = f"{names.get(cls_id, cls_id)} {cf:.2f}"
        cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_bgr, (int(x1), int(y1) - th - 8), (int(x1) + tw, int(y1)), (0, 255, 0), -1)
        cv2.putText(img_bgr, label, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def label_tile(img_bgr, text: str):
    """Prepend a caption bar so compare grids say which arm is which."""
    import cv2
    import numpy as np

    bar = np.zeros((36, img_bgr.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return np.vstack([bar, img_bgr])


def layer_tag(layers: list[int], layer_types: dict[int, str]) -> str:
    return "+".join(f"{i}{layer_types.get(i, '')}" for i in layers)


def auto_sppf(det_model) -> list[int]:
    """Default target: the last SPPF layer (backbone exit).

    Index 9 in the A-E/H arms, 10 in the LSK arms (F_lsk, G_lsk,
    I_eiou) where the inserted LSKBlock shifts everything downstream by +1.
    """
    sppfs = [i for i, mod in enumerate(det_model.model)
             if type(mod).__name__ == "SPPF"]
    assert sppfs, "no SPPF layer found for auto target resolution"
    return [sppfs[-1]]


def main():
    import sys

    import cv2
    import numpy as np
    import torch

    sys.path.insert(0, str(ROOT))
    from src.explain.gradcam import (
        gradcam_heatmap,
        letterbox,
        overlay_heatmap,
        predict_boxes,
        resolve_target_layers,
    )
    from src.model import install

    install()  # custom layers (SPDConv, C2f_Faster, ...) must resolve on load
    from ultralytics import YOLO

    args = parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "compare").mkdir(parents=True, exist_ok=True)

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    images = pick_images(args)
    print(f"images ({len(images)}): " + ", ".join(p.name for p in images), flush=True)

    layer_types: dict[int, str] = {}
    arm_layers: dict[str, list[int]] = {}
    ckpt_src: dict[str, str] = {}
    overrides = {}
    for item in args.arm_paths or []:
        label, _, path = item.partition("=")
        if not label or not path:
            raise ValueError(f"bad --arm-paths entry {item!r}, want LABEL=DIR")
        overrides[label] = pathlib.Path(path)
    for arm in args.arms:
        ckpt = (overrides.get(arm, pathlib.Path(args.runs_dir) / arm)
                / "weights" / "best.pt")
        assert ckpt.exists(), f"missing {ckpt} — train {arm} first"
        print(f"[{arm}] loading {ckpt}", flush=True)
        yolo = YOLO(str(ckpt))
        det = yolo.model.eval()
        layers = resolve_target_layers(det, args.layers or auto_sppf(det))
        arm_layers[arm] = layers
        ckpt_src[arm] = str(ckpt)
        dev = next(det.parameters()).device
        for i in layers:
            layer_types.setdefault(i, type(det.model[i]).__name__)
        names = yolo.model.names

        arm_dir = out_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        tag = layer_tag(layers, layer_types)

        for img_path in images:
            stem = img_path.stem
            out_path = arm_dir / f"{stem}_overlay.jpg"
            if out_path.exists() and not args.force:
                continue
            orig = cv2.imread(str(img_path))
            assert orig is not None, f"cannot read {img_path}"
            padded, ratio, pad = letterbox(orig, args.imgsz)
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            tensor = (
                torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            ).to(dev)

            cams = [
                gradcam_heatmap(det, tensor, li, conf=args.conf,
                                target_class=args.target_class)
                for li in layers
            ]
            cam = cams[0] if len(cams) == 1 else np.mean(cams, axis=0)
            overlay, heat = overlay_heatmap(orig, cam, ratio, pad)
            draw_boxes(overlay, predict_boxes(yolo, img_path, args.conf, args.imgsz), names)
            cv2.imwrite(str(out_path), overlay)
            cv2.imwrite(str(arm_dir / f"{stem}_heat.jpg"), heat)
            print(f"[{arm}] {stem} -> {out_path.name}", flush=True)

        del yolo, det
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Side-by-side compare grids: same image, every arm.
    for img_path in images:
        stem = img_path.stem
        tiles = []
        for arm in args.arms:
            p = out_dir / arm / f"{stem}_overlay.jpg"
            if not p.exists():
                break
            tiles.append(label_tile(cv2.imread(str(p)), arm))
        else:
            h = max(t.shape[0] for t in tiles)
            normed = []
            for t in tiles:
                if t.shape[0] != h:  # same source size, but be safe
                    w = round(t.shape[1] * h / t.shape[0])
                    t = cv2.resize(t, (w, h))
                normed.append(t)
            cv2.imwrite(str(out_dir / "compare" / f"{stem}_compare.jpg"),
                        np.hstack(normed))
    print(f"compare grids -> {out_dir / 'compare'}", flush=True)

    (out_dir / "GRADCAM.md").write_text(
        "# Grad-CAM — all trained arms\n\n"
        + "".join(f"- {arm}: layer(s) {layer_tag(ls, layer_types)} "
                  f"from `{ckpt_src[arm]}`\n"
                  for arm, ls in arm_layers.items())
        + "\nThe default target is each arm's SPPF backbone exit (index 9, "
        "or 10 where an inserted LSKBlock shifts it), so heatmaps stay "
        "comparable across variants.\n\n"
        f"Target score: summed per-anchor best class scores above conf={args.conf}"
        + (" (all classes)." if args.target_class is None
           else f" (class {args.target_class} only).")
        + "\n\n"
        f"Images ({len(images)}): " + ", ".join(p.name for p in images) + "\n\n"
        "Re-run:\n\n"
        "```bash\n"
        ".venv/bin/python scripts/gradcam_all_arms.py "
        f"--conf {args.conf} --imgsz {args.imgsz}"
        + (f" --layers {' '.join(map(str, args.layers))}" if args.layers else "")
        + (" --force" if args.force else "") + "\n"
        "```\n"
    )
    print(f"wrote {out_dir}/GRADCAM.md", flush=True)


if __name__ == "__main__":
    main()
