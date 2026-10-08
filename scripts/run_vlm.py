"""VLM explanations for manually selected ablation arms.

Frozen pipeline — no training, ever: each arm's ``best.pt`` predicts boxes,
each kept box is cropped and captioned (Florence-2, or mock placeholders when
weights are not cached offline), and a keep/reject second opinion is stored:

- ``runs/vlm/<arm>/<stem>_vlm.json`` — per-box caption + verdict
- ``runs/vlm/VLM.md`` — what was run and how to re-run it

Manual variant selector (nothing runs unless you name it):

    .venv/bin/python scripts/run_vlm.py --arms baseline E_all --num-images 4
    .venv/bin/python scripts/run_vlm.py --arms G_lsk --mock          # no download
    .venv/bin/python scripts/run_vlm.py --arms H_eiou --force        # after training

Adding a NEW ablation arm later needs NO retraining of anything: Grad-CAM
hooks a frozen SPPF layer and this script reads a frozen ``best.pt``. Just:
1. train the arm once (any loss/arch), 2. add its name to ``ARMS`` below,
3. run this script + ``scripts/gradcam_all_arms.py --arms <new>``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib

os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARMS = [
    "baseline", "A_spd", "B_neck", "C_faster", "D_wiou", "E_all",
    "F_lsk", "G_lsk", "H_eiou", "I_eiou",
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="*", default=None,
                    help="MANUAL selector: arm names to explain (default: all 10). "
                         "Nothing runs unless named here or by default list.")
    ap.add_argument("--list-arms", action="store_true",
                    help="print the 10 known arms and exit")
    ap.add_argument("--runs-dir", default=str(ROOT / "runs" / "ablation"),
                    help="project dir holding <arm>/weights/best.pt")
    ap.add_argument("--images", nargs="*", default=None)
    ap.add_argument("--test-dir", default=str(ROOT / "data" / "rdd2022-cn" / "images" / "test"))
    ap.add_argument("--num-images", type=int, default=4)
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "vlm"))
    ap.add_argument("--model-id", default="microsoft/Florence-2-base")
    ap.add_argument("--mock", action="store_true",
                    help="skip VLM download; deterministic placeholder captions")
    ap.add_argument("--topk", type=int, default=3,
                    help="max boxes per image sent to the VLM (highest conf first)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--pad-ratio", type=float, default=1.0,
                    help="context around the box (1.0 = one box-size each side)")
    ap.add_argument("--context-min", type=int, default=224,
                    help="minimum square crop side (anti-hallucination floor)")
    ap.add_argument("--caption-task", default="<CAPTION>",
                    help="<CAPTION> hallucinates less than <DETAILED_CAPTION>")
    ap.add_argument("--device", default="cpu",
                    help="VLM device; keep 'cpu' while the GPU trains H_eiou")
    ap.add_argument("--force", action="store_true")
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


def main():
    import sys

    import torch
    from PIL import Image

    sys.path.insert(0, str(ROOT))
    from src.explain.gradcam import predict_boxes
    from src.model import install
    from src.vlm import (
        caption_crop, crop_box, load_florence, resolve_checkpoint,
        second_opinion,
    )

    install()
    from ultralytics import YOLO

    args = parse_args()
    if args.list_arms:
        print(" ".join(ARMS))
        return
    arms = args.arms or ARMS
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise ValueError(f"unknown arms {unknown}; known: {' '.join(ARMS)}")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = pick_images(args)
    print(f"arms: {' '.join(arms)}", flush=True)
    print(f"images ({len(images)}): " + ", ".join(p.name for p in images), flush=True)

    vlm = None if args.mock else load_florence(args.model_id, args.device)
    mode = "mock" if vlm is None else "florence2"
    if not args.mock and vlm is None:
        print("VLM weights unavailable offline → mock captions", flush=True)

    summary = {}
    for arm in arms:
        ckpt = resolve_checkpoint(arm, args.runs_dir)
        print(f"[{arm}] loading {ckpt}", flush=True)
        yolo = YOLO(str(ckpt))
        names = yolo.model.names
        arm_dir = out_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        done, kept, rejected = 0, 0, 0
        for img_path in images:
            stem = img_path.stem
            out_path = arm_dir / f"{stem}_vlm.json"
            if out_path.exists() and not args.force:
                continue
            boxes = predict_boxes(yolo, img_path, args.conf)[: args.topk]
            image = Image.open(img_path).convert("RGB")
            records = []
            for x1, y1, x2, y2, cls_id, cf in boxes:
                cap = caption_crop(
                    vlm, crop_box(image, (x1, y1, x2, y2, cls_id, cf),
                                  args.pad_ratio, args.context_min),
                    args.device, task=args.caption_task)
                verdict = second_opinion(cf, cap["caption"], args.conf)
                kept += verdict["keep"]
                rejected += not verdict["keep"]
                records.append({
                    "box": [x1, y1, x2, y2], "cls": names.get(cls_id, cls_id),
                    "yolo_conf": cf, "caption": cap["caption"],
                    "vlm_mode": cap["mode"], **verdict,
                })
            out_path.write_text(json.dumps(
                {"arm": arm, "image": img_path.name, "vlm_mode": mode,
                 "boxes": records}, indent=1))
            done += 1
        summary[arm] = {"images": done, "kept": kept, "rejected": rejected}
        print(f"[{arm}] {done} images, kept={kept} rejected={rejected}", flush=True)
        del yolo
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (out_dir / "VLM.md").write_text(
        "# VLM — per-arm language explanations\n\n"
        f"Mode default: {mode} (--mock forces placeholders; real Florence-2 "
        "when weights are cached). Top-K boxes/image captioned.\n\n"
        + "".join(f"- {a}: {s['images']} images, kept={s['kept']} "
                   f"rejected={s['rejected']}\n" for a, s in summary.items())
        + f"\nImages: " + ", ".join(p.name for p in images) + "\n\n"
        "New ablation arms need NO retraining for explainability: Grad-CAM hooks "
        "a frozen SPPF layer and this script reads a frozen best.pt. Train the arm "
        "once, append its name to ARMS in scripts/run_vlm.py + "
        "scripts/gradcam_all_arms.py, and re-run both with --arms <new>.\n"
    )
    print(f"wrote {out_dir}/VLM.md", flush=True)


if __name__ == "__main__":
    main()
