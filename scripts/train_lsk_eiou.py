"""Train the LSK+EIoU variant (arm I_eiou): paper-LSK model with EIoU loss.

Clone of scripts/train_lsk.py (arm G_lsk: WIoU v1 + LSK) with exactly one
experimental change: WIoU -> EIoU (plain, Focal=False). All other training
settings are identical and must stay that way:

    YAML models/arms/yolov8s-I_eiou.yaml, data/rdd2022-cn.yaml (5 classes),
    COCO init from weights/yolov8s.pt with LSK index-shift remap (+1 for
    layer index >= 4), SGD lr0 0.01, batch 4, imgsz 640, 150 epochs.

Usage:
    .venv/bin/python scripts/train_lsk_eiou.py [--epochs 150] [--data data/rdd2022-cn.yaml]
Resumable: an existing runs/ablation/I_eiou/weights/last.pt (or newer
periodic snapshot) is resumed.
"""

import argparse
import pathlib
import re
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.checkpoint import checkpoint_every, CKPT_DIRNAME  # noqa: E402
from src.model import install  # noqa: E402

install()
from src.modules.eiou import install_eiou  # noqa: E402
from ultralytics import YOLO  # noqa: E402

YAML = "models/arms/yolov8s-I_eiou.yaml"
COCO = "weights/yolov8s.pt"
PROJECT = ROOT / "runs" / "ablation"
NAME = "I_eiou"
SHIFT_FROM = 4  # LSK row id: stock layers >= 4 move to +1


def coco_init_lsk(yolo: YOLO, coco_path: str = COCO) -> tuple[int, int]:
    """Copy COCO weights into the LSK model with index-shift remapping.

    Returns (transferred, total_tensors).
    """
    ckpt = torch.load(coco_path, map_location="cpu", weights_only=False)
    src_sd = ckpt["model"].float().state_dict()
    pat = re.compile(r"^model\.(\d+)\.")
    remapped = {}
    for k, v in src_sd.items():
        m = pat.match(k)
        if m and int(m.group(1)) >= SHIFT_FROM:
            remapped[f"model.{int(m.group(1)) + 1}.{k[m.end():]}"] = v
        else:
            remapped[k] = v
    dst = yolo.model
    dst_sd = dst.state_dict()
    n_hit = 0
    with torch.no_grad():
        for k, v in remapped.items():
            if k in dst_sd and dst_sd[k].shape == v.shape:
                dst_sd[k].copy_(v)
                n_hit += 1
    return n_hit, len(dst_sd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--data", default="data/rdd2022-cn.yaml")
    ap.add_argument("--project", default=str(PROJECT))
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--checkpoint-every", type=int, default=5)
    args = ap.parse_args()

    install_eiou(focal=False)  # plain EIoU; must precede criterion construction, also on resume
    project = pathlib.Path(args.project)
    run_dir = project / args.name
    snaps = sorted((run_dir / CKPT_DIRNAME).glob("epoch*.pt")) if (run_dir / CKPT_DIRNAME).is_dir() else []
    resume_from = snaps[-1] if snaps else (run_dir / "weights" / "last.pt")
    if not resume_from.exists():
        resume_from = None

    if resume_from:
        model = YOLO(str(resume_from))
        print(f"RESUME {args.name} from {resume_from}", flush=True)
    else:
        model = YOLO(YAML)
        n_hit, n_total = coco_init_lsk(model)
        print(f"COCO-INIT {args.name} from {COCO}: transferred {n_hit}/{n_total} tensors", flush=True)
    model.add_callback("on_train_epoch_end", checkpoint_every(args.checkpoint_every))

    if resume_from:
        model.train(resume=True, device=0)
    else:
        model.train(
            data=args.data, epochs=args.epochs, imgsz=640, batch=4, workers=2,
            optimizer="SGD", lr0=0.01, lrf=0.01, device=0, channels_last=False,
            project=str(project), name=args.name, exist_ok=True, verbose=False,
        )
    print("DONE", args.name, flush=True)


if __name__ == "__main__":
    main()
