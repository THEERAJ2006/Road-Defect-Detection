"""Train arms F_lsk and H_eiou sequentially, at the paper's exact settings.

Same methodology as scripts/run_ablation.py (and therefore comparable with
arms A-E): COCO init from weights/yolov8s.pt, SGD lr0 0.01, batch 4,
imgsz 640, 150 epochs, data/rdd2022-cn.yaml, workers 2, channels_last=False.

Per-arm specifics:
    F_lsk: stock CIoU loss; the LSK row shifts backbone/neck indices >= 4 by
        +1, so COCO weights are remapped before the shape-matched copy (same
        routine as scripts/train_lsk.py) — only the LSK block starts random.
    H_eiou: yaml-identical to baseline; plain EIoU (Focal=False, matching arm
        I_eiou) installed via src.modules.eiou.install_eiou() before the
        criterion is constructed, including on resume.

Resumable: an arm whose weights/last.pt (or newer periodic snapshot) exists
is resumed. A finished arm (>= epochs in results.csv) is skipped.

Usage:
    nohup .venv/bin/python scripts/run_ablation_gh.py > runs/ablation/F_H.log 2>&1 &
"""

import argparse
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "ablation"
ARMS = ["F_lsk", "H_eiou"]

VENV_PY = ROOT / ".venv" / "bin" / "python"
INTERPRETER = str(VENV_PY) if VENV_PY.exists() else sys.executable

CHILD_ENV = {
    **os.environ,
    "YOLO_OFFLINE": "1",
    "PYTHONUNBUFFERED": "1",
    "PYTHONPATH": str(ROOT),
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:128",
}

WORKER = r'''
import pathlib, re, sys
sys.path.insert(0, ".")
import torch
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.35, 0)
from src.model import install
install()
from src.checkpoint import checkpoint_every, CKPT_DIRNAME
from ultralytics import YOLO

arm, loss, project, epochs, every, data = (
    sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]),
    int(sys.argv[5]), sys.argv[6])
run_dir = pathlib.Path(project) / arm

if loss == "eiou":
    from src.modules.eiou import install_eiou
    install_eiou(focal=False)  # must precede criterion construction, also on resume
elif loss != "ciou":
    raise ValueError(f"unknown loss {loss!r}")

snaps = sorted((run_dir / CKPT_DIRNAME).glob("epoch*.pt")) if (run_dir / CKPT_DIRNAME).is_dir() else []
resume_from = snaps[-1] if snaps else (run_dir / "weights" / "last.pt")
if not resume_from.exists():
    resume_from = None

yaml_path = f"models/arms/yolov8s-{arm}.yaml"
if resume_from:
    model = YOLO(str(resume_from))
    print(f"RESUME {arm} from {resume_from}", flush=True)
elif arm == "F_lsk":
    # LSK row shifts stock layers >= 4 to +1: remap COCO keys before copying
    # so every shared layer initialises exactly as in the other arms.
    model = YOLO(yaml_path)
    ckpt = torch.load("weights/yolov8s.pt", map_location="cpu", weights_only=False)
    src_sd = ckpt["model"].float().state_dict()
    pat = re.compile(r"^model\.(\d+)\.")
    remapped = {}
    for k, v in src_sd.items():
        m = pat.match(k)
        if m and int(m.group(1)) >= 4:
            remapped[f"model.{int(m.group(1)) + 1}.{k[m.end():]}"] = v
        else:
            remapped[k] = v
    dst_sd = model.model.state_dict()
    n_hit = 0
    with torch.no_grad():
        for k, v in remapped.items():
            if k in dst_sd and dst_sd[k].shape == v.shape:
                dst_sd[k].copy_(v)
                n_hit += 1
    print(f"COCO-INIT {arm} from weights/yolov8s.pt: transferred {n_hit}/{len(dst_sd)} tensors", flush=True)
else:
    model = YOLO(yaml_path).load("weights/yolov8s.pt")
    print(f"COCO-INIT {arm} from weights/yolov8s.pt (loss={loss})", flush=True)
model.add_callback("on_train_epoch_end", checkpoint_every(every))

if resume_from:
    model.train(resume=True, device=0)
else:
    model.train(
        data=data, epochs=epochs, imgsz=640, batch=4, workers=2,
        optimizer="SGD", lr0=0.01, lrf=0.01, device=0, channels_last=False,
        project=project, name=arm, exist_ok=True, verbose=False,
    )
print("DONE", arm, flush=True)
'''

LOSS = {"F_lsk": "ciou", "H_eiou": "eiou"}


def epochs_done(project: pathlib.Path, arm: str) -> int:
    csv = project / arm / "results.csv"
    if not csv.exists():
        return 0
    with csv.open() as fh:
        return max(0, sum(1 for _ in fh) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--project", default=str(RUNS))
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--data", default="data/rdd2022-cn.yaml")
    args = ap.parse_args()

    project = pathlib.Path(args.project)
    project.mkdir(parents=True, exist_ok=True)
    script = project / "_worker_gh.py"
    script.write_text(WORKER)

    todo = []
    for arm in args.arms:
        done = epochs_done(project, arm)
        if done >= args.epochs:
            print(f"skip {arm}: already {done}/{args.epochs} epochs", flush=True)
            continue
        if done:
            print(f"queue {arm}: resuming from {done}/{args.epochs} epochs", flush=True)
        todo.append(arm)

    print(f"arms to run: {len(todo)}  epochs: {args.epochs}", flush=True)
    started = time.time()
    for arm in todo:
        log = open(project / f"{arm}.log", "w")
        proc = subprocess.Popen(
            [INTERPRETER, str(script), arm, LOSS[arm], str(project),
             str(args.epochs), str(args.checkpoint_every), args.data],
            cwd=ROOT, env=CHILD_ENV, stdout=log, stderr=subprocess.STDOUT,
        )
        print(f"  start {arm} (pid {proc.pid})", flush=True)
        rc = proc.wait()
        log.close()
        mins = (time.time() - started) / 60
        state = "ok" if rc == 0 else f"FAILED rc={rc}"
        print(f"  {arm} {state}  (elapsed {mins:.0f}m, "
              f"{epochs_done(project, arm)}/{args.epochs} epochs)", flush=True)
        if rc != 0:
            break
    print(f"finished in {(time.time() - started) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
