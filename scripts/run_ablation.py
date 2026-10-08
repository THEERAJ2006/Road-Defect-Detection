"""Run the six ablation arms concurrently, at the paper's exact settings.

Concurrency, not batch size, is what raises GPU utilisation here: at batch 4 each
run peaks at 1.13 GiB, so four fit in the card's 8.19 GiB, whereas batch 16 needs
4.00 GiB and allows only one. Batch stays at the paper's 4, so lr0=0.01 and the
training dynamics are untouched — only the hardware is used better.

Paper Table 1 settings, all applied verbatim: epochs 150, batchsize 4, workers 4,
imgsz 640, optimizer SGD, lr0 0.01.
`workers` is lowered to 3 only to avoid CPU over-subscription across concurrent
runs; it is a dataloader-throughput knob and has no effect on results.
`channels_last=False` is required because SSFF's Conv3d has rank-5 weights and
NHWC is rank-4 only. AMP stays on.

All arms initialise from weights/yolov8s.pt (uniform policy, so the ablation is
internally consistent). The paper does not state whether it fine-tunes from COCO
or trains from scratch; COCO init is the usual choice and the only one plausibly
consistent with its reported 0.746 mAP50 on a few thousand images.

Resumable: an arm whose weights/last.pt exists is skipped.
"""

import argparse
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "ablation"
ARMS = ["baseline", "A_spd", "B_neck", "C_faster", "D_wiou", "E_all"]
# 4 GB VRAM budget: one arm peaks at ~1.45 GiB at batch 4, plus ~1.1 GiB desktop
# compositor = ~2.6 GiB total. Two arms (~4.0 GiB) sit exactly on the cap and spike
# over it, so the safe default is 1. Override with --concurrent only if you have
# headroom and keep --max-vram-mib at 4096 so the launcher still defers.
CONCURRENT = 1

# Pin the venv interpreter explicitly. sys.executable is correct when this script is
# started by .venv/bin/python, but it degrades to the system interpreter when the
# runner is relaunched from a shell that has not activated the venv, and the worker
# then dies with "No module named 'torch'". Resolving the path here removes that
# whole class of failure.
VENV_PY = ROOT / ".venv" / "bin" / "python"
INTERPRETER = str(VENV_PY) if VENV_PY.exists() else sys.executable

# Fully offline: skip every DNS/online probe and any asset fetch. Everything the run
# needs (dataset, weights, font) is already on disk.
CHILD_ENV = {
    **os.environ,
    "YOLO_OFFLINE": "1",
    "PYTHONUNBUFFERED": "1",
    "PYTHONPATH": str(ROOT),
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    # Reduce CUDA fragmentation so peak reserved stays close to peak allocated.
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:128",
}

WORKER = r'''
import pathlib, sys
sys.path.insert(0, ".")
import torch
# Self-limit: hard-cap this process to ~2.8 GB VRAM (0.35 x 7.9 GB card) so that
# training + desktop compositor (~1.1 GB) stays under 4 GB total. Observed
# per-arm need is only ~1.45 GB at batch 4, so the cap never binds training —
# it only turns a config mistake into a clean CUDA OOM instead of starving
# the desktop / other procs. Must run before any CUDA allocation.
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.35, 0)
from src.model import install
install()
from src.modules.wiou import install_wiou
from src.checkpoint import checkpoint_every, CKPT_DIRNAME
from ultralytics import YOLO

arm, use_wiou, project, epochs, every, init, data = (
    sys.argv[1], sys.argv[2] == "1", sys.argv[3], int(sys.argv[4]),
    int(sys.argv[5]), sys.argv[6], sys.argv[7])
run_dir = pathlib.Path(project) / arm

# Prefer the newest periodic snapshot over last.pt: if the machine died mid-epoch,
# last.pt is whatever survived the overwrite race, whereas snapshots are named.
snaps = sorted((run_dir / CKPT_DIRNAME).glob("epoch*.pt")) if (run_dir / CKPT_DIRNAME).is_dir() else []
resume_from = snaps[-1] if snaps else (run_dir / "weights" / "last.pt")
if not resume_from.exists():
    resume_from = None

if use_wiou:
    install_wiou("v1")   # must precede criterion construction, also on resume

if resume_from:
    model = YOLO(str(resume_from))
    print(f"RESUME {arm} from {resume_from}", flush=True)
elif init == "scratch":
    # Paper-faithful: the paper never mentions COCO pretraining, so build
    # from the yaml with random init instead of transferring COCO weights.
    model = YOLO(f"models/arms/yolov8s-{arm}.yaml")
    print(f"SCRATCH {arm} from yaml (no COCO weights)", flush=True)
else:
    model = YOLO(f"models/arms/yolov8s-{arm}.yaml").load("weights/yolov8s.pt")
    print(f"COCO-INIT {arm} from weights/yolov8s.pt", flush=True)
model.add_callback("on_train_epoch_end", checkpoint_every(every))

if resume_from:
    model.train(resume=True, device=0)   # settings come from the checkpoint
else:
    model.train(
        data=data, epochs=epochs, imgsz=640, batch=4, workers=2,
        optimizer="SGD", lr0=0.01, lrf=0.01, device=0, channels_last=False,
        project=project, name=arm, exist_ok=True, verbose=False,
    )
print("DONE", arm, flush=True)
'''


def epochs_done(project: pathlib.Path, arm: str) -> int:
    """Epochs already recorded for this arm, or 0 if it never ran."""
    csv = project / arm / "results.csv"
    if not csv.exists():
        return 0
    with csv.open() as fh:
        return max(0, sum(1 for _ in fh) - 1)  # minus the header


EPOCHS = 150


def free_ram_gib():
    """Memory actually available, which is what bounds concurrency."""
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return 0.0


def used_vram_mib():
    """Total VRAM currently used on GPU 0, or None if nvidia-smi is absent."""
    import shutil
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--concurrent", type=int, default=CONCURRENT)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--project", default=str(RUNS))
    ap.add_argument("--checkpoint-every", type=int, default=5,
                    help="snapshot weights every N epochs so a crash loses at most N")
    ap.add_argument("--max-launch-gib", type=float, default=10.0,
                    help="refuse to start another run unless this much RAM is available")
    ap.add_argument("--max-vram-mib", type=float, default=4096.0,
                    help="keep total GPU memory.used under this (nvidia-smi MiB); "
                         "refuse to start another run above it")
    ap.add_argument("--init", choices=["coco", "scratch"], default="coco",
                    help="coco: transfer weights/yolov8s.pt (default, preserves "
                         "existing runs); scratch: random init from the arm yaml "
                         "(paper-faithful — the paper never mentions pretraining)")
    ap.add_argument("--data", default="data/rdd2022-cn.yaml",
                    help="dataset descriptor (use data/rdd2022-cn-balanced.yaml "
                         "for the augmented train split; val/test identical)")
    args = ap.parse_args()

    project = pathlib.Path(args.project)
    if not project.is_absolute():
        # Ultralytics nests relative project dirs under runs/detect/; anchor
        # at the repo root so outputs land where epochs_done() looks.
        project = ROOT / project
    project.mkdir(parents=True, exist_ok=True)
    script = project / "_worker.py"
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

    print(f"arms to run: {len(todo)}  concurrency: {args.concurrent}  "
          f"epochs: {args.epochs}  init: {args.init}  MemAvailable: {free_ram_gib():.1f} GiB  "
          f"VRAM used: {used_vram_mib()} MiB (cap {args.max_vram_mib:.0f})", flush=True)
    running = []
    started = time.time()
    while todo or running:
        while todo and len(running) < args.concurrent:
            # A run costs ~2.2 GiB anon plus ~0.35 GiB shmem per dataloader worker.
            # Launching blind is what got the OOM killer involved last time.
            if running and free_ram_gib() < args.max_launch_gib:
                break
            # VRAM self-limit: each arm needs ~1.5 GiB at batch 4; never start
            # another while total used VRAM is already at the cap.
            vram = used_vram_mib()
            if vram is not None and vram > args.max_vram_mib - 1600:
                print(f"  defer launch: VRAM {vram:.0f} MiB near cap "
                      f"{args.max_vram_mib:.0f} MiB", flush=True)
                break
            arm = todo.pop(0)
            log = open(project / f"{arm}.log", "w")
            proc = subprocess.Popen(
                [INTERPRETER, str(script), arm,
                 "1" if arm in ("D_wiou", "E_all") else "0", str(project),
                 str(args.epochs), str(args.checkpoint_every), args.init,
                 args.data],
                cwd=ROOT, env=CHILD_ENV, stdout=log, stderr=subprocess.STDOUT,
            )
            running.append((arm, proc, log))
            print(f"  start {arm} (pid {proc.pid})  MemAvailable now {free_ram_gib():.1f} GiB",
                  flush=True)
        time.sleep(10)
        for entry in list(running):
            arm, proc, log = entry
            if proc.poll() is not None:
                log.close()
                running.remove(entry)
                mins = (time.time() - started) / 60
                state = "ok" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
                print(f"  {arm} {state}  (elapsed {mins:.0f}m, {len(todo)} queued, "
                      f"{epochs_done(project, arm)}/{args.epochs} epochs)", flush=True)

    print(f"all arms finished in {(time.time() - started) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()