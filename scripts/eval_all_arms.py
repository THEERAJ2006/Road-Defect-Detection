"""Full evaluation report for all six ablation arms.

Per arm (from runs/ablation/<arm>/weights/best.pt):
  - val + test splits: P, R, F1, mAP50, mAP50-95, per-class AP50/AP50-95,
    confusion matrix + PR-curve PNGs (saved under runs/eval/)
  - efficiency: params, GFLOPs @640 (via thop through model.info()),
    best.pt size, batch-1 latency (mean+/-std) + FPS, batch-8 throughput

Writes runs/eval/summary.json, runs/eval/summary.csv and runs/eval/EVAL.md.
Sequential, one arm at a time; peak VRAM ~2.5 GB (under the 4 GB cap).

Usage:
    .venv/bin/python scripts/eval_all_arms.py [--arms baseline A_spd ...]
"""

import argparse
import csv
import glob
import json
import os
import pathlib
import statistics
import time

# Offline: nothing may phone home.
os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# Fragmentation killed the first eval run (5.7 GB single alloc); expandable
# segments keep reserved close to allocated.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "ablation"
OUT = ROOT / "runs" / "eval"
ARMS = ["baseline", "A_spd", "B_neck", "C_faster", "D_wiou", "E_all",
        "F_lsk", "G_lsk", "H_eiou", "I_eiou"]
DATA = "data/rdd2022-cn.yaml"
IMGSZ = 640


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def eval_split(model, split, save_dir):
    """Run model.val on one split; return dict of metrics + per-class AP."""
    res = model.val(
        data=DATA, split=split, imgsz=IMGSZ, batch=4, device=0,
        plots=True, project=str(OUT), name=save_dir, exist_ok=True,
        verbose=False,
    )
    b = res.box
    nc = len(model.names)
    # class_result(i) -> (precision, recall, AP50, AP50-95) for class i.
    per_class = {}
    for i in range(nc):
        p_i, r_i, ap50_i, ap_i = b.class_result(i)
        per_class[model.names[i]] = {
            "precision": float(p_i),
            "recall": float(r_i),
            "ap50": float(ap50_i),
            "ap50_95": float(ap_i),
        }
    p, r = float(b.mp), float(b.mr)
    return {
        "precision": p,
        "recall": r,
        "f1": f1(p, r),
        "map50": float(b.map50),
        "map50_95": float(b.map),
        "per_class": per_class,
        "plots_dir": f"runs/eval/{save_dir}",
    }


def bench_fps(model, img_paths):
    """Batch-1 latency (mean/std, end-to-end incl. pre/post) + batch-8 throughput."""
    import torch

    # Warmup.
    for p in img_paths[:20]:
        _ = model(p, imgsz=IMGSZ, device=0, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    lat = []
    for k, p in enumerate(img_paths):
        t0 = time.perf_counter()
        _ = model(p, imgsz=IMGSZ, device=0, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000.0)
        if k % 100 == 99 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    mean_ms = statistics.mean(lat)
    std_ms = statistics.stdev(lat) if len(lat) > 1 else 0.0

    # Bulk throughput at batch 8 over a 64-image subset (the full 437 at
    # once OOMs: predict materialises the whole source list otherwise).
    sub = img_paths[:64]
    t0 = time.perf_counter()
    _ = model(sub, imgsz=IMGSZ, device=0, batch=8, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    tput_fps = len(sub) / (time.perf_counter() - t0)
    return {
        "latency_ms_mean": mean_ms,
        "latency_ms_std": std_ms,
        "fps_batch1": 1000.0 / mean_ms,
        "fps_batch8": tput_fps,
    }


def main():
    global RUNS, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--force", action="store_true",
                    help="re-evaluate arms that already have runs/eval/<arm>.json")
    ap.add_argument("--runs-dir", default=str(ROOT / "runs" / "ablation"),
                    help="project dir holding <arm>/weights/best.pt")
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "eval"),
                    help="where to write plots, summaries and EVAL inputs")
    ap.add_argument("--tag", default="",
                    help="prefix for plot subdirs and json keys, e.g. 'scratch_'")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(ROOT))
    from src.model import install
    install()
    from src.modules.wiou import install_wiou  # noqa: F401 (import keeps parity)
    from ultralytics import YOLO

    global RUNS, OUT
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS = pathlib.Path(args.runs_dir)
    OUT = pathlib.Path(args.out_dir)
    TAG = args.tag
    OUT.mkdir(parents=True, exist_ok=True)
    test_imgs = sorted(glob.glob(str(ROOT / "data" / "rdd2022-cn" / "images" / "test" / "*")))
    test_imgs = sorted(glob.glob(str(ROOT / "data" / "rdd2022-cn" / "images" / "test" / "*")))
    assert test_imgs, "no test images found"

    summary = {}
    for arm in args.arms:
        key = f"{TAG}{arm}"
        arm_json = OUT / f"{key}.json"
        if arm_json.exists() and not args.force:
            print(f"[{key}] skip: {arm_json} exists (use --force to redo)", flush=True)
            summary[key] = next(iter(json.loads(arm_json.read_text()).values()))
            continue
        ckpt = RUNS / arm / "weights" / "best.pt"
        assert ckpt.exists(), f"missing {ckpt}"
        print(f"[{arm}] loading {ckpt}", flush=True)
        model = YOLO(str(ckpt))

        import torch as _torch
        import thop as _thop
        # GFLOPs on CPU with ultralytics' own convention (MACs/1e9*2):
        # model_info()/get_flops() profiles on the model's device (GPU) where
        # one conv workspace alloc hits ~5.7 GB and OOMs (then returns 0.0).
        model.to("cpu")
        _flops = _thop.profile(
            model.model.eval(),
            inputs=[_torch.empty(1, 3, IMGSZ, IMGSZ)], verbose=False)[0] / 1e9 * 2
        gflops = round(float(_flops), 2)
        n_params = sum(p.numel() for p in model.model.parameters())
        model.to("cuda:0")

        out = {
            "params": int(n_params),
            "gflops_640": gflops,
            "size_mb": ckpt.stat().st_size / 2**20,
            "val": eval_split(model, "val", f"{key}_val"),
            "test": eval_split(model, "test", f"{key}_test"),
        }
        out.update(bench_fps(model, test_imgs))
        (OUT / f"{key}.json").write_text(json.dumps({key: out}, indent=1))
        summary[key] = out
        print(f"[{key}] val mAP50={out['val']['map50']:.4f} "
              f"test mAP50={out['test']['map50']:.4f} "
              f"fps_b1={out['fps_batch1']:.1f} gflops={gflops}", flush=True)

        # Free GPU memory between arms.
        import torch, gc
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))

    with (OUT / "summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "split", "P", "R", "F1", "mAP50", "mAP50-95",
                    "params", "GFLOPs", "size_MB",
                    "lat_ms_mean", "lat_ms_std", "FPS_b1", "FPS_b8"])
        for arm, s in summary.items():
            for split in ("val", "test"):
                m = s[split]
                w.writerow([arm, split, f"{m['precision']:.4f}", f"{m['recall']:.4f}",
                            f"{m['f1']:.4f}", f"{m['map50']:.4f}",
                            f"{m['map50_95']:.4f}", s["params"], s["gflops_640"],
                            f"{s['size_mb']:.2f}", f"{s['latency_ms_mean']:.2f}",
                            f"{s['latency_ms_std']:.2f}",
                            f"{s['fps_batch1']:.1f}", f"{s['fps_batch8']:.1f}"])

    with (OUT / "per_class.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        first = summary[f"{TAG}{args.arms[0]}"]
        classes = list(first["val"]["per_class"])
        w.writerow(["arm", "split"] + [f"{c}_AP50" for c in classes]
                   + [f"{c}_AP50-95" for c in classes])
        for arm, s in summary.items():
            for split in ("val", "test"):
                pc = s[split]["per_class"]
                w.writerow([arm, split]
                           + [f"{pc[c]['ap50']:.4f}" for c in classes]
                           + [f"{pc[c]['ap50_95']:.4f}" for c in classes])

    print(f"wrote {OUT}/summary.{{json,csv}} {OUT}/per_class.csv", flush=True)


if __name__ == "__main__":
    main()
