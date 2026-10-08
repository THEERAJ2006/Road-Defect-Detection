# Model comparison — best epoch per run (val split, `data/rdd2022-cn.yaml`)

Updated: 2026-10-07. Source: each run's `results.csv` (best row by `metrics/mAP50(B)`).
All runs: 150-epoch config, SGD, imgsz 640, batch 4, unless noted.
Params counted from each run's `weights/best.pt`. FPS = raw model forward,
batch 1 @640 on this machine (RTX 4060 Laptop GPU, 50 timed iters after warmup) —
relative ranking is comparable across rows, but absolute FPS is NOT comparable
to the paper's end-to-end numbers (RTX 3090, incl. pre/post-processing).

| Model | Run dir | Best ep | Precision | Recall | mAP50 | mAP50-95 | Params (M) | FPS* | Status |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (stock YOLOv8s, CIoU) | `runs/ablation/baseline` | 126 | 0.823 | 0.818 | 0.850 | 0.558 | 11.14 | 103.2 | done |
| A_spd (SPD-Conv only) | `runs/ablation/A_spd` | 148 | 0.831 | 0.807 | 0.845 | 0.556 | 10.27 | 100.7 | done |
| B_neck (ASF-YOLO neck only) | `runs/ablation/B_neck` | 122 | 0.865 | 0.773 | 0.850 | 0.554 | 10.49 | 92.5 | done |
| C_faster (C2f_Faster only) | `runs/ablation/C_faster` | 150 | 0.853 | 0.767 | 0.840 | 0.548 | 9.48 | 100.3 | done |
| D_wiou (WIoU loss only) | `runs/ablation/D_wiou` | 138 | 0.849 | 0.798 | 0.851 | 0.561 | 11.14 | 100.3 | done |
| E_all (paper full: SPD + neck + Faster + WIoU) | `runs/ablation/E_all` | 147 | 0.826 | 0.757 | 0.827 | 0.527 | 8.77 | 100.3 | done |
| F_lsk (stock + LSK only, CIoU) | `runs/ablation/F_lsk` | — | — | — | — | — | 11.34 | 78.5 | done (150/150 epochs; metrics TBD) |
| G_lsk (paper + LSK, WIoU) | `runs/ablation/G_lsk` | 147 | 0.844 | 0.759 | 0.831 | 0.523 | 8.98 | 81.1 | done (renamed from ablation-lsk/F_lsk) |
| H_eiou (stock + EIoU loss only) | `runs/ablation/H_eiou` | — | — | — | — | — | ~11.14 | — | training (ep 55/150; loss-only change adds 0 params, same arch as baseline) |
| I_eiou (paper + LSK + EIoU) | `runs/ablation/I_eiou` | 145 | 0.797 | 0.770 | 0.824 | 0.526 | 8.98 | 80.7 | done (renamed from ablation-lsk/F_lsk_EIoU) |

## Paper reference — Sun et al. Tables 2 & 3 (RTX 3090, end-to-end)

| Paper arm | P | R | mAP50 | Params (M) | FPS |
|---|---|---|---|---|---|
| YOLOv8 (baseline) | 0.754 | 0.667 | 0.746 | 11.13 | 38 |
| A (SPD-Conv) | 0.759 | 0.711 | 0.766 | 10.21 | — |
| B (neck) | 0.758 | 0.704 | 0.764 | 10.92 | — |
| C (C2f_Faster) | 0.756 | 0.664 | 0.744 | 9.75 | — |
| D (WIoU) | 0.733 | 0.701 | 0.758 | 11.13 | — |
| E (improved, all four) | 0.803 | 0.675 | 0.774 | 8.62 | 43 |

Paper's efficiency claim: E cuts params 11.13M → 8.62M (−22.6%) and runs 38 → 43 FPS.
Our reproduction: E_all is 8.77M (−21.3% vs our 11.14M baseline) at 100.3 FPS* —
the lightweighting reproduces, but the accuracy gain (+2.8 mAP in the paper) does
not: our E_all mAP50 (0.827) trails our baseline (0.850). LSK adds ~0.21M params
over E_all and costs ~20 FPS* (100.3 → ~81).

Excluded from this table: `F_lsk_EIoU_smoke1` (1-epoch smoke test, old name for I_eiou),
`runs/ablation_wrongscale/*` (broken-scale debug), `runs/ablation-dronebal/*`
(different dataset yaml), `runs/detect/runs/*` (partial/debug copies).
