# Third-party code

This project vendors two upstream excerpts verbatim and reimplements the third
change from its source paper. Nothing in the installed `ultralytics` package or
in the stock model yamls is modified.

## `src/vendor/space_to_depth.py` — AGPL-3.0

| | |
|---|---|
| Upstream | [Cateners/yolov8-spd](https://github.com/Cateners/yolov8-spd) (`main`, last pushed 2023-06-06) |
| Excerpt | `ultralytics/nn/modules/block.py:271-283`, class `space_to_depth` |
| Also useful there | `ultralytics/nn/tasks.py:651-652` (`c2 = 4*ch[f]`) and `ultralytics/models/v8/yolov8-spd.yaml`, which shows the intended yaml pattern: a **stride-1** `Conv` followed by `space_to_depth` at every downsampling point, backbone *and* neck |
| Changes made here | none — copied verbatim, dead `dimension` argument and commented-out alternative included |
| Note | That fork predates the `ultralytics/cfg/models/` layout (it still uses `ultralytics/models/`), so its `parse_model` wiring does not apply to ultralytics 8.4.170 |

## `src/vendor/asf_yolo_blocks.py` — AGPL-3.0

| | |
|---|---|
| Upstream | [mkang315/ASF-YOLO](https://github.com/mkang315/ASF-YOLO) (`main`, last pushed 2026-09-22) |
| Excerpt | `models/common.py:721-844` — `DownSample`, `Zoom_cat`, `ScalSeq`, `Add`, `channel_att`, `local_att`, `attention_model` |
| Reference yaml | `models/segment/asf-yolo.yaml` — shows `Zoom_cat` → `C3` for the FPN, then `ScalSeq` over `[P3,P4,P5]`, then `attention_model` over `[P3_down, ScalSeq_out]` |
| Changes made here | none — copied verbatim. `Conv` is imported from the installed ultralytics package rather than vendored a second time; the signature is identical to the YOLOv5 copy these blocks were written against |
| Known issue | `ScalSeq` hardcodes `Conv(512, …)` / `Conv(1024, …)`, i.e. YOLOv5 full-width P4/P5. YOLOv8s is width 0.5 (P4=256, P5=512), so the vendored class raises a shape error — use `src/modules/ssff.py:ScaleSequenceFusion` instead |

ASF-YOLO is the official implementation of "ASF-YOLO: A Novel YOLO Model with
Attentional Scale Sequence Fusion for Cell Instance Segmentation", Image and
Vision Computing 2024, [10.1016/j.imavis.2024.105057](https://doi.org/10.1016/j.imavis.2024.105057).

## `src/modules/wiou.py` — written here, no code copied

The [Instinct323/Wise-IoU](https://github.com/Instinct323/Wise-IoU) repository
(v2 branch) has **no license file**, so its code is not vendored. This module is
written from the equations in Tong et al., "Wise-IoU: Bounding Box Regression Loss
with Dynamic Focusing Mechanism", [arXiv:2301.10051](https://arxiv.org/abs/2301.10051):
Eq. 13 (wise attention), Eq. 15 (non-monotonic focusing, v1) and Eq. 18
(monotonic focusing, v2), with the paper's `alpha=1.7`, `delta=2.7`.

One deliberate difference: community ports keep `beta = L_IoU / mean` in **both**
branches, the reciprocal of the paper's v1 definition. This module follows the
paper. Anyone comparing loss values against a published run should expect a
small discrepancy for that reason.

## `src/modules/ssff.py` — written here

Subclasses the vendored `ScalSeq` and rebuilds its two input projections for the
actual P4/P5 widths. Everything else (3D conv, BN, LeakyReLU, 3D max-pool over
the scale axis) is inherited unchanged.

## `src/modules/lsk.py` — written here, no code copied

The [zcablii/LSKNet](https://github.com/zcablii/LSKNet) repository (IJCV 2024
"LSKNet: A Foundation Lightweight Backbone for Remote Sensing",
[arXiv:2403.11735](https://arxiv.org/abs/2403.11735)) is under
**CC BY-NC 4.0 (non-commercial)**, so its backbone code
(`mmrotate/models/backbones/lsknet.py`: `LSKblock`, `Attention`, `Block`,
`Mlp`, `DWConv`) is **not vendored**. This module is a clean-room
reimplementation of the paper's selective-kernel mechanism (5x5 depthwise +
7x7 dilated-3 depthwise, half-width projections, avg/max scale pooling, 7x7
squeeze + sigmoid selection) and its `Block` stage wrapper (BatchNorm,
1x1-GELU attention, 1x1-DW3x3-GELU-1x1 MLP, 1e-2 layer scales), with own
naming/structure, no `mmcv`/`timm` dependencies (inline stochastic depth),
and a `(c1, c2)` constructor that enforces channel preservation for
`parse_model`. Anyone using the trained LSK-variant weights commercially
should check the upstream non-commercial terms.

## Checked, not vendored

[oracl4/RoadDamageDetection](https://github.com/oracl4/RoadDamageDetection) has no
license either. It was read for dataset conventions only, and confirmed the
class order used in `data/rdd2022.yaml`: `0 Longitudinal Crack (D00)`,
`1 Transverse Crack (D10)`, `2 Alligator Crack (D20)`, `3 Potholes (D40)`. It also
shows RDD2022 as a multi-source `train:` list (one entry per country) — the Sun et
al. paper uses a Chinese subset, so that layout is the one to follow. Its
checkpoint (`models/YOLOv8_Small_RDD.pt`) is YOLOv8s trained on Japan+India, not
the paper's China baseline.
