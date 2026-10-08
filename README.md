# Improved YOLOv8 for road-surface defect detection

Reproduction scaffold for **"Road Surface Defect Detection Algorithm Based on
YOLOv8"** — Zhen Sun, Lingxi Zhu, Su Qin, Yongbo Yu, Ruiwen Ju, Qingdang Li —
*Electronics* **2024**, 13(12), 2413. DOI
[10.3390/electronics13122413](https://doi.org/10.3390/electronics13122413),
open access (CC BY 4.0).

> **Status: architecture complete and validated against the paper's Table 2.**
> `models/yolov8s-paper.yaml` encodes all four changes, builds to a working
> detector (forward, backward, gradients and WIoU all verified), and sits at
> **8,769,692 parameters against the paper's 8.62M (+1.7%)**, with each isolated
> ablation arm within 3% of the reported figure. The paper's full text is in
> `references/paper-fulltext.md` and is the spec every decision is checked
> against. Not done: the RDD2022 dataset is absent, so nothing has been trained
> on real data and no accuracy figure has been reproduced.

## What the paper changes in stock YOLOv8

Baseline is **YOLOv8s bounding-box detection** on **RDD2022** (≈2000 images,
7:2:1 split). Reported result: **+2.8% mAP, 43 FPS** over the baseline.

| # | Change | Replaces | Where in stock YOLOv8 |
|---|--------|----------|------------------------|
| 1 | **SPD-Conv** — space-to-depth reshape followed by a **stride-1** conv, in place of conventional downsampling convs | `Conv [c, 3, 2]` | backbone, the stride-2 rows producing P1/2, P2/4, P3/8, P4/16, P5/32 |
| 2 | **ASF-YOLO neck** — attentional scale-sequence feature fusion + triple-feature encoding; its attention component swapped for element-wise addition | the FPN+PAN `nn.Upsample` / `Concat` / `C2f` block | `yolov8.yaml` head rows 10–21 |
| 3 | **FasterNet block inside C2f** — partial conv (PConv) on a channel subset + 3×3 conv, cutting redundant computation | `Bottleneck`'s `cv1` / `cv2` pair | every `C2f` instance |
| 4 | **Wise-IoU (WIoU)** box loss — sample-quality-aware focusing | **CIoU** | `ultralytics/utils/loss.py:131-134` (`bbox_iou(..., CIoU=True)`) |

Number to match: parameters reportedly drop **11.13M → 8.62M** (the stock
`yolov8s` checkpoint downloaded here is 11,166,560 parameters).

## Why this cannot be done by editing a yaml

Verified against the installed `ultralytics 8.4.170`:

- Layer names in a model yaml are resolved with `globals()[m]`
  (`ultralytics/nn/tasks.py:2144`). A name that does not already exist in
  `ultralytics.nn.tasks` raises `KeyError` — `SPDConv`, `C2f_Faster` and an
  ASF-YOLO neck do not exist in the package (grep over `nn/modules/*.py`: 0 hits).
- The channel-inference chain ends in a bare `else: c2 = ch[f]`
  (`tasks.py:2236-2237`), so an injected custom module gets no `c1` argument, no
  width/depth scaling and no repeat handling. A custom module must take complete
  literal args straight from the yaml.
- The box loss is constructed in Python, not config: `BboxLoss` is instantiated
  at `utils/loss.py:382`. WIoU is not reachable from any yaml or train argument.

So changes 1–3 need new `nn.Module` classes injected into
`ultralytics.nn.tasks`' namespace, and change 4 needs a loss swap — all of it
from this project's own `src/`, with the installed package left untouched.

**The installed package is never modified.** `ultralytics/cfg/models/v8/yolov8.yaml`
stays `md5 ee36bfc920d04460e4c081000697961f`, as shipped. All edits land in the
project's own copy at `models/yolov8s.yaml`, which currently has that identical
md5 — so "what did we change?" is answerable by diffing against the package file:

```bash
diff -u .venv/lib/python3.12/site-packages/ultralytics/cfg/models/v8/yolov8.yaml \
        models/yolov8s.yaml
```

`models/yolov8s.yaml` resolves correctly through `yaml_model_load`
(`nn/tasks.py:2257-2277`): the given path is checked first, and
`guess_model_scale` reads `s` from the stem. Verified — building from the copy
gives `scale=s`, `nc=80`, 11,166,560 parameters, i.e. exactly the stock model.

## Repo layout

```
README.md                    # this file
requirements.txt             # pinned environment (torch CUDA wheels, ultralytics 8.4.170)
configs/train.yaml           # training hyperparameters (source of truth, UNVERIFIED — see file header)
data/rdd2022.yaml            # dataset descriptor: splits + class names
data/rdd2022/images/{train,val,test}/    # dataset skeleton, empty
data/rdd2022/labels/{train,val,test}/    # dataset skeleton, empty
models/yolov8s.yaml          # verbatim stock YOLOv8 yaml — the untouched baseline
models/yolov8s-paper.yaml    # the paper's model: SPD-Conv backbone, ASF-YOLO neck, C2f_Faster
src/vendor/                  # verbatim upstream excerpts (AGPL-3.0)
src/modules/                 # spd.py, c2f_faster.py, ssff.py, wiou.py — this project's adaptations
src/model/                   # parse_model copy + install() that injects the custom layers
THIRD_PARTY.md               # licences, provenance, known incompatibilities
weights/yolov8s.pt           # base checkpoint, downloaded
runs/                        # training / evaluation output
```

## The wired model

```python
from src.model import install       # injects custom layers + patches parse_model
install()                            # must run before any YOLO()/DetectionModel() build
from ultralytics import YOLO
model = YOLO("models/yolov8s-paper.yaml")
model.train(data="data/rdd2022.yaml")
```

`install()` is the only coupling to ultralytics. It does two reversible things
(`src/model/parse_model.py:317`): injects the six custom layer classes into
`ultralytics.nn.tasks.__dict__` so `globals()[m]` at `nn/tasks.py:2144`
resolves them, and rebinds `ultralytics.nn.tasks.parse_model` to our copy of
that function — `DetectionModel._from_yaml` looks it up in that module's globals
(`nn/tasks.py:886`). The installed package is never modified on disk, and
`install()` returns the previous function for restoring.

The yaml itself only encodes changes 1–3; change 4 is not expressible in a
model yaml because `BboxLoss` is built in Python. Call
`src.modules.wiou.install_wiou()` before constructing the trainer.

Verified by building and running the model:

| Check | Result |
|-------|--------|
| Structure | 23 layers: 5 `SPDConv`, 8 `C2f_Faster`, 2 `Zoom_cat`, 1 `ScaleSequenceFusion`, 1 `Add`, `Detect` last |
| Eval forward (640x640) | `(1, 8, 8400)`, strides 8/16/32 |
| Backward | gradients reach every parameter the stock model reaches, plus more (199/224 vs 159/224; the 25 gap is the Detect DFL branch, identical in both) |
| WIoU | box loss term 2.0158 vs CIoU's 2.4470 on the same synthetic batch; `cls`/`dfl` untouched |

### Validation against the paper's own ablation table

The full text is now in this repo (`references/paper-fulltext.md`, CC BY 4.0),
so every decision below is checked against the paper rather than its abstract —
and the parameter column of the paper's **Table 2** works as a pass/fail test,
since it is a number the model either reproduces or does not.

| Arm | Paper | This repo | Delta |
|-----|-------|-----------|-------|
| stock YOLOv8s | 11.13M | 11,137,148 | baseline |
| A — SPD-Conv only | 10.21M (−8.2%) | 10,266,268 (−7.8%) | **+0.5%** |
| B — improved neck only | 10.92M (−1.9%) | 10,487,292 (−4.0%) | −4.0% |
| C — C2f_Faster only | 9.75M (−12.4%) | 9,484,028 (−14.8%) | −2.8% |
| D — WIoU only | 11.13M (unchanged) | 11,137,148 (unchanged) | exact |
| **E — all four (shipped)** | **8.62M (−22.6%)** | **8,769,692 (−21.3%)** | **+1.7%** |

Four of the five arms land within 3%, and arm D confirms the loss adds zero
parameters, as it must.

### What the paper actually specifies

The kernel size behind SPD-Conv was the open question, and the paper settles it
directly. Figure 3(e) is captioned *"the feature map obtained by **1 × 1**
convolution"*. That is why `SPDConv` uses a 1x1 kernel:

| Kernel after space-to-depth | Cost vs the stride-2 conv it replaces |
|------------------------------|---------------------------------------|
| 3x3 | 4x more expensive (what this repo first did) |
| **1x1 (shipped)** | **0.44x — cheaper** |

A stride-2 3x3 conv costs $9c_1c_{out}$; space-to-depth followed by a 1x1 conv
costs $4c_1c_{out}$. All four spatial phases survive the rearrangement and the
convolution gets cheaper — the actual selling point of SPD-Conv in Du et al.
Only at k=1 is the paper's arm-A figure (10.21M) reproducible at all.

Three other details the full text fixed, two of which had been guessed:

- **Neck**: the paper keeps ASF-YOLO's SSFF and TFE but states explicitly that
  *"the CPAM attention mechanism is replaced with the Add operation"*. The
  `Add` layer was already correct; this confirms it rather than assuming it.
- **WIoU**: Eq. (2) is $r = \beta/(\delta\alpha^{\beta-\delta})$ with
  $\beta = \bar L_{IoU}/L_{IoU}$ — that is **WIoU v1** (non-monotonic focusing),
  which is what `install_wiou()` defaults to.
- **FasterNet block**: the prose says "PConv and pointwise convolution
  (PWConv)", but implementing that literally gives arm C at **−29.9%** against
  the paper's **−12.4%** — a 17-point miss. PConv followed by a 3x3 conv gives
  −14.8%, within 2.8 points, and is a genuine FasterNet stage-2-to-4 block. The
  code keeps the 3x3 and this note records the conflict.

## Reference implementations (verified, vendored)

Each of the four changes has a public reference implementation. All four repos
were cloned and read; licences and provenance are in [THIRD_PARTY.md](THIRD_PARTY.md).

| Change | Source | What is in `src/` now |
|--------|--------|------------------------|
| SPD-Conv | [Cateners/yolov8-spd](https://github.com/Cateners/yolov8-spd) (AGPL-3.0) | `vendor/space_to_depth.py` — verbatim. Verified: (1,256,40,40) → (1,1024,20,20) |
| ASF-YOLO neck | [mkang315/ASF-YOLO](https://github.com/mkang315/ASF-YOLO) (AGPL-3.0) | `vendor/asf_yolo_blocks.py` — verbatim `DownSample`, `Zoom_cat`, `ScalSeq`, `Add`, `channel_att`, `local_att`, `attention_model` |
| SSFF for YOLOv8s | — | `modules/ssff.py:ScaleSequenceFusion` — rebuilds `ScalSeq`'s two projections, which upstream hardcodes to YOLOv5 widths |
| WIoU | [Instinct323/Wise-IoU](https://github.com/Instinct323/Wise-IoU) (**no licence** — read only) | `modules/wiou.py` — written from arXiv:2301.10051 instead of vendored |
| RDD2022 conventions | [oracl4/RoadDamageDetection](https://github.com/oracl4/RoadDamageDetection) (**no licence** — read only) | confirmed the class order in `data/rdd2022.yaml` |

Two things the references settled, both handled in the implementation:

- Upstream's `ScalSeq` hardcodes `Conv(512, …)`/`Conv(1024, …)`, i.e. YOLOv5
  full-width P4/P5. YOLOv8s is width 0.5 (P4=256, P5=512) and the vendored class
  raises `expected input[1, 256, 40, 40] to have 512 channels` — reproduced in
  testing. `ScaleSequenceFusion` is the fix.
- The SPD fork's yaml pattern is a **stride-1 `Conv` followed by
  `space_to_depth`** at every downsampling point, backbone *and* neck — not a
  straight substitution of stride-2 convs. Its `parse_model` branch
  (`c2 = 4*ch[f]`) belongs to a pre-`cfg/models` ultralytics and does not apply
  here, so `SPDConv` is a single fused layer with the same args as the `Conv` it
  replaces. That fusion is what makes change 1 expensive; see the parameter
  section above.

## Baseline in place

| Item | Value |
|------|-------|
| Checkpoint | `weights/yolov8s.pt`, md5 `0a1d5d0619d5a24848a267ec1e3a3118` (22,588,772 B) |
| Verified | loads as `task=detect`, `nc=80`, 11,166,560 parameters — stock YOLOv8s |
| ultralytics | 8.4.170 |
| torch / torchvision | 2.14.1+cu130 / 0.29.1+cu130 |

Refetch with:

```bash
curl -L -o weights/yolov8s.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt
```

## Dataset

RDD2022 is **not** included in this repo (multi-GB). Fetch it from the
official release and prepare the Chinese subset locally:

- Release hub: [sekilab/RoadDamageDetector](https://github.com/sekilab/RoadDamageDetector) (see "CRDDC'2022 has been released")
- Citable archive: [RDD2022 on figshare](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547) (DOI `10.6084/m9.figshare.21431547.v1`)
- Data article: Arya et al., "RDD2022: A multi-national image dataset for automatic Road Damage Detection" ([arXiv:2209.08538](https://arxiv.org/abs/2209.08538)) — 47,420 road images from six countries, four damage types (D00 longitudinal crack, D10 transverse crack, D20 alligator crack, D40 pothole)
- Challenge site: [crddc2022.sekilab.global/data](https://crddc2022.sekilab.global/data/)

This project uses only the **China_Drone + China_MotorBike** subsets, per Sun et
al., converted from VOC XML to YOLO detect format (`cls xc yc w h`, normalised
0–1) with a 7:2:1 split. To reproduce `data/rdd2022-cn/` from the downloaded
release archive:

```bash
# place RDD2022_released_through_CRDDC2022.zip in the repo root, then:
python scripts/prepare_rdd2022_cn.py
```

`data/rdd2022.yaml` carries the core damage codes `D00 D10 D20 D40`;
**class ids are conversion-specific** — verify them against the label files
before training. That order is confirmed by the
`oracl4/RoadDamageDetection` reference (`0 Longitudinal Crack`, `1 Transverse
Crack`, `2 Alligator Crack`, `3 Potholes`), which also shows RDD2022 laid out as
a multi-source `train:` list, one entry per country. The paper's fifth category
is never named and has no equivalent in the core 4-class conversion; the local
`data/rdd2022-cn.yaml` conversion adds `Repair` as a fifth class.

If you use the dataset, cite Arya et al. (arXiv:2209.08538) and the figshare
record above.

## Next steps

1. **Add the RDD2022 dataset** (absent; see `data/rdd2022.yaml`) and confirm the
   class ids against the converted labels. Everything downstream is blocked on it.
2. **Write the training entry point**: call `src.model.install()` then
   `src.modules.wiou.install_wiou("v1")`, load `configs/train.yaml`, train.
   `install()` must run before any model is constructed. The settings in
   `configs/train.yaml` are now confirmed against the paper's Table 1
   (epochs 150, batch 4, workers 4, imgsz 640, SGD, lr0 0.01).
3. **Reproduce the numbers**: mAP@0.5 0.774, precision 0.803, recall 0.675,
   8.62M parameters, 43 FPS — plus the per-arm ablations A–E.
4. **Decide the arm-B and arm-C gaps.** The neck (−4.0% vs −1.9%) and C2f_Faster
   (−14.8% vs −12.4%) each reduce parameters more than the paper reports, so
   each is slightly over-applied relative to their implementation.
5. **Drop WIoU v3.** It is not what the paper uses (Eq. 2 is v1), and its
   learnable scale does not survive the trainer's model rebuild, so it never
   trains. Keeping a non-functional variant is worse than removing it.

## Unresolved in the paper

Even with the full text, three things are not pinned down and need a judgement
call rather than a lookup:

- The exact layer spec of the improved neck. The paper names SSFF, TFE and the
  Add substitution but publishes no yaml; this repo ports ASF-YOLO's
  `Zoom_cat`/`ScalSeq` blocks to YOLOv8 widths, which is an adaptation, not a
  transcription.
- Whether `C2f_Faster` replaces every `C2f` or only the neck's. This repo does
  all of them; the arm-C gap suggests their scope was narrower.
- Whether the post-SPD convolution is applied at all five downsampling points.
  All five are used here, and only that choice reproduces arm A.

