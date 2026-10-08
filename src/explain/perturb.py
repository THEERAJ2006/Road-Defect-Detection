"""Perturbation-based explanations for YOLO detectors: DRISE + LIME.

Both are black-box (no gradients, no architecture dependence), so the
custom backbone/neck layers cannot distort them the way they can gradient
methods. Both explain ONE target box via detection similarity::

    S = max over anchors of IoU(decoded box, target) * class score

DRISE (Petsiuk et al., 2021, detection adaptation): random low-res masks
(Bernoulli, bilinear-upsampled with random crop shift) occlude a blurred
copy of the image; saliency = similarity-weighted mean mask.
LIME (Ribeiro et al., 2016) via the ``lime`` package: quickshift
superpixels toggled against the image mean colour, Ridge surrogate on the
same similarity score; superpixel weights form the dense map.

Raw :class:`~ultralytics.nn.tasks.DetectionModel` forwards are used
(``torch.no_grad``); decoded boxes are cxcywh in letterbox pixels and are
converted to xyxy internally, matching ``src/explain/gradcam.py``.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ("raw_forward", "anchor_similarity", "drise_heatmap", "lime_heatmap")


@torch.no_grad()
def raw_forward(det_model, batch: torch.Tensor):
    """Decoded ``(boxes_xyxy, scores)`` for a batch, letterbox-pixel boxes.

    Returns:
        boxes ``(B, A, 4)`` xyxy, scores ``(B, nc, A)``.
    """
    det_model.eval()
    y, _ = det_model(batch)
    cx, cy, bw, bh = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
    boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1)
    return boxes, y[:, 4:, :]


def anchor_similarity(boxes: torch.Tensor, scores: torch.Tensor,
                      target_box, target_class: int | None = None) -> torch.Tensor:
    """Per-image detection similarity ``max_j IoU(box_j, target) * score_j``."""
    tb = torch.as_tensor(target_box, dtype=boxes.dtype, device=boxes.device)
    x1 = torch.maximum(boxes[..., 0], tb[0])
    y1 = torch.maximum(boxes[..., 1], tb[1])
    x2 = torch.minimum(boxes[..., 2], tb[2])
    y2 = torch.minimum(boxes[..., 3], tb[3])
    inter = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
    area = (boxes[..., 2] - boxes[..., 0]).clamp_min(0) * (boxes[..., 3] - boxes[..., 1]).clamp_min(0)
    tarea = ((tb[2] - tb[0]) * (tb[3] - tb[1])).clamp_min(1e-8)
    ious = inter / (area + tarea - inter + 1e-8)
    s = scores[:, target_class, :] if target_class is not None else scores.max(dim=1).values
    return (ious * s).max(dim=1).values


def drise_heatmap(
    det_model,
    img_tensor: torch.Tensor,
    target_box_letterbox,
    target_class: int | None = None,
    n_masks: int = 1000,
    grid: int = 16,
    p: float = 0.5,
    batch: int = 25,
    seed: int = 0,
) -> np.ndarray:
    """DRISE saliency for one target box, letterbox-space float map in 0-1."""
    import cv2

    g = torch.Generator().manual_seed(seed)
    _, _, H, W = img_tensor.shape
    dev = next(det_model.parameters()).device
    with torch.no_grad():
        base = img_tensor.to(dev)
        blur = _blur(base)
        sal = torch.zeros(1, 1, H, W, device=dev)
        seen = torch.zeros(1, 1, H, W, device=dev)
        for start in range(0, n_masks, batch):
            b = min(batch, n_masks - start)
            m = (torch.rand(b, 1, grid, grid, generator=g, device=dev) < p).float()
            m = torch.nn.functional.interpolate(m, size=(H + grid, W + grid),
                                                mode="bilinear", align_corners=False)
            ox = torch.randint(0, grid + 1, (1,), generator=g).item()
            oy = torch.randint(0, grid + 1, (1,), generator=g).item()
            m = m[:, :, oy:oy + H, ox:ox + W]
            mixed = base * m + blur * (1 - m)
            boxes, scores = raw_forward(det_model, mixed)
            sim = anchor_similarity(boxes, scores, target_box_letterbox, target_class)
            sal += (sim.view(-1, 1, 1, 1) * m).sum(dim=0, keepdim=True)
            seen += m.sum(dim=0, keepdim=True)
        sal = sal / seen.clamp_min(1e-8)
    sal = sal[0, 0].cpu().numpy().astype(np.float64)
    sal -= sal.min()
    if sal.max() > 0:
        sal /= sal.max()
    return sal.astype(np.float32)


def _blur(tensor: torch.Tensor) -> torch.Tensor:
    import cv2

    arr = (tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    arr = cv2.GaussianBlur(arr, (31, 31), 0)
    out = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return out.to(tensor.device)


def lime_heatmap(
    det_model,
    orig_bgr: np.ndarray,
    ratio: float,
    pad: tuple[int, int],
    target_box_orig,
    target_class: int | None = None,
    n_samples: int = 1000,
    batch_size: int = 25,
    seed: int = 0,
) -> np.ndarray:
    """LIME superpixel-weight map for one target box, original-size 0-1 float."""
    import cv2
    from lime import lime_image

    import sys
    sys.path.insert(0, "/home/sandy/Projects/PEP-26")
    from src.explain.gradcam import letterbox

    dev = next(det_model.parameters()).device
    rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    dw, dh = pad
    tb = target_box_orig
    target_lb = (tb[0] * ratio + dw, tb[1] * ratio + dh,
                 tb[2] * ratio + dw, tb[3] * ratio + dh)
    mean_color = tuple(int(v) for v in rgb.reshape(-1, 3).mean(axis=0))

    def batch_predict(rgb_batch: np.ndarray) -> np.ndarray:
        tensors = []
        for arr in rgb_batch:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            padded, _r, _p = letterbox(bgr, rgb.shape[1])
            t = torch.from_numpy(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
            tensors.append(t.permute(2, 0, 1))
        bt = torch.stack(tensors).float().to(dev) / 255.0
        boxes, scores = raw_forward(det_model, bt)
        sim = anchor_similarity(boxes, scores, target_lb, target_class)
        sim = sim.cpu().numpy().astype(np.float64)
        return np.stack([1 - sim, sim], axis=1)

    explainer = lime_image.LimeImageExplainer(random_state=seed)
    exp = explainer.explain_instance(
        rgb, batch_predict, labels=(1,), hide_color=mean_color,
        num_samples=n_samples, batch_size=batch_size,
        segmentation_fn=None,
    )
    segs = exp.segments  # (H, W) superpixel ids
    weights = dict(exp.local_exp[1])
    heat = np.zeros(segs.shape, dtype=np.float64)
    for sid, w in weights.items():
        heat[segs == sid] = max(w, 0.0)
    if heat.max() > 0:
        heat /= heat.max()
    return heat.astype(np.float32)
