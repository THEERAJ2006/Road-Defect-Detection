"""Grad-CAM for YOLO detection models, shared by all ablation variants.

Method: hook the output of a convolutional feature map (default: the SPPF
layer that terminates the backbone, index 9 in every arm yaml, so cross-arm
comparisons are apples-to-apples), forward a letterboxed image through the
raw :class:`~ultralytics.nn.tasks.DetectionModel`, and use the summed class
scores of the decoded predictions as the Grad-CAM target::

    score = sum of per-anchor best class scores above ``conf``
    (falls back to the top-10 anchors when nothing passes ``conf``)

``torch.autograd.grad(score, activation)`` gives the channel weights; the
ReLU'd weighted sum is the heatmap. It lives in letterbox space and is
un-padded back to the original image before overlaying, so it aligns with
the boxes drawn by ``YOLO.predict``.

Checkpoints load with frozen parameters (``requires_grad=False``), so the
input tensor is marked ``requires_grad_(True)`` to keep the autograd graph
alive through the frozen weights. No training state is changed.
"""

from __future__ import annotations

import pathlib

import cv2
import numpy as np
import torch

__all__ = (
    "letterbox",
    "resolve_target_layers",
    "gradcam_heatmap",
    "cam_heatmap",
    "box_target_cam",
    "multiscale_cam",
    "backbone_pyramid",
    "all_cam_layers",
    "detect_head_layers",
    "fused_perbox_cam",
    "unpad_cam",
    "threshold_cam",
    "predict_boxes",
    "overlay_heatmap",
)

# Padding colour used by ultralytics' own letterbox.
_PAD = (114, 114, 114)


def letterbox(img_bgr: np.ndarray, new_shape: int = 640):
    """Resize + pad to a square, replicating ultralytics' letterbox.

    Returns:
        (padded, ratio, pad): padded ``(new_shape, new_shape, 3)`` image,
        the resize ratio, and the ``(dw, dh)`` half-padding applied on each
        side (before rounding, matching ultralytics' convention).
    """
    h, w = img_bgr.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dh, dw = new_shape - nh, new_shape - nw
    top, bottom = round(dh / 2 - 0.1), round(dh / 2 + 0.1)
    left, right = round(dw / 2 - 0.1), round(dw / 2 + 0.1)
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=_PAD
    )
    return padded, r, (left, top)


def resolve_target_layers(det_model, spec) -> list[int]:
    """Validate layer indices in ``spec`` against a DetectionModel.

    Args:
        det_model: the ``.model`` (Sequential) of a loaded YOLO model.
        spec: int or list of ints, e.g. ``9`` or ``[9, 21]``.

    Raises:
        ValueError: on an out-of-range index.
    """
    idxs = [spec] if isinstance(spec, int) else list(spec)
    n = len(det_model.model)
    for i in idxs:
        if not 0 <= i < n:
            raise ValueError(f"target layer {i} out of range for {n}-layer model")
    return idxs


def _decode_scores(y: torch.Tensor, conf: float, target_class: int | None,
                   topk: int | None = None) -> torch.Tensor:
    """Reduce decoded predictions ``(1, 4+nc, 8400)`` to a scalar target.

    ``topk`` focuses the target on the strongest anchors (default None keeps
    the legacy behaviour: every anchor above ``conf``, top-10 fallback).
    A diffuse sum over thousands of background anchors is the main reason
    vanilla detection CAMs smear; ``topk=8``-ish concentrates the gradient
    on what the model actually fires on.
    """
    cls_scores = y[0, 4:, :]  # (nc, anchors)
    vals = cls_scores[target_class] if target_class is not None else cls_scores.max(dim=0).values
    if topk is not None:
        return vals.topk(min(topk, vals.numel())).values.sum()
    keep = vals[vals > conf]
    if keep.numel():
        return keep.sum()
    k = min(10, vals.numel())
    return vals.topk(k).values.sum()


def _forward_with_hook(det_model, img_tensor, layer_idx):
    """Forward pass capturing one layer's output; returns (decoded, act, hook)."""
    captured: dict[str, torch.Tensor] = {}
    handle = det_model.model[layer_idx].register_forward_hook(
        lambda _mod, _inp, out: captured.setdefault("act", out)
    )
    x = img_tensor.requires_grad_(True)
    with torch.enable_grad():
        y, _ = det_model(x)
    return y, captured["act"], handle


def _norm_cam(cam: torch.Tensor, out_hw) -> np.ndarray:
    cam = cam[0, 0].detach().cpu().numpy().astype(np.float64)
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    h, w = out_hw
    return cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def cam_heatmap(
    det_model,
    img_tensor: torch.Tensor,
    layer_idx: int,
    conf: float = 0.25,
    target_class: int | None = None,
    method: str = "hirescam",
    topk: int | None = 8,
) -> np.ndarray:
    """Class-activation heatmap with selectable method and target focus.

    Args:
        method: ``gradcam`` (GAP-weighted, legacy), ``hirescam``
            (element-wise grad weighting — much tighter on small objects),
            ``layercam`` (ReLU'd element-wise grads), ``gplusplus``
            (Grad-CAM++, 2nd/3rd-order weighting, best for multiple
            same-class instances), ``eigencam`` (gradient-free first-PC,
            ignores ``conf``/``topk``).
        topk: anchors forming the target score (None = legacy sum-all).

    Returns:
        float32 ``(imgsz, imgsz)`` heatmap in 0-1, letterbox space.
    """
    method = method.lower()
    out_hw = img_tensor.shape[-2:]
    if method == "eigencam":
        _, act, handle = _forward_with_hook(det_model, img_tensor, layer_idx)
        try:
            a = act[0].detach().cpu().numpy().astype(np.float64)  # (C,H,W)
            m = a.reshape(a.shape[0], -1)
            m = m - m.mean(axis=1, keepdims=True)
            _, _, vh = np.linalg.svd(m, full_matrices=False)
            pc = vh[0].reshape(a.shape[1:])
            base = a.mean(axis=0)
            if float((pc * base).mean()) < 0:  # fix SVD sign ambiguity
                pc = -pc
            cam = torch.from_numpy(pc)[None, None]
        finally:
            handle.remove()
        return _norm_cam(cam.float(), out_hw)

    y, act, handle = _forward_with_hook(det_model, img_tensor, layer_idx)
    try:
        with torch.enable_grad():
            score = _decode_scores(y, conf, target_class, topk)
            if method == "gplusplus":
                g1 = torch.autograd.grad(score, act, create_graph=True)[0]
                g2 = torch.autograd.grad(g1.sum(), act, retain_graph=True)[0]
                g3 = torch.autograd.grad(g2.sum(), act)[0]
                denom = 2 * g2 + (act * g3).sum(dim=(2, 3), keepdim=True) + 1e-8
                alpha = g2 / denom
                weights = (torch.relu(g1) * alpha).sum(dim=(2, 3), keepdim=True)
                cam = (weights * act).sum(dim=1, keepdim=True).clamp_min(0)
            else:
                (grad,) = torch.autograd.grad(score, act)
                if method == "hirescam":
                    cam = (act * grad).sum(dim=1, keepdim=True).clamp_min(0)
                elif method == "layercam":
                    cam = (act * torch.relu(grad)).sum(dim=1, keepdim=True).clamp_min(0)
                elif method == "gradcam":
                    weights = grad.mean(dim=(2, 3), keepdim=True)
                    cam = (weights * act).sum(dim=1, keepdim=True).clamp_min(0)
                else:
                    raise ValueError(f"unknown CAM method {method!r}")
    finally:
        handle.remove()
    return _norm_cam(cam, out_hw)


def gradcam_heatmap(
    det_model,
    img_tensor: torch.Tensor,
    layer_idx: int,
    conf: float = 0.25,
    target_class: int | None = None,
) -> np.ndarray:
    """Legacy vanilla Grad-CAM (sum-all target). Prefer :func:`cam_heatmap`."""
    return cam_heatmap(det_model, img_tensor, layer_idx, conf=conf,
                       target_class=target_class, method="gradcam", topk=None)


def backbone_pyramid(det_model) -> list[int]:
    """Backbone P3/P4/P5 feature layers: ``[sppf-5, sppf-3, sppf]``.

    Verified across all arm architectures (stock, SPD, LSK-shifted, Faster):
    P3 = last C2f(*)-level backbone feature at /8, P4 at /16, SPPF at /32.
    """
    n = len(det_model.model)
    types = [type(m).__name__ for m in det_model.model]
    s = max(i for i, t in enumerate(types) if t == "SPPF")
    pyr = [s - 5, s - 3, s]
    if pyr[0] < 0 or not all(t in ("C2f", "C2f_Faster") for t in (types[pyr[0]], types[pyr[1]])):
        raise ValueError(f"backbone pyramid not found (SPPF={s}, types={types})")
    return pyr


def all_cam_layers(det_model, min_stride: int = 8) -> list[int]:
    """Every hookable layer at stride >= ``min_stride`` (default: P3/8 and deeper).

    SPPF-only CAM misses every path that bypasses SPPF to Detect: the
    LSKBlock, the backbone P3/P4 taps, and the whole ASF neck (Zoom_cat,
    ScaleSequenceFusion, Add, neck C2f_Faster/Concat/Conv). All of those
    emit 4D activations, so ``cam_heatmap`` can hook them directly —
    including the non-parametric fusion layers (Concat/Add/Zoom_cat), whose
    gradients are well-defined even without weights.

    Excludes the Detect head (list output, not a feature map) and, by
    default, the P1/2 + P2/4 stem (layers 0-2: 320x320/160x160 edge noise
    that hurts the pointing game). For the proposed arm (SPPF@10) this
    returns layers 3..22; for stock arms (SPPF@9) layers 3..21.
    """
    skip = {"Detect", "v10Detect", "WorldDetect", "YOLOEDetect", "Segment",
            "Pose", "OBB", "Classify", "ImagePoolingAttn", "RTDETRDecoder"}
    layers = [i for i, m in enumerate(det_model.model)
              if type(m).__name__ not in skip]
    if min_stride >= 8:  # drop the full-res stem: indices before the P3/8 downsampler
        types = [type(m).__name__ for m in det_model.model]
        s = max(i for i, t in enumerate(types) if t == "SPPF")
        # P3/8 downsampler sits 6 (stock) or 7 (LSK-shifted) rows before SPPF
        p3_down = s - (7 if "LSKBlock" in types else 6)
        layers = [i for i in layers if i >= p3_down]
    return layers


def detect_head_layers(det_model) -> list[int]:
    """Indices feeding the Detect head, i.e. ``[P3-head, P4-head, P5-head]``.

    Read off the Detect module's stored ``from`` indices (attached as
    ``.f`` by ``parse_model``), so this is correct on every arm without
    hardcoding: ``[16, 19, 22]`` on the proposed/LSK-shifted arms,
    ``[15, 18, 21]`` on stock-scale arms. Each head owns one anchor
    scale, so a box predicted by the P3 head has (numerically) zero
    gradient w.r.t. the P5 head's input — per-box CAMs must target the
    box's own head (see :func:`fused_perbox_cam`).
    """
    detect = det_model.model[-1]
    f = getattr(detect, "f", None)
    if f is None:
        raise ValueError("Detect head has no stored 'from' indices (.f)")
    heads = list(f) if isinstance(f, (list, tuple)) else [f]
    n = len(det_model.model)
    for i in heads:
        if not 0 <= i < n:
            raise ValueError(f"Detect head input {i} out of range for {n}-layer model")
    return heads


def multiscale_cam(    det_model,
    img_tensor: torch.Tensor,
    layers: list[int] | None = None,
    method: str = "hirescam",
    topk: int | None = 8,
    conf: float = 0.25,
    target_class: int | None = None,
) -> np.ndarray:
    """Mean-fused CAM over backbone pyramid layers (default: P3+P4+P5).

    Fine layers (P3/8) localise thin cracks; coarse layers (P5/32) carry
    semantics. Averaging after upsampling keeps both.
    """
    layers = layers if layers is not None else backbone_pyramid(det_model)
    cams = [cam_heatmap(det_model, img_tensor, li, conf=conf,
                        target_class=target_class, method=method, topk=topk)
            for li in layers]
    fused = sum(cams) / len(cams)
    fused -= fused.min()
    if fused.max() > 0:
        fused /= fused.max()
    return fused.astype(np.float32)


def unpad_cam(cam_letterboxed: np.ndarray, ratio: float, pad: tuple[int, int],
              orig_hw: tuple[int, int]) -> np.ndarray:
    """Map a letterbox-space heatmap back to original-image pixels (0-1 float)."""
    dw, dh = pad
    h, w = orig_hw
    nh, nw = round(h * ratio), round(w * ratio)
    cropped = cam_letterboxed[dh:dh + nh, dw:dw + nw]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def box_target_cam(
    det_model,
    img_tensor: torch.Tensor,
    layer_idx: int,
    target_box_letterbox,
    target_class: int | None = None,
    method: str = "layercam",
    iou_thr: float = 0.5,
    topk: int = 3,
    conf: float = 0.0,
) -> np.ndarray:
    """CAM for ONE box: target = anchors whose decoded box matches it.

    Instead of summing global anchors (which blends every object in the
    scene), this scores only the anchors behind a single predicted/GT box,
    so each TP or FP gets its own explanation ("why did the model fire
    HERE?"). Raw YOLOv8 decoded boxes are cxcywh in letterbox pixels and
    are converted to xyxy internally.

    Args:
        target_box_letterbox: ``(x1, y1, x2, y2)`` in letterbox pixels —
            map original-pixel boxes with ``x * ratio + pad``.
        target_class: class whose score forms the target (None = best class
            per anchor).
        iou_thr: anchors below this IoU with the target are ignored; if
            none pass, the top-``topk`` IoU anchors are used.
        topk: cap on anchors forming the target.
        conf: minimum anchor score (0 = no filtering).

    Returns:
        float32 ``(imgsz, imgsz)`` heatmap in 0-1, letterbox space.
    """
    out_hw = img_tensor.shape[-2:]
    y, act, handle = _forward_with_hook(det_model, img_tensor, layer_idx)
    try:
        with torch.enable_grad():
            boxes = y[0, :4, :].detach()  # (4, anchors) cxcywh, letterbox px
            cx, cy, bw, bh = boxes[0], boxes[1], boxes[2], boxes[3]
            boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
            tb = torch.tensor(target_box_letterbox, dtype=boxes.dtype, device=boxes.device)
            x1 = torch.maximum(boxes[0], tb[0])
            y1 = torch.maximum(boxes[1], tb[1])
            x2 = torch.minimum(boxes[2], tb[2])
            y2 = torch.minimum(boxes[3], tb[3])
            inter = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
            area = (boxes[2] - boxes[0]).clamp_min(0) * (boxes[3] - boxes[1]).clamp_min(0)
            tarea = ((tb[2] - tb[0]) * (tb[3] - tb[1])).clamp_min(1e-8)
            ious = inter / (area + tarea - inter + 1e-8)
            cls_scores = y[0, 4:, :]
            vals = cls_scores[target_class] if target_class is not None else cls_scores.max(dim=0).values
            vals = vals.detach()
            keep = (ious >= iou_thr) & (vals > conf)
            if int(keep.sum()) == 0:
                keep = torch.zeros_like(keep)
                keep[torch.topk(ious, min(topk, ious.numel())).indices] = True
            idx = torch.nonzero(keep).squeeze(1)[:topk]
            # Re-attach: recompute the score WITH grad through the graph.
            cls_g = y[0, 4:, :]
            v = cls_g[target_class, idx] if target_class is not None else cls_g[:, idx].max(dim=0).values
            score = v.sum()
            if method == "eigencam":
                raise ValueError("eigencam is gradient-free; use cam_heatmap instead")
            if method == "gplusplus":
                g1 = torch.autograd.grad(score, act, create_graph=True)[0]
                g2 = torch.autograd.grad(g1.sum(), act, retain_graph=True)[0]
                g3 = torch.autograd.grad(g2.sum(), act)[0]
                denom = 2 * g2 + (act * g3).sum(dim=(2, 3), keepdim=True) + 1e-8
                alpha = g2 / denom
                weights = (torch.relu(g1) * alpha).sum(dim=(2, 3), keepdim=True)
                cam = (weights * act).sum(dim=1, keepdim=True).clamp_min(0)
            else:
                (grad,) = torch.autograd.grad(score, act)
                if method in ("hirescam", "layercam"):
                    g = torch.relu(grad) if method == "layercam" else grad
                    cam = (act * g).sum(dim=1, keepdim=True).clamp_min(0)
                elif method == "gradcam":
                    weights = grad.mean(dim=(2, 3), keepdim=True)
                    cam = (weights * act).sum(dim=1, keepdim=True).clamp_min(0)
                else:
                    raise ValueError(f"unknown CAM method {method!r}")
    finally:
        handle.remove()
    return _norm_cam(cam, out_hw)


def _box_energy(cam_letterboxed: np.ndarray, ratio: float, pad: tuple[int, int],
                orig_hw: tuple[int, int], box_orig) -> float:
    """Heat-mass fraction inside ``box_orig`` (original-pixel coords)."""
    cam = unpad_cam(cam_letterboxed, ratio, pad, orig_hw)
    total = float(cam.sum())
    if total <= 0:
        return 0.0
    x1, y1, x2, y2 = box_orig[:4]
    h, w = cam.shape
    ys, xs = np.mgrid[0:h, 0:w]
    inside = (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)
    return float(cam[inside].sum()) / total


def fused_perbox_cam(
    det_model,
    img_tensor: torch.Tensor,
    boxes_orig: list[tuple[float, float, float, float]],
    ratio: float,
    pad: tuple[int, int],
    orig_hw: tuple[int, int],
    method: str = "layercam",
    conf: float = 0.25,
    context_weight: float = 0.25,
    min_energy: float = 0.10,
) -> tuple[np.ndarray, list[dict]]:
    """One fused heatmap in which every box carries its own gradient.

    For each box, a per-box CAM (:func:`box_target_cam`) is computed at
    every Detect-head input (:func:`detect_head_layers`) and the
    max-energy head is kept — the box's own scale, the only head with a
    non-zero gradient path to it. The per-box maps are combined with a
    pixel-wise max (every box keeps full-strength heat), then blended
    with a faint global top-8 context map at the P5 head so background
    scene context is not pitch black::

        final = (1 - w) * max(per_box) + w * global_top8

    Args:
        boxes_orig: ``(x1, y1, x2, y2[, ...])`` targets in original
            pixels; mapped to letterbox space internally.
        context_weight: blend weight ``w`` for the global map (0 = boxes only).
        min_energy: boxes below this own-box energy are reported
            ``supported=False`` (regression gate, default 0.10).

    Returns:
        (fused, report): float32 ``(imgsz, imgsz)`` heatmap in 0-1
        (letterbox space) and per-box ``{head, energy, supported}`` rows.
    """
    heads = detect_head_layers(det_model)
    p5 = heads[-1]
    dw, dh = pad
    per_box: list[np.ndarray] = []
    report: list[dict] = []
    for bi, b in enumerate(boxes_orig):
        lb = (b[0] * ratio + dw, b[1] * ratio + dh,
              b[2] * ratio + dw, b[3] * ratio + dh)
        cands = [box_target_cam(det_model, img_tensor, li, lb, method=method)
                 for li in heads]
        energies = [_box_energy(c, ratio, pad, orig_hw, b) for c in cands]
        k = int(np.argmax(energies))
        per_box.append(cands[k])
        report.append({"box": bi, "head": heads[k], "energy": float(energies[k]),
                       "supported": bool(energies[k] >= min_energy)})
    fused = np.maximum.reduce(per_box) if len(per_box) > 1 else per_box[0].copy()
    if context_weight > 0:
        ctx = cam_heatmap(det_model, img_tensor, p5, conf=conf,
                          method="layercam", topk=8)
        fused = (1.0 - context_weight) * fused + context_weight * ctx
    fused -= fused.min()
    if fused.max() > 0:
        fused /= fused.max()
    return fused.astype(np.float32), report


def threshold_cam(cam: np.ndarray, keep: float = 0.4) -> np.ndarray:
    """Zero everything but the top-``keep`` heat-mass fraction, renormalise.

    Display-side cleanup: blobs shrink to their cores instead of washing
    the whole road. Does not change localisation metrics, only rendering.
    """
    flat = np.sort(cam.ravel())[::-1]
    thresh = flat[min(int(keep * flat.size), flat.size - 1)]
    out = np.where(cam >= thresh, cam, 0).astype(np.float32)
    mx = out.max()
    return (out / mx).astype(np.float32) if mx > 0 else out


def predict_boxes(yolo_model, image_path: str | pathlib.Path, conf: float = 0.25, imgsz: int = 640):
    """Run the standard NMS predict path; return boxes in original pixels.

    Returns:
        list of ``(x1, y1, x2, y2, cls_id, confidence)`` tuples.
    """
    res = yolo_model.predict(
        source=str(image_path), conf=conf, imgsz=imgsz, verbose=False
    )[0]
    out = []
    if res.boxes is not None and len(res.boxes):
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        cf = res.boxes.conf.cpu().numpy()
        out = [
            (float(x1), float(y1), float(x2), float(y2), int(c), float(p))
            for (x1, y1, x2, y2), c, p in zip(xyxy, cls, cf)
        ]
    return out


def overlay_heatmap(
    orig_bgr: np.ndarray,
    cam_letterboxed: np.ndarray,
    ratio: float,
    pad: tuple[int, int],
    alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a letterbox-space heatmap back onto the original image.

    Returns:
        (overlay_bgr, heat_bgr): the blended overlay and the colour-mapped
        heat alone, both at the original image resolution.
    """
    imgsz = cam_letterboxed.shape[0]
    dw, dh = pad
    h, w = orig_bgr.shape[:2]
    nh, nw = round(h * ratio), round(w * ratio)
    # Crop the padding, then stretch to the original size.
    cropped = cam_letterboxed[dh : dh + nh, dw : dw + nw]
    cam = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    heat_u8 = (np.clip(cam, 0, 1) * 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_bgr, 1.0 - alpha, heat_bgr, alpha, 0)
    return overlay, heat_bgr
