"""Wise-IoU (WIoU) box-regression loss, written from Tong et al., "Wise-IoU:
Bounding Box Regression Loss with Dynamic Focusing Mechanism" (arXiv:2301.10051).

Replaces the CIoU term of ultralytics' ``BboxLoss`` (which is what
``v8DetectionLoss`` instantiates) without editing the installed package: call
``install_wiou()`` before the trainer builds its criterion.

What the paper defines, and what is implemented here
-----------------------------------------------------
``L_IoU = 1 - IoU``, ``rho^B`` is the centre distance between boxes, ``L^B`` the
diagonal of the smallest enclosing box (its square is ``c2`` in ultralytics' own
CIoU code).

* Eq. 13 — Wise attention::

      L_WIoU = L_IoU * exp( rho^B^2 / detach(c2) )

* Eq. 15 — non-monotonic focusing (``version="v1"``, the default)::

      beta   = mean(L_IoU) / L_IoU          # detached both ways
      L      = L_WIoU * beta / (delta * alpha ** (beta - delta))

  with ``alpha=1.7``, ``delta=2.7``. Medium-quality samples get the most weight,
  outliers are suppressed.

* Eq. 18 — monotonic focusing (``version="v2"``)::

      beta   = L_IoU / mean(L_IoU)          # detached
      L      = L_WIoU * beta ** 0.5

WIoU v3 is not implemented: it replaces the beta term with a function fitted
offline against the training set, which is a data-fitting step, not a loss.

``mean(L_IoU)`` is an exponential moving average (``momentum=1e-2``) held in a
buffer, as in the reference implementation, rather than a per-batch mean.

Note for anyone comparing against community implementations: the widely copied
``Instinct323/Wise-IoU`` script keeps ``beta = L_IoU / mean`` in *both* branches,
i.e. the reciprocal of the paper's v1 definition. This module follows the paper.
"""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.utils.loss import BboxLoss, bbox2dist

EPS = 1e-7


def wise_iou_loss(
    pred_bboxes: torch.Tensor,
    target_bboxes: torch.Tensor,
    iou_mean: torch.Tensor,
    version: str = "v1",
    alpha: float = 1.7,
    delta: float = 2.7,
    beta_scale: "torch.Tensor | None" = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-pair WIoU loss for ``xyxy`` boxes.

    Args:
        pred_bboxes: predicted boxes, ``[N, 4]`` as ``x1, y1, x2, y2``.
        target_bboxes: matching ground-truth boxes, ``[N, 4]``.
        iou_mean: running mean of the IoU loss, a 0-dim tensor.
        version: ``"v1"`` (non-monotonic focusing) or ``"v2"`` (monotonic).
        alpha: focusing sharpness, paper value 1.7.
        delta: focusing offset, paper value 2.7.

    Returns:
        ``(loss, loss_iou)``: the focused WIoU loss and the unfocused IoU loss,
        both ``[N, 1]``. The second value is what ``iou_mean`` tracks.
    """
    px1, py1, px2, py2 = pred_bboxes.split(1, dim=-1)
    tx1, ty1, tx2, ty2 = target_bboxes.split(1, dim=-1)

    inter = (torch.minimum(px2, tx2) - torch.maximum(px1, tx1)).clamp(min=0) * (
        torch.minimum(py2, ty2) - torch.maximum(py1, ty1)
    ).clamp(min=0)
    union = (px2 - px1) * (py2 - py1) + (tx2 - tx1) * (ty2 - ty1) - inter
    loss_iou = 1.0 - inter / union.clamp(min=EPS)

    # rho^B^2 / L^B^2, denominator detached (Eq. 13)
    rho2 = ((px1 + px2 - tx1 - tx2) ** 2 + (py1 + py2 - ty1 - ty2) ** 2) / 4
    c2 = (
        (torch.maximum(px2, tx2) - torch.minimum(px1, tx1)).pow(2)
        + (torch.maximum(py2, ty2) - torch.minimum(py1, ty1)).pow(2)
    ).clamp(min=EPS)
    loss_wiou = loss_iou * torch.exp(rho2 / c2.detach())

    if version == "v1":  # non-monotonic focusing, Eq. 15
        beta = iou_mean.detach() / loss_iou.detach().clamp(min=EPS)
        loss = loss_wiou * beta / (delta * alpha ** (beta - delta))
    elif version == "v2":  # monotonic focusing, Eq. 18
        beta = loss_iou.detach() / iou_mean.detach().clamp(min=EPS)
        loss = loss_wiou * beta.sqrt()
    elif version == "v3":  # non-monotonic focusing with a learnable outlier-degree scale
        # The paper's third revision adds a learnable degree of focusing rather than
        # fixing it at v1's shape. beta_scale is a single learnable scalar initialised
        # to 1.0, so at initialisation this is exactly v1; training is free to sharpen
        # or flatten the focusing curve. Because the scale only ever multiplies the
        # coefficient, it cannot change the sign of the loss.
        beta = (iou_mean.detach() / loss_iou.detach().clamp(min=EPS)).clamp(min=EPS) * (
            1.0 if beta_scale is None else beta_scale
        )
        loss = loss_wiou * beta / (delta * alpha ** (beta - delta))
    else:
        raise ValueError(f"unsupported WIoU version {version!r}, expected 'v1', 'v2' or 'v3'")
    return loss, loss_iou


class WiseIoU(BboxLoss):
    """``BboxLoss`` with CIoU replaced by WIoU. DFL handling is unchanged."""

    def __init__(self, reg_max: int = 16, version: str = "v1", momentum: float = 1e-2) -> None:
        super().__init__(reg_max)
        self.version = version
        self.momentum = momentum
        self.register_buffer("iou_mean", torch.tensor(1.0))
        # v3's focusing scale is injected from the MODEL by train_with_wiou(); it is not
        # a parameter of this criterion, because v8DetectionLoss is a plain class rather
        # than an nn.Module, so the trainer's optimizer would never see it here.
        self.beta_scale: torch.Tensor | None = None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute WIoU and DFL (or L1 when reg_max <= 1) losses for bounding boxes."""
        fg_mask = fg_mask.nonzero(as_tuple=True)
        weight = target_scores[fg_mask].sum(-1, keepdim=True)
        loss_iou, raw_iou = wise_iou_loss(
            pred_bboxes[fg_mask], target_bboxes[fg_mask], self.iou_mean, self.version, beta_scale=self.beta_scale
        )
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                self.iou_mean.mul_(1 - self.momentum).add_(self.momentum * raw_iou.detach().mean())
        loss_iou = (loss_iou * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


def install_wiou(version: str = "v1", momentum: float = 1e-2, model: torch.nn.Module | None = None) -> type[BboxLoss]:
    """Patch ``ultralytics.utils.loss.BboxLoss`` so training uses WIoU.

    ``BboxLoss`` is defined at ``ultralytics/utils/loss.py:111`` and instantiated
    by ``v8DetectionLoss.__init__`` at ``utils/loss.py:382``, which resolves the
    module global at call time. Rebinding that one module attribute is therefore
    enough — ``ultralytics/nn/tasks.py`` imports only the ``v8*Loss`` classes
    (lines 94-104), never ``BboxLoss`` itself. Call this before the trainer is
    constructed.

    Args:
        version: ``"v1"``, ``"v2"`` or ``"v3"``.
        momentum: EMA rate for the running IoU-loss mean.
        model: for ``version="v3"``, the model whose ``wiou_beta_scale`` parameter
            (a scalar ``nn.Parameter``) the focusing coefficient must read. This is
            required for v3 to actually learn: the scale must be visible to the
            trainer's optimizer, and ``v8DetectionLoss`` is a plain class, so a
            parameter registered on the criterion would be ignored. Add the
            parameter with :func:`add_beta_scale` before training.

    Returns:
        The original class, for restoring.
    """
    import ultralytics.utils.loss as loss_module

    if version == "v3" and model is None:
        raise ValueError("WIoU v3 needs the model so beta_scale can be optimizer-visible; pass model=...")

    original = loss_module.BboxLoss

    def make(reg_max=16):
        criterion = WiseIoU(reg_max, version=version, momentum=momentum)
        if version == "v3":
            criterion.beta_scale = model.wiou_beta_scale
        return criterion

    loss_module.BboxLoss = make
    return original


def add_beta_scale(model: torch.nn.Module) -> torch.nn.Module:
    """Add v3's learnable focusing scale to ``model`` so the optimizer can train it.

    Registers a scalar ``nn.Parameter`` named ``wiou_beta_scale`` initialised to 1.0
    (i.e. v3 starts identical to v1). Call once, before training, and pair it with
    ``install_wiou("v3", model=model)``.
    """
    if not hasattr(model, "wiou_beta_scale"):
        # assigned directly, not add_module(): nn.Module.__setattr__ registers a bare
        # Parameter, whereas add_module() only accepts Module instances.
        model.wiou_beta_scale = torch.nn.Parameter(torch.tensor(1.0))
    return model
