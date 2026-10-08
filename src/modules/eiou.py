"""Efficient-IoU (EIoU) box-regression loss, adapted for ultralytics 8.4.170.

Replaces the CIoU term of ultralytics' ``BboxLoss`` (which is what
``v8DetectionLoss`` instantiates) without editing the installed package: call
``install_eiou()`` before the trainer builds its criterion. Same pattern as
``src/modules/wiou.py:install_wiou()``.

What is implemented
--------------------
EIoU (Zhang et al., "Focal and Efficient IoU Loss for Accurate Bounding Box
Regression", Neurocomputing 2022) splits the CIoU aspect-ratio penalty into
separate width and height terms::

    EIoU = IoU - (rho_b^2 / c^2 + rho_w^2 / cw^2 + rho_h^2 / ch^2)

where ``rho_b`` is the centre distance, ``c`` the enclosing-box diagonal,
``rho_w``/``rho_h`` the width/height differences, and ``cw``/``ch`` the
enclosing-box width/height. Reference math ported from the
``arojsubedi/Focal-EIoU`` community adaptation of ``ultralytics/utils/metrics.py``
(``bbox_iou(..., EIoU=True, Focal=...)``); the loss weighting follows its
``loss.py::BboxLoss.forward``:

* plain EIoU (``focal=False``): ``loss = (1 - EIoU) * weight``
* Focal-EIoU (``focal=True``): ``loss = (1 - EIoU) * IoU^gamma * weight``
  (``gamma=0.5`` default, IoU factor detached), down-weighting easy samples.

Only the box-IoU term changes; DFL/L1 handling is inherited verbatim from
the installed ``BboxLoss.forward`` (``ultralytics/utils/loss.py:119-156``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import BboxLoss, bbox2dist

EPS = 1e-7


def eiou(
    pred_bboxes: torch.Tensor,
    target_bboxes: torch.Tensor,
    focal: bool = False,
    gamma: float = 0.5,
    alpha: float = 1.0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Per-pair EIoU for ``xyxy`` boxes.

    Args:
        pred_bboxes: predicted boxes, ``[..., 4]`` as ``x1, y1, x2, y2``.
        target_bboxes: matching ground-truth boxes, same shape.
        focal: if True also return the ``IoU^gamma`` re-weighting factor
            (Focal-EIoU); if False return plain EIoU only.
        gamma: Focal-EIoU focusing exponent on IoU.
        alpha: power applied to the enclosing ``cw``/``ch`` denominators
            (community script default 1.0; kept for parity).

    Returns:
        EIoU tensor ``[..., 1]``, or ``(EIoU, IoU^gamma)`` when focal.
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = pred_bboxes.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = target_bboxes.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + EPS
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + EPS

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(min=0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(min=0)
    union = w1 * h1 + w2 * h2 - inter + EPS
    iou = inter / union

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # enclosing width
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # enclosing height
    c2 = cw.pow(2) + ch.pow(2) + EPS
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4
    rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)).pow(2)
    rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)).pow(2)
    cw2 = torch.pow(cw.pow(2) + EPS, alpha)
    ch2 = torch.pow(ch.pow(2) + EPS, alpha)
    eiou_val = iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)
    if focal:
        return eiou_val, torch.pow(iou.clamp(min=0.0), gamma)
    return eiou_val


class EIoULoss(BboxLoss):
    """``BboxLoss`` with CIoU replaced by EIoU (or Focal-EIoU). DFL unchanged."""

    def __init__(self, reg_max: int = 16, focal: bool = False, gamma: float = 0.5) -> None:
        super().__init__(reg_max)
        self.focal = focal
        self.gamma = gamma

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
        """Compute EIoU and DFL (or L1 when reg_max <= 1) losses for bounding boxes."""
        fg_mask = fg_mask.nonzero(as_tuple=True)
        weight = target_scores[fg_mask].sum(-1, keepdim=True)
        iou = eiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], focal=self.focal, gamma=self.gamma)
        if isinstance(iou, tuple):
            loss_iou = ((1.0 - iou[0]) * iou[1].detach() * weight).sum() / target_scores_sum
        else:
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss (verbatim from installed BboxLoss)
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


def install_eiou(focal: bool = False, gamma: float = 0.5) -> type[BboxLoss]:
    """Patch ``ultralytics.utils.loss.BboxLoss`` so training uses EIoU.

    Same mechanism as :func:`src.modules.wiou.install_wiou`: ``BboxLoss`` is
    defined at ``ultralytics/utils/loss.py:111`` and instantiated by
    ``v8DetectionLoss.__init__`` at ``utils/loss.py:382``, which resolves the
    module global at call time. Rebinding that one attribute is enough. Call
    this before the trainer is constructed (and again on resume, before the
    trainer rebuilds its criterion).

    Args:
        focal: ``False`` for plain EIoU (``I_eiou`` default — clean
            single-variable ablation vs WIoU); ``True`` for Focal-EIoU.
        gamma: Focal-EIoU focusing exponent (only used when focal=True).

    Returns:
        The original class, for restoring.
    """
    import ultralytics.utils.loss as loss_module

    original = loss_module.BboxLoss

    def make(reg_max=16):
        return EIoULoss(reg_max, focal=focal, gamma=gamma)

    loss_module.BboxLoss = make
    return original
