"""Large Selective Kernel (LSK) block, reimplemented for this project's YOLOv8s neck.

Clean-room reimplementation of the selective-kernel mechanism from Li et al.,
"Large Selective Kernel Network for Remote Sensing Object Detection"
(ICCV 2023) and its journal extension "LSKNet: A Foundation Lightweight
Backbone for Remote Sensing" (IJCV 2024,
https://arxiv.org/abs/2403.11735). No code is copied from the official
`zcablii/LSKNet` repository (CC BY-NC 4.0, non-commercial): only the
architecture described in the paper is reproduced here, with own naming and
structure, and without the ``mmcv``/``timm`` dependencies of the reference
implementation.

Mechanism (paper §2): two depthwise convolutions with growing receptive
fields (5x5, then 7x7 dilated-3) produce two candidate responses; each is
projected to half width, concatenated, and pooled along the scale axis with
both average and max to form a 2-channel descriptor; a 7x7 convolution +
sigmoid turns the descriptor into per-position selection weights; the
weighted sum is projected back to full width and applied multiplicatively::

    attn = proj_out(w1 * half1 + w2 * half2)
    out = x * attn            # same channels, same resolution

The full stage wrapper used here is the paper's transformer-style block:
``norm -> Attention -> layer-scale -> stochastic-depth`` residual, then
``norm -> MLP(1x1 -> DW3x3 -> GELU -> 1x1) -> layer-scale ->
stochastic-depth`` residual. ``Attention`` is a 1x1 projection, GELU, the
selective kernel above, a second 1x1 projection, and a residual.

Yaml / ``parse_model`` contract: the block is channel- and
resolution-preserving, so it is constructed as ``LSKBlock(c1, c2, ...)``
with ``c1 == c2`` enforced after width scaling (a ``ValueError`` otherwise).
A yaml row ``[-1, 1, LSKBlock, [256]]`` at width 0.5 therefore yields
``c1 = c2 = 128``.
"""

from __future__ import annotations

import torch
from torch import nn


class _DropPath(nn.Module):
    """Stochastic depth per sample: randomly drops the whole residual branch.

    Stand-in for ``timm.models.layers.DropPath`` so this module needs no new
    dependency. With ``p == 0`` (the default used here) it is the identity.
    """

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"drop_path probability must be in [0, 1], got {p}.")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x / keep * mask


class _SelectiveKernel(nn.Module):
    """The large-selective-kernel spatial gate (paper's ``LSKblock``).

    Args:
        channels: input (= output) width. Must be even and >= 2, because the
            two scale responses are each projected to ``channels // 2``.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels < 2 or channels % 2:
            raise ValueError(f"selective-kernel width must be even and >= 2, got {channels}.")
        half = channels // 2
        self.dw_small = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.dw_large = nn.Conv2d(channels, channels, 7, padding=9, dilation=3, groups=channels)
        self.proj_small = nn.Conv2d(channels, half, 1)
        self.proj_large = nn.Conv2d(channels, half, 1)
        self.selector = nn.Conv2d(2, 2, 7, padding=3)
        self.proj_out = nn.Conv2d(half, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        small = self.dw_small(x)
        large = self.dw_large(small)
        half_small = self.proj_small(small)
        half_large = self.proj_large(large)
        stacked = torch.cat([half_small, half_large], dim=1)
        pooled_avg = torch.mean(stacked, dim=1, keepdim=True)
        pooled_max, _ = torch.max(stacked, dim=1, keepdim=True)
        weights = self.selector(torch.cat([pooled_avg, pooled_max], dim=1)).sigmoid()
        fused = half_small * weights[:, 0:1] + half_large * weights[:, 1:2]
        return x * self.proj_out(fused)


class _LSKAttention(nn.Module):
    """Projection + GELU + selective kernel + projection, with a residual."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj_in = nn.Conv2d(channels, channels, 1)
        self.act = nn.GELU()
        self.gate = _SelectiveKernel(channels)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        return shortcut + self.proj_out(self.gate(self.act(self.proj_in(x))))


class _LSKFeedForward(nn.Module):
    """1x1 expansion, depthwise 3x3, GELU, 1x1 contraction (paper's ``Mlp``)."""

    def __init__(self, channels: int, hidden: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden, channels, 1)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dw(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return self.drop2(x)


class LSKBlock(nn.Module):
    """Full LSK stage block (paper's ``Block``): attention + feed-forward.

    Channel- and resolution-preserving: input ``(N, c, H, W)`` yields
    ``(N, c, H, W)``. ``c1`` must equal ``c2`` after the yaml width scaling;
    a mismatch raises ``ValueError`` rather than silently projecting, so a
    mis-wired yaml row fails loudly.

    Args:
        c1: input channels (from ``parse_model``).
        c2: output channels (from ``parse_model``, width-scaled yaml arg).
        mlp_ratio: feed-forward hidden width as a multiple of ``c2``.
            The reference backbone uses 8/8/4/4 across its four stages;
            P3-level features correspond to a late stage, hence the
            default of 4.
        drop: dropout rate inside the feed-forward network.
        drop_path: stochastic-depth rate on both residuals (0 = identity).
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if c1 != c2:
            raise ValueError(
                f"LSKBlock is channel-preserving, got c1={c1} != c2={c2}. "
                "Point it at a single width (e.g. [-1, 1, LSKBlock, [256]] at width 0.5)."
            )
        dim = c2
        hidden = int(dim * mlp_ratio)
        if hidden < 1:
            raise ValueError(f"LSKBlock hidden width must be >= 1, got {hidden} (dim={dim}, ratio={mlp_ratio}).")
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.attn = _LSKAttention(dim)
        self.ffn = _LSKFeedForward(dim, hidden, drop)
        self.drop_path1 = _DropPath(drop_path)
        self.drop_path2 = _DropPath(drop_path)
        init = 1e-2
        self.scale1 = nn.Parameter(init * torch.ones(dim), requires_grad=True)
        self.scale2 = nn.Parameter(init * torch.ones(dim), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.scale1.view(1, -1, 1, 1)
        s2 = self.scale2.view(1, -1, 1, 1)
        x = x + self.drop_path1(s1 * self.attn(self.norm1(x)))
        x = x + self.drop_path2(s2 * self.ffn(self.norm2(x)))
        return x
