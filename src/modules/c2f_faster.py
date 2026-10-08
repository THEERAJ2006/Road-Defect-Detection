"""FasterNet blocks for use inside the ultralytics ``C2f`` stage wrapper.

Implements the partial-convolution (PConv) mechanism from "FasterNet: A Fast Lightweight ConvNet
Enabling High-Predictive and Computational Efficiency" (Chen et al., CVPR 2023), and a copy of
ultralytics ``C2f`` whose bottlenecks are ``FasterBlock`` instead of ``Bottleneck``.
"""

from __future__ import annotations

import torch
from torch import nn

from ultralytics.nn.modules import Conv


class PConv(nn.Module):
    """Partial convolution: run a 3x3 convolution over only a fraction of the input channels.

    Standard convolutions recompute the same full-density features for every channel at every
    spatial position, which is largely redundant. PConv exploits that redundancy directly: it
    convolves only the first ``c_ = int(c2 * ratio)`` input channels with a 3x3 kernel, pushes the
    remaining channels through a cheap pointwise 1x1 convolution, and concatenates the two
    results. The 3x3 -- the expensive, spatially dense part -- is therefore applied to only
    ``ratio`` of the channels, which is the mechanism that cuts redundant computation; the 1x1
    keeps the remaining channels in the output without paying for spatial mixing.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        ratio (float): Fraction of the output channels produced by the 3x3 branch.

    Attributes:
        c_ (int): Channels handled by the 3x3 convolution, ``int(c2 * ratio)``.
        cv1 (Conv): 3x3 convolution over the ``c_``-wide leading slice ``x[:, :c_]``.
        cv2 (Conv): 1x1 convolution over the remaining ``x[:, c_:]``.
    """

    def __init__(self, c1: int, c2: int, ratio: float = 0.25) -> None:
        """Initialize the partial 3x3 and the pointwise 1x1 convolution."""
        super().__init__()
        self.c_ = int(c2 * ratio)
        # cv1 sees only the c_-wide slice, so it consumes and produces c_ channels; cv2 maps the
        # remaining c1 - c_ input channels to the remaining c2 - c_ output channels.
        self.cv1 = Conv(self.c_, self.c_, 3, 1)
        self.cv2 = Conv(c1 - self.c_, c2 - self.c_, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve the leading channels at 3x3 and the remainder at 1x1, then concatenate.

        Args:
            x (torch.Tensor): Input tensor of shape ``(N, c1, H, W)``.

        Returns:
            torch.Tensor: Output tensor of shape ``(N, c2, H, W)``.
        """
        c_ = self.c_
        return torch.cat((self.cv1(x[:, :c_]), self.cv2(x[:, c_:])), 1)


class FasterBlock(nn.Module):
    """FasterNet block: a partial convolution followed by a full 3x3 convolution.

    The pairing is what matters. The trailing full 3x3 mixes every channel spatially, while the
    preceding PConv has already confined the dense 3x3 work to a fraction of the channels.
    Against the standard ``Bottleneck`` it replaces, this removes redundant per-channel
    computation rather than changing what the stage can express (Chen et al., CVPR 2023).

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        shortcut (bool): Add an identity connection when the input and output widths match.
        g (int): Groups for convolutions. Accepted for signature compatibility with ``Bottleneck``;
            the PConv branches are dense convolutions, so it does not alter their grouping.

    Attributes:
        pconv (PConv): Partial convolution.
        conv (Conv): Full 3x3 convolution applied to the partial-convolution output.
        shortcut (bool): Stored identity-connection flag.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1) -> None:
        """Initialize the partial convolution and the full 3x3 convolution."""
        super().__init__()
        self.shortcut = shortcut
        self.pconv = PConv(c1, c2)
        self.conv = Conv(c2, c2, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the PConv + 3x3 stack, adding the identity shortcut when enabled.

        Args:
            x (torch.Tensor): Input tensor of shape ``(N, c1, H, W)``.

        Returns:
            torch.Tensor: Output tensor of shape ``(N, c2, H, W)``.
        """
        y = self.conv(self.pconv(x))
        return x + y if self.shortcut and x.shape[1] == y.shape[1] else y


class C2f_Faster(nn.Module):
    """FasterNet-flavoured ``C2f``: same CSP plumbing, ``FasterBlock`` instead of ``Bottleneck``.

    Structurally identical to ultralytics ``C2f`` -- same ``c`` / ``cv1`` / ``cv2`` / ``m`` layout,
    same forward and forward_split -- with one substitution: the repeated internal block is
    ``FasterBlock``, whose partial convolution trims the redundant computation that the standard
    3x3 bottleneck spends on every channel (Chen et al., CVPR 2023). This is deliberately not a
    subclass of ``C2f``: the bottleneck type is the whole point of the change.

    The signature matches ``C2f`` exactly, so it is both base-module and repeat-module compatible
    with ultralytics ``parse_model``: a yaml row ``C2f_Faster, [128, True]`` is constructed as
    ``C2f_Faster(c1, c2, n, True)``.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of ``FasterBlock`` blocks.
        shortcut (bool): Whether to use shortcut connections inside the blocks.
        g (int): Groups for convolutions.
        e (float): Expansion ratio.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5) -> None:
        """Initialize the CSP split, the parallel 1x1 convolutions and the ``FasterBlock`` list."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(FasterBlock(self.c, self.c, shortcut, g) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the C2f layer.

        Args:
            x (torch.Tensor): Input tensor of shape ``(N, c1, H, W)``.

        Returns:
            torch.Tensor: Output tensor of shape ``(N, c2, H, W)``.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using ``split()`` instead of ``chunk()``.

        Args:
            x (torch.Tensor): Input tensor of shape ``(N, c1, H, W)``.

        Returns:
            torch.Tensor: Output tensor of shape ``(N, c2, H, W)``.
        """
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
