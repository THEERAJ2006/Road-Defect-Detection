"""SPD-Conv: space-to-depth followed by a non-strided convolution.

Replaces a stride-2 ``Conv`` with a lossless space-to-depth rearrangement plus a stride-1
convolution of the same kernel size. At ``s=2`` the rearrangement carries exactly the same
information as the vendored ``space_to_depth`` in ``src/vendor/space_to_depth.py``; see
:class:`SPDConv` for the exact (permuted) channel correspondence.
"""

from __future__ import annotations

import torch
from torch import nn

from ultralytics.nn.modules import Conv


class SPDConv(nn.Module):
    """Space-to-depth plus a non-strided convolution, a drop-in replacement for a strided ``Conv``.

    Stride-2 convolutions destroy fine spatial detail: every output position sees a strided sample
    of its input window, so thin high-frequency structures -- cracks, pothole rims, lane
    markings, i.e. precisely the small road defects this model targets -- can be missed outright.
    SPD-Conv (Du et al., "No More Strided Convolutions or Pooling", 2022) instead rearranges the
    input into a lower-resolution, higher-channel tensor with no information loss and applies a
    stride-1 convolution to it, so each output position still integrates a contiguous receptive
    field over the original pixels.

    The rearrangement is ``torch.nn.functional.pixel_unshuffle``, which is used in preference to
    the vendored ``space_to_depth`` because it generalises to any ``s``. At ``s=2`` both emit the
    same four sub-sampled planes and differ only in channel order: ``pixel_unshuffle`` interleaves
    them as ``(c, i, j)`` while the vendored 4-way strided slice and channel concat concatenates by
    ``(i, j)``, so the vendored output equals ``pixel_unshuffle(x, 2)[:, [0, 2, 1, 3]]``. The
    ordering is immaterial because the following convolution can permute channels through its
    weights.

    The signature and channel bookkeeping match stock ``Conv``, so this is base-module compatible
    with ultralytics ``parse_model``: a yaml row ``SPDConv, [64, 3, 2]`` -- args identical to the
    stock ``Conv, [64, 3, 2]`` it replaces -- is constructed as ``SPDConv(c1, c2, 3, 2)`` and
    yields ``c2`` channels at 1/s resolution.

    Non-divisible inputs are rejected rather than padded: padding would invent data that is not
    in the image and shift this feature map's grid relative to the rest of the network, so a clear
    ``ValueError`` is the safer failure mode. Models here only ever see ``s=2`` on divisible
    inputs.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        k (int | tuple): Kernel size of the stride-1 convolution.
        s (int): Space-to-depth factor, and hence the block's total downsampling stride.
        p (int | tuple | str | None, optional): Padding; ``None`` (auto) by default. When
            ``None`` or ``"auto"``, ultralytics ``Conv`` derives padding from the kernel size.
        g (int): Groups for the convolution.
        d (int): Dilation for the convolution.
        act (bool | nn.Module): Activation used by the convolution.

    Attributes:
        s (int): Stored space-to-depth factor, and hence the block stride.
        conv (Conv): The stride-1 convolution applied to the rearranged tensor.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 2,
        p: int | tuple | str | None = None,
        g: int = 1,
        d: int = 1,
        act: bool | nn.Module = True,
    ) -> None:
        """Initialize the space-to-depth factor and the stride-1 convolution over the reshaped input."""
        super().__init__()
        self.s = s
        self.conv = Conv(c1 * s * s, c2, k, 1, p, g, d, act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply space-to-depth, then the stride-1 convolution.

        Args:
            x (torch.Tensor): Input tensor of shape ``(N, c1, H, W)``.

        Returns:
            torch.Tensor: Output tensor of shape ``(N, c2, H // s, W // s)``.

        Raises:
            ValueError: If ``s`` is less than 1, or if ``H`` or ``W`` is not divisible by ``s``.
        """
        s = self.s
        if s < 1:
            raise ValueError(f"SPDConv space-to-depth factor must be >= 1, got {s}.")
        h, w = x.shape[-2], x.shape[-1]
        if h % s or w % s:
            raise ValueError(
                f"SPDConv requires input spatial size divisible by s={s}, got (H={h}, W={w}). "
                "Space-to-depth cannot be applied without cropping or padding."
            )
        return self.conv(torch.nn.functional.pixel_unshuffle(x, s))
