"""Project-side adaptation of ASF-YOLO's scale-sequence fusion (SSFF).

``ScalSeq`` upstream hardcodes its input projections to ``Conv(512, …)`` and
``Conv(1024, …)`` — YOLOv5 full-width P4/P5. YOLOv8s is width 0.5, so its P4/P5
are 256/512 and the vendored class raises a shape error. ``ScaleSequenceFusion``
rebuilds the two projections for the actual widths and is otherwise identical:
P3, P4 and P5 are projected to a common width, P4/P5 are resized to P3's
resolution, the three are stacked along a scale axis and fused with a 3D
convolution, batch norm, LeakyReLU and a 3D max-pool over the scale axis.
"""

from __future__ import annotations

from ultralytics.nn.modules import Conv
from torch import nn

from ..vendor.asf_yolo_blocks import ScalSeq


class ScaleSequenceFusion(ScalSeq):
    """SSFF sized for a given model scale.

    Args:
        c4: channels of the P4/16 feature map (input of the first projection).
        c5: channels of the P5/32 feature map (input of the second projection).
        channel: common output width of the fusion. The P3/8 map is consumed
            as-is — it sets the target resolution, and its width must match.
    """

    def __init__(self, c4: int, c5: int, channel: int) -> None:
        super().__init__(channel)
        self.conv1 = Conv(c4, channel, 1)
        self.conv2 = Conv(c5, channel, 1)
        # Paper §2.2.2: SSFF ends in "three-dimensional convolution, batch
        # normalization, and SiLU". The vendored ScalSeq uses LeakyReLU(0.1);
        # override to match the paper. Vendor file stays verbatim.
        self.act = nn.SiLU()
