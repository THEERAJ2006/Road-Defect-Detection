# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# Verbatim excerpt of `space_to_depth` from Cateners/yolov8-spd (AGPL-3.0):
#   https://github.com/Cateners/yolov8-spd  ->  ultralytics/nn/modules/block.py:271-283
#   commit 83ebc51-era tree, default branch `main`, last pushed 2023-06-06
#
# Unmodified, including its dead `dimension` argument and the commented-out
# alternative implementation. Do not edit: it is vendored so the diff against
# upstream stays empty. Project-side adaptations belong in src/modules/.
#
# Upstream wiring, for reference (ultralytics 8.0-era layout, NOT this repo):
#   - ultralytics/nn/modules/__init__.py:17,29  export `space_to_depth`
#   - ultralytics/nn/tasks.py:651-652           `elif m is space_to_depth: c2 = 4*ch[f]`
#   - ultralytics/models/v8/yolov8-spd.yaml     stride-1 Conv + space_to_depth[1] per stage

import torch
from torch import nn


class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
         return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
#         size_tensor = x.size()
#         return torch.cat([x[...,0:size_tensor[2]//2,0:size_tensor[3]//2],
#                          x[...,0:size_tensor[2]//2,size_tensor[3]//2:],
#                          x[...,size_tensor[2]//2:,0:size_tensor[3]//2],
#                          x[...,size_tensor[2]//2:,size_tensor[3]//2:]  ],1)
