"""Project-side ``parse_model`` with the branches this paper's yaml needs.

Ultralytics resolves every module name in a ``*.yaml`` layer row through
``globals()[m]`` inside :func:`ultralytics.nn.tasks.parse_model` and tracks the
output channel count with a hardcoded ``if``/``elif`` chain whose fallback is
``c2 = ch[f]``. Both facts are wrong for the layers this project adds, so the
stock function is copied verbatim here and extended, rather than patching the
installed package. :func:`install` makes the installed ultralytics use the copy.
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch
from ultralytics.nn.modules import (
    A2C2f,
    ADown,
    AConv,
    AIFI,
    Bottleneck,
    BottleneckCSP,
    C1,
    C2,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C2PSA,
    C3,
    C3Ghost,
    C3k2,
    C3TR,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    ConvTranspose,
    Depth,
    Detect,
    DWConv,
    DWConvTranspose2d,
    ELAN1,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Index,
    OBB,
    OBB26,
    Pose,
    Pose26,
    PSA,
    RepC3,
    RepNCSPELAN4,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    Segment26,
    SemanticSegment,
    SPP,
    SPPELAN,
    SPPF,
    TorchVision,
    v10Detect,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    YOLOESegment26,
)
from ultralytics.nn.tasks import _SafeLoad
from ultralytics.utils import LOGGER, colorstr, emojis
from ultralytics.utils.ops import make_divisible

from ..modules.c2f_faster import C2f_Faster
from ..modules.lsk import LSKBlock
from ..modules.spd import SPDConv
from ..modules.ssff import ScaleSequenceFusion
from ..vendor.asf_yolo_blocks import Add, DownSample, Zoom_cat

__all__ = ("parse_model", "install")


def parse_model(d, ch, verbose=True):
    """Parse a YOLO model.yaml dictionary into a PyTorch model.

    Args:
        d (dict): Model dictionary.
        ch (int): Input channels.
        verbose (bool): Whether to print model details.

    Returns:
        model (torch.nn.Sequential): PyTorch model.
        save (list): Sorted list of layer indices whose outputs need to be saved.
    """
    import ast

    # Args
    legacy = True  # backward compatibility for v3/v5/v8/v9 models
    max_channels = float("inf")
    nc, act, scales, end2end = (d.get(x) for x in ("nc", "activation", "scales", "end2end"))
    reg_max = d.get("reg_max", 16)
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    scale = d.get("scale")
    if scales:
        if not scale:
            scale = next(iter(scales.keys()))
            LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    restricted = _SafeLoad.restricted()
    default_act = Conv.default_act  # restore before returning: Conv.default_act is process-wide state
    if act:
        # redefine default activation, i.e. Conv.default_act = torch.nn.SiLU(). Under restricted loading, resolve the
        # spec without eval() (see _SafeLoad.activation).
        Conv.default_act = _SafeLoad.activation(act) if restricted else eval(act)
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    base_modules = frozenset(
        {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            torch.nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
            A2C2f,
            SPDConv,
            DownSample,
            C2f_Faster,
            LSKBlock,
        }
    )
    repeat_modules = frozenset(  # modules with 'repeat' arguments
        {
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            C3x,
            RepC3,
            C2fPSA,
            C2fCIB,
            C2PSA,
            A2C2f,
            C2f_Faster,
        }
    )
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = (
            getattr(torch.nn, m[3:])
            if m.startswith("nn.")
            else getattr(__import__("torchvision").ops, m[16:])
            if m.startswith("torchvision.ops.")
            else globals()[m]
        )  # get module
        if restricted and not (isinstance(m, type) and issubclass(m, torch.nn.Module)):
            # Under restricted loading, only known model layers may be named here.
            raise TypeError(emojis(f"ERROR ❌️ module '{m}' is not a permitted model layer under restricted loading."))
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if m is not Classify:  # Classify() output must stay at nc; every other layer scales by width
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:  # set 1) embed channels and 2) num heads
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])
                hidden_channels = int(c2 * (args[6] if len(args) > 6 else 0.5))
                if hidden_channels % args[2]:
                    raise ValueError(
                        f"C2fAttn hidden channels {hidden_channels} (from c2={c2}) must be divisible by nh={args[2]}; "
                        "adjust width_multiple, nh, or C2fAttn expansion"
                    )
                args[1] = hidden_channels

            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)  # number of repeats
                n = 1
            if m is C3k2:  # for M/L/X sizes
                legacy = False
                if scale in {"m", "l", "x"}:
                    args[3:4] = [True]  # slice assignment also supplies c3k when the YAML omits it
            if m is A2C2f:
                legacy = False
                if scale in {"l", "x"}:  # for L/X sizes
                    args.extend((True, 1.2))
            if m is C2fCIB:
                legacy = False
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in frozenset(
            {
                Detect,
                WorldDetect,
                YOLOEDetect,
                Segment,
                Segment26,
                YOLOESegment,
                YOLOESegment26,
                Pose,
                Pose26,
                OBB,
                OBB26,
            }
        ):
            args.extend([reg_max, end2end, [ch[x] for x in f]])
            if m is Segment or m is YOLOESegment or m is Segment26 or m is YOLOESegment26:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, YOLOEDetect, Segment, Segment26, YOLOESegment, YOLOESegment26, Pose, Pose26, OBB, OBB26}:
                m.legacy = legacy
        elif m is Depth:
            args = [*args[:1], [ch[x] for x in f]]  # c_mid, ch tuple; drops the legacy mode arg old checkpoints store
        elif m is SemanticSegment:
            args.append([ch[x] for x in f])  # nc, ch tuple
        elif m is v10Detect:
            args.append([ch[x] for x in f])
        elif m is ImagePoolingAttn:
            args.insert(1, [ch[x] for x in f])  # channels as second arg
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m in frozenset({TorchVision, Index}):
            c2 = args[0]
            c1 = ch[f]
            args = [*args[1:]]
        elif m is Zoom_cat:  # project: ASF-YOLO neck, concatenates max/avg-pooled large + medium + upsampled small
            c2 = sum(ch[x] for x in f)  # args[0] is the unused `in_dim` and is passed through unchanged
        elif m is Add:  # project: element-wise sum of two same-resolution feature maps
            c2 = ch[f[0]]
        elif m is ScaleSequenceFusion:  # project: f is [P3, P4, P5] -> ScaleSequenceFusion(c4, c5, channel)
            c1, c5 = ch[f[1]], ch[f[2]]
            c2 = make_divisible(min(args[0], max_channels) * width, 8)
            args = [c1, c5, c2]
        else:
            c2 = ch[f]

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        if m is SPPF and len(args) <= 3:  # Legacy YAML rows predate the unactivated YOLO26 SPPF.
            for block in m_ if n > 1 else [m_]:
                block.cv1.act = Conv.default_act
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    Conv.default_act = default_act
    return torch.nn.Sequential(*layers), sorted(save)


def install() -> Any:
    """Make the installed ultralytics build models with this project's :func:`parse_model`.

    Two independent patches are applied, both by rebinding module attributes on
    ``ultralytics.nn.tasks`` (the installed package is never edited on disk):

    1. Every project layer class is injected into ``ultralytics.nn.tasks.__dict__``.
       That module's namespace is the ``globals()`` a yaml module name is looked
       up in, so this is what lets a row such as ``- [SPDConv, [64, 3, 2]]``
       resolve instead of raising ``KeyError`` while its channels are still being
       resolved by the stock function, and what keeps any other ultralytics code
       that string-resolves model-layer names working on these layers.
    2. ``ultralytics.nn.tasks.parse_model`` is rebound to the copy above, which
       adds the ``Zoom_cat``/``Add``/``ScaleSequenceFusion`` channel branches the
       stock if/elif chain cannot express. ``DetectionModel._from_yaml`` looks the
       function up in that module's globals, so the rebinding is what makes
       ``YOLO("...yaml")`` use it.

    Returns:
        previous (Callable): The ``parse_model`` that was bound to
            ``ultralytics.nn.tasks`` before the rebinding. Undo by assigning it
            back: ``ultralytics.nn.tasks.parse_model = install()``.
    """
    from ultralytics.nn import tasks

    previous = tasks.parse_model
    for cls in (SPDConv, C2f_Faster, LSKBlock, Add, DownSample, Zoom_cat, ScaleSequenceFusion):
        setattr(tasks, cls.__name__, cls)
    tasks.parse_model = parse_model
    return previous
