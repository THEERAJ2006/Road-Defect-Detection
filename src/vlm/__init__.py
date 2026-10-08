"""VLM second stage: language-grounded explanations over frozen YOLO boxes.

Design: the detector stays frozen (no retraining, ever). For each kept YOLO
box we crop the original image (with padding), ask Florence-2 for a short
caption, and record a keep/reject second opinion. When the Florence weights
are not cached (offline box, GPU busy training), the module degrades to
mock mode: deterministic placeholder captions so the pipeline, JSON schema
and eval harness keep working without a download.
"""

from __future__ import annotations

import pathlib

import torch
from PIL import Image

ARMS = [
    "baseline", "A_spd", "B_neck", "C_faster", "D_wiou", "E_all",
    "F_lsk", "G_lsk", "H_eiou", "I_eiou",
]

# Legacy layout (pre 2026-10-07 rename) kept as fallback so old
# runs/ablation-lsk checkpoints still resolve.
_LEGACY_RUNS = {
    "G_lsk": ("ablation-lsk", "F_lsk"),
    "I_eiou": ("ablation-lsk", "F_lsk_EIoU"),
}

DEFAULT_MODEL_ID = "microsoft/Florence-2-base"

# Short captions confabulate less than detailed ones (A/B 2026-10-07:
# 0/8 domain-correct hallucinations for <CAPTION> vs 3/8 for <DETAILED_CAPTION>).
DEFAULT_TASK = "<CAPTION>"

# Caption must mention at least one of these, else the VLM is deemed to be
# describing something else (person, tree, wall) and the verdict abstains.
DEFECT_LEXICON = (
    "road", "street", "asphalt", "concrete", "pavement", "crack",
    "pothole", "repair", "patch", "surface", "lane", "sidewalk",
    "parking", "floor",
)

__all__ = (
    "ARMS",
    "DEFAULT_MODEL_ID",
    "DEFAULT_TASK",
    "DEFECT_LEXICON",
    "resolve_checkpoint",
    "crop_box",
    "load_florence",
    "caption_crop",
    "second_opinion",
)


def resolve_checkpoint(arm: str, runs_dir: str | pathlib.Path) -> pathlib.Path:
    """Return ``<runs_dir>/<arm>/weights/best.pt``, with legacy fallback.

    Raises:
        FileNotFoundError: when neither the canonical nor the legacy path
            holds a ``best.pt`` (i.e. train the arm first).
    """
    runs_dir = pathlib.Path(runs_dir)
    ckpt = runs_dir / arm / "weights" / "best.pt"
    if ckpt.exists():
        return ckpt
    if arm in _LEGACY_RUNS:
        legacy_proj, legacy_arm = _LEGACY_RUNS[arm]
        legacy = runs_dir.parent / legacy_proj / legacy_arm / "weights" / "best.pt"
        if legacy.exists():
            return legacy
    raise FileNotFoundError(f"missing {ckpt} — train {arm} first")


def crop_box(
    image: Image.Image, box: tuple, pad_ratio: float = 1.0, min_side: int = 224
) -> Image.Image:
    """Square context crop around ``(x1, y1, x2, y2, cls_id, conf)``.

    Tight thin cracks (e.g. 63x13px) starve Florence-2 of scene context and
    it confabulates (people, trees, walls — A/B 2026-10-07). A square crop
    of at least ``min_side`` centered on the box keeps road context in frame;
    the box stays findable inside it. Clipped to the image.
    """
    w, h = image.size
    x1, y1, x2, y2, _, _ = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(min_side, max(x2 - x1, y2 - y1) * (1 + 2 * pad_ratio))
    side = min(side, min(w, h))
    x1 = min(max(0, cx - side / 2), w - side)
    y1 = min(max(0, cy - side / 2), h - side)
    return image.crop((int(x1), int(y1), int(x1 + side), int(y1 + side)))


def load_florence(model_id: str = DEFAULT_MODEL_ID, device: str = "cpu"):
    """Load Florence-2, or return ``None`` for mock mode when unavailable.

    Never raises for missing weights offline: returns ``None`` so callers
    emit deterministic placeholder captions instead of crashing a batch job.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype="auto"
        ).to(device).eval()
        return processor, model
    except Exception:
        return None


@torch.no_grad()
def caption_crop(
    processor_model, crop: Image.Image, device: str = "cpu",
    task: str = DEFAULT_TASK, max_tokens: int = 32,
) -> dict:
    """Caption one crop. ``processor_model=None`` → deterministic mock."""
    if processor_model is None:
        w, h = crop.size
        return {
            "caption": f"mock defect crop {w}x{h} (VLM weights not cached)",
            "mode": "mock",
        }
    processor, model = processor_model
    prompt = task
    inputs = processor(text=prompt, images=crop, return_tensors="pt")
    dtype = next(model.parameters()).dtype
    input_ids = inputs["input_ids"].to(device)
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=max_tokens,
            num_beams=1,
            do_sample=False,
        )
    text = processor.batch_decode(out, skip_special_tokens=True)[0]
    return {"caption": text.strip(), "mode": "florence2"}


def second_opinion(
    yolo_conf: float, caption: str, keep_threshold: float = 0.25
) -> dict:
    """Combine YOLO confidence with the VLM caption into a keep/reject call.

    Anti-hallucination gate: when the caption contains no road/defect
    vocabulary the VLM is describing something else, so it ABSTAINS —
    ``keep`` defers to YOLO instead of laundering a confabulation into a
    verdict. Returns ``{"keep", "reason", "abstained"}``.
    """
    low = caption.lower()
    if not any(t in low for t in DEFECT_LEXICON):
        return {
            "keep": yolo_conf >= keep_threshold,
            "reason": f"VLM abstained (no defect lexicon); defer to yolo {yolo_conf:.2f}",
            "abstained": True,
        }
    if yolo_conf >= keep_threshold + 0.25:
        return {"keep": True, "reason": f"yolo {yolo_conf:.2f} confident",
                "abstained": False}
    filler = ("mock defect", "empty road", "no defect")
    if any(f in low for f in filler):
        return {"keep": False,
                "reason": f"yolo {yolo_conf:.2f} marginal + VLM filler",
                "abstained": False}
    return {"keep": True,
            "reason": f"yolo {yolo_conf:.2f} marginal, VLM kept",
            "abstained": False}
