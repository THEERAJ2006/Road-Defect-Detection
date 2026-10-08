"""Interactive RSD-LSK demo: upload any road image -> detect -> Grad-CAM -> VLM.

INFERENCE-ONLY. No training code runs here; the frozen F_lsk checkpoint is
loaded once and reused across uploads. Never writes under runs/ablation/.

Pipeline (mirrors scripts/run_single_image_pipeline.py, but interactive):
    upload -> RSD-LSK detect -> per-box target-specific Grad-CAM (fused)
           -> defect crop (src.vlm.crop_box) -> Florence-2 caption

Usage:
    cd C:\\Crack\\PEP-26
    streamlit run scripts/demo_app.py
"""

from __future__ import annotations

import datetime
import io
import json
import os
import pathlib
import sys

# Offline flags match the rest of the project: never attempt a surprise
# download mid-demo; Florence-2 simply reports "unavailable" instead.
os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model import install

install()  # custom layers (SPDConv, C2f_Faster, LSKBlock, ASF neck) — required before YOLO()

from src.explain.gradcam import (  # noqa: E402
    box_target_cam,
    detect_head_layers,
    letterbox,
    overlay_heatmap,
    predict_boxes,
    unpad_cam,
)
from src.vlm import (  # noqa: E402
    DEFAULT_MODEL_ID,
    caption_crop,
    crop_box,
    load_florence,
    resolve_checkpoint,
    second_opinion,
)

IMGSZ = 640
ARM = "F_lsk"
DEMO_DIR = ROOT / "runs" / "demo"


def resolve_demo_checkpoint() -> pathlib.Path:
    """F_lsk checkpoint via the project's own resolver, with legacy fallback.

    Primary: runs/ablation/F_lsk/weights/best.pt (resolve_checkpoint).
    Fallback: weights/F_lsk_proposed.pt (used by the proposed-model scripts).
    """
    try:
        return resolve_checkpoint(ARM, ROOT / "runs" / "ablation")
    except FileNotFoundError:
        legacy = ROOT / "weights" / "F_lsk_proposed.pt"
        if legacy.exists():
            return legacy
        raise


def pick_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def draw_boxes(orig_bgr, preds, names):
    """Return BGR copies: (detection overlay, plain boxes)."""
    import cv2

    det = orig_bgr.copy()
    for (x1, y1, x2, y2, c, cf) in preds:
        cv2.rectangle(det, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            det,
            f"{names[c]} {cf:.2f}",
            (int(x1), max(0, int(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    return det


def fused_perbox_gradcam(dm, tensor, preds, ratio, pad, orig_hw):
    """Per-box target-specific Grad-CAM fused with pixel-wise max.

    Same logic as scripts/run_single_image_pipeline.py: each box is explained
    at every Detect-head input (src.explain.gradcam.detect_head_layers) with
    box_target_cam on its own predicted-class score; the max-energy head is
    kept per box; boxes fuse via np.maximum. Returns (fused_cam, reports).
    """
    import numpy as np
    import torch

    heads = detect_head_layers(dm)
    types = [type(m).__name__ for m in dm.model]
    dw, dh = pad
    fused = None
    reports = []
    for bi, (x1, y1, x2, y2, c, cf) in enumerate(preds):
        lb = (
            x1 * ratio + dw,
            y1 * ratio + dh,
            x2 * ratio + dw,
            y2 * ratio + dh,
        )
        cams, energies = {}, {}
        for li in heads:
            cam = box_target_cam(
                dm, tensor, li, lb, target_class=c,
                method="gradcam", iou_thr=0.5, topk=3,
            )
            co = unpad_cam(cam, ratio, pad, orig_hw)
            tot = float(co.sum())
            ys, xs = np.mgrid[0 : co.shape[0], 0 : co.shape[1]]
            inside_m = (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)
            energies[li] = float(co[inside_m].sum()) / tot if tot > 0 else 0.0
            cams[li] = cam
        li_best = max(heads, key=lambda li: round(energies[li], 6))
        fused = cams[li_best] if fused is None else np.maximum(fused, cams[li_best])
        reports.append(
            {
                "box": bi,
                "layer": int(li_best),
                "layer_type": types[li_best],
                "inside_activation": round(float(energies[li_best]), 4),
            }
        )
    return fused, reports


def run_pipeline(yolo, dm, adev, img_path, conf, topk, vlm, vlm_device):
    """Full inference pipeline on one saved image. Returns a result dict."""
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    names = [yolo.model.names[i] for i in range(len(yolo.model.names))]
    orig = cv2.imread(str(img_path))
    if orig is None:
        raise ValueError(f"cannot read image {img_path}")

    preds = sorted(predict_boxes(yolo, img_path, conf, IMGSZ),
                   key=lambda r: -r[5])[:topk]
    detections = [
        {
            "class": names[c],
            "class_id": int(c),
            "confidence": round(float(cf), 4),
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
        }
        for (x1, y1, x2, y2, c, cf) in preds
    ]
    detection_bgr = draw_boxes(orig, preds, names)

    cam_reports: list[dict] = []
    overlay_bgr = heat_bgr = None
    if preds:
        padded, ratio, pad = letterbox(orig, IMGSZ)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        ).to(adev)
        fused, cam_reports = fused_perbox_gradcam(
            dm, tensor, preds, ratio, pad, orig.shape[:2]
        )
        overlay_bgr, heat_bgr = overlay_heatmap(orig, fused, ratio, pad, alpha=0.5)

    # VLM on the top-confidence box crop. Real Florence-2 only: when weights
    # are unavailable vlm is None and NO caption is fabricated (see app body).
    vlm_record: dict | None = None
    crop_rgb = None
    if preds and vlm is not None:
        image = Image.open(img_path).convert("RGB")
        x1, y1, x2, y2, c, cf = preds[0]
        crop = crop_box(image, (x1, y1, x2, y2, c, cf))  # project default pad/context
        crop_rgb = np.array(crop)
        cap = caption_crop(vlm, crop, vlm_device)
        verdict = second_opinion(cf, cap["caption"], conf)
        vlm_record = {
            "class": names[c],
            "confidence": round(float(cf), 4),
            "caption": cap["caption"],
            "vlm_mode": cap["mode"],
            **{k: verdict[k] for k in ("keep", "reason", "abstained")},
        }
    elif preds:
        # Crop is still shown (it comes from the detector, not the VLM).
        image = Image.open(img_path).convert("RGB")
        x1, y1, x2, y2, c, cf = preds[0]
        crop_rgb = np.array(crop_box(image, (x1, y1, x2, y2, c, cf)))

    return {
        "names": names,
        "orig_bgr": orig,
        "detection_bgr": detection_bgr,
        "overlay_bgr": overlay_bgr,
        "heat_bgr": heat_bgr,
        "crop_rgb": crop_rgb,
        "detections": detections,
        "cam_reports": cam_reports,
        "vlm": vlm_record,
        "vlm_available": vlm is not None,
    }


def save_demo(result, upload_name, ckpt_path, out_root=DEMO_DIR):
    """Persist the latest result under runs/demo/<timestamp>/ (never ablation)."""
    import cv2

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_root / stamp
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "input.jpg"), result["orig_bgr"])
    cv2.imwrite(str(out / "detection.jpg"), result["detection_bgr"])
    if result["overlay_bgr"] is not None:
        cv2.imwrite(str(out / "gradcam_overlay.jpg"), result["overlay_bgr"])
        cv2.imwrite(str(out / "gradcam_heat.jpg"), result["heat_bgr"])
    if result["crop_rgb"] is not None:
        import cv2 as _cv

        cv2.imwrite(
            str(out / "crop.jpg"),
            _cv.cvtColor(result["crop_rgb"], _cv.COLOR_RGB2BGR),
        )
    meta = {
        "upload": upload_name,
        "checkpoint": str(ckpt_path),
        "classes": result["names"],
        "detections": result["detections"],
        "gradcam": result["cam_reports"],
        "vlm": result["vlm"],
        "vlm_available": result["vlm_available"],
    }
    (out / "result.json").write_text(json.dumps(meta, indent=1))
    return out


# --------------------------------------------------------------------------- UI
def main():
    import cv2
    import torch
    import streamlit as st
    from ultralytics import YOLO

    st.set_page_config(page_title="RSD-LSK Road Defect Detection", layout="wide")
    st.title("ROAD SURFACE DEFECT DETECTION")
    st.subheader("RSD-LSK + Explainable AI + Vision-Language")

    ckpt_path = resolve_demo_checkpoint()

    @st.cache_resource(show_spinner="Loading RSD-LSK detector (once per session)...")
    def load_detector():
        yolo = YOLO(str(ckpt_path))
        dm = yolo.model.eval()
        if torch.cuda.is_available():
            dm.to(torch.device("cuda:0"))
        # Clone inference-cached anchors/strides so backward() works for CAM
        # (same as scripts/proposed_gradcam_5classes.py).
        det_head = dm.model[-1]
        det_head.anchors = det_head.anchors.clone()
        det_head.strides = det_head.strides.clone()
        with torch.enable_grad():
            dev = next(dm.parameters()).device
            dm(torch.zeros(1, 3, IMGSZ, IMGSZ, device=dev))
        return yolo

    @st.cache_resource(show_spinner="Loading Florence-2 (once per session)...")
    def load_vlm_cached(device: str):
        return load_florence(DEFAULT_MODEL_ID, device)

    yolo = load_detector()
    names = [yolo.model.names[i] for i in range(len(yolo.model.names))]
    dm = yolo.model
    adev = next(dm.parameters()).device
    device_label = f"{adev.type}{'' if adev.index is None else f':{adev.index}'}"

    with st.sidebar:
        st.header("Settings")
        conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
        topk = st.slider("Max boxes to explain", 1, 10, 5, 1)
        save = st.checkbox("Save results to runs/demo/", value=True)
        dev_mock = st.checkbox(
            "Dev only: allow mock captions when Florence-2 is missing",
            value=False,
            help="Off by default. Mock text is NEVER presented as real VLM output.",
        )
        st.divider()
        st.caption(f"Checkpoint: `{ckpt_path.name}`")
        st.caption(f"Classes: {', '.join(names)}")
        st.caption(f"Detector device: {device_label}")
        st.caption(f"VLM: {DEFAULT_MODEL_ID} (inference-time only)")

    vlm_device = "cuda" if torch.cuda.is_available() else "cpu"
    vlm = load_vlm_cached(vlm_device)
    vlm_real = vlm is not None
    if vlm_real:
        st.success("Florence-2 loaded — VLM interpretation is REAL model output.")
    else:
        st.warning(
            "Florence-2 weights are not available locally. "
            "VLM interpretation is unavailable. "
            "Detection + Grad-CAM still work; no placeholder text will be "
            "shown as a VLM result."
        )

    uploaded = st.file_uploader("Upload Road Image", type=["jpg", "jpeg", "png"])
    run = st.button("Run Inference", type="primary", disabled=uploaded is None)

    if uploaded is not None:
        raw = uploaded.getvalue()
        st.image(raw, caption=f"INPUT IMAGE — {uploaded.name}", width="stretch")

    if uploaded is not None and run:
        tmp = ROOT / "runs" / "demo" / "_upload_tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        img_path = tmp / uploaded.name
        img_path.write_bytes(raw)

        with st.spinner("Running RSD-LSK detection + Grad-CAM + VLM..."):
            # load_florence() returns None when weights are missing; only a
            # dev-opt-in flag may substitute mock captions for pipeline testing.
            effective_vlm = vlm if (vlm_real or dev_mock) else None
            result = run_pipeline(
                yolo, dm, adev, img_path, conf, topk, effective_vlm, vlm_device
            )
        try:
            img_path.unlink(missing_ok=True)
        except OSError:
            pass

        st.header("DETECTION RESULT")
        det_rgb = cv2.cvtColor(result["detection_bgr"], cv2.COLOR_BGR2RGB)
        st.image(det_rgb, caption="Detection (RSD-LSK)", width="stretch")
        if result["detections"]:
            st.subheader("Detected Defects")
            st.table(
                {
                    "Class": [d["class"] for d in result["detections"]],
                    "Confidence": [d["confidence"] for d in result["detections"]],
                    "BBox (x1, y1, x2, y2)": [
                        ", ".join(f"{v:.0f}" for v in d["bbox"])
                        for d in result["detections"]
                    ],
                }
            )
            for d, rep in zip(result["detections"], result["cam_reports"]):
                st.caption(
                    f"{d['class']} conf={d['confidence']:.2f} — Grad-CAM layer "
                    f"{rep['layer']} ({rep['layer_type']}), "
                    f"inside-activation {rep['inside_activation']:.3f}"
                )
        else:
            st.info(f"No defects detected at confidence ≥ {conf:.2f}.")

        if result["overlay_bgr"] is not None:
            st.header("GRAD-CAM EXPLANATION")
            c1, c2 = st.columns(2)
            c1.image(
                cv2.cvtColor(result["overlay_bgr"], cv2.COLOR_BGR2RGB),
                caption="Grad-CAM overlay (per-box target-specific, fused)",
                width="stretch",
            )
            c2.image(
                cv2.cvtColor(result["heat_bgr"], cv2.COLOR_BGR2RGB),
                caption="Grad-CAM heat",
                width="stretch",
            )

        if result["crop_rgb"] is not None:
            st.header("DETECTED DEFECT REGION")
            st.image(
                result["crop_rgb"],
                caption=f"Crop of top detection: {result['detections'][0]['class']} "
                f"{result['detections'][0]['confidence']:.2f}",
                width=420,
            )

        st.header("VISION-LANGUAGE INTERPRETATION")
        st.caption("Detector classifies; Florence-2 only describes what it sees.")
        if result["vlm"] is not None and result["vlm"]["vlm_mode"] == "florence2":
            st.success("Source: REAL Florence-2 output")
            st.write(f"Florence-2: \"{result['vlm']['caption']}\"")
            st.caption(f"Second opinion: {result['vlm']['reason']}")
        elif result["vlm"] is not None:
            st.warning("Source: MOCK caption (dev testing only — not a model result)")
            st.write(f"Mock: \"{result['vlm']['caption']}\"")
        else:
            st.warning(
                "Florence-2 weights are not available locally. "
                "VLM interpretation is unavailable."
            )

        with st.expander("Inference Summary"):
            st.json(
                {
                    "model": f"RSD-LSK ({ckpt_path.name})",
                    "detector": "YOLO-based (frozen inference)",
                    "classes": names,
                    "confidence_threshold": conf,
                    "gradcam": "per-box target-specific Grad-CAM "
                    "(src.explain.gradcam.box_target_cam @ Detect-head inputs)",
                    "vlm": DEFAULT_MODEL_ID,
                    "vlm_usage": "inference-time only (not trained, not evaluated)",
                    "vlm_status": "real" if vlm_real else "unavailable",
                }
            )

        if save:
            out = save_demo(result, uploaded.name, ckpt_path)
            st.success(f"Saved to `{out}` (input/detection/gradcam/crop/result.json)")

    st.divider()
    st.caption(
        "Inference-only demo around the existing F_lsk checkpoint. "
        "Nothing here retrains, fine-tunes, or overwrites research results."
    )


if __name__ == "__main__":
    main()
