"""Explainability helpers: Grad-CAM for the trained detection variants."""

from .gradcam import (
    backbone_pyramid,
    box_target_cam,
    cam_heatmap,
    gradcam_heatmap,
    letterbox,
    multiscale_cam,
    overlay_heatmap,
    predict_boxes,
    resolve_target_layers,
    threshold_cam,
    unpad_cam,
)

__all__ = (
    "letterbox",
    "resolve_target_layers",
    "gradcam_heatmap",
    "cam_heatmap",
    "box_target_cam",
    "threshold_cam",
    "multiscale_cam",
    "backbone_pyramid",
    "unpad_cam",
    "predict_boxes",
    "overlay_heatmap",
)
