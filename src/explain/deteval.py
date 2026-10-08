"""Shared detection-evaluation helpers: GT loading, matching, marked drawing.

Factored out of scripts/proposed_best.py so CAM-comparison scripts can reuse
the exact same TP/FP (same class, IoU >= threshold) / FN (unmatched GT)
semantics and colour scheme: green TP, red FP, magenta missed-GT.
"""

from __future__ import annotations

import pathlib

TP_COLOR = (0, 255, 0)
FP_COLOR = (0, 0, 255)
FN_COLOR = (255, 0, 255)

__all__ = (
    "TP_COLOR",
    "FP_COLOR",
    "FN_COLOR",
    "box_iou",
    "load_gt",
    "match",
    "match_detailed",
    "draw_marks",
    "draw_legend",
    "caption_tile",
    "grid",
)


def box_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter)


def load_gt(label_dir: str | pathlib.Path, stem: str, img_w: int, img_h: int):
    """GT boxes ``(x1, y1, x2, y2, cls)`` in pixels from a YOLO label file."""
    gts = []
    p = pathlib.Path(label_dir) / f"{stem}.txt"
    if p.exists():
        for line in p.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            c, xc, yc, w, h = int(parts[0]), *(float(v) for v in parts[1:5])
            gts.append(((xc - w / 2) * img_w, (yc - h / 2) * img_h,
                        (xc + w / 2) * img_w, (yc + h / 2) * img_h, c))
    return gts


def match(preds, gts, iou_thr: float):
    """Greedy match of conf-sorted preds to GT. Returns (tp, fp, fn) indices.

    NOTE: a second prediction on an already-matched GT (duplicate fragment)
    counts as FP here (COCO convention). Use :func:`match_detailed` to
    separate such duplicates / near-misses from true background FPs.
    """
    d = match_detailed(preds, gts, iou_thr=iou_thr)
    return d["tp"], d["fp"], d["fn"]


def match_detailed(preds, gts, iou_thr: float = 0.5, near_thr: float = 0.3):
    """Greedy match with FP breakdown.

    Returns dict with index lists:
      tp      - matched GT (same class, IoU >= iou_thr)
      fp      - all unmatched preds (same as match() would give)
      fn      - unmatched GT
      fp_dup  - subset of fp: same class, IoU >= iou_thr with an
                already-matched GT (duplicate fragment / sub-box of a TP)
      fp_near - subset of fp: best same-class IoU in [near_thr, iou_thr)
                (localisation near-miss, e.g. IoU 0.47)
      fp_cls  - subset of fp: IoU >= iou_thr with a GT of another class
                (right box, wrong label)
      fp_bg   - remaining subset: true background FPs
    Also ``fp_info``: {pi: (kind, best_iou)} for every fp.
    """
    used = [False] * len(gts)
    tp, fp = [], []
    for pi, (x1, y1, x2, y2, c, _cf) in enumerate(preds):
        best, bj = iou_thr, -1
        for gi, (gx1, gy1, gx2, gy2, gc) in enumerate(gts):
            if used[gi] or gc != c:
                continue
            v = box_iou((x1, y1, x2, y2), (gx1, gy1, gx2, gy2))
            if v > best:
                best, bj = v, gi
        if bj >= 0:
            used[bj] = True
            tp.append(pi)
        else:
            fp.append(pi)
    fn = [gi for gi, u in enumerate(used) if not u]

    fp_dup, fp_near, fp_cls, fp_bg, fp_info = [], [], [], [], {}
    for pi in fp:
        x1, y1, x2, y2, c, _cf = preds[pi]
        best_same, best_any, same_used = 0.0, 0.0, False
        for gi, (gx1, gy1, gx2, gy2, gc) in enumerate(gts):
            v = box_iou((x1, y1, x2, y2), (gx1, gy1, gx2, gy2))
            if gc == c:
                if v > best_same:
                    best_same = v
                if v >= iou_thr and used[gi]:
                    same_used = True
            if v > best_any:
                best_any = v
        if same_used or best_same >= iou_thr:
            # IoU good enough for TP but the GT was already taken by a
            # higher-confidence box -> duplicate fragment of a TP.
            fp_dup.append(pi)
            fp_info[pi] = ("dup", round(float(best_same), 3))
        elif best_same >= near_thr:
            fp_near.append(pi)
            fp_info[pi] = ("near", round(float(best_same), 3))
        elif best_any >= iou_thr:
            fp_cls.append(pi)
            fp_info[pi] = ("cls", round(float(best_any), 3))
        else:
            fp_bg.append(pi)
            fp_info[pi] = ("bg", round(float(best_same), 3))
    return {"tp": tp, "fp": fp, "fn": fn, "fp_dup": fp_dup,
            "fp_near": fp_near, "fp_cls": fp_cls, "fp_bg": fp_bg,
            "fp_info": fp_info}


def draw_marks(img_bgr, preds, gts, tp, fp, fn, names):
    import cv2

    for pi in tp:
        x1, y1, x2, y2, c, cf = preds[pi]
        cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), TP_COLOR, 2)
        cv2.putText(img_bgr, f"{names[c]} {cf:.2f} TP", (int(x1), max(0, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TP_COLOR, 2)
    for pi in fp:
        x1, y1, x2, y2, c, cf = preds[pi]
        cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), FP_COLOR, 2)
        cv2.putText(img_bgr, f"{names[c]} {cf:.2f} FP", (int(x1), max(0, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, FP_COLOR, 2)
    for gi in fn:
        x1, y1, x2, y2, c = gts[gi]
        cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), FN_COLOR, 3)
        cv2.putText(img_bgr, f"missed {names[c]} FN", (int(x1), max(0, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, FN_COLOR, 2)


def draw_legend(img_bgr):
    import cv2
    import numpy as np

    items = [((0, 255, 0), "TP"), ((0, 0, 255), "FP"), ((255, 0, 255), "FN=missed GT")]
    w = 40 + max(len(t) for _, t in items) * 11 + 24 * len(items)
    legend = np.zeros((30, w, 3), dtype=np.uint8)
    x = 6
    for color, text in items:
        cv2.rectangle(legend, (x, 8), (x + 14, 22), color, -1)
        cv2.putText(legend, text, (x + 18, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        x += 18 + len(text) * 11 + 16
    h, ww = img_bgr.shape[:2]
    legend = legend[:, : min(w, ww)]
    img_bgr[h - 30:h, 0:legend.shape[1]] = legend
    return img_bgr


def caption_tile(img_bgr, text: str):
    import cv2
    import numpy as np

    bar = np.zeros((34, img_bgr.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return np.vstack([bar, img_bgr])


def grid(tiles, per_row=5, tile_h=420):
    """Captioned tiles into rows of `per_row`, aspect preserved."""
    import cv2
    import numpy as np

    normed = []
    for t in tiles:
        w = round(t.shape[1] * tile_h / t.shape[0])
        normed.append(cv2.resize(t, (w, tile_h)))
    rows = []
    for i in range(0, len(normed), per_row):
        chunk = normed[i:i + per_row]
        wmax = max(t.shape[1] for t in chunk)
        padded = [cv2.copyMakeBorder(t, 0, 0, 0, wmax - t.shape[1],
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0)) for t in chunk]
        rows.append(np.hstack(padded))
    wmax = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, wmax - r.shape[1],
                               cv2.BORDER_CONSTANT, value=(0, 0, 0)) for r in rows]
    return np.vstack(rows)
