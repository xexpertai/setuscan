"""Annotated overlays: crack mask, skeleton, width callouts, scale bar."""

from __future__ import annotations

import cv2
import numpy as np

from .measure import thin
from .pipeline import Analysis

# BGR, chosen to stay distinguishable for common colour-vision deficiencies
CRACK = (0, 140, 255)  # orange
SKEL = (255, 255, 0)  # cyan
NEW = (60, 20, 220)  # crimson
OLD = (230, 160, 0)  # blue
TEXT_BG = (20, 20, 20)


def _label(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.5, placed: list | None = None) -> bool:
    """Draw a label with a dark background, kept inside the image. With `placed`, a label that would overlap an
    earlier one is moved down (up to 3 times) or skipped, so text never overlaps text. Returns True if drawn."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    th = max(1, int(round(scale * 2)))
    (tw, tht), base = cv2.getTextSize(text, font, scale, th)
    x, y = int(org[0]), int(org[1])
    x = int(np.clip(x, 2, img.shape[1] - tw - 4))
    for _ in range(4):
        yy = int(np.clip(y, tht + 4, img.shape[0] - 4))
        rect = (x - 5, yy - tht - 6, x + tw + 5, yy + base + 2)
        hit = placed is not None and any(rect[0] < r[2] and r[0] < rect[2] and rect[1] < r[3] and r[1] < rect[3] for r in placed)
        if not hit:
            break
        y += tht + 12
    else:
        return False
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), TEXT_BG, -1)
    cv2.putText(img, text, (x, yy), font, scale, (255, 255, 255), th, cv2.LINE_AA)
    if placed is not None:
        placed.append(rect)
    return True


def _scale_bar(img: np.ndarray, px_per_mm: float) -> None:
    h, w = img.shape[:2]
    for mm in (100, 50, 20, 10, 5, 2, 1):
        if mm * px_per_mm <= 0.3 * w:
            break
    L = int(round(mm * px_per_mm))
    x0, y0 = 16, h - 18
    cv2.rectangle(img, (x0 - 6, y0 - 26), (x0 + L + 6, y0 + 8), TEXT_BG, -1)
    cv2.line(img, (x0, y0), (x0 + L, y0), (255, 255, 255), 3)
    for x in (x0, x0 + L):
        cv2.line(img, (x, y0 - 6), (x, y0 + 4), (255, 255, 255), 2)
    cv2.putText(img, f"{mm} mm", (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _focus_box(a: Analysis, pad: int = 90) -> tuple[int, int, int, int]:
    """Region worth showing: cracks plus the marker card, else the valid photo area."""
    h, w = a.mask.shape
    boxes = [c.bbox for c in a.cracks]
    if a.scale.found and a.view.unit == "mm":
        from .marker import card_polygon_mm
        poly = (card_polygon_mm(a.scale) - np.array(a.view.box[:2])) * a.view.scale
        x0, y0 = poly.min(axis=0)
        x1, y1 = poly.max(axis=0)
        boxes.append([int(x0), int(y0), int(x1 - x0), int(y1 - y0)])
    if not a.cracks:
        ys, xs = np.nonzero(a.valid)
        if xs.size:
            boxes = [[int(xs.min()), int(ys.min()), int(np.ptp(xs)), int(np.ptp(ys))]]
    if not boxes:
        return 0, 0, w, h
    b = np.array(boxes)
    x0 = int(b[:, 0].min()) - pad
    y0 = int(b[:, 1].min()) - pad
    x1 = int((b[:, 0] + b[:, 2]).max()) + pad
    y1 = int((b[:, 1] + b[:, 3]).max()) + pad
    # a crop may extend past the photo; draw_analysis pads that with dark margin so nothing is clipped at the edge
    return x0, y0, x1, y1


def draw_analysis(a: Analysis, n_callouts: int = 3, crop: bool = True) -> np.ndarray:
    img = a.work.copy()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if a.valid is not None:  # dim areas with no photo data (outside the frame) and the marker card
        img[a.valid == 0] = (img[a.valid == 0] * 0.35).astype(np.uint8)
    tint = img.copy()
    tint[a.mask > 0] = CRACK
    img = cv2.addWeighted(tint, 0.55, img, 0.45, 0)
    skel = thin(a.mask) if a.mask.any() else a.mask
    img[skel > 0] = SKEL
    k = a.mm_per_px
    fs = max(0.45, min(img.shape[:2]) / 1100)
    labels = []  # drawn after the crop so a label is never cut off by the crop edge
    for c in a.cracks:
        x, y, w, h = c.bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1, cv2.LINE_AA)
        if k:
            text = f"#{c.id}  L {c.length_px * k:.0f} mm  W med {c.width_median_px * k:.2f} / p95 {c.width_max_px * k:.2f} mm"
        else:
            text = f"#{c.id}  L {c.length_px:.0f} px  W med {c.width_median_px:.1f} / p95 {c.width_max_px:.1f} px (no scale)"
        labels.append((text, (x, y - 6), fs))
        widest = []  # the widest samples, kept well apart so their labels cannot touch
        for s in sorted(c.samples, key=lambda s: -s.width_px):
            if all(np.hypot(s.x - t.x, s.y - t.y) >= 70 for t in widest):
                widest.append(s)
            if len(widest) == n_callouts:
                break
        for s in widest:
            cv2.circle(img, (int(s.x), int(s.y)), 5, (255, 255, 255), 2, cv2.LINE_AA)
            val = f"sample {s.width_px * k:.2f} mm" if k else f"sample {s.width_px:.1f} px"
            labels.append((val, (s.x + 8, s.y + 4), fs * 0.85))
    x0, y0 = 0, 0
    if crop:
        x0, y0, x1, y1 = _focus_box(a)
        h, w = img.shape[:2]
        # only the left/top margin is padded (so the marker and its label are never clipped there)
        pl, pt = max(0, -x0), max(0, -y0)
        img = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)].copy()
        if pl or pt:
            img = cv2.copyMakeBorder(img, pt, 0, pl, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    placed: list = []
    for text, (lx, ly), sc in labels:
        _label(img, text, (lx - x0, ly - y0), sc, placed)
    if k:
        _scale_bar(img, 1.0 / k)
    return img


def draw_marker_on_photo(photo: np.ndarray, a: Analysis) -> np.ndarray:
    img = photo.copy()
    if a.scale.found:
        q = np.array(a.scale.corners, np.int32)
        cv2.polylines(img, [q], True, (0, 220, 0), max(2, img.shape[1] // 400), cv2.LINE_AA)
        _label(img, f"marker {a.scale.marker_mm:.0f} mm", (int(q[:, 0].min()), int(q[:, 1].min()) - 8),
               max(0.6, img.shape[1] / 1600))
    return img


def draw_comparison(a: Analysis, warped_b: np.ndarray, new: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(cv2.cvtColor(a.work, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    img = (img * 0.7).astype(np.uint8)
    img[a.mask > 0] = OLD
    img[(warped_b > 0) & (a.mask == 0)] = CRACK
    img[new > 0] = NEW
    fs = max(0.5, min(img.shape[:2]) / 1100)
    _label(img, "blue: visit A   orange: visit B   red: new since A", (12, 28), fs)
    if a.mm_per_px:
        _scale_bar(img, 1.0 / a.mm_per_px)
    return img
