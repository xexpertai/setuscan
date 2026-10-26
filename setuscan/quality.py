"""Image-quality gate: decide whether a photo is good enough to measure.

Three cheap, explainable OpenCV checks:
* sharpness: variance of the Laplacian of a lightly smoothed image at a fixed working
  size (low value = blurry). This is only a coarse gate: a plain painted wall has
  little texture even when sharp. The precise check is the blur measured on the
  printed marker's edges (marker.edge_blur_sigma), which the agent applies next;
* exposure: share of very dark / very bright pixels and the mean grey level;
* glare: share of saturated, low-texture blobs (specular highlights on wet or
  glossy surfaces hide cracks).

Each failed check returns a human-readable reason that the agent turns into a
re-shoot request.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .marker import CARD_MARGIN, detect_markers

WORK_SIDE = 1024  # quality metrics are computed at this longest side


@dataclass
class QualityThresholds:
    # variance of the Laplacian of a sigma-1 smoothed image at WORK_SIDE. Texture-dependent, so it is only a
    # guard against near-featureless frames: real KrakN patches have 1st percentile 3.1, median 9.2.
    min_sharpness: float = 2.0
    min_mean: float = 45.0
    max_mean: float = 225.0
    max_dark_frac: float = 0.35  # pixels < 25
    max_bright_frac: float = 0.25  # pixels > 245
    max_glare_frac: float = 0.04  # saturated low-texture blobs


@dataclass
class QualityReport:
    sharpness: float
    mean: float
    dark_frac: float
    bright_frac: float
    glare_frac: float
    ok: bool
    issues: list[str] = field(default_factory=list)
    advice: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_max(img: np.ndarray, side: int) -> tuple[np.ndarray, float]:
    """Resize so the longest side is at most `side`. Returns (image, scale)."""
    h, w = img.shape[:2]
    s = min(1.0, side / max(h, w))
    if s >= 1.0:
        return img, 1.0
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA), s


def assess_quality(img: np.ndarray, th: QualityThresholds | None = None) -> QualityReport:
    th = th or QualityThresholds()
    gray, _ = resize_max(to_gray(img), WORK_SIDE)
    # smoothing first keeps sensor noise from looking like detail
    sharp = float(cv2.Laplacian(cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0), cv2.CV_32F, ksize=3).var())
    mean = float(gray.mean())
    dark = float((gray < 25).mean())
    bright = float((gray > 245).mean())

    # glare: saturated pixels whose neighbourhood has almost no texture
    sat = (gray >= 250).astype(np.uint8)
    local_std = cv2.GaussianBlur(gray.astype(np.float32) ** 2, (0, 0), 3) - cv2.GaussianBlur(
        gray.astype(np.float32), (0, 0), 3
    ) ** 2
    flat = (np.sqrt(np.maximum(local_std, 0)) < 4).astype(np.uint8)
    glare_mask = cv2.morphologyEx(sat & flat, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # the printed white marker card is flat and bright by design: never call it glare
    for _, quad in detect_markers(gray):
        c = quad.mean(axis=0)
        grown = c + (quad - c) * (1 + 2 * CARD_MARGIN + 0.3)
        cv2.fillConvexPoly(glare_mask, np.round(grown).astype(np.int32), 0)
    n, _, stats, _ = cv2.connectedComponentsWithStats(glare_mask, connectivity=8)
    big = sum(int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 30)
    glare = big / gray.size

    issues, advice = [], []
    if sharp < th.min_sharpness:
        issues.append("blurry")
        advice.append("Hold the phone steady, tap to focus on the crack, and retake.")
    if mean < th.min_mean or dark > th.max_dark_frac:
        issues.append("too_dark")
        advice.append("Add light (torch at an angle) or move to daylight and retake.")
    if mean > th.max_mean or bright > th.max_bright_frac:
        issues.append("overexposed")
        advice.append("Avoid direct sun on the surface; shade it or lower exposure.")
    if glare > th.max_glare_frac:
        issues.append("glare")
        advice.append("Change the angle so the light does not reflect straight into the camera.")
    return QualityReport(sharp, mean, dark, bright, glare, not issues, issues, advice)
