"""Classical crack segmentation with OpenCV 5 (no learned weights).

Cracks are thin, elongated structures that are darker than the surrounding surface.
The detector combines two complementary cues:

1. Depth map: morphological black-hat (closing minus image) with an elliptical
   kernel wider than the widest crack we expect. It gives "how much darker than the
   local surface" each pixel is, which is also what the width measurement uses.
2. Ridge map: multi-scale Hessian (second derivatives of a Gaussian-smoothed image).
   For a dark line the largest eigenvalue is strongly positive while the other is
   near zero, which suppresses round pores and stains.

The product of the two is thresholded with hysteresis relative to a robust noise
level of the image, so the same parameters work on bright and dull surfaces.
Connected components are then kept only if they are long and thin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class SegParams:
    max_width_px: int = 21  # black-hat kernel must be wider than any crack
    sigmas: tuple[float, ...] = (1.0, 2.0, 3.5)
    k_high: float = 5.0  # seed threshold, in robust-noise units (tuned on KrakN dev split)
    k_low: float = 3.0  # grow threshold
    min_contrast: float = 6.0  # grey levels: weaker seeds are ignored
    min_len_px: float = 40.0  # minimum crack extent (max Feret diameter)
    min_elongation: float = 10.0  # extent^2 / area; a disc is about 1.3 (tuned on KrakN dev split)
    min_fill_contrast: float = 0.5  # mask = depth >= this * local peak depth (FWHM-like)
    # a crack is darker than BOTH sides, so the intensity gradient on its centre line is
    # small; on the dark side of a step edge (object outline, shadow) it is large.
    max_asymmetry: float | None = 0.13  # median(|grad| * sigma / depth) on the centre line (tuned on dev)


@dataclass
class SegResult:
    mask: np.ndarray  # uint8 0/255, final crack pixels
    depth: np.ndarray  # float32 black-hat depth (grey levels), smoothed, used for detection
    depth_raw: np.ndarray  # float32 black-hat depth on the raw image, used for width
    score: np.ndarray  # float32 combined response (noise units)
    components: list[dict] = field(default_factory=list)
    rejected: int = 0
    noise: float = 0.0


def _ellipse(k: int) -> np.ndarray:
    k = max(3, k | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def black_hat_depth(gray: np.ndarray, max_width_px: int, smooth: bool = True) -> np.ndarray:
    """Local darkness below the surface envelope (morphological closing).

    smooth=True (detection) subtracts a lightly smoothed image; smooth=False
    (measurement) subtracts the raw image so thin cracks are not widened.
    """
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    closed = cv2.morphologyEx(g, cv2.MORPH_CLOSE, _ellipse(max_width_px + 4))
    return (closed.astype(np.float32) - (g if smooth else gray).astype(np.float32)).clip(0, None)


def ridge_strength(gray: np.ndarray, sigmas: tuple[float, ...]) -> np.ndarray:
    """Scale-normalised dark-line ridge measure (max over scales)."""
    f = gray.astype(np.float32)
    best = np.zeros_like(f)
    for s in sigmas:
        g = cv2.GaussianBlur(f, (0, 0), s)
        dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
        dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
        dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
        tmp = np.sqrt((dxx - dyy) ** 2 + 4 * dxy**2)
        l1 = 0.5 * (dxx + dyy + tmp)  # larger eigenvalue (dark line -> large positive)
        l2 = 0.5 * (dxx + dyy - tmp)
        r = (l1 - np.abs(l2)).clip(0, None) * (s**2)
        np.maximum(best, r, out=best)
    return best


def _robust_noise(x: np.ndarray, valid: np.ndarray) -> float:
    v = x[valid > 0]
    if v.size == 0:
        return 1.0
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    return max(mad, 0.5) + 0.0 * med


def _component_shape(cnt_mask: np.ndarray) -> tuple[float, float]:
    pts = cv2.findNonZero(cnt_mask)
    hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.float32)
    if len(hull) < 2:
        return 1.0, 1.0
    d = np.sqrt(((hull[:, None, :] - hull[None, :, :]) ** 2).sum(-1)).max()  # max Feret
    return float(d), float(len(pts))


def segment(gray: np.ndarray, valid: np.ndarray | None = None, p: SegParams | None = None) -> SegResult:
    p = p or SegParams()
    if valid is None:
        valid = np.full(gray.shape, 255, np.uint8)
    depth = black_hat_depth(gray, p.max_width_px)
    ridge = ridge_strength(gray, p.sigmas)
    depth[valid == 0] = 0
    ridge[valid == 0] = 0

    nd = _robust_noise(depth, valid)
    nr = _robust_noise(ridge, valid)
    # geometric mean of the two normalised cues: both must agree
    score = np.sqrt((depth / nd) * (ridge / nr)).astype(np.float32)
    # hysteresis: grow from seeds that are both strong AND visibly dark
    weak = (score >= p.k_low).astype(np.uint8)
    strong = (score >= p.k_high) & (depth >= p.min_contrast)
    n0, lab0 = cv2.connectedComponents(weak, connectivity=8)
    keep = np.zeros(n0, np.uint8)
    keep[np.unique(lab0[strong])] = 255
    keep[0] = 0
    cand = keep[lab0]

    grad = None
    if p.max_asymmetry is not None:
        gs = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.5)
        grad = cv2.magnitude(cv2.Sobel(gs, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gs, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    mask = np.zeros_like(cand)
    comps, rejected = [], 0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if max(w, h) < p.min_len_px * 0.5:
            rejected += 1
            continue
        sub = (labels[y : y + h, x : x + w] == i).astype(np.uint8) * 255
        extent, a = _component_shape(sub)
        elong = extent**2 / max(a, 1.0)
        if extent < p.min_len_px or elong < p.min_elongation:
            rejected += 1
            continue
        asym = None
        if grad is not None:
            sk = cv2.ximgproc.thinning(sub) > 0
            d_sk = depth[y : y + h, x : x + w][sk]
            if d_sk.size:
                asym = float(np.median(grad[y : y + h, x : x + w][sk] * 1.5 / np.maximum(d_sk, 1.0)))
                if asym > p.max_asymmetry:
                    rejected += 1
                    continue
        # refine: keep pixels at least half as dark as the component's typical core
        dsub = depth[y : y + h, x : x + w]
        core = float(np.percentile(dsub[sub > 0], 90))
        refined = sub & ((dsub >= p.min_fill_contrast * core).astype(np.uint8) * 255)
        mask[y : y + h, x : x + w] |= refined
        comps.append({"bbox": [int(x), int(y), int(w), int(h)], "extent_px": round(extent, 1),
                      "elongation": round(elong, 1), "core_contrast": round(core, 1),
                      "asymmetry": None if asym is None else round(asym, 3)})
    depth_raw = black_hat_depth(gray, p.max_width_px, smooth=False)
    return SegResult(mask=mask, depth=depth, depth_raw=depth_raw, score=score, components=comps,
                     rejected=rejected, noise=float(nd))
