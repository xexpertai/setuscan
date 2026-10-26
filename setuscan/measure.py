"""Crack geometry: skeleton, length, and width in mm.

* Skeleton: Zhang-Suen thinning (cv2.ximgproc.thinning, opencv-contrib).
* Length: chain length of the 8-connected skeleton (orthogonal steps count 1,
  diagonal steps sqrt(2)).
* Width, two estimators:
  - "dt": 2 x distance transform of the binary mask at the skeleton (fast, coarse
    for cracks only a few pixels wide);
  - "profile": sub-pixel full-width-at-half-depth of the black-hat depth profile
    sampled perpendicular to the local crack direction (cv2.remap with bilinear
    interpolation). This is the reported width; the dt value is kept as a check.
Widths below about 2 px are unreliable at any resolution; the agent uses the
`reliable` flag to decide whether to zoom in or ask for a closer photo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

MIN_RELIABLE_WIDTH_PX = 3.0  # from eval/results/synthetic_mm_accuracy.json: <5 % error above ~2.5 px true width
PROFILE_HALF_LEN = 3.0  # profile length on each side, in multiples of the dt width


@dataclass
class WidthSample:
    x: float
    y: float
    width_px: float
    dt_width_px: float
    angle_deg: float
    raw_width_px: float = 0.0  # before blur correction


@dataclass
class CrackMeasure:
    id: int
    length_px: float
    width_max_px: float  # 95th percentile of profile widths (robust "max")
    width_median_px: float
    orientation_deg: float  # 0 = horizontal, 90 = vertical
    bbox: list[int]
    n_samples: int
    samples: list[WidthSample] = field(default_factory=list)
    n_below_blur: int = 0  # samples narrower than the blur limit (left uncorrected)
    raw_width_median_px: float = 0.0


def thin(mask: np.ndarray) -> np.ndarray:
    return cv2.ximgproc.thinning(mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)


def skeleton_length_chain(skel: np.ndarray) -> float:
    """Naive 8-connected chain length (orthogonal 1, diagonal sqrt 2). Over-reads
    oblique lines by up to ~8 % (staircase effect); kept for comparison."""
    s = (skel > 0).astype(np.uint8)
    orth = int((s[:, 1:] & s[:, :-1]).sum() + (s[1:, :] & s[:-1, :]).sum())
    d1 = s[1:, 1:] & s[:-1, :-1]
    d2 = s[1:, :-1] & s[:-1, 1:]
    c1 = d1 & ~(s[1:, :-1] | s[:-1, 1:]) & 1
    c2 = d2 & ~(s[1:, 1:] | s[:-1, :-1]) & 1
    return float(orth + np.sqrt(2) * (int(c1.sum()) + int(c2.sum())))


def skeleton_length(skel: np.ndarray, epsilon: float = 1.0) -> float:
    """Length of a 1-px skeleton without the staircase bias.

    The outer contour of a 1-px-wide skeleton walks along every branch and back, so
    half its perimeter is the skeleton length. Simplifying that contour with
    Douglas-Peucker (epsilon = 1 px) replaces pixel staircases by straight segments.
    """
    s = (skel > 0).astype(np.uint8)
    if not s.any():
        return 0.0
    contours, _ = cv2.findContours(s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    total = 0.0
    for c in contours:
        if len(c) < 2:
            continue
        approx = cv2.approxPolyDP(c, epsilon, True)
        total += cv2.arcLength(approx, True) / 2.0
    return float(total)


def _local_direction(pts: np.ndarray, i: int, radius: float = 6.0) -> float:
    p = pts[i]
    near = pts[np.abs(pts - p).max(axis=1) <= radius]
    if len(near) < 3:
        return 0.0
    c = near - near.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    dx, dy = vt[0]
    return float(np.degrees(np.arctan2(dy, dx)))


def deblur_width(ratio: float, sigma: float) -> float | None:
    """Invert the blur model for the area/peak width estimate.

    Model: a crack with a flat-bottomed (box) profile of width w, blurred by a
    Gaussian of sigma s (camera optics + resampling, measured on the printed marker in
    the same view). Its area is preserved and its peak drops by erf(w / (2 sqrt2 s)), so
        area / peak = w / erf(w / (2 sqrt2 s)).
    Solved for w by bisection. Returns None when the ratio is at or below the
    blur-only limit sqrt(2 pi) s, i.e. the crack is narrower than the blur can resolve.
    """
    if sigma <= 0:
        return ratio
    k = 2.0 * math.sqrt(2.0) * sigma
    if ratio <= math.sqrt(2.0 * math.pi) * sigma * 1.02:
        return None
    lo, hi = 0.0, ratio
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if mid - ratio * math.erf(mid / k) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


WIDTH_METHOD = "area"  # "area" (equivalent width) or "fwhm"


def profile_width(depth: np.ndarray, x: float, y: float, angle_deg: float, half_len: float,
                  method: str | None = None) -> float | None:
    """Crack width (px) from the black-hat depth profile perpendicular to the crack.

    The profile is sampled every 0.1 px with bilinear interpolation. A baseline (the
    median of the outer quarter on each side) is subtracted, then either
    * "fwhm": full width at half of the peak depth, or
    * "area": equivalent width = integral of the profile / peak depth, integrated
      between the half-depth crossings extended by half the FWHM on each side.
    For a crack whose interior reaches full depth, the area estimate is unbiased under
    pixel sampling and symmetric blur; both estimators over-read cracks thinner
    than about 2 px.
    """
    method = method or WIDTH_METHOD
    a = np.radians(angle_deg + 90.0)
    ta = np.radians(angle_deg)
    step = 0.1
    t = np.arange(-half_len, half_len + step, step, dtype=np.float32)
    # average 5 parallel profiles spaced 0.75 px along the crack: less noise, no widening
    offs = np.arange(-2, 3, dtype=np.float32) * 0.75
    mx = (x + offs[:, None] * np.cos(ta) + t[None, :] * np.cos(a)).astype(np.float32)
    my = (y + offs[:, None] * np.sin(ta) + t[None, :] * np.sin(a)).astype(np.float32)
    prof = cv2.remap(depth, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).mean(axis=0).astype(np.float64)
    q = max(3, len(prof) // 8)
    base = float(np.median(np.concatenate([prof[:q], prof[-q:]])))
    prof = prof - base
    c = len(t) // 2
    win = max(1, int(1.5 / step))
    k = c - win + int(np.argmax(prof[c - win : c + win + 1]))
    peak = float(prof[k])
    if peak <= 0:
        return None
    half = peak / 2.0
    left = k
    while left > 0 and prof[left] > half:
        left -= 1
    right = k
    while right < len(prof) - 1 and prof[right] > half:
        right += 1
    if left == 0 or right == len(prof) - 1:
        return None  # profile did not fall to half depth: neighbouring crack or texture

    def cross(i0: int, i1: int) -> float:
        v0, v1 = prof[i0], prof[i1]
        return i0 + (half - v0) / (v1 - v0) if v1 != v0 else float(i0)

    tl = cross(left, left + 1)
    tr = cross(right - 1, right)
    fwhm = (tr - tl) * step
    if method == "fwhm":
        return float(fwhm)
    ext = 0.5 * (tr - tl)
    i0, i1 = max(0, int(np.floor(tl - ext))), min(len(prof) - 1, int(np.ceil(tr + ext)))
    area = float(np.clip(prof[i0 : i1 + 1], 0, None).sum() * step)
    return area / peak


def measure(mask: np.ndarray, depth: np.ndarray, sample_every_px: float = 4.0,
            min_len_px: float = 20.0, blur_sigma_px: float | None = None) -> list[CrackMeasure]:
    """Measure each connected crack in `mask` (uint8 0/255).

    If `blur_sigma_px` (the view's blur, measured on the marker) is given, each width
    sample is blur-corrected with `deblur_width`; samples narrower than the blur
    limit are kept at their raw value and counted in `n_below_blur`.
    """
    if not mask.any():
        return []
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    skel = thin(mask)
    n, labels = cv2.connectedComponents(skel, connectivity=8)
    out: list[CrackMeasure] = []
    depth = depth.astype(np.float32)
    for i in range(1, n):
        ys, xs = np.nonzero(labels == i)
        comp = np.zeros_like(skel)
        comp[ys, xs] = 255
        length = skeleton_length(comp)
        if length < min_len_px:
            continue
        pts = np.stack([xs, ys], axis=1).astype(np.float64)
        stride = max(1, int(round(sample_every_px)))
        order = np.lexsort((ys, xs)) if (np.ptp(xs) >= np.ptp(ys)) else np.lexsort((xs, ys))
        samples: list[WidthSample] = []
        below = 0
        for j in order[::stride]:
            x, y = pts[j]
            dtw = 2.0 * float(dt[int(y), int(x)])
            ang = _local_direction(pts, j)
            raw = profile_width(depth, x, y, ang, max(4.0, PROFILE_HALF_LEN * dtw))
            if raw is None:
                continue
            pw = raw
            if blur_sigma_px:
                d = deblur_width(raw, blur_sigma_px)
                if d is None:
                    below += 1
                else:
                    pw = d
            samples.append(WidthSample(float(x), float(y), pw, dtw, ang, raw))
        if not samples:
            continue
        w = np.array([s.width_px for s in samples])
        c = pts - pts.mean(axis=0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        orient = float(np.degrees(np.arctan2(abs(vt[0][1]), abs(vt[0][0]))))
        x0, y0 = xs.min(), ys.min()
        out.append(CrackMeasure(
            id=len(out) + 1,
            length_px=length,
            width_max_px=float(np.percentile(w, 95)),
            width_median_px=float(np.median(w)),
            orientation_deg=orient,
            bbox=[int(x0), int(y0), int(xs.max() - x0 + 1), int(ys.max() - y0 + 1)],
            n_samples=len(samples),
            samples=samples,
            n_below_blur=below,
            raw_width_median_px=float(np.median([s.raw_width_px for s in samples])),
        ))
    out.sort(key=lambda m: -m.length_px)
    for k, m in enumerate(out, 1):
        m.id = k
    return out
