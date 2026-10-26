"""Two-visit change detection: align visit B to visit A and measure crack growth.

1. Both photos are analysed; when both carry the scale marker they are rendered on
   their metric wall planes at the SAME px/mm, so the remaining difference between
   them is an in-plane rotation + translation (the card may be stuck in a slightly
   different place each visit).
2. SIFT keypoints (features2d) on the surface texture, outside the printed card,
   are matched with Lowe's ratio test; a similarity transform is fitted with
   RANSAC (calib3d estimateAffinePartial2D). Without markers a full homography is
   fitted with USAC MAGSAC instead and results stay in pixels.
3. Visit B's crack mask is warped into visit A's frame. Pixels of B that are not
   near any crack pixel of A are "new" crack; their skeleton length is the
   extension. Width change compares the robust max widths of the two visits.

Registration quality (inliers, residual) is reported so the agent can refuse to
compare when the alignment is not trustworthy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .measure import skeleton_length, thin
from .pipeline import Analysis, View, analyze_view
from .quality import to_gray

MIN_INLIERS = 25
MAX_RMS_PX = 3.0
SIGNIFICANT_WIDTH_MM = 0.10
SIGNIFICANT_LENGTH_MM = 5.0


@dataclass
class Registration:
    ok: bool
    method: str
    n_keypoints: tuple[int, int]
    n_matches: int
    n_inliers: int
    rms_px: float
    scale: float
    rotation_deg: float
    M: list[list[float]]  # 3x3, B working px -> A working px
    reason: str = ""


def register(work_a: np.ndarray, work_b: np.ndarray, valid_a: np.ndarray, valid_b: np.ndarray,
             similarity: bool) -> Registration:
    sift = cv2.SIFT_create(nfeatures=6000)
    ga, gb = to_gray(work_a), to_gray(work_b)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ka, da = sift.detectAndCompute(clahe.apply(ga), valid_a)
    kb, db = sift.detectAndCompute(clahe.apply(gb), valid_b)
    nk = (len(ka), len(kb))
    fail = Registration(False, "sift", nk, 0, 0, float("inf"), 0.0, 0.0, np.eye(3).tolist())
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        fail.reason = "too few keypoints"
        return fail
    matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 64})
    pairs = matcher.knnMatch(db, da, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    fail.n_matches = len(good)
    if len(good) < MIN_INLIERS:
        fail.reason = f"only {len(good)} feature matches"
        return fail
    pb = np.float32([kb[m.queryIdx].pt for m in good])
    pa = np.float32([ka[m.trainIdx].pt for m in good])
    if similarity:
        A, inl = cv2.estimateAffinePartial2D(pb, pa, method=cv2.RANSAC, ransacReprojThreshold=3.0,
                                             maxIters=5000, confidence=0.999)
        M = np.vstack([A, [0, 0, 1]]) if A is not None else None
        method = "sift+ransac-similarity"
    else:
        M, inl = cv2.findHomography(pb, pa, cv2.USAC_MAGSAC, 3.0)
        method = "sift+magsac-homography"
    if M is None or inl is None:
        fail.reason = "no consistent transform"
        return fail
    inl = inl.ravel().astype(bool)
    proj = cv2.perspectiveTransform(pb[inl].reshape(-1, 1, 2), M).reshape(-1, 2)
    rms = float(np.sqrt(((proj - pa[inl]) ** 2).sum(axis=1).mean())) if inl.any() else float("inf")
    scale = float(np.sqrt(abs(np.linalg.det(M[:2, :2]))))
    rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    ok = int(inl.sum()) >= MIN_INLIERS and rms <= MAX_RMS_PX and (not similarity or 0.9 < scale < 1.1)
    reason = "" if ok else (f"inliers={int(inl.sum())}, rms={rms:.2f}px, scale={scale:.3f}")
    return Registration(ok, method, nk, len(good), int(inl.sum()), rms, scale, rot, M.tolist(), reason)


@dataclass
class Comparison:
    ok: bool
    unit: str
    registration: Registration
    length_a: float
    length_b: float
    width_max_a: float
    width_max_b: float
    extension: float  # skeleton length of B's crack not present in A
    length_change: float
    width_change: float
    significant_growth: bool
    note: str = ""
    # median width of the same (widest) crack that width_max_* (p95) refers to; same unit as width_max_*
    width_median_a: float = 0.0
    width_median_b: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 3)
        r = d["registration"]
        r["rms_px"] = round(r["rms_px"], 3) if np.isfinite(r["rms_px"]) else None
        r["scale"], r["rotation_deg"] = round(r["scale"], 4), round(r["rotation_deg"], 2)
        r["M"] = [[round(x, 5) for x in row] for row in r["M"]]
        return d


def _totals(a: Analysis) -> tuple[float, float]:
    k = a.mm_per_px or 1.0
    length = sum(c.length_px for c in a.cracks) * k
    width = max((c.width_max_px for c in a.cracks), default=0.0) * k
    return length, width


def _median_of_widest(a: Analysis) -> float:
    """Median profile width of the crack whose p95 width is the largest (the crack width_max refers to)."""
    if not a.cracks:
        return 0.0
    c = max(a.cracks, key=lambda c: c.width_max_px)
    return c.width_median_px * (a.mm_per_px or 1.0)


def compare_visits(img_a: np.ndarray, a: Analysis, img_b: np.ndarray, b: Analysis) -> tuple[Comparison, np.ndarray, np.ndarray]:
    """Returns (comparison, B mask warped into A's frame, new-crack mask in A's frame)."""
    both_mm = a.view.unit == "mm" and b.view.unit == "mm"
    if both_mm and abs(b.view.scale - a.view.scale) > 1e-6:
        vb = View(b.view.H_base, "mm", b.view.box, a.view.scale, b.view.native_scale)
        b = analyze_view(img_b, vb, b.scale, b.quality)
    reg = register(a.work, b.work, a.valid, b.valid, similarity=both_mm)
    unit = "mm" if both_mm else "px"
    la, wa = _totals(a)
    lb, wb = _totals(b)
    if not both_mm:
        # without a metric plane, B lengths are only comparable after scaling by the registration
        lb *= reg.scale if reg.ok else 1.0
        wb *= reg.scale if reg.ok else 1.0
    ma, mb = _median_of_widest(a), _median_of_widest(b) * (1.0 if both_mm or not reg.ok else reg.scale)
    empty = np.zeros_like(a.mask)
    if not reg.ok:
        return Comparison(False, unit, reg, la, lb, wa, wb, 0.0, lb - la, wb - wa, False,
                          note="visits could not be aligned: " + reg.reason, width_median_a=ma, width_median_b=mb), empty, empty
    M = np.array(reg.M)
    h, w = a.mask.shape
    warped_b = cv2.warpPerspective(b.mask, M, (w, h), flags=cv2.INTER_NEAREST)
    tol = max(3, int(round(1.5 / (a.mm_per_px or 0.5)))) if both_mm else 5
    near_a = cv2.dilate(a.mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1)))
    new = cv2.bitwise_and(warped_b, cv2.bitwise_not(near_a))
    new = cv2.morphologyEx(new, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ext = skeleton_length(thin(new)) * (a.mm_per_px or 1.0) if new.any() else 0.0
    sig_w = SIGNIFICANT_WIDTH_MM if both_mm else 1.5
    sig_l = SIGNIFICANT_LENGTH_MM if both_mm else 20.0
    significant = (wb - wa) > sig_w or ext > sig_l
    return Comparison(True, unit, reg, la, lb, wa, wb, ext, lb - la, wb - wa, significant, width_median_a=ma, width_median_b=mb), warped_b, new
