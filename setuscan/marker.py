"""Printed scale marker: ArUco detection (OpenCV objdetect) -> px/mm and rectification.

The printable sheet (docs/marker-sheet.pdf) carries ArUco markers from the
DICT_4X4_50 dictionary with known printed side lengths (MARKER_SIZES_MM). The user
sticks one flat on the surface next to the crack.

From the four detected corners we compute a homography (calib3d) from the image to
a fronto-parallel metric plane. Warping the photo with it removes perspective, so
every pixel in the rectified image has the same size in mm. This is only valid for
points on the same plane as the marker, which is true for cracks on a flat wall,
slab or pier face, and is stated as a limitation in the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

DICT_ID = cv2.aruco.DICT_4X4_50
# printed side length (black square, outer edge) per marker id, in mm
MARKER_SIZES_MM: dict[int, float] = {0: 40.0, 1: 40.0, 2: 20.0, 3: 20.0}
CARD_MARGIN = 0.45  # white card border around the marker, as a fraction of the side


def _detector() -> "cv2.aruco.ArucoDetector":
    dictionary = cv2.aruco.getPredefinedDictionary(DICT_ID)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    return cv2.aruco.ArucoDetector(dictionary, params)


@dataclass
class ScaleResult:
    found: bool
    marker_id: int | None = None
    marker_mm: float | None = None
    corners: list[list[float]] = field(default_factory=list)  # 4x2 image px (TL,TR,BR,BL)
    px_per_mm_native: float | None = None  # mean marker side length in px / marker_mm
    tilt_ratio: float | None = None  # longest/shortest side (1.0 = facing straight on)
    H_mm: list[list[float]] | None = None  # image px -> mm on the marker plane (marker TL = 0,0)
    view_mm: list[float] | None = None  # visible plane area [x0, y0, x1, y1] in mm (clipped)
    n_markers: int = 0
    edge_blur_px: float | None = None  # Gaussian-equivalent blur sigma measured on the marker edges
    edge_blur_mm: float | None = None
    note: str = ""

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        for k in ("px_per_mm_native", "tilt_ratio", "marker_mm", "edge_blur_px", "edge_blur_mm"):
            if d[k] is not None:
                d[k] = round(float(d[k]), 4)
        return d


def detect_markers(img: np.ndarray) -> list[tuple[int, np.ndarray]]:
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _detector().detectMarkers(gray)
    if ids is None:
        return []
    out = []
    for c, i in zip(corners, ids.ravel()):
        if int(i) in MARKER_SIZES_MM:
            out.append((int(i), c.reshape(4, 2).astype(np.float64)))
    return out


def side_lengths(quad: np.ndarray) -> np.ndarray:
    return np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)


def estimate_scale(img: np.ndarray) -> ScaleResult:
    """Detect the largest known marker and build the image -> mm-plane homography."""
    found = detect_markers(img)
    if not found:
        return ScaleResult(found=False, note="no SetuScan marker detected")
    mid, quad = max(found, key=lambda t: cv2.contourArea(t[1].astype(np.float32)))
    mm = MARKER_SIZES_MM[mid]
    sides = side_lengths(quad)
    native = float(sides.mean() / mm)
    tilt = float(sides.max() / max(sides.min(), 1e-6))
    dst = np.array([[0, 0], [mm, 0], [mm, mm], [0, mm]], np.float64)
    H, _ = cv2.findHomography(quad, dst, 0)
    h, w = img.shape[:2]
    frame = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(frame, H).reshape(-1, 2)
    # strong perspective can explode the far corners: keep a window around the marker
    lim = 0.75 * max(h, w) / native  # 1.5x the photo's extent at the marker's scale
    cx = cy = mm / 2
    x0, y0 = warped.min(axis=0)
    x1, y1 = warped.max(axis=0)
    view = [max(x0, cx - lim), max(y0, cy - lim), min(x1, cx + lim), min(y1, cy + lim)]
    blur_px = edge_blur_sigma(img, quad)
    return ScaleResult(
        edge_blur_px=blur_px,
        edge_blur_mm=None if blur_px is None else blur_px / native,
        found=True,
        marker_id=mid,
        marker_mm=mm,
        corners=quad.tolist(),
        px_per_mm_native=native,
        tilt_ratio=tilt,
        H_mm=H.tolist(),
        view_mm=[float(v) for v in view],
        n_markers=len(found),
    )


def edge_blur_sigma(img: np.ndarray, quad: np.ndarray, n_per_side: int = 9) -> float | None:
    """Blur estimate from the marker's outer black-to-white edges (slanted-edge idea).

    Intensity profiles are sampled across each of the four outer edges (black
    square inside, white card outside). For a Gaussian blur of sigma s the 10 %-90 %
    rise distance of an ideal step is 2.563 s. Returns the median sigma in px over
    the profiles, or None if the edges are too small or too low-contrast.
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32)
    side = float(side_lengths(quad).mean())
    if side < 24:
        return None
    reach = min(0.12 * side, 12.0)  # stay inside the first black cell and the white margin
    c = quad.mean(axis=0)
    sig = []
    t = np.arange(-reach, reach + 0.05, 0.1, dtype=np.float32)
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        d = (b - a) / np.linalg.norm(b - a)
        nrm = np.array([-d[1], d[0]])
        if np.dot(nrm, (a + b) / 2 - c) < 0:
            nrm = -nrm  # point outwards (black -> white)
        for f in np.linspace(0.2, 0.8, n_per_side):
            p = a + f * (b - a)
            xs = (p[0] + t * nrm[0]).astype(np.float32).reshape(1, -1)
            ys = (p[1] + t * nrm[1]).astype(np.float32).reshape(1, -1)
            prof = cv2.remap(g, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()
            lo, hi = float(np.percentile(prof[: len(prof) // 4], 50)), float(np.percentile(prof[-len(prof) // 4 :], 50))
            if hi - lo < 40:
                continue
            n = (prof - lo) / (hi - lo)
            i10 = int(np.argmax(n >= 0.1))
            i90 = int(np.argmax(n >= 0.9))
            if i90 <= i10:
                continue
            sig.append((i90 - i10) * 0.1 / 2.563)
    return float(np.median(sig)) if len(sig) >= 6 else None


def card_polygon_mm(scale: ScaleResult, extra_mm: float = 0.0) -> np.ndarray:
    """Printed card outline (marker plus white border), grown by extra_mm, in mm-plane coordinates."""
    mm = scale.marker_mm or 0.0
    m = CARD_MARGIN * mm + extra_mm
    return np.array([[-m, -m], [mm + m, -m], [mm + m, mm + m], [-m, mm + m]], np.float64)


def render_marker(marker_id: int, side_px: int) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(DICT_ID)
    return cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)
