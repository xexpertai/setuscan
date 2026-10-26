"""End-to-end single-photo analysis: quality -> scale -> view -> segment -> measure.

A `View` is a rectangular window on a "base plane" rendered at a chosen scale:
* with a marker, the base plane is the wall plane in mm (perspective removed);
* without a marker, the base plane is the original photo in pixels.
The default view covers the whole photo at a bounded working size (fast). A zoom
view covers one crack's bounding box at the photo's native resolution, which is
how the agent "looks closer" without asking the user for a new photo.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import marker as mk
from .measure import MIN_RELIABLE_WIDTH_PX, CrackMeasure, measure
from .quality import QualityReport, assess_quality, to_gray
from .segment import SegParams, segment

WORK_SIDE = 1400  # longest side of the default working view
ZOOM_SIDE = 2000  # longest side allowed for a zoom view
MAX_CRACK_MM = 8.0  # black-hat kernel covers cracks up to this width
MIN_CRACK_LEN_MM = 12.0
CARD_GUARD_MM = 6.0


@dataclass
class View:
    H_base: np.ndarray  # image px -> base plane
    unit: str  # "mm" or "px"
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in base units
    scale: float  # working px per base unit
    native_scale: float  # max useful scale (photo resolution) in working px per base unit

    @property
    def M(self) -> np.ndarray:
        x0, y0 = self.box[0], self.box[1]
        S = np.array([[self.scale, 0, -self.scale * x0], [0, self.scale, -self.scale * y0], [0, 0, 1]])
        return S @ self.H_base

    @property
    def size(self) -> tuple[int, int]:
        x0, y0, x1, y1 = self.box
        return max(1, int(np.ceil((x1 - x0) * self.scale))), max(1, int(np.ceil((y1 - y0) * self.scale)))

    @property
    def mm_per_px(self) -> float | None:
        return 1.0 / self.scale if self.unit == "mm" else None

    def to_base(self, x: float, y: float) -> tuple[float, float]:
        return self.box[0] + x / self.scale, self.box[1] + y / self.scale

    def describe(self) -> dict:
        return {"unit": self.unit, "box": [round(v, 2) for v in self.box], "scale_px_per_unit": round(self.scale, 4),
                "native_scale": round(self.native_scale, 4), "size": list(self.size)}


def base_view(img: np.ndarray, scale: mk.ScaleResult, work_side: int = WORK_SIDE) -> View:
    h, w = img.shape[:2]
    if scale.found:
        box = tuple(scale.view_mm)
        native = float(scale.px_per_mm_native)
        span = max(box[2] - box[0], box[3] - box[1])
        return View(np.array(scale.H_mm), "mm", box, min(native, work_side / span), native)
    box = (0.0, 0.0, float(w), float(h))
    return View(np.eye(3), "px", box, min(1.0, work_side / max(w, h)), 1.0)


def zoom_view(v: View, bbox_px: list[int], pad_frac: float = 0.15, zoom_side: int = ZOOM_SIDE) -> View:
    """View of one crack's bbox (given in v's working px) at up to native resolution."""
    x, y, bw, bh = bbox_px
    pad = pad_frac * max(bw, bh) + 10
    bx0, by0 = v.to_base(x - pad, y - pad)
    bx1, by1 = v.to_base(x + bw + pad, y + bh + pad)
    bx0, by0 = max(bx0, v.box[0]), max(by0, v.box[1])
    bx1, by1 = min(bx1, v.box[2]), min(by1, v.box[3])
    span = max(bx1 - bx0, by1 - by0)
    return View(v.H_base, v.unit, (bx0, by0, bx1, by1), min(v.native_scale, zoom_side / span), v.native_scale)


def render(img: np.ndarray, v: View) -> tuple[np.ndarray, np.ndarray]:
    """Warp the photo into the view; also return the valid-pixel mask (255 = real data)."""
    M = v.M
    out = cv2.warpPerspective(img, M, v.size, flags=cv2.INTER_AREA if v.scale < 1 else cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    valid = cv2.warpPerspective(np.full(img.shape[:2], 255, np.uint8), M, v.size, flags=cv2.INTER_NEAREST)
    valid = cv2.erode(valid, np.ones((7, 7), np.uint8))
    return out, valid


def view_blur_sigma(img: np.ndarray, v: View, scale: mk.ScaleResult) -> float | None:
    """Blur sigma (in this view's px) measured on the marker rendered at this view's scale.

    It includes the camera blur and every resampling step of the view, so it is the
    right sigma for blur-correcting widths measured in this view.
    """
    if v.unit != "mm" or not scale.found:
        return None
    mm = scale.marker_mm
    pad = 0.3 * mm
    mv = View(v.H_base, "mm", (-pad, -pad, mm + pad, mm + pad), v.scale, v.native_scale)
    patch, _ = render(img, mv)
    quad = (np.array([[0, 0], [mm, 0], [mm, mm], [0, mm]], np.float64) + pad) * v.scale
    return mk.edge_blur_sigma(patch, quad)


def seg_params_for(v: View) -> SegParams:
    if v.unit == "mm":
        return SegParams(max_width_px=int(np.clip(MAX_CRACK_MM * v.scale, 11, 61)),
                         min_len_px=max(25.0, MIN_CRACK_LEN_MM * v.scale))
    return SegParams(max_width_px=int(np.clip(21 * v.scale, 11, 41)), min_len_px=max(25.0, 40 * v.scale))


@dataclass
class Analysis:
    quality: QualityReport
    scale: mk.ScaleResult
    view: View
    work: np.ndarray  # BGR working image
    mask: np.ndarray
    valid: np.ndarray  # 255 where the working image holds real photo data (card excluded)
    cracks: list[CrackMeasure]
    rejected_components: int
    timings_ms: dict = field(default_factory=dict)
    blur_sigma_px: float | None = None  # view blur measured on the marker (None without marker)

    @property
    def mm_per_px(self) -> float | None:
        return self.view.mm_per_px

    def crack_summary(self, c: CrackMeasure) -> dict:
        k = self.mm_per_px
        d = {
            "id": c.id,
            "length_px": round(c.length_px, 1),
            "width_max_px": round(c.width_max_px, 2),
            "width_median_px": round(c.width_median_px, 2),
            "orientation_deg": round(c.orientation_deg, 1),
            "bbox_px": c.bbox,
            "n_width_samples": c.n_samples,
            "width_reliable": c.width_max_px >= MIN_RELIABLE_WIDTH_PX and c.n_below_blur <= 0.2 * c.n_samples,
            "raw_width_median_px": round(c.raw_width_median_px, 2),
            "samples_below_blur_limit": c.n_below_blur,
        }
        if k:
            d.update({"length_mm": round(c.length_px * k, 1), "width_max_mm": round(c.width_max_px * k, 3),
                      "width_median_mm": round(c.width_median_px * k, 3)})
        return d

    def to_dict(self) -> dict:
        return {
            "quality": self.quality.to_dict(),
            "scale": self.scale.to_dict(),
            "view": self.view.describe(),
            "mm_per_px": round(self.mm_per_px, 5) if self.mm_per_px else None,
            "view_blur_sigma_px": None if self.blur_sigma_px is None else round(self.blur_sigma_px, 3),
            "n_cracks": len(self.cracks),
            "cracks": [self.crack_summary(c) for c in self.cracks],
            "rejected_components": self.rejected_components,
            "timings_ms": self.timings_ms,
        }


def analyze_view(img: np.ndarray, v: View, scale: mk.ScaleResult, quality: QualityReport) -> Analysis:
    t0 = time.perf_counter()
    work, valid = render(img, v)
    if scale.found and v.unit == "mm":
        # the paper edge and its shadow read as a dark step: exclude a band around the card
        card = mk.card_polygon_mm(scale, extra_mm=CARD_GUARD_MM)
        poly = (card - np.array(v.box[:2])) * v.scale
        cv2.fillConvexPoly(valid, np.round(poly).astype(np.int32), 0)
    t1 = time.perf_counter()
    seg = segment(to_gray(work), valid, seg_params_for(v))
    sigma = view_blur_sigma(img, v, scale)
    t2 = time.perf_counter()
    cracks = measure(seg.mask, seg.depth_raw, sample_every_px=max(2.0, 0.5 * v.scale if v.unit == "mm" else 3.0),
                     min_len_px=seg_params_for(v).min_len_px * 0.6, blur_sigma_px=sigma)
    t3 = time.perf_counter()
    timings = {"render": round((t1 - t0) * 1000, 1), "segment": round((t2 - t1) * 1000, 1),
               "measure": round((t3 - t2) * 1000, 1)}
    return Analysis(quality, scale, v, work, seg.mask, valid, cracks, seg.rejected, timings, sigma)


def analyze(img: np.ndarray, work_side: int = WORK_SIDE) -> Analysis:
    t0 = time.perf_counter()
    q = assess_quality(img)
    t1 = time.perf_counter()
    s = mk.estimate_scale(img)
    t2 = time.perf_counter()
    a = analyze_view(img, base_view(img, s, work_side), s, q)
    a.timings_ms = {"quality": round((t1 - t0) * 1000, 1), "marker": round((t2 - t1) * 1000, 1), **a.timings_ms}
    return a


def decode_image(data: bytes, max_side: int = 6000) -> np.ndarray:
    """Decode an uploaded photo; EXIF orientation applied, metadata (incl. GPS) dropped."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # applies EXIF orientation, keeps pixels only
    if img is None:
        raise ValueError("not a decodable image")
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return img
