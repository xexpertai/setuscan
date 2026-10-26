"""SYNTHETIC test scenes with exact ground truth (clearly labelled as synthetic).

A flat concrete-like wall patch is rendered on a metric plane (WORLD_PPM px per mm):
procedural texture with pores and stains, one or more cracks drawn from an exact
point-to-segment distance field (so crack width is exact to sub-pixel level), and a
printed ArUco card. The plane is then "photographed": downscaled to the camera
scale, warped by a perspective homography, blurred, lit unevenly, noised and JPEG
encoded.

Ground truth returned with each scene: crack width (max and per-point) in mm, crack
centreline length in mm, and the image-space crack mask.

These images are NOT real photos. They measure measurement error under controlled
conditions only; real-world accuracy must be validated with real photos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .marker import CARD_MARGIN, MARKER_SIZES_MM, render_marker

WORLD_PPM = 12.0  # render resolution of the wall plane, px per mm


@dataclass
class CrackSpec:
    start_mm: tuple[float, float]
    heading_deg: float
    length_mm: float
    width_mm: float  # maximum width
    taper: float = 0.0  # 0 = constant width; 0.6 = ends are 40% of max width
    wiggle: float = 0.09  # heading noise (radians per sqrt(mm))
    step_mm: float = 0.5
    darkness: float = 0.62  # fraction of local brightness removed inside the crack


@dataclass
class SceneSpec:
    seed: int = 0
    wall_mm: tuple[float, float] = (300.0, 220.0)
    cracks: list[CrackSpec] = field(default_factory=list)
    marker_id: int | None = 0
    marker_at_mm: tuple[float, float] = (45.0, 45.0)  # top-left of the black square
    cam_ppm: float = 6.0  # image px per mm at the plane centre
    tilt: float = 0.08  # perspective strength (0 = fronto-parallel)
    rotate_deg: float = 4.0
    blur_sigma: float = 0.7  # optical blur in image px
    motion_blur_px: int = 0
    gain: float = 1.0  # exposure multiplier
    light_gradient: float = 0.25
    noise_sigma: float = 2.5
    glare: bool = False
    jpeg_quality: int = 92
    pores: int = 260
    stains: int = 3


def _smooth_noise(rng: np.random.Generator, h: int, w: int, sigma: float) -> np.ndarray:
    """Unit-variance Gaussian-filtered noise; large sigmas are made at low resolution."""
    f = max(1, int(sigma / 3))
    sh, sw = max(4, h // f), max(4, w // f)
    n = cv2.GaussianBlur(rng.standard_normal((sh, sw)).astype(np.float32), (0, 0), sigma / f)
    if f > 1:
        n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    return n / (n.std() + 1e-6)


def _texture(rng: np.random.Generator, h: int, w: int, n_pores: int, n_stains: int) -> np.ndarray:
    base = rng.uniform(150, 185)
    tex = np.full((h, w), base, np.float32)
    for sigma, amp in ((60, 10), (15, 5), (3, 4), (0.8, 5)):
        tex += amp * _smooth_noise(rng, h, w, sigma)
    pores = np.zeros((h, w), np.float32)
    for _ in range(n_pores):  # small dark voids and light aggregate grains
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(0.15, 0.9) * WORLD_PPM
        col = float(rng.choice([-45.0, -30.0, 25.0]))
        cv2.ellipse(pores, (int(cx), int(cy)), (int(r), int(r * rng.uniform(0.5, 1.0))), rng.uniform(0, 180),
                    0, 360, col, -1, cv2.LINE_AA)
    tex += cv2.GaussianBlur(pores, (0, 0), 1.0)
    f = 8  # broad water stains: dark but not thin (drawn at 1/8 resolution, then upsampled)
    stains = np.zeros((h // f, w // f), np.float32)
    for _ in range(n_stains):
        cv2.ellipse(stains, (int(rng.uniform(0, w / f)), int(rng.uniform(0, h / f))),
                    (int(rng.uniform(8, 25) * WORLD_PPM / f), int(rng.uniform(5, 15) * WORLD_PPM / f)),
                    rng.uniform(0, 180), 0, 360, -float(rng.uniform(10, 25)), -1)
    stains = cv2.GaussianBlur(stains, (0, 0), 4 * WORLD_PPM / f)
    tex += cv2.resize(stains, (w, h), interpolation=cv2.INTER_LINEAR)
    return tex


def crack_polyline(c: CrackSpec, rng: np.random.Generator) -> np.ndarray:
    n = max(2, int(c.length_mm / c.step_mm) + 1)
    a = np.radians(c.heading_deg)
    pts = [np.array(c.start_mm, np.float64)]
    for _ in range(n - 1):
        a += rng.normal(0, c.wiggle * np.sqrt(c.step_mm))
        pts.append(pts[-1] + c.step_mm * np.array([np.cos(a), np.sin(a)]))
    return np.array(pts)


def polyline_length(pts: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def _width_at(c: CrackSpec, t: np.ndarray) -> np.ndarray:
    """Width (mm) at normalised arc position t in [0,1]."""
    return c.width_mm * (1.0 - c.taper * np.abs(2 * t - 1))


def crack_coverage(pts_mm: np.ndarray, c: CrackSpec, h: int, w: int) -> np.ndarray:
    """Fraction of each world pixel covered by the crack (exact distance, 1 px AA)."""
    pix = 1.0 / WORLD_PPM
    seg_len = np.linalg.norm(np.diff(pts_mm, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg_len)])
    total = cum[-1]
    cov = np.zeros((h, w), np.float32)
    band = c.width_mm / 2 + 2 * pix
    for i in range(len(pts_mm) - 1):
        a, b = pts_mm[i], pts_mm[i + 1]
        x0 = int(max(0, (min(a[0], b[0]) - band) * WORLD_PPM))
        x1 = int(min(w, (max(a[0], b[0]) + band) * WORLD_PPM + 2))
        y0 = int(max(0, (min(a[1], b[1]) - band) * WORLD_PPM))
        y1 = int(min(h, (max(a[1], b[1]) + band) * WORLD_PPM + 2))
        if x1 <= x0 or y1 <= y0:
            continue
        xs = (np.arange(x0, x1) + 0.5) / WORLD_PPM
        ys = (np.arange(y0, y1) + 0.5) / WORLD_PPM
        X, Y = np.meshgrid(xs, ys)
        ab = b - a
        L2 = float(ab @ ab) or 1e-12
        u = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / L2, 0, 1)
        d = np.hypot(X - (a[0] + u * ab[0]), Y - (a[1] + u * ab[1]))
        half = _width_at(c, (cum[i] + u * seg_len[i]) / max(total, 1e-9)) / 2
        local = np.clip((half - d) / pix + 0.5, 0, 1).astype(np.float32)
        np.maximum(cov[y0:y1, x0:x1], local, out=cov[y0:y1, x0:x1])
    return cov


def _camera_homography(spec: SceneSpec, rng: np.random.Generator, wpx: int, hpx: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Homography from the cam-scale plane image to the photo, plus photo size."""
    s = spec.cam_ppm / WORLD_PPM
    pw, ph = wpx * s, hpx * s
    src = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], np.float32)
    c = src.mean(axis=0)
    rot = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), spec.rotate_deg, 1.0)
    dst = cv2.transform(src.reshape(-1, 1, 2), rot).reshape(-1, 2)
    jitter = rng.uniform(-1, 1, (4, 2)) * spec.tilt * min(pw, ph) * 0.5
    jitter[:, 0] += np.array([1, -1, -1, 1]) * spec.tilt * pw * 0.15  # keystone
    dst = dst + jitter
    # the camera frame is the largest axis-aligned box inside the photographed wall
    x0 = max(dst[0, 0], dst[3, 0]) + 2
    x1 = min(dst[1, 0], dst[2, 0]) - 2
    y0 = max(dst[0, 1], dst[1, 1]) + 2
    y1 = min(dst[2, 1], dst[3, 1]) - 2
    dst -= np.array([x0, y0])
    size = (int(x1 - x0), int(y1 - y0))
    return cv2.getPerspectiveTransform(src, dst.astype(np.float32)), size


def render_scene(spec: SceneSpec, rng_cracks: np.random.Generator | None = None) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(spec.seed)
    rng_c = rng_cracks or np.random.default_rng(spec.seed + 1000)
    w_mm, h_mm = spec.wall_mm
    wpx, hpx = int(w_mm * WORLD_PPM), int(h_mm * WORLD_PPM)
    tex = _texture(rng, hpx, wpx, spec.pores, spec.stains)

    gt_cracks, cov_all = [], np.zeros((hpx, wpx), np.float32)
    for c in spec.cracks:
        pts = crack_polyline(c, rng_c)
        cov = crack_coverage(pts, c, hpx, wpx)
        tex = tex * (1 - c.darkness * cov)
        np.maximum(cov_all, cov, out=cov_all)
        gt_cracks.append({"_pts": pts, "length_mm": round(polyline_length(pts), 2), "width_max_mm": c.width_mm,
                          "taper": c.taper, "centreline_mm": pts[:: max(1, len(pts) // 60)].round(2).tolist()})

    img = np.clip(tex, 0, 255).astype(np.uint8)
    if spec.marker_id is not None:
        mm = MARKER_SIZES_MM[spec.marker_id]
        side = int(round(mm * WORLD_PPM))
        m = render_marker(spec.marker_id, side)
        margin = int(round(CARD_MARGIN * mm * WORLD_PPM))
        card = cv2.copyMakeBorder(m, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=245)
        x = int(round(spec.marker_at_mm[0] * WORLD_PPM)) - margin
        y = int(round(spec.marker_at_mm[1] * WORLD_PPM)) - margin
        img[y : y + card.shape[0], x : x + card.shape[1]] = card
        cov_all[y : y + card.shape[0], x : x + card.shape[1]] = 0

    # photograph it
    s = spec.cam_ppm / WORLD_PPM
    small = cv2.resize(img, (round(wpx * s), round(hpx * s)), interpolation=cv2.INTER_AREA)
    cov_small = cv2.resize(cov_all, (round(wpx * s), round(hpx * s)), interpolation=cv2.INTER_AREA)
    H, size = _camera_homography(spec, rng, wpx, hpx)
    for g in gt_cracks:  # length of the centre line that is actually inside the photo
        pts = np.array(g.pop("_pts"))
        img_pts = cv2.perspectiveTransform((pts * WORLD_PPM * s).reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
        inside = (img_pts[:, 0] >= 2) & (img_pts[:, 0] <= size[0] - 3) & (img_pts[:, 1] >= 2) & (img_pts[:, 1] <= size[1] - 3)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        g["visible_length_mm"] = round(float(seg[inside[:-1] & inside[1:]].sum()), 2)
    photo = cv2.warpPerspective(small, H, size, flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    gt_mask = (cv2.warpPerspective(cov_small, H, size, flags=cv2.INTER_LINEAR) >= 0.35).astype(np.uint8) * 255

    yy, xx = np.mgrid[0 : size[1], 0 : size[0]].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    ramp = (xx * np.cos(ang) + yy * np.sin(ang)) / max(size)
    photo *= spec.gain * (1 - spec.light_gradient / 2 + spec.light_gradient * (ramp - ramp.min()) / (np.ptp(ramp) + 1e-6))
    if spec.glare:
        cx, cy = rng.uniform(0.3, 0.7) * size[0], rng.uniform(0.3, 0.7) * size[1]
        r = 0.18 * min(size)
        g = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r**2)))
        photo = photo + 400 * g
    if spec.blur_sigma > 0:
        photo = cv2.GaussianBlur(photo, (0, 0), spec.blur_sigma)
    if spec.motion_blur_px > 1:
        k = np.zeros((spec.motion_blur_px, spec.motion_blur_px), np.float32)
        k[spec.motion_blur_px // 2, :] = 1.0 / spec.motion_blur_px
        photo = cv2.filter2D(photo, -1, k)
    photo += rng.normal(0, spec.noise_sigma, photo.shape).astype(np.float32)
    gray = np.clip(photo, 0, 255).astype(np.uint8)
    bgr = cv2.merge([np.clip(gray.astype(np.int16) + d, 0, 255).astype(np.uint8) for d in (-4, 0, 6)])
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, spec.jpeg_quality])
    bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    marker_in_frame = None
    if spec.marker_id is not None:
        mm = MARKER_SIZES_MM[spec.marker_id]
        mx, my = spec.marker_at_mm
        corners = np.array([[mx, my], [mx + mm, my], [mx + mm, my + mm], [mx, my + mm]]) * WORLD_PPM * s
        ic = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).reshape(-1, 2)
        marker_in_frame = bool(((ic >= 0) & (ic < np.array(size))).all())
    gt = {"synthetic": True, "seed": spec.seed, "cam_ppm": spec.cam_ppm, "cracks": gt_cracks,
          "marker_in_frame": marker_in_frame,
          "marker_id": spec.marker_id, "image_size": list(size)}
    return bgr, {**gt, "mask": gt_mask}


def default_crack(rng: np.random.Generator, wall_mm=(300.0, 220.0), width_mm: float = 0.8,
                  length_mm: float = 150.0, taper: float = 0.0) -> CrackSpec:
    """A crack starting in the right part of the wall, away from the marker corner."""
    return CrackSpec(start_mm=(wall_mm[0] * rng.uniform(0.42, 0.55), wall_mm[1] * rng.uniform(0.18, 0.32)),
                     heading_deg=float(rng.uniform(25, 65)), length_mm=length_mm, width_mm=width_mm, taper=taper)


def two_visits(seed: int, width1: float, width2: float, len1: float, len2: float,
               cam_ppm: float = 6.0) -> tuple[tuple[np.ndarray, dict], tuple[np.ndarray, dict]]:
    """Same wall photographed twice; the crack has grown between visits (SYNTHETIC)."""
    rng = np.random.default_rng(seed)
    base = default_crack(rng, width_mm=width1, length_mm=len1)
    v1 = SceneSpec(seed=seed, cracks=[base], cam_ppm=cam_ppm, rotate_deg=3.0)
    grown = CrackSpec(base.start_mm, base.heading_deg, len2, width2, base.taper, base.wiggle, base.step_mm,
                      base.darkness)
    v2 = SceneSpec(seed=seed, cracks=[grown], cam_ppm=cam_ppm * 1.1, rotate_deg=-6.0, tilt=0.12,
                   marker_at_mm=(52.0, 40.0))
    # identical texture (same seed) and identical crack path prefix (same crack rng seed)
    a = render_scene(v1, np.random.default_rng(seed + 1000))
    b = render_scene(v2, np.random.default_rng(seed + 1000))
    return a, b
