import cv2
import numpy as np
import pytest

from setuscan import marker as mk
from setuscan.measure import deblur_width
from setuscan.synth import SceneSpec, render_scene


def card_on_wall(px_per_mm=5.0, marker_id=0, blur=0.0):
    mm = mk.MARKER_SIZES_MM[marker_id]
    side = int(round(mm * px_per_mm))
    m = mk.render_marker(marker_id, side)
    pad = int(side * 0.6)
    card = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=245)
    canvas = np.full((card.shape[0] + 200, card.shape[1] + 300), 150, np.uint8)
    canvas[100:100 + card.shape[0], 150:150 + card.shape[1]] = card
    if blur:
        canvas = cv2.GaussianBlur(canvas, (0, 0), blur)
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def test_scale_fronto_parallel():
    s = mk.estimate_scale(card_on_wall(5.0))
    assert s.found and s.marker_id == 0
    assert s.px_per_mm_native == pytest.approx(5.0, rel=0.02)
    assert s.tilt_ratio == pytest.approx(1.0, abs=0.03)


def test_scale_small_marker_id():
    s = mk.estimate_scale(card_on_wall(8.0, marker_id=2))
    assert s.found and s.marker_mm == 20.0
    assert s.px_per_mm_native == pytest.approx(8.0, rel=0.03)


def test_no_marker():
    img = np.full((400, 600, 3), 140, np.uint8)
    s = mk.estimate_scale(img)
    assert not s.found and s.H_mm is None


def test_homography_maps_marker_to_metric_square_under_perspective():
    img = card_on_wall(6.0)
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[40, 10], [w - 10, 50], [w - 60, h - 5], [5, h - 40]])
    P = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, P, (w, h), borderValue=(150, 150, 150))
    s = mk.estimate_scale(warped)
    assert s.found
    H = np.array(s.H_mm)
    corners = np.array(s.corners).reshape(-1, 1, 2)
    mm_pts = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    assert np.allclose(mm_pts, [[0, 0], [40, 0], [40, 40], [0, 40]], atol=0.05)


@pytest.mark.parametrize("sigma", [1.0, 2.0, 3.0])
def test_edge_blur_measured_on_marker(sigma):
    s = mk.estimate_scale(card_on_wall(8.0, blur=sigma))
    assert s.found
    assert s.edge_blur_px == pytest.approx(sigma, abs=0.3)


def test_deblur_inverts_blur_model():
    import math
    for w, sig in [(1.5, 0.8), (3.0, 1.0), (6.0, 1.2)]:
        ratio = w / math.erf(w / (2 * math.sqrt(2) * sig))
        assert deblur_width(ratio, sig) == pytest.approx(w, rel=1e-3)
    assert deblur_width(1.0, 1.0) is None  # below the blur limit: refuse to invent a width


def test_synthetic_scene_marker_found_and_scale():
    img, gt = render_scene(SceneSpec(seed=5, cam_ppm=6.0, tilt=0.0, rotate_deg=0.0))
    s = mk.estimate_scale(img)
    assert gt["marker_in_frame"] and s.found
    assert s.px_per_mm_native == pytest.approx(6.0, rel=0.03)
