import cv2
import numpy as np
import pytest

from setuscan.measure import measure, profile_width, skeleton_length, skeleton_length_chain


def bar_depth(h, w, width_px, depth=80.0, angle_deg=0.0, ss=8):
    """Anti-aliased straight dark bar (as a black-hat depth map), exact width."""
    H, W = h * ss, w * ss
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32) + 0.5
    cx, cy = W / 2, H / 2
    a = np.radians(angle_deg)
    d = np.abs(-(xx - cx) * np.sin(a) + (yy - cy) * np.cos(a)) / ss
    along = np.abs((xx - cx) * np.cos(a) + (yy - cy) * np.sin(a)) / ss
    inside = ((d <= width_px / 2) & (along <= 0.35 * w)).astype(np.float32)
    small = cv2.resize(inside, (w, h), interpolation=cv2.INTER_AREA)
    return small * depth


def test_skeleton_length_straight_and_diagonal():
    s = np.zeros((50, 50), np.uint8)
    s[10, 5:45] = 255
    assert skeleton_length(s) == pytest.approx(39.0)
    d = np.zeros((50, 50), np.uint8)
    for i in range(30):
        d[10 + i, 10 + i] = 255
    assert skeleton_length(d) == pytest.approx(29 * np.sqrt(2))


def test_skeleton_length_corner_within_tolerance():
    s = np.zeros((60, 60), np.uint8)
    s[10, 5:45] = 255  # 39 px run
    s[10:50, 44] = 255  # then 40 px down: an L shape
    # the polygon simplification may cut the corner by about 1 px; never double counts
    assert skeleton_length(s) == pytest.approx(79, abs=2.0)


def test_skeleton_length_oblique_no_staircase_bias():
    s = np.zeros((200, 200), np.uint8)
    cv2.line(s, (10, 50), (190, 125), 255, 1)  # about 22.6 degrees
    true = float(np.hypot(180, 75))
    assert skeleton_length(s) == pytest.approx(true, rel=0.01)
    assert skeleton_length_chain(s) > true * 1.03  # the naive chain count over-reads


@pytest.mark.parametrize("w", [2.0, 3.0, 4.5, 7.25])
def test_profile_width_subpixel(w):
    depth = bar_depth(80, 120, w)
    got = profile_width(depth, 60.0, 40.0, 0.0, half_len=12.0)
    assert got == pytest.approx(w, abs=0.15)


@pytest.mark.parametrize("angle", [0.0, 30.0, 63.0, 90.0])
def test_profile_width_any_orientation(angle):
    depth = bar_depth(160, 160, 4.0, angle_deg=angle)
    got = profile_width(depth, 80.0, 80.0, angle, half_len=12.0)
    assert got == pytest.approx(4.0, abs=0.2)


def test_measure_bar_length_and_width():
    depth = bar_depth(100, 200, 5.0)
    mask = (depth >= 40).astype(np.uint8) * 255
    out = measure(mask, depth, sample_every_px=3)
    assert len(out) == 1
    m = out[0]
    true_len = 0.7 * 200
    assert m.length_px == pytest.approx(true_len, rel=0.06)
    assert m.width_median_px == pytest.approx(5.0, abs=0.25)
    assert m.orientation_deg < 3


def test_measure_empty():
    assert measure(np.zeros((20, 20), np.uint8), np.zeros((20, 20), np.float32)) == []
