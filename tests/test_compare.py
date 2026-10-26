import numpy as np
import pytest

from setuscan.compare import compare_visits
from setuscan.pipeline import analyze
from setuscan.synth import SceneSpec, default_crack, render_scene, two_visits


@pytest.fixture(scope="module")
def grown():
    (a, ga), (b, gb) = two_visits(21, 0.6, 1.2, 80, 130)
    return a, b, ga, gb


def test_growth_detected_and_sized(grown):
    a_img, b_img, ga, gb = grown
    comp, warped, new = compare_visits(a_img, analyze(a_img), b_img, analyze(b_img))
    assert comp.ok and comp.unit == "mm"
    assert comp.registration.n_inliers >= 25 and comp.registration.rms_px < 2.0
    assert 0.95 < comp.registration.scale < 1.05
    assert comp.significant_growth
    true_ext = gb["cracks"][0]["visible_length_mm"] - ga["cracks"][0]["visible_length_mm"]
    assert comp.extension == pytest.approx(true_ext, abs=12.0)
    assert comp.width_change == pytest.approx(0.6, abs=0.25)
    assert new.any()


def test_same_crack_two_poses_no_growth():
    (a, _), (b, _) = two_visits(22, 0.8, 0.8, 120, 120)
    comp, _, _ = compare_visits(a, analyze(a), b, analyze(b))
    assert comp.ok
    assert not comp.significant_growth
    assert abs(comp.width_change) < 0.1
    assert comp.extension < 5.0


def test_unrelated_walls_refused():
    rng = np.random.default_rng(1)
    a, _ = render_scene(SceneSpec(seed=1, cracks=[default_crack(rng)]))
    rng = np.random.default_rng(2)
    b, _ = render_scene(SceneSpec(seed=777, cracks=[default_crack(rng)]))
    comp, _, _ = compare_visits(a, analyze(a), b, analyze(b))
    assert not comp.ok
    assert "could not be aligned" in comp.note
