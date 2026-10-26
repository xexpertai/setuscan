import numpy as np

from setuscan.quality import assess_quality
from setuscan.synth import SceneSpec, default_crack, render_scene


def scene(**kw):
    rng = np.random.default_rng(4)
    return render_scene(SceneSpec(seed=4, cracks=[default_crack(rng)], **kw))[0]


def test_good_photo_passes_and_marker_card_is_not_glare():
    q = assess_quality(scene())
    assert q.ok, q.issues


def test_dark_photo():
    q = assess_quality(scene(gain=0.2))
    assert "too_dark" in q.issues and q.advice


def test_glare_photo():
    q = assess_quality(scene(glare=True))
    assert "glare" in q.issues


def test_heavy_blur_is_caught_by_marker_edges():
    from setuscan import agent
    c = agent.run(agent.new_case(scene(blur_sigma=5.0, cam_ppm=4.0)))
    assert c.status == "needs_reshoot" and c.reshoot["reasons"] == ["blurry_marker"]


def test_flat_grey_is_blurry():
    q = assess_quality(np.full((600, 800, 3), 128, np.uint8))
    assert "blurry" in q.issues
