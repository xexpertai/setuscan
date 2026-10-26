import os

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["SETUSCAN_RUNS"] = str(tmp_path_factory.mktemp("runs"))
    from setuscan import api
    from setuscan.store import LocalStore
    api.STORE = LocalStore(os.environ["SETUSCAN_RUNS"])
    return TestClient(api.app)


def jpeg(img):
    return cv2.imencode(".jpg", img)[1].tobytes()


def wide_scene():
    from setuscan.synth import SceneSpec, default_crack, render_scene
    rng = np.random.default_rng(9)
    return render_scene(SceneSpec(seed=9, cracks=[default_crack(rng, width_mm=2.6)]))[0]


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["opencv"].startswith("5.")
    assert r.headers["X-Content-Type-Options"] == "nosniff"


def test_index(client):
    r = client.get("/")
    assert r.status_code == 200 and "SetuScan" in r.text


def test_upload_inspect_and_approve(client):
    r = client.post("/api/inspect", files={"current": ("wall.jpg", jpeg(wide_scene()), "image/jpeg")})
    assert r.status_code == 200, r.text
    c = r.json()
    assert c["status"] == "awaiting_approval"
    assert c["labels"]["source"] == "User upload"
    img = client.get(c["image_urls"]["current_overlay"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    bad = client.post(f"/api/cases/{c['id']}/approve", data={"reviewer": ""})
    assert bad.status_code in (409, 422)
    ok = client.post(f"/api/cases/{c['id']}/approve", data={"reviewer": "Engineer A", "note": "ok"})
    assert ok.status_code == 200 and ok.json()["status"] == "ticket_raised"
    again = client.post(f"/api/cases/{c['id']}/reject", data={"reviewer": "Engineer A"})
    assert again.status_code == 409


def test_rejects_non_image(client):
    r = client.post("/api/inspect", files={"current": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 415


def test_unknown_case_and_path_safety(client):
    assert client.get("/api/cases/zzzz").status_code == 404
    assert client.get("/api/cases/0123456789ab/images/..%2Fcase.jpg").status_code == 404
