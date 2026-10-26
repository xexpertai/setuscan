"""Plan B browser tests: the static site in dist-planb/ is served locally and driven with Chromium (Playwright).

Run: scripts/build_planb.sh && python -m pytest tests/test_planb_browser.py   (needs `pip install playwright pytest`
and an installed Chromium). Checks that OpenCV 5 loads as WebAssembly, that every bundled sample reaches the same
final status as the desktop Python agent, that the worst-crack width agrees with Python within a stated tolerance,
and (with PLANB_SHOTS=1) writes screenshots to docs/screenshots/planb-*.png.
"""
import functools, http.server, json, os, threading
from pathlib import Path

import pytest

pw = pytest.importorskip("playwright.sync_api")
ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist-planb"
pytestmark = pytest.mark.skipif(not (DIST / "index.html").exists(), reason="run scripts/build_planb.sh first")
REF = json.loads((ROOT / "planb/data/python_reference.json").read_text()) if (ROOT / "planb/data/python_reference.json").exists() else {}
WIDTH_TOL = 0.05  # relative; measured, see docs/PLAN-B.md


@pytest.fixture(scope="module")
def server():
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DIST))
    h.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture(scope="module")
def browser():
    with pw.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


def run_sample(browser, base, sid, size=(1440, 900)):
    page = browser.new_page(viewport={"width": size[0], "height": size[1]})
    errs = []
    page.on("pageerror", lambda e: errs.append(str(e)))
    page.goto(f"{base}/?sample={sid}")
    page.wait_for_function("document.querySelector('#status') && document.querySelector('#status').textContent.length > 0", timeout=240000)
    return page, errs


def test_opencv_version_in_footer(browser, server):
    page = browser.new_page()
    page.goto(server)
    page.wait_for_function("document.querySelector('#foot').textContent.includes('OpenCV 5.')", timeout=120000)
    assert "WebAssembly" in page.inner_text("#foot")
    page.close()


@pytest.mark.parametrize("sid", sorted(REF))
def test_sample_matches_python(browser, server, sid):
    page, errs = run_sample(browser, server, sid)
    status = page.evaluate("current.status")
    ref = REF[sid]
    assert not errs
    assert status == ref["status"], (status, ref["status"])
    cracks = page.evaluate("current.cracks")
    if ref["cracks"]:
        key = "width_max_mm" if ref["cracks"][0].get("width_max_mm") is not None else "width_max_px"
        got, want = cracks[0][key], ref["cracks"][0][key]
        assert abs(got - want) / want <= WIDTH_TOL, (sid, got, want)
    else:
        assert cracks == []
    page.close()


def test_human_gate_no_ticket_before_approval(browser, server):
    page, _ = run_sample(browser, server, "wide")
    assert page.evaluate("current.ticket") is None
    assert page.is_visible("#approval-card")
    page.fill("#rev", "Test Engineer")
    page.click("button[data-act=approve]")
    assert page.evaluate("current.status") == "ticket_raised"
    assert page.is_visible("#ticket-card")
    page.close()


def test_bedrock_panel_is_honest(browser, server):
    page, _ = run_sample(browser, server, "blurry")
    page.wait_for_selector("#bedrock .status")
    txt = page.inner_text("#bedrock")
    assert ("RECORDED" in txt) or ("Not available" in txt)
    page.close()


@pytest.mark.skipif(not os.environ.get("PLANB_SHOTS"), reason="set PLANB_SHOTS=1 to write screenshots")
@pytest.mark.parametrize("name,size", [("desktop", (1440, 900)), ("mobile", (390, 844))])
def test_screenshots(browser, server, name, size):
    out = ROOT / "docs/screenshots"
    page = browser.new_page(viewport={"width": size[0], "height": size[1]})
    page.goto(server)
    page.wait_for_selector("[data-sample]")
    page.screenshot(path=str(out / f"planb-home-{name}.png"))
    page.close()
    for sid in ("wide", "blurry"):
        page, _ = run_sample(browser, server, sid, size)
        assert page.evaluate("document.documentElement.scrollWidth") <= size[0] + 1, "horizontal overflow"
        page.wait_for_selector("#bedrock .status")
        page.locator("#result").screenshot(path=str(out / f"planb-{sid}-{name}.png"))
        page.close()
