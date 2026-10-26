"""Render real pipeline outputs (overlays, comparison, trace snippet) for the video slides, from the same code as the deployed app.
Run: PYTHONPATH=. .venv/bin/python video/make_vis.py   -> video/build/vis/*.jpg, trace_wide.json"""
import json, pathlib, cv2
from setuscan import agent
H = pathlib.Path(__file__).parent; O = H / "build/vis"; O.mkdir(parents=True, exist_ok=True)
S = H.parent / "web/samples/"
for e in json.load(open(S / "index.json")):
    if e["id"] not in ("wide", "thin", "growth"): continue
    f = e["files"]; cur = cv2.imread(str(S / f["current"])); prev = cv2.imread(str(S / f["previous"])) if "previous" in f else None
    c = agent.new_case(cur, prev); agent.run(c)
    print(e["id"], c.status, sorted(c.images))
    for n, im in c.images.items():
        if n in ("current_overlay", "comparison"): cv2.imwrite(str(O / f"{e['id']}_{n}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if e["id"] == "wide":
        json.dump([{"action": t["action"], "rationale": t.get("rationale")} for t in c.trace], open(O / "trace_wide.json", "w"), indent=1)
