"""Run the Python (desktop OpenCV 5.0.0.93) agent with the rules decider on every web sample and save the
reference results the in-browser port is compared against (planb/data/python_reference.json)."""
import json, sys
from pathlib import Path
import cv2
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from setuscan import agent  # noqa: E402

out = {}
for s in json.loads((ROOT / "web/samples/index.json").read_text()):
    cur = cv2.imread(str(ROOT / "web/samples" / s["files"]["current"]))
    prev = cv2.imread(str(ROOT / "web/samples" / s["files"]["previous"])) if "previous" in s["files"] else None
    c = agent.run(agent.new_case(cur, prev), None)
    cr = c.visits["current"].cracks
    out[s["id"]] = {"status": c.status, "actions": [t["action"] for t in c.trace], "band": (c.classification or {}).get("band"),
                    "growth": (c.classification or {}).get("growth"),
                    "cracks": [{k: x.get(k) for k in ("id", "length_mm", "width_max_mm", "width_max_px", "width_source")} for x in cr],
                    "comparison": None if c.comparison is None else {k: v for k, v in c.comparison.to_dict().items() if k != "registration"},
                    "opencv": cv2.__version__, "trace": [{k: t.get(k) for k in ("step", "decider", "action", "args", "rationale")} for t in c.trace]}
    print(s["id"], c.status, out[s["id"]]["actions"], [x.get("width_max_mm", x.get("width_max_px")) for x in cr])
(ROOT / "planb/data").mkdir(parents=True, exist_ok=True)
(ROOT / "planb/data/python_reference.json").write_text(json.dumps(out, indent=1, default=str))
