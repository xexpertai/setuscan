"""Run the SetuScan agent with Amazon Bedrock as the decider and record each case as a trace.

The OpenCV 5 tools run locally (Python); only the "decide" step calls Bedrock (Converse API through
the aws CLI, so no extra Python deps). Output: planb/data/traces/<sample>.json plus planb/data/bedrock_calls.log.
Every trace is labelled RECORDED: pre-run from the build laptop, not a live call from the viewer.
Hard cap on calls: MAX_CALLS (default 200 across all runs, counted in the log).
"""
from __future__ import annotations

import json, os, subprocess, sys, time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from setuscan import agent  # noqa: E402
from setuscan.deciders import LLMDecider, RulesDecider  # noqa: E402

MODEL = os.environ.get("BEDROCK_MODEL", "amazon.nova-micro-v1:0")
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
MAX_CALLS = int(os.environ.get("MAX_CALLS", "200"))
OUT = ROOT / "planb" / "data"
LOG = OUT / "bedrock_calls.log"
AWS = os.path.expanduser("~/.local/bin/aws")


def n_calls() -> int:
    return sum(1 for _ in LOG.open()) if LOG.exists() else 0


def invoke(prompt: str) -> str:
    if n_calls() >= MAX_CALLS:
        raise RuntimeError("Bedrock call cap reached")
    msgs = json.dumps([{"role": "user", "content": [{"text": prompt}]}])
    cfg = json.dumps({"maxTokens": 300, "temperature": 0})
    t0 = time.time()
    p = subprocess.run([AWS, "bedrock-runtime", "converse", "--region", REGION, "--model-id", MODEL,
                        "--messages", msgs, "--inference-config", cfg, "--output", "json"],
                       capture_output=True, text=True, timeout=60)
    with LOG.open("a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": MODEL,
                            "rc": p.returncode, "ms": round((time.time() - t0) * 1000)}) + "\n")
    if p.returncode:
        raise RuntimeError(p.stderr[-200:])
    r = json.loads(p.stdout)
    invoke.usage = r.get("usage", {})
    return r["output"]["message"]["content"][0]["text"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "traces").mkdir(exist_ok=True)
    samples = json.loads((ROOT / "web/samples/index.json").read_text())
    only = sys.argv[1:]
    for s in samples:
        if only and s["id"] not in only:
            continue
        cur = cv2.imread(str(ROOT / "web/samples" / s["files"]["current"]))
        prev = cv2.imread(str(ROOT / "web/samples" / s["files"]["previous"])) if "previous" in s["files"] else None
        decider = LLMDecider(invoke, MODEL)
        case = agent.new_case(cur, prev, labels={"sample": s["id"]})
        agent.run(case, decider)
        # what the deterministic rules would have done at the same step is not re-derived here;
        # fallbacks are visible in each step's "decider" field.
        steps = [{k: t.get(k) for k in ("step", "decider", "action", "args", "rationale", "allowed")}
                 | {"result_brief": t.get("result")} for t in case.trace]
        fb = sum(1 for t in case.trace if str(t.get("decider", "")).startswith("rules"))
        rec = {"label": "RECORDED", "note": "Pre-run on the build laptop on 25 Oct 2026. OpenCV 5 tools ran locally in "
               "Python; each decision was made by Amazon Bedrock. Not a live call from this page.",
               "model": MODEL, "region": REGION, "sample": s["id"], "sample_label": s["label"],
               "status": case.status, "steps": steps, "n_steps": len(steps), "n_fallbacks_to_rules": fb,
               "classification": case.classification,
               "python_opencv": cv2.__version__,
               "python_cracks": case.visits["current"].cracks}
        (OUT / "traces" / f"{s['id']}.json").write_text(json.dumps(rec, indent=1, default=str))
        print(s["id"], case.status, len(steps), "steps", fb, "fallbacks", "calls so far", n_calls())


if __name__ == "__main__":
    main()
