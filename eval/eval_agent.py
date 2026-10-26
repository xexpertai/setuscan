"""Agent scenario evaluation on SYNTHETIC scenes (labelled as such).

Each scenario has an expected terminal status and, where relevant, an action that
must (or must not) appear in the trace. Scenarios cover the happy paths and the
failure-handling paths: blur, darkness, glare, missing marker, thin cracks that
need a closer look, two-visit growth, two visits that cannot be aligned, and an
undecodable/blank input.

Metrics: terminal-status accuracy, required-action rate, false escalations (sent
to an engineer when the expected outcome was to close), missed escalations, mean
steps and wall time. The decider used is recorded (phase 1: deterministic rules).

Usage: uv run python eval/eval_agent.py [--seeds 5]
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import provenance, write_json  # noqa: I001
from setuscan import agent
from setuscan.synth import CrackSpec, SceneSpec, default_crack, render_scene, two_visits


def scene(seed: int, width: float, **kw) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cracks = [] if width <= 0 else [default_crack(rng, width_mm=width, length_mm=float(rng.uniform(110, 150)))]
    return render_scene(SceneSpec(seed=seed, cracks=cracks, **kw))[0]


SCENARIOS = {
    # name: (builder(seed) -> (current, previous|None), expected_status, must_have_action, must_not_have_action)
    "fine_crack_closes": (lambda s: (scene(s, 0.6), None), "closed", None, "request_approval"),
    "hairline_closes": (lambda s: (scene(s, 0.18, cam_ppm=9.0), None), "closed", None, "request_approval"),
    "moderate_escalates": (lambda s: (scene(s, 1.5), None), "awaiting_approval", "request_approval", None),
    "wide_escalates": (lambda s: (scene(s, 3.0), None), "awaiting_approval", "request_approval", None),
    "no_crack_closes": (lambda s: (scene(s, 0.0), None), "closed", None, "request_approval"),
    "thin_far_needs_zoom": (lambda s: (scene(s, 0.35, cam_ppm=13.0, wall_mm=(300.0, 220.0)), None),
                            None, "zoom_measure", None),
    "blurry_reshoot": (lambda s: (scene(s, 1.0, blur_sigma=5.0), None), "needs_reshoot", "request_reshoot", "segment_measure"),
    "dark_reshoot": (lambda s: (scene(s, 1.0, gain=0.22), None), "needs_reshoot", "request_reshoot", "segment_measure"),
    "glare_reshoot": (lambda s: (scene(s, 1.0, glare=True), None), "needs_reshoot", "request_reshoot", "segment_measure"),
    "no_marker_reshoot": (lambda s: (scene(s, 1.0, marker_id=None), None), "needs_reshoot", "request_reshoot", "classify"),
    "growth_escalates": (lambda s: tuple(x[0] for x in two_visits(s, 0.5, 0.9, 90, 140))[::-1],
                         "awaiting_approval", "compare_visits", None),
    "stable_two_visits_close": (lambda s: tuple(x[0] for x in two_visits(s, 0.5, 0.5, 120, 120))[::-1],
                                "closed", "compare_visits", "request_approval"),
    "unrelated_walls_reshoot": (lambda s: (scene(s, 0.6), scene(s + 5000, 0.6)), "needs_reshoot", "compare_visits",
                                "request_approval"),
    "blank_image_handled": (lambda s: (np.full((900, 1200, 3), 128, np.uint8), None), "needs_reshoot", None, None),
}


def run_one(args):
    name, seed = args
    build, exp_status, must, must_not = SCENARIOS[name]
    cur, prev = build(seed)
    t0 = time.perf_counter()
    case = agent.run(agent.new_case(cur, prev))
    ms = (time.perf_counter() - t0) * 1000
    actions = [t.get("action") for t in case.trace]
    ok_status = exp_status is None or case.status == exp_status
    ok_must = must is None or must in actions
    ok_not = must_not is None or must_not not in actions
    return {"scenario": name, "seed": seed, "status": case.status, "expected_status": exp_status,
            "actions": actions, "ok_status": ok_status, "ok_required_action": ok_must, "ok_forbidden_action": ok_not,
            "success": ok_status and ok_must and ok_not, "steps": len(case.trace), "ms": round(ms, 1),
            "deciders": sorted({t.get("decider", "") for t in case.trace}),
            "reshoot": case.reshoot, "classification": case.classification,
            "zoom_results": [t["result"] for t in case.trace if t.get("action") == "zoom_measure"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    jobs = [(n, 100 + s) for n in SCENARIOS for s in range(args.seeds)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(run_one, jobs))
    per = {}
    for n in SCENARIOS:
        sub = [r for r in rows if r["scenario"] == n]
        per[n] = {"n": len(sub), "success_rate": round(np.mean([r["success"] for r in sub]), 3),
                  "statuses": dict(Counter(r["status"] for r in sub)), "expected_status": SCENARIOS[n][1],
                  "required_action": SCENARIOS[n][2], "forbidden_action": SCENARIOS[n][3]}
    close_expected = [r for r in rows if r["expected_status"] == "closed"]
    esc_expected = [r for r in rows if r["expected_status"] == "awaiting_approval"]
    res = {
        "provenance": provenance("eval/eval_agent.py"),
        "label": "SYNTHETIC scenarios; deterministic rules decider (no LLM)",
        "n_runs": len(rows),
        "task_success_rate": round(np.mean([r["success"] for r in rows]), 4),
        "terminal_status_accuracy": round(np.mean([r["ok_status"] for r in rows]), 4),
        "false_escalation_rate": round(np.mean([r["status"] == "awaiting_approval" for r in close_expected]), 4)
        if close_expected else None,
        "missed_escalation_rate": round(np.mean([r["status"] != "awaiting_approval" for r in esc_expected]), 4)
        if esc_expected else None,
        "error_status_runs": sum(r["status"] == "error" for r in rows),
        "mean_steps": round(np.mean([r["steps"] for r in rows]), 2),
        "mean_ms": round(np.mean([r["ms"] for r in rows]), 1),
        "per_scenario": per,
        "failures": [r for r in rows if not r["success"]],
        "rows": rows,
        "wall_time_s": round(time.time() - t0, 1),
    }
    write_json("agent_scenarios.json", res)
    print({k: res[k] for k in ("n_runs", "task_success_rate", "terminal_status_accuracy", "false_escalation_rate",
                               "missed_escalation_rate", "error_status_runs", "mean_steps")})
    for n, d in per.items():
        print(f"  {n:28s} {d['success_rate']:.2f} {d['statuses']}")


if __name__ == "__main__":
    main()
