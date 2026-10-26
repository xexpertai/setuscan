"""Pluggable deciders for the inspection agent.

The agent loop is: observe (a compact JSON view of what the OpenCV tools have
found so far) -> decide (pick ONE next action from the allowed set) -> act (run the
tool) -> repeat. Only the "decide" step is pluggable:

* RulesDecider: deterministic, auditable policy. This is what runs in phase 1.
* LLMDecider: wraps any text model behind an `invoke(prompt) -> str` callable
  (phase 2: Amazon Bedrock, if the account has access). Its output is parsed and
  validated against the allowed actions; anything invalid falls back to the rules
  decider and the trace records that the fallback happened.

Whichever decider produced each step is written into the trace, so a demo can
never claim an LLM made a decision that the rules made.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Protocol

from .measure import MIN_RELIABLE_WIDTH_PX


@dataclass
class Decision:
    action: str
    args: dict
    rationale: str
    decider: str  # which decider actually produced it ("rules", "llm:<model>", "rules (llm fallback)")


class Decider(Protocol):
    name: str

    def decide(self, obs: dict, allowed: dict[str, dict]) -> Decision: ...


class RulesDecider:
    """Deterministic decision policy over the observation."""

    name = "rules"

    def decide(self, obs: dict, allowed: dict[str, dict]) -> Decision:
        d = self._decide(obs, allowed)
        if d.action not in allowed:  # defensive: the rules must respect the guardrails too
            raise RuntimeError(f"rules decider chose disallowed action {d.action}")
        return d

    def _decide(self, obs: dict, allowed: dict[str, dict]) -> Decision:
        V = obs["visits"]
        visits = [v for v in ("current", "previous") if v in V]
        R = lambda a, args, why: Decision(a, args, why, self.name)  # noqa: E731

        for v in visits:
            if V[v]["quality"] is None:
                return R("check_quality", {"visit": v}, f"No quality check yet for the {v} photo.")
        cq = V["current"]["quality"]
        if not cq["ok"]:
            return R("request_reshoot", {"visit": "current", "reasons": cq["issues"]},
                     f"Current photo failed the quality gate ({', '.join(cq['issues'])}); "
                     "measuring it would give unreliable widths.")
        usable = [v for v in visits if V[v]["quality"]["ok"]]
        for v in usable:
            if V[v]["scale"] is None:
                return R("detect_scale", {"visit": v}, f"Need px-to-mm scale for the {v} photo.")
        cs = V["current"]["scale"]
        if cs["found"] and cs.get("edge_blur_px") is not None and cs["edge_blur_px"] > obs["max_edge_blur_px"]:
            return R("request_reshoot", {"visit": "current", "reasons": ["blurry_marker"]},
                     f"The marker's printed edges are blurred by sigma {cs['edge_blur_px']:.1f} px "
                     f"({cs['edge_blur_mm']:.2f} mm), above the {obs['max_edge_blur_px']} px limit: crack widths "
                     "would be over-read.")
        for v in usable:
            if not V[v]["measured"]:
                return R("segment_measure", {"visit": v}, f"Segment and measure cracks in the {v} photo.")

        cur = V["current"]
        if not cur["scale"]["found"]:
            if cur["cracks"]:
                return R("request_reshoot", {"visit": "current", "reasons": ["no_marker"]},
                         f"{len(cur['cracks'])} crack(s) found but no scale marker, so widths are only in "
                         "pixels and cannot be compared with the mm policy.")
            return R("close_case", {"outcome": "no_crack_detected"},
                     "No crack found and no marker present; nothing to measure.")

        if "zoom_measure" in allowed:
            for c in sorted(cur["cracks"], key=lambda c: -c["width_max_mm"]):
                if c["id"] not in allowed["zoom_measure"]["crack_ids"]:
                    continue
                why = []
                if not c["width_reliable"]:
                    why.append(f"only {c['width_max_px']:.1f} px wide at the working scale")
                if c["ambiguous"]:
                    why.append(f"{c['width_max_mm']:.2f} mm is close to the {c['nearest_edge_mm']} mm band edge")
                if why:
                    return R("zoom_measure", {"visit": "current", "crack_id": c["id"]},
                             f"Crack #{c['id']}: " + " and ".join(why) +
                             "; re-measuring it at the photo's native resolution.")

        for c in cur["cracks"]:
            if c.get("zoomed") and c["width_max_px"] < MIN_RELIABLE_WIDTH_PX and c["ambiguous"]:
                return R("request_reshoot", {"visit": "current", "reasons": ["too_far"], "crack_id": c["id"]},
                         f"Even at native resolution crack #{c['id']} is {c['width_max_px']:.1f} px wide and its "
                         f"{c['width_max_mm']:.2f} mm reading sits on a band edge: a closer photo is needed.")

        if "compare_visits" in allowed:
            return R("compare_visits", {}, "Both visits are measured; align them to check for growth.")
        comp = obs.get("comparison")
        if comp is not None and not comp["ok"]:
            return R("request_reshoot", {"visit": "current", "reasons": ["misaligned"]},
                     "The two visits could not be aligned (" + comp["note"] + "); retake from the same spot.")

        if obs.get("classification") is None:
            return R("classify", {}, "All measurements are in; apply the severity policy.")
        cl = obs["classification"]
        if cl["needs_engineer"]:
            return R("request_approval", {},
                     f"Policy result '{cl['band']}'" + (" with significant growth" if cl["growth"] else "") +
                     ": an engineer must approve before a ticket is raised.")
        return R("close_case", {"outcome": cl["band"]},
                 f"Policy result '{cl['band']}' needs no engineer action; schedule re-inspection in "
                 f"{cl['reinspect_days']} days.")


SYSTEM_PROMPT = """You are the planning step of a crack-inspection agent. You never see images.
You see JSON produced by OpenCV tools and must choose exactly ONE next action from ALLOWED.
Rules: never invent measurements; prefer gathering evidence (quality, scale, measurement,
zoom, comparison) before classifying; a human engineer must approve any ticket.
Reply with JSON only: {"action": "...", "args": {...}, "rationale": "one sentence citing the numbers"}."""


class LLMDecider:
    """Model-backed decider with strict validation and a rules fallback."""

    def __init__(self, invoke: Callable[[str], str], model_name: str, fallback: RulesDecider | None = None):
        self.invoke = invoke
        self.name = f"llm:{model_name}"
        self.fallback = fallback or RulesDecider()
        self.last_error: str | None = None

    def prompt(self, obs: dict, allowed: dict[str, dict]) -> str:
        return SYSTEM_PROMPT + "\n\nOBSERVATION:\n" + json.dumps(obs, sort_keys=True) + \
            "\n\nALLOWED:\n" + json.dumps(allowed, sort_keys=True)

    @staticmethod
    def validate(raw: str, allowed: dict[str, dict]) -> Decision:
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`").split("\n", 1)[-1]
        d = json.loads(text[text.index("{"): text.rindex("}") + 1])
        action, args = d.get("action"), d.get("args") or {}
        if action not in allowed:
            raise ValueError(f"action {action!r} not allowed")
        spec = allowed[action]
        if "visits" in spec and args.get("visit") not in spec["visits"]:
            raise ValueError(f"visit {args.get('visit')!r} not allowed for {action}")
        if "crack_ids" in spec and args.get("crack_id") not in spec["crack_ids"]:
            raise ValueError(f"crack_id {args.get('crack_id')!r} not allowed for {action}")
        if not isinstance(d.get("rationale"), str) or not d["rationale"].strip():
            raise ValueError("missing rationale")
        return Decision(action, args, d["rationale"].strip()[:400], "llm")

    def decide(self, obs: dict, allowed: dict[str, dict]) -> Decision:
        try:
            d = self.validate(self.invoke(self.prompt(obs, allowed)), allowed)
            d.decider = self.name
            self.last_error = None
            return d
        except Exception as e:  # any model/transport/validation failure -> deterministic fallback
            self.last_error = f"{type(e).__name__}: {e}"[:200]
            d = self.fallback.decide(obs, allowed)
            d.decider = f"rules (fallback: {self.name} failed: {self.last_error})"
            return d
