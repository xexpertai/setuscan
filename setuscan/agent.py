"""SetuScan inspection agent: observe -> decide -> act, with a human approval gate.

Tools (all OpenCV 5 work happens inside them):
  check_quality, detect_scale, segment_measure, zoom_measure, compare_visits,
  classify, request_reshoot, request_approval, close_case.

Guardrails live HERE, not in the decider: `allowed_actions()` derives from the case
state which actions are legal next (e.g. no comparison before both photos are
measured, no closing a case that the policy says needs an engineer). There is no
"raise_ticket" tool at all: a ticket can only be created by `approve()`, which is
called by a human through the UI/API.

Every step appends a trace record (what was observed, which decider chose what and
why, what the tool returned, timings). The trace is the evidence that the vision
output changed what the system did next.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

from . import marker as mk
from .compare import Comparison, compare_visits
from .deciders import Decider, Decision, RulesDecider
from .measure import MIN_RELIABLE_WIDTH_PX
from .overlay import draw_analysis, draw_comparison, draw_marker_on_photo
from .pipeline import Analysis, analyze_view, base_view, zoom_view
from .policy import SeverityPolicy
from .quality import QualityReport, assess_quality

MAX_STEPS = 24
MAX_ZOOMS = 3
TERMINAL = {"needs_reshoot", "awaiting_approval", "closed", "ticket_raised", "rejected", "error"}

MAX_EDGE_BLUR_PX = 2.5  # Gaussian sigma measured on the marker edges

RESHOOT_ADVICE = {
    "blurry_marker": "The photo is out of focus or shaken: tap to focus on the crack, hold steady (or brace the "
                     "phone), and retake.",
    "no_marker": "Stick the printed SetuScan marker flat on the surface next to the crack and retake.",
    "too_far": "Move closer (about half the distance) keeping the marker and the crack in view.",
    "misaligned": "Retake from roughly the same position and angle as the previous visit's photo.",
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Visit:
    name: str
    image: np.ndarray
    quality: QualityReport | None = None
    scale: mk.ScaleResult | None = None
    analysis: Analysis | None = None
    cracks: list[dict] = field(default_factory=list)


@dataclass
class Case:
    id: str
    visits: dict[str, Visit]
    policy: SeverityPolicy
    status: str = "open"
    trace: list[dict] = field(default_factory=list)
    comparison: Comparison | None = None
    classification: dict | None = None
    images: dict[str, np.ndarray] = field(default_factory=dict)  # named overlays for the UI/report
    reshoot: dict | None = None
    approval: dict | None = None
    ticket: dict | None = None
    created: str = field(default_factory=utcnow)
    zooms: int = 0
    labels: dict = field(default_factory=dict)  # e.g. {"source": "SYNTHETIC sample"}


# ---------------------------------------------------------------- observation

def _crack_obs(c: dict) -> dict:
    keys = ("id", "length_mm", "width_max_mm", "width_max_px", "width_reliable", "ambiguous", "nearest_edge_mm",
            "zoomed", "zoomable", "length_px")
    return {k: c[k] for k in keys if k in c}


def observe(case: Case) -> dict:
    vis = {}
    for name, v in case.visits.items():
        vis[name] = {
            "quality": None if v.quality is None else {"ok": v.quality.ok, "issues": v.quality.issues,
                                                       "sharpness": round(v.quality.sharpness, 1)},
            "scale": None if v.scale is None else {"found": v.scale.found,
                                                   "px_per_mm": v.scale.to_dict()["px_per_mm_native"],
                                                   "tilt_ratio": v.scale.to_dict()["tilt_ratio"],
                                                   "edge_blur_px": v.scale.to_dict()["edge_blur_px"],
                                                   "edge_blur_mm": v.scale.to_dict()["edge_blur_mm"]},
            "measured": v.analysis is not None,
            "mm_per_px": None if v.analysis is None or not v.analysis.mm_per_px else round(v.analysis.mm_per_px, 4),
            "cracks": [_crack_obs(c) for c in v.cracks],
        }
    comp = None
    if case.comparison is not None:
        c = case.comparison
        comp = {"ok": c.ok, "note": c.note, "extension": round(c.extension, 2), "width_change": round(c.width_change, 3),
                "significant_growth": c.significant_growth, "unit": c.unit}
    return {"visits": vis, "comparison": comp, "classification": case.classification,
            "policy": {"label": case.policy.label, "band_edges_mm": case.policy.edges()},
            "steps": len(case.trace), "zooms_used": case.zooms, "max_edge_blur_px": MAX_EDGE_BLUR_PX}


def allowed_actions(case: Case) -> dict[str, dict]:
    """Legal next actions given the state. Safety invariants are enforced here."""
    V = case.visits
    a: dict[str, dict] = {"request_reshoot": {"visits": ["current"]}}
    q_missing = [n for n, v in V.items() if v.quality is None]
    if q_missing:
        a["check_quality"] = {"visits": q_missing}
    usable = [n for n, v in V.items() if v.quality is not None and v.quality.ok]
    s_missing = [n for n in usable if V[n].scale is None]
    if s_missing:
        a["detect_scale"] = {"visits": s_missing}
    m_missing = [n for n in usable if V[n].scale is not None and V[n].analysis is None]
    if m_missing:
        a["segment_measure"] = {"visits": m_missing}
    cur = V["current"]
    if cur.analysis is not None and case.zooms < MAX_ZOOMS:
        ids = [c["id"] for c in cur.cracks if c.get("zoomable") and not c.get("zoomed")]
        if ids:
            a["zoom_measure"] = {"visits": ["current"], "crack_ids": ids}
    prev = V.get("previous")
    if (prev is not None and prev.analysis is not None and cur.analysis is not None and case.comparison is None
            and cur.scale.found == prev.scale.found):
        a["compare_visits"] = {}
    if cur.analysis is not None and case.classification is None:
        a["classify"] = {}
    if case.classification is not None:
        if case.classification["needs_engineer"]:
            a["request_approval"] = {}
        else:
            a["close_case"] = {}
    elif cur.analysis is not None and not cur.cracks:
        a["close_case"] = {}
    return a


# ---------------------------------------------------------------- tools

def _decorate_cracks(case: Case, v: Visit) -> None:
    a = v.analysis
    out = []
    for c in a.cracks:
        d = a.crack_summary(c)
        if a.mm_per_px:
            e, _ = case.policy.nearest_edge(d["width_max_mm"])
            d["nearest_edge_mm"] = e
            d["ambiguous"] = case.policy.is_ambiguous(d["width_max_mm"], a.mm_per_px)
        else:
            d["nearest_edge_mm"], d["ambiguous"] = None, False
        d["zoomable"] = a.view.scale < 0.9 * a.view.native_scale
        d["zoomed"] = False
        d["width_source"] = "working view"
        out.append(d)
    v.cracks = out


def t_check_quality(case: Case, visit: str) -> dict:
    v = case.visits[visit]
    v.quality = assess_quality(v.image)
    return v.quality.to_dict()


def t_detect_scale(case: Case, visit: str) -> dict:
    v = case.visits[visit]
    v.scale = mk.estimate_scale(v.image)
    case.images[f"{visit}_marker"] = draw_marker_on_photo(v.image, _stub_analysis(v))
    d = v.scale.to_dict()
    d.pop("H_mm", None)
    return d


def _stub_analysis(v: Visit):
    class _S:  # minimal object for draw_marker_on_photo
        scale = v.scale
    return _S()


def t_segment_measure(case: Case, visit: str) -> dict:
    v = case.visits[visit]
    v.analysis = analyze_view(v.image, base_view(v.image, v.scale), v.scale, v.quality)
    _decorate_cracks(case, v)
    case.images[f"{visit}_overlay"] = draw_analysis(v.analysis)
    return {"n_cracks": len(v.cracks), "mm_per_px": v.analysis.to_dict()["mm_per_px"],
            "cracks": [_crack_obs(c) for c in v.cracks], "view": v.analysis.view.describe(),
            "rejected_components": v.analysis.rejected_components, "timings_ms": v.analysis.timings_ms}


def t_zoom_measure(case: Case, visit: str, crack_id: int) -> dict:
    v = case.visits[visit]
    case.zooms += 1
    c = next(c for c in v.cracks if c["id"] == crack_id)
    z = zoom_view(v.analysis.view, c["bbox_px"])
    az = analyze_view(v.image, z, v.scale, v.quality)
    c["zoomed"] = True
    if not az.cracks:
        return {"crack_id": crack_id, "found_in_zoom": False, "view": z.describe(),
                "note": "crack not re-detected at native resolution; keeping working-view values"}
    # the zoomed crack nearest to where this crack was in the working view
    bx, by, bw, bh = c["bbox_px"]
    cx, cy = v.analysis.view.to_base(bx + bw / 2, by + bh / 2)
    zx, zy = (cx - z.box[0]) * z.scale, (cy - z.box[1]) * z.scale

    def dist(m):
        x, y, w, h = m.bbox
        return np.hypot(x + w / 2 - zx, y + h / 2 - zy) - 0.5 * m.length_px  # prefer long and central

    best = min(az.cracks, key=dist)
    zd = az.crack_summary(best)
    before = {k: c[k] for k in ("width_max_mm", "width_max_px", "width_median_mm") if k in c}
    k = az.mm_per_px
    # widths come from the sharper view; length stays from the full view (zoom may crop)
    c["width_max_mm"] = zd.get("width_max_mm", c.get("width_max_mm"))
    c["width_median_mm"] = zd.get("width_median_mm", c.get("width_median_mm"))
    c["width_max_px"] = zd["width_max_px"]
    c["width_reliable"] = zd["width_reliable"]
    c["width_source"] = f"zoom view at {1 / k:.2f} px/mm" if k else "zoom view"
    if k:
        e, _ = case.policy.nearest_edge(c["width_max_mm"])
        c["nearest_edge_mm"], c["ambiguous"] = e, case.policy.is_ambiguous(c["width_max_mm"], k)
    case.images[f"{visit}_zoom_{crack_id}"] = draw_analysis(az)
    return {"crack_id": crack_id, "found_in_zoom": True, "view": z.describe(), "before": before,
            "after": {kk: c[kk] for kk in before}, "width_reliable": c["width_reliable"],
            "ambiguous": c["ambiguous"]}


def t_compare_visits(case: Case) -> dict:
    p, c = case.visits["previous"], case.visits["current"]
    comp, warped, new = compare_visits(p.image, p.analysis, c.image, c.analysis)
    case.comparison = comp
    if comp.ok:
        case.images["comparison"] = draw_comparison(p.analysis, warped, new)
    d = comp.to_dict()
    d["registration"].pop("M", None)
    return d


def t_classify(case: Case) -> dict:
    cur = case.visits["current"]
    growth = bool(case.comparison and case.comparison.ok and case.comparison.significant_growth)
    if not cur.cracks:
        case.classification = {"band": "no_crack_detected", "width_max_mm": None, "needs_engineer": growth,
                               "growth": growth, "reinspect_days": 365, "policy": case.policy.label}
        return case.classification
    worst = max(cur.cracks, key=lambda c: c["width_max_mm"])
    band = case.policy.band_for(worst["width_max_mm"])
    case.classification = {
        "band": band.name, "action": band.action, "width_max_mm": worst["width_max_mm"], "worst_crack": worst["id"],
        "width_source": worst["width_source"], "growth": growth,
        "needs_engineer": case.policy.needs_engineer(worst["width_max_mm"], growth),
        "reinspect_days": band.reinspect_days, "policy": case.policy.label,
    }
    return case.classification


def t_request_reshoot(case: Case, visit: str, reasons: list[str], crack_id: int | None = None) -> dict:
    v = case.visits[visit]
    advice = list(v.quality.advice) if v.quality is not None and not v.quality.ok else []
    advice += [RESHOOT_ADVICE[r] for r in reasons if r in RESHOOT_ADVICE]
    case.reshoot = {"visit": visit, "reasons": reasons, "advice": advice, "crack_id": crack_id}
    if "blurry_marker" in reasons:  # so the UI can state the numbers behind the reshoot (the marker check, not the photo check)
        case.reshoot["max_edge_blur_px"] = MAX_EDGE_BLUR_PX
    case.status = "needs_reshoot"
    return case.reshoot


def t_request_approval(case: Case) -> dict:
    case.approval = {"state": "pending", "requested": utcnow(), "summary": summary_text(case)}
    case.status = "awaiting_approval"
    return case.approval


def t_close_case(case: Case, outcome: str = "") -> dict:
    days = (case.classification or {}).get("reinspect_days", 365)
    due = (datetime.now(timezone.utc) + timedelta(days=days)).date().isoformat()
    case.status = "closed"
    case.approval = None
    return {"outcome": outcome or (case.classification or {}).get("band", "closed"), "reinspect_by": due}


TOOLS = {
    "check_quality": t_check_quality, "detect_scale": t_detect_scale, "segment_measure": t_segment_measure,
    "zoom_measure": t_zoom_measure, "compare_visits": t_compare_visits, "classify": t_classify,
    "request_reshoot": t_request_reshoot, "request_approval": t_request_approval, "close_case": t_close_case,
}


def summary_text(case: Case) -> str:
    cl = case.classification or {}
    parts = [f"Worst crack max width {cl.get('width_max_mm')} mm -> band '{cl.get('band')}' ({cl.get('policy')})."]
    if case.comparison is not None and case.comparison.ok:
        c = case.comparison
        parts.append(f"Since previous visit: extension {c.extension:.1f} {c.unit}, width change "
                     f"{c.width_change:+.2f} {c.unit}.")
    return " ".join(parts)


# ---------------------------------------------------------------- loop

def new_case(current: np.ndarray, previous: np.ndarray | None = None, policy: SeverityPolicy | None = None,
             labels: dict | None = None) -> Case:
    visits = {"current": Visit("current", current)}
    if previous is not None:
        visits["previous"] = Visit("previous", previous)
    return Case(id=uuid.uuid4().hex[:12], visits=visits, policy=policy or SeverityPolicy(), labels=labels or {})


def _brief(result: dict, limit: int = 1200) -> dict:
    """Keep trace records small (no arrays)."""
    import json
    s = json.dumps(result, default=str)
    return result if len(s) <= limit else {"truncated": s[:limit]}


def run(case: Case, decider: Decider | None = None, max_steps: int = MAX_STEPS) -> Case:
    decider = decider or RulesDecider()
    t_start = time.perf_counter()
    while case.status not in TERMINAL and len(case.trace) < max_steps:
        obs = observe(case)
        allowed = allowed_actions(case)
        t0 = time.perf_counter()
        try:
            d: Decision = decider.decide(obs, allowed)
        except Exception as e:  # pragma: no cover - rules decider should not fail
            case.status = "error"
            case.trace.append({"step": len(case.trace) + 1, "error": f"decider failed: {e}"})
            break
        t1 = time.perf_counter()
        rec = {"step": len(case.trace) + 1, "t": utcnow(), "decider": d.decider, "action": d.action,
               "args": d.args, "rationale": d.rationale, "allowed": sorted(allowed),
               "decide_ms": round((t1 - t0) * 1000, 1)}
        if d.action not in allowed:  # never execute a disallowed action, whoever proposed it
            rec.update(result={"blocked": True}, status=case.status)
            case.trace.append(rec)
            case.status = "error"
            break
        try:
            result = TOOLS[d.action](case, **d.args)
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"}
            case.status = "error"
        rec.update(result=_brief(result), act_ms=round((time.perf_counter() - t1) * 1000, 1), status=case.status)
        case.trace.append(rec)
    if case.status not in TERMINAL:
        case.status = "error"
        case.trace.append({"step": len(case.trace) + 1, "error": "step budget exhausted"})
    case.labels["agent_total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
    return case


# ---------------------------------------------------------------- human gate

def approve(case: Case, reviewer: str, note: str = "") -> dict:
    """Human-only path to a ticket. Called from the UI/API, never from the decider."""
    if case.status != "awaiting_approval":
        raise ValueError(f"case is {case.status}, not awaiting approval")
    if not reviewer.strip():
        raise ValueError("reviewer name is required")
    cl = case.classification or {}
    case.approval = {**(case.approval or {}), "state": "approved", "by": reviewer.strip()[:80],
                     "note": note.strip()[:500], "at": utcnow()}
    due = (datetime.now(timezone.utc) + timedelta(days=cl.get("reinspect_days", 30))).date().isoformat()
    case.ticket = {"ticket_id": "T-" + case.id[:8].upper(), "case_id": case.id, "created": utcnow(),
                   "band": cl.get("band"), "width_max_mm": cl.get("width_max_mm"), "growth": cl.get("growth"),
                   "action": cl.get("action"), "review_by": due, "approved_by": case.approval["by"],
                   "summary": summary_text(case), "policy": cl.get("policy"),
                   "sink": "case record in the app store (S3 on AWS, 24 h expiry); no external notification"}
    case.status = "ticket_raised"
    case.trace.append({"step": len(case.trace) + 1, "t": utcnow(), "decider": "human", "action": "raise_ticket",
                       "args": {"reviewer": case.approval["by"]}, "rationale": note or "approved by engineer",
                       "result": case.ticket, "status": case.status})
    return case.ticket


def reject(case: Case, reviewer: str, note: str = "") -> dict:
    if case.status != "awaiting_approval":
        raise ValueError(f"case is {case.status}, not awaiting approval")
    if not reviewer.strip():
        raise ValueError("reviewer name is required")
    case.approval = {**(case.approval or {}), "state": "rejected", "by": reviewer.strip()[:80],
                     "note": note.strip()[:500], "at": utcnow()}
    case.status = "rejected"
    case.trace.append({"step": len(case.trace) + 1, "t": utcnow(), "decider": "human", "action": "reject_ticket",
                       "args": {"reviewer": case.approval["by"]}, "rationale": note or "rejected by engineer",
                       "result": {"ticket": None}, "status": case.status})
    return case.approval


def case_to_dict(case: Case) -> dict:
    return {
        "id": case.id, "created": case.created, "status": case.status, "labels": case.labels,
        "policy": case.policy.to_dict(),
        "visits": {n: {"quality": None if v.quality is None else v.quality.to_dict(),
                       "scale": None if v.scale is None else {k: x for k, x in v.scale.to_dict().items() if k != "H_mm"},
                       "analysis": None if v.analysis is None else
                       {k: x for k, x in v.analysis.to_dict().items() if k not in ("cracks", "quality", "scale")},
                       "cracks": v.cracks} for n, v in case.visits.items()},
        "comparison": None if case.comparison is None else case.comparison.to_dict(),
        "classification": case.classification, "reshoot": case.reshoot, "approval": case.approval,
        "ticket": case.ticket, "trace": case.trace, "images": sorted(case.images),
    }
