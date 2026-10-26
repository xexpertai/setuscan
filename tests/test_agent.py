import json

import numpy as np
import pytest

from setuscan import agent
from setuscan.deciders import LLMDecider, RulesDecider
from setuscan.synth import SceneSpec, default_crack, render_scene


def scene(seed=3, width=1.5, **kw):
    rng = np.random.default_rng(seed)
    cracks = [] if width <= 0 else [default_crack(rng, width_mm=width, length_mm=130)]
    return render_scene(SceneSpec(seed=seed, cracks=cracks, **kw))[0]


@pytest.fixture(scope="module")
def wide_case():
    return agent.run(agent.new_case(scene(width=2.6)))


def actions(case):
    return [t.get("action") for t in case.trace]


def test_wide_crack_waits_for_engineer(wide_case):
    c = wide_case
    assert c.status == "awaiting_approval"
    assert actions(c)[:4] == ["check_quality", "detect_scale", "segment_measure", "classify"] or \
        "zoom_measure" in actions(c)
    assert c.ticket is None  # the agent itself never raises a ticket
    assert all(t["decider"] == "rules" for t in c.trace)
    assert c.classification["needs_engineer"]
    json.dumps(agent.case_to_dict(c), default=str)  # serialisable


def test_approval_gate():
    c = agent.run(agent.new_case(scene(width=2.6)))
    with pytest.raises(ValueError):
        agent.approve(c, "   ")  # reviewer name required
    t = agent.approve(c, "Engineer A", "site visit booked")
    assert c.status == "ticket_raised" and t["approved_by"] == "Engineer A"
    assert c.trace[-1]["decider"] == "human" and c.trace[-1]["action"] == "raise_ticket"
    with pytest.raises(ValueError):
        agent.approve(c, "Engineer A")  # cannot approve twice


def test_reject_path():
    c = agent.run(agent.new_case(scene(width=2.6)))
    agent.reject(c, "Engineer B", "joint, not a crack")
    assert c.status == "rejected" and c.ticket is None


def test_blurry_photo_asks_for_reshoot_before_measuring():
    c = agent.run(agent.new_case(scene(width=1.0, blur_sigma=5.0)))
    assert c.status == "needs_reshoot"
    assert c.reshoot["reasons"] == ["blurry_marker"]  # caught on the marker edges
    assert "segment_measure" not in actions(c)


def test_missing_marker_asks_for_marker():
    c = agent.run(agent.new_case(scene(width=1.0, marker_id=None)))
    assert c.status == "needs_reshoot"
    assert c.reshoot["reasons"] == ["no_marker"]


def test_fine_crack_closes_with_reinspection_date():
    c = agent.run(agent.new_case(scene(width=0.6)))
    assert c.status == "closed"
    assert "reinspect_by" in c.trace[-1]["result"]


def test_guardrail_close_not_allowed_when_engineer_needed(wide_case):
    allowed = agent.allowed_actions(wide_case)
    assert "close_case" not in allowed
    assert "raise_ticket" not in allowed  # no such tool exists for the decider


class _Disallowed:
    name = "bad"

    def decide(self, obs, allowed):
        from setuscan.deciders import Decision
        return Decision("close_case", {}, "skip the engineer", "bad")


def test_disallowed_action_is_blocked_not_executed():
    c = agent.new_case(scene(width=2.6))
    agent.run(c, _Disallowed())
    assert c.status == "error"
    assert c.trace[-1]["result"] == {"blocked": True}


def test_llm_decider_valid_output_is_used():
    # FAKE model (test double): always asks for the quality check first
    fake = lambda prompt: '{"action": "check_quality", "args": {"visit": "current"}, "rationale": "start"}'  # noqa: E731
    d = LLMDecider(fake, "fake-model")
    c = agent.new_case(scene(width=2.6))
    agent.run(c, d, max_steps=1)
    assert c.trace[0]["decider"] == "llm:fake-model"


@pytest.mark.parametrize("raw", ["not json", '{"action": "raise_ticket", "args": {}, "rationale": "x"}',
                                 '{"action": "check_quality", "args": {"visit": "elsewhere"}, "rationale": "x"}'])
def test_llm_decider_invalid_output_falls_back_to_rules(raw):
    d = LLMDecider(lambda p: raw, "fake-model")
    c = agent.run(agent.new_case(scene(width=2.6)), d)
    assert c.status == "awaiting_approval"
    assert all(t["decider"].startswith("rules (fallback") for t in c.trace)


def test_llm_transport_error_falls_back():
    def boom(prompt):
        raise TimeoutError("bedrock timeout")
    c = agent.run(agent.new_case(scene(width=0.6)), LLMDecider(boom, "fake-model"))
    assert c.status == "closed"
    assert "TimeoutError" in c.trace[0]["decider"]


def test_rules_decider_name():
    assert RulesDecider().name == "rules"
