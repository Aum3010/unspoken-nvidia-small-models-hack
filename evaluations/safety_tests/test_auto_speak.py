"""Demo auto-speak covers accepted readings only: ambiguous readings, lip-only matches, lip/voice disagreement
and the coordinator's corrections always wait for the person's Yes/No.

Run: .venv/bin/python -m pytest evaluations/safety_tests -q
"""

import asyncio

import pytest

import services.gateway.app as gateway
from services.gateway.state import Session
from shared.schemas import Evidence, Hypothesis, Provenance, ToolCall

PROV = Provenance(service="test", checkpoint="test", preprocess="test")


def ev(modality, status, *texts):
    return Evidence(modality=modality, status=status, provenance=PROV,
                    hypotheses=[Hypothesis(text=t, score=-1.0 - i) for i, t in enumerate(texts)])


def plan_and_check(*evidence, correction=False) -> bool:
    """Apply a plan to a session holding this evidence; True if auto-speak was scheduled."""
    scheduled = []

    async def run():
        s = Session(id="t")
        s.settings["auto_speak"] = True
        for e in evidence:
            s.add_evidence(e)
        m = s.message
        if correction:
            call = ToolCall(name="propose_correction", rationale="x", by="llm",
                            arguments={"text": "I need water now", "source_candidate_ids": [m.candidates[0].id], "input_method": "button"})
        else:
            call = ToolCall(name="ask_confirmation", rationale="x", by="deterministic_fallback",
                            arguments={"candidate_ids": [c.id for c in m.candidates[:3]], "input_method": "button"})
        gateway.apply_plan(s, call)
        scheduled.append(s.auto_speak_candidate_id is not None)

    original = gateway._auto_speak

    async def noop(*a, **k):
        return None

    gateway._auto_speak = noop
    try:
        asyncio.run(run())
    finally:
        gateway._auto_speak = original
    return scheduled[0]


def test_accepted_voice_reading_is_auto_spoken():
    assert plan_and_check(ev("audio", "proposed", "I need water"))


def test_lips_and_voice_agreeing_is_auto_spoken():
    assert plan_and_check(ev("lip", "proposed", "I need water"), ev("audio", "proposed", "I need water"))


@pytest.mark.parametrize("evidence", [
    [ev("audio", "ambiguous", "I need water", "I need suction")],
    [ev("lip", "proposed", "I need water")],
    [ev("lip", "proposed", "I'm cold"), ev("audio", "proposed", "I need water")],
    [ev("audio", "proposed", "I need water"), ev("lip", "ambiguous", "I need water", "I'm cold")],
], ids=["voice ambiguous", "lip only", "lip/voice disagree", "lip unsure"])
def test_everything_else_waits_for_yes_or_no(evidence):
    assert not plan_and_check(*evidence)


def test_corrections_are_never_auto_spoken():
    assert not plan_and_check(ev("audio", "proposed", "I need wader"), correction=True)
