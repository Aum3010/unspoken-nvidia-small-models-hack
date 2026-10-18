"""UNSPOKEN_AUTO_SPEAK_DELAY_MS: a countdown before an accepted reading is auto-spoken. No (or a shake) inside the
window means nothing is spoken; no answer means it is spoken exactly once; the default (0) speaks at once.

Run: .venv/bin/python -m pytest evaluations/safety_tests -q
"""

import asyncio

import pytest

import services.gateway.app as gateway
from services.gateway.state import Session
from shared.schemas import Evidence, Hypothesis, Provenance, ToolCall

PROV = Provenance(service="test", checkpoint="test", preprocess="test")


def accepted_session() -> Session:
    s = Session(id="t")
    s.settings["auto_speak"] = True
    s.add_evidence(Evidence(modality="audio", status="proposed", provenance=PROV,
                            hypotheses=[Hypothesis(text="I need water", score=-1.0)]))
    return s


def plan(s: Session):
    gateway.apply_plan(s, ToolCall(name="ask_confirmation", rationale="x", by="deterministic_fallback",
                                   arguments={"candidate_ids": [s.message.candidates[0].id], "input_method": "button"}))


@pytest.fixture
def spoken(monkeypatch):
    calls = []

    async def fake_speak(s, ev, demo_auto=False):
        calls.append((s.message.confirmed_text, demo_auto))
        s.mark_spoken(0)
        return {}

    monkeypatch.setattr(gateway, "_speak", fake_speak)
    return calls


def run(coro):
    return asyncio.run(coro)


def test_no_inside_the_window_speaks_nothing(monkeypatch, spoken):
    monkeypatch.setattr(gateway, "AUTO_SPEAK_DELAY_MS", 80)

    async def go():
        s = accepted_session()
        plan(s)
        assert s.snapshot()["auto_speak_in_ms"] > 0
        await asyncio.sleep(0.02)
        s.reject_all()                      # what No / a shake does while a reading is pending
        await asyncio.sleep(0.2)
        return s
    s = run(go())
    assert spoken == [] and s.message.status == "rejected"


def test_no_answer_speaks_exactly_once(monkeypatch, spoken):
    monkeypatch.setattr(gateway, "AUTO_SPEAK_DELAY_MS", 50)

    async def go():
        s = accepted_session()
        plan(s)
        await asyncio.sleep(0.02)
        assert spoken == []                 # still counting down
        await asyncio.sleep(0.2)
        plan(s)                             # a late plan for the same message must not speak it again
        await asyncio.sleep(0.2)
        return s
    s = run(go())
    assert spoken == [("I need water", True)] and s.message.status == "spoken"


def test_yes_inside_the_window_is_not_doubled(monkeypatch, spoken):
    monkeypatch.setattr(gateway, "AUTO_SPEAK_DELAY_MS", 80)

    async def go():
        s = accepted_session()
        plan(s)
        s.select(s.message.candidates[0].id, "button")
        s.confirm("accessible_button_confirmed")   # the person said Yes first
        await asyncio.sleep(0.2)
        return s
    s = run(go())
    assert spoken == [] and s.message.status == "confirmed"


def test_default_has_no_countdown(monkeypatch, spoken):
    monkeypatch.setattr(gateway, "AUTO_SPEAK_DELAY_MS", 0)

    async def go():
        s = accepted_session()
        plan(s)
        snap = s.snapshot()
        await asyncio.sleep(0.05)
        return snap
    snap = run(go())
    assert snap["auto_speak_in_ms"] is None and spoken == [("I need water", True)]
