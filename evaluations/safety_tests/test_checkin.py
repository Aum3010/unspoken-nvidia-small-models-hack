"""Vision preview check-in (UNSPOKEN_CHECKIN, off by default): absent when off; when on, the nurse alert comes
before the question, the answer never reaches the policy gate or TTS and creates no message, it waits while a
message is pending, and nothing is auto-spoken over it.

Run: .venv/bin/python -m pytest evaluations/safety_tests -q
"""

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import services.gateway.app as gateway
from services.gateway import checkin
from services.gateway.state import Session
from shared.schemas import Evidence, Hypothesis, Provenance, ToolCall, now_ms

PROV = Provenance(service="test", checkpoint="test", preprocess="test")


def ev(status, *texts):
    return Evidence(modality="audio", status=status, provenance=PROV,
                    hypotheses=[Hypothesis(text=t, score=-1.0 - i) for i, t in enumerate(texts)])


class Pushes:
    def __init__(self):
        self.snaps = []

    async def __call__(self, s):
        self.snaps.append(s.snapshot())


@pytest.fixture
def no_speech(monkeypatch):
    """Fail loudly if anything reaches the policy gate or TTS."""
    hits = []

    def gate(*a, **k):
        hits.append("authorize_speech")
        raise AssertionError("check-in reached authorize_speech")

    async def speak(*a, **k):
        hits.append("tts")
        raise AssertionError("check-in reached TTS")

    monkeypatch.setattr(gateway, "authorize_speech", gate)
    monkeypatch.setattr(gateway, "_speak", speak)
    return hits


def test_flag_off_has_no_routes():
    assert not gateway.CHECKIN
    c = TestClient(gateway.app)
    assert c.post("/api/t-off/vision/checkin").status_code == 404
    assert c.post("/api/t-off/vision/checkin/answer", json={"answer": "yes"}).status_code == 404
    assert c.get("/api/vision/data").status_code == 404
    assert "checkin" not in Session(id="t").snapshot()


def test_flag_on_routes_answer_without_speech(no_speech):
    app, sessions, push = FastAPI(), {}, Pushes()
    get = lambda sid: sessions.setdefault(sid, Session(id=sid))  # noqa: E731
    checkin.register(app, get, push)
    c = TestClient(app)
    r = c.post("/api/t1/vision/checkin")
    assert r.status_code == 200 and r.json()["checkin"]["status"] == "open"
    r = c.post("/api/t1/vision/checkin/answer", json={"answer": "yes", "method": "deliberate_blink_confirmed"})
    assert r.status_code == 200 and r.json()["checkin"]["answer"] == "yes"
    s = sessions["t1"]
    assert no_speech == []
    assert s.message.candidates == [] and s.spoken == [] and s.message.status == "collecting"
    assert any(e.kind == "checkin_response" and e.data["answer"] == "yes" for e in s.audit)
    assert any(t["title"] == "Nurse notified: patient answered YES" for t in s.timeline)
    assert c.post("/api/t1/vision/checkin/answer", json={"answer": "no"}).status_code == 409   # already answered


def test_nurse_alert_precedes_the_question():
    async def go():
        s, push = Session(id="t"), Pushes()
        await checkin.start(s, push)
        return s, push
    s, push = asyncio.run(go())
    titles = [t["title"] for t in s.timeline]
    assert titles.index("🔔 Nurse alerted (vision preview)") < titles.index("Check-in question shown")
    first = push.snaps[0]
    assert any(t["title"].startswith("🔔 Nurse alerted") for t in first["timeline"]) and first["checkin"]["status"] == "queued"
    assert push.snaps[-1]["checkin"]["status"] == "open"


def test_queued_while_a_message_is_pending(monkeypatch):
    monkeypatch.setattr(checkin, "IDLE_POLL_S", 0.02)

    async def go():
        s, push = Session(id="t"), Pushes()
        s.add_evidence(ev("ambiguous", "I need water", "I need suction"))
        assert s.message.status == "ambiguous"
        await checkin.start(s, push)
        queued = s.checkin["status"]
        await asyncio.sleep(0.1)
        still = s.checkin["status"]
        s.reject_all()                       # the person answers the pending message
        await asyncio.sleep(0.1)
        return queued, still, s.checkin["status"]
    assert asyncio.run(go()) == ("queued", "queued", "open")


def test_queued_while_speech_plays(monkeypatch):
    monkeypatch.setattr(checkin, "IDLE_POLL_S", 0.02)
    monkeypatch.setattr(checkin, "SPEECH_GRACE_MS", 150)

    async def go():
        s, push = Session(id="t"), Pushes()
        s.spoken.append({"message_id": "m", "text": "I need water", "ts": now_ms()})
        await checkin.start(s, push)
        queued = s.checkin["status"]
        await asyncio.sleep(0.3)
        return queued, s.checkin["status"]
    assert asyncio.run(go()) == ("queued", "open")


def test_nothing_is_auto_spoken_over_an_open_checkin(no_speech):
    async def go():
        s, push = Session(id="t"), Pushes()
        s.settings["auto_speak"] = True
        await checkin.start(s, push)
        s.add_evidence(ev("proposed", "I need water"))   # a stray accepted utterance
        gateway.apply_plan(s, ToolCall(name="ask_confirmation", rationale="x", by="deterministic_fallback",
                                       arguments={"candidate_ids": [s.message.candidates[0].id], "input_method": "button"}))
        await asyncio.sleep(0.05)
        return s
    s = asyncio.run(go())
    assert s.auto_speak_candidate_id is None and s.spoken == [] and no_speech == []
