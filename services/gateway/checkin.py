"""Vision preview: a proactive check-in (UNSPOKEN_CHECKIN=1; off by default, and then nothing here is registered).

The system starts the conversation and the person answers with the same inputs they already use:
  1. the timeline shows "Nurse alerted (vision preview)" first;
  2. then a fixed question (no model) appears with the usual Yes / No;
  3. the answer (gaze, blink, nod/shake, click, Space) is logged as `checkin_response` and shown as
     "Nurse notified: patient answered YES/NO".
The answer is a reply to the nurse, not a message: it never goes through authorize_speech or TTS and never
creates a candidate or a message. If a message is pending or speech just played, the check-in waits until the
session is idle; while it is open, auto-speak is paused.

Data: data/vision/mimic_event.json (recorded, de-identified MIMIC-IV demo; scripts/vision/fetch_mimic_event.py) and
data/vision/profile.yaml (a SYNTHETIC profile). Both are labelled as such wherever they are shown.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shared.provenance import ROOT
from shared.schemas import new_id, now_ms

DATA = ROOT / "data" / "vision"
QUESTION = "Your nurse is coming to check on you. Do you feel shaky?"
BUSY = ("proposed", "ambiguous", "clarifying", "selected", "confirmed")
SPEECH_GRACE_MS = 3000      # treat speech as still playing this long after it was spoken
IDLE_POLL_S = 0.2
QUEUE_MAX_S = 600


def is_open(s) -> bool:
    return bool(s.checkin) and s.checkin.get("status") == "open"


def _idle(s) -> bool:
    if s.message.status in BUSY:
        return False
    last = s.spoken[-1]["ts"] if s.spoken else 0
    return now_ms() - last >= SPEECH_GRACE_MS


def load_data() -> dict:
    """What the panel shows: the recorded event (numbers only) and the synthetic profile."""
    out: dict = {"label": "Recorded ICU data (MIMIC-IV demo) · synthetic profile · vision preview"}
    ev = DATA / "mimic_event.json"
    if ev.exists():
        d = json.loads(ev.read_text())
        out["event"] = {k: d[k] for k in ("source", "glucose", "window_hours", "heart_rate", "spo2") if k in d}
    prof = DATA / "profile.yaml"
    if prof.exists():
        import yaml
        out["profile"] = yaml.safe_load(prof.read_text())
    return out


async def start(s, push) -> dict:
    """Nurse alert first, then the question (now, or once the session is idle)."""
    if s.checkin and s.checkin.get("status") in ("queued", "open"):
        return s.checkin
    g = (load_data().get("event") or {}).get("glucose") or {}
    detail = f"recorded glucose {g.get('value')} {g.get('unit', '')} (MIMIC-IV demo)" if g else "vision preview"
    s.checkin = {"id": new_id("chk"), "status": "queued", "question": QUESTION, "ts": now_ms(), "answer": None}
    s.trace("vision", "🔔 Nurse alerted (vision preview)", detail)
    s.log("system", "checkin_started", {"checkin_id": s.checkin["id"], "question": QUESTION})
    await push(s)
    if _idle(s):
        await _open(s, push)
    else:
        s.trace("vision", "Check-in waits until the current message is finished", level="pending")
        await push(s)
        asyncio.get_running_loop().create_task(_open_when_idle(s, push, s.checkin["id"]))
    return s.checkin


async def _open(s, push):
    s.timeline = [t for t in s.timeline if not (t["agent"] == "vision" and t["level"] == "pending")]
    s.checkin["status"] = "open"
    s.checkin["opened_ts"] = now_ms()
    s.trace("vision", "Check-in question shown", QUESTION)
    await push(s)


async def _open_when_idle(s, push, checkin_id: str):
    waited = 0.0
    while waited < QUEUE_MAX_S:
        if not s.checkin or s.checkin.get("id") != checkin_id or s.checkin.get("status") != "queued":
            return
        if _idle(s):
            await _open(s, push)
            return
        await asyncio.sleep(IDLE_POLL_S)
        waited += IDLE_POLL_S


async def answer(s, push, value: str, method: str) -> dict:
    if not is_open(s):
        raise HTTPException(409, "no open check-in")
    if value not in ("yes", "no"):
        raise HTTPException(400, "answer must be yes or no")
    s.checkin.update(status="answered", answer=value, method=method, answered_ts=now_ms())
    s.log("user", "checkin_response", {"checkin_id": s.checkin["id"], "question": s.checkin["question"],
                                       "answer": value, "method": method})
    s.trace("vision", f"Nurse notified: patient answered {value.upper()}", f"by {method.replace('_', ' ')}")
    await push(s)
    return s.checkin


class Answer(BaseModel):
    answer: str
    method: str = "accessible_button_confirmed"


def register(app: FastAPI, get_session, push):
    @app.get("/api/vision/data")
    async def vision_data():
        return load_data()

    @app.post("/api/{sid}/vision/checkin")
    async def vision_checkin(sid: str):
        s = get_session(sid)
        await start(s, push)
        return {"checkin": s.checkin, "state": s.snapshot()}

    @app.post("/api/{sid}/vision/checkin/answer")
    async def vision_checkin_answer(sid: str, body: Answer):
        s = get_session(sid)
        await answer(s, push, body.answer, body.method)
        return {"checkin": s.checkin, "state": s.snapshot()}
