"""Gateway: session state, specialist routing, coordinator calls, policy-gated speech.

The frontend talks only to this service (REST + one WebSocket per session).
Every optional specialist may be down; the phrase/spelling board keeps working.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import httpx
from fastapi import File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel

from services.gateway.state import Session, TransitionError
from shared.confirmation import PolicyViolation, authorize_speech
from shared.provenance import MODEL_LEDGER, ROOT, service_url
from shared.schemas import ALLOWED_CONFIRMATION_METHODS, Evidence, ToolCall
from shared.service import make_app

app = make_app("unspoken-gateway")
SPECIALISTS = ["vsr", "audio", "eeg", "tts", "coordinator"]

sessions: dict[str, Session] = {}
sockets: dict[str, set[WebSocket]] = {}
_health_cache: dict = {"ts": 0, "data": {}}


def get_session(sid: str) -> Session:
    if sid not in sessions:
        sessions[sid] = Session(id=sid)
    return sessions[sid]


async def push(s: Session):
    snap = s.snapshot()
    dead = []
    for ws in sockets.get(s.id, set()):
        try:
            await ws.send_json({"type": "state", "state": snap})
        except Exception:
            dead.append(ws)
    for ws in dead:
        sockets[s.id].discard(ws)


# ---- health ---------------------------------------------------------------

async def _probe(client: httpx.AsyncClient, name: str) -> dict:
    try:
        r = await client.get(f"{service_url(name)}/health", timeout=1.5)
        return r.json()
    except Exception as exc:
        return {"service": name, "state": "unavailable", "error": type(exc).__name__}


@app.get("/api/health")
async def health():
    if time.time() - _health_cache["ts"] > 2:
        async with httpx.AsyncClient() as client:
            results = await asyncio.gather(*[_probe(client, n) for n in SPECIALISTS])
        _health_cache.update(ts=time.time(), data={n: r for n, r in zip(SPECIALISTS, results)})
    # where the agents run; the UI must say so when clips leave the person's device
    return {"gateway": {"state": "ready", "host": os.environ.get("UNSPOKEN_HOST_LABEL", "this device")}, **_health_cache["data"]}


@app.get("/api/ledger")
async def ledger():
    verified = ROOT / "evaluations" / "results" / "weights.json"
    if verified.exists():
        return json.loads(verified.read_text())
    return {"budget_b": 33.0, "components": MODEL_LEDGER, "verified": False}


@app.get("/api/evals")
async def evals():
    path = ROOT / "evaluations" / "results" / "summary.json"
    if not path.exists():
        raise HTTPException(404, "no results yet")
    return json.loads(path.read_text())


# ---- coordination -----------------------------------------------------------

async def coordinate(s: Session):
    m = s.message
    req = {
        "message_id": m.id,
        "candidates": [{"id": c.id, "text": c.text, "modality": c.modality, "score": c.score,
                        "supporting_modalities": c.supporting_modalities} for c in m.candidates],
        "evidence": [{"modality": e.modality, "status": e.status, "quality_flags": e.quality_flags,
                      "top2_margin": e.quality.get("top2_margin"), "n_hypotheses": len(e.hypotheses)}
                     for e in (s.evidence[i] for i in m.evidence_ids)],
        "available_inputs": s.settings["available_inputs"],
        "gaze_calibrated": s.settings["gaze_calibrated"],
        "history": s.history,
        "use_llm": s.settings["use_llm"],
    }
    # Instant deterministic plan so the person never waits on the model;
    # the coordinator model refines it when it answers.
    from services.coordinator.app import PlanRequest, deterministic_plan, validate
    pr = PlanRequest(**req)
    if not s.settings["use_llm"]:
        apply_plan(s, validate(deterministic_plan(pr), pr))
        return
    apply_plan(s, validate(deterministic_plan(pr), pr), quiet=True)
    s.trace("coordinator", "Nemotron is planning the next question", f"{len(m.candidates)} candidate(s) · {len(m.evidence_ids)} evidence event(s)", level="pending")
    task = asyncio.create_task(_refine_with_model(s, req, _plan_version(s)))
    _background.add(task)
    task.add_done_callback(_background.discard)


_background: set[asyncio.Task] = set()


def _plan_version(s: Session) -> tuple:
    user_events = sum(1 for e in s.audit if e.actor in ("user", "operator") and e.message_id == s.message.id)
    return s.message.id, len(s.message.evidence_ids), user_events


async def _refine_with_model(s: Session, req: dict, plan_version: tuple):
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(f"{service_url('coordinator')}/plan", json=req)
            call = ToolCall(**r.json())
    except Exception as exc:
        s.timeline = [t for t in s.timeline if t["level"] != "pending"]
        s.trace("coordinator", "Coordinator offline — deterministic policy kept", type(exc).__name__, level="warn")
        await push(s)
        return
    s.timeline = [t for t in s.timeline if t["level"] != "pending"]
    # The person may already have acted; never override their selection.
    now = _plan_version(s)
    if now[0] != plan_version[0] or now[2] != plan_version[2] or s.message.status in ("selected", "confirmed", "spoken"):
        s.trace("coordinator", "Model plan arrived after the person acted — discarded", call.rationale, latency_ms=call.latency_ms)
    elif now[1] != plan_version[1]:
        pass  # newer evidence arrived; a newer plan is already on its way
    else:
        apply_plan(s, call)
    await push(s)


def apply_plan(s: Session, call: ToolCall, quiet: bool = False):
    m = s.message
    s.last_plan = call
    who = "Nemotron coordinator" if call.by == "llm" else "Deterministic policy"
    rejected = (call.raw or {}).get("rejected_llm_call")
    if rejected:
        s.trace("policy", "Rejected coordinator tool call", "; ".join(rejected.get("validation_errors", [])), level="warn",
                data={"call": rejected})
    if (call.raw or {}).get("llm_error"):
        s.trace("coordinator", "Model unavailable, using fallback", call.raw["llm_error"], level="warn")
    s.log("coordinator", "tool_call", call.model_dump())
    if quiet:
        if call.name == "ask_confirmation":
            s.shown_candidate_ids = call.arguments["candidate_ids"]
            if m.status in ("proposed", "ambiguous"):
                m.status = "clarifying"
        return
    if call.name == "ask_confirmation":
        s.shown_candidate_ids = call.arguments["candidate_ids"]
        if m.status in ("proposed", "ambiguous"):
            m.status = "clarifying"
        s.trace("coordinator", f"{who}: ask_confirmation", f"{len(s.shown_candidate_ids)} option(s) via {call.arguments['input_method']} — {call.rationale}",
                latency_ms=call.latency_ms, data={"call": call.model_dump()})
    elif call.name == "request_retake":
        s.history.append("retake")
        s.trace("coordinator", f"{who}: request_retake", f"{call.arguments.get('reason')} — {call.rationale}", latency_ms=call.latency_ms,
                level="warn", data={"call": call.model_dump()})
    elif call.name == "ignore_input":
        # noise, not a message: drop the readings so they neither show nor pile up with the next clip
        s.new_message()
        s.last_plan = call
        s.trace("coordinator", f"{who}: ignore_input", f"{call.arguments.get('reason')} — {call.rationale}",
                latency_ms=call.latency_ms, data={"call": call.model_dump()})
    elif call.name == "switch_input_method":
        s.trace("coordinator", f"{who}: switch_input_method → {call.arguments.get('method')}", call.rationale,
                latency_ms=call.latency_ms, data={"call": call.model_dump()})


# ---- captures ---------------------------------------------------------------

# Capture is always on, so most clips are not messages. Evidence the specialists themselves mark unusable
# (the voice agent's VAD and frozen thresholds, the lip agent's face/motion checks) is kept in the audit log
# but never becomes a candidate and never prompts the person. A rule in code, not a model judgement.
UNUSABLE_FLAGS = {"no_face", "low_mouth_motion", "too_short", "no_speech_detected"}


def screen_evidence(ev: Evidence) -> str | None:
    """Why this evidence should be ignored, or None to let it through."""
    if ev.modality not in ("lip", "audio"):
        return None
    bad = UNUSABLE_FLAGS & set(ev.quality_flags)
    if bad:
        return ", ".join(sorted(f.replace("_", " ") for f in bad))
    if ev.status == "unknown" or not ev.hypotheses:
        return "no usable reading"
    return None


async def _forward_capture(s: Session, service: str, field: str, upload: UploadFile, label: str, extra: dict | None = None):
    data = await upload.read()
    s.trace(service, f"{label} received", f"{len(data) / 1024:.0f} KB · processing", level="pending")
    await push(s)
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{service_url(service)}/infer",
                                  files={field: (upload.filename or field, data, upload.content_type or "application/octet-stream")},
                                  data={"session_id": s.id, **(extra or {})})
        if r.status_code != 200:
            raise HTTPException(502, f"{service} error: {r.text[:200]}")
        ev = Evidence(**r.json())
    except HTTPException:
        raise
    except Exception as exc:
        s.timeline = [t for t in s.timeline if t["level"] != "pending"]
        s.trace(service, f"{service} unavailable", f"{type(exc).__name__}. The board is still available.", level="error")
        await push(s)
        raise HTTPException(503, f"{service} unavailable")
    s.timeline = [t for t in s.timeline if t["level"] != "pending"]
    ignored = screen_evidence(ev)
    if ignored:
        s.log("policy", "evidence_ignored", {"modality": ev.modality, "event_id": ev.event_id, "reason": ignored,
                                             "raw_hypotheses": [h.model_dump() for h in ev.hypotheses],
                                             "quality_flags": ev.quality_flags})
        s.trace("policy", f"Ignored {label.lower()}", ignored, latency_ms=ev.provenance.latency_ms,
                data={"event_id": ev.event_id})
        await push(s)
        return {"event": ev.model_dump(), "ignored": ignored, "state": s.snapshot()}
    s.add_evidence(ev)
    top = ev.hypotheses[0].text if ev.hypotheses else "—"
    s.trace(service, f"{len(ev.hypotheses)} raw hypothesis(es)", f"top: “{top}” · flags: {', '.join(ev.quality_flags) or 'none'}",
            latency_ms=ev.provenance.latency_ms, level="warn" if ev.quality_flags else "info",
            data={"event_id": ev.event_id})
    if s.agreement():
        a = s.agreement()
        s.trace("policy", "Cross-modal check", f"lip vs audio word overlap {a['word_overlap']:.0%}" +
                (" — disagreement kept, not merged" if a["word_overlap"] < 1 else " — agree"),
                level="warn" if a["word_overlap"] < 1 else "info")
    await coordinate(s)
    await push(s)
    return {"event": ev.model_dump(), "state": s.snapshot()}


@app.post("/api/{sid}/capture/lip")
async def capture_lip(sid: str, video: UploadFile = File(...)):
    return await _forward_capture(get_session(sid), "vsr", "video", video, "Mouth clip")


@app.post("/api/{sid}/capture/audio")
async def capture_audio(sid: str, audio: UploadFile = File(...), enhance: bool = Form(True)):
    return await _forward_capture(get_session(sid), "audio", "audio", audio, "Voice clip", {"enhance": str(enhance).lower()})


class EegCapture(BaseModel):
    subject: str
    chars: list[int]


@app.post("/api/{sid}/capture/eeg")
async def capture_eeg(sid: str, body: EegCapture):
    s = get_session(sid)
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{service_url('eeg')}/evidence", json={**body.model_dump(), "session_id": sid})
    if r.status_code != 200:
        raise HTTPException(502, r.text[:200])
    ev = Evidence(**r.json())
    s.add_evidence(ev)
    s.trace("eeg", "Recorded EEG replay decoded", f"“{ev.hypotheses[0].text}” · {ev.source_ref}", latency_ms=ev.provenance.latency_ms)
    await coordinate(s)
    await push(s)
    return s.snapshot()


# ---- user actions -----------------------------------------------------------

class TextBody(BaseModel):
    text: str
    origin: str = "board"


class SelectBody(BaseModel):
    candidate_id: str
    method: str = "accessible_button_confirmed"


class EditBody(BaseModel):
    candidate_id: str
    text: str


class ConfirmBody(BaseModel):
    method: str


def _guard(fn):
    try:
        return fn()
    except TransitionError as exc:
        raise HTTPException(409, str(exc))


@app.post("/api/{sid}/board")
async def board(sid: str, body: TextBody):
    s = get_session(sid)
    origin = body.origin if body.origin in ("board", "spelling") else "board"
    c = _guard(lambda: s.add_user_candidate(body.text, origin, "board"))
    s.select(c.id, origin)
    s.trace("user", f"{'Spelled' if origin == 'spelling' else 'Chose phrase'}", f"“{c.text}” — waiting for explicit confirmation")
    await push(s)
    return s.snapshot()


@app.post("/api/{sid}/select")
async def select(sid: str, body: SelectBody):
    s = get_session(sid)
    c = _guard(lambda: s.select(body.candidate_id, body.method))
    s.trace("user", "Selected", f"“{c.text}” via {body.method.replace('_', ' ')} — confirm to speak")
    await push(s)
    return s.snapshot()


@app.post("/api/{sid}/edit")
async def edit(sid: str, body: EditBody):
    s = get_session(sid)
    c = _guard(lambda: s.edit(body.candidate_id, body.text))
    s.select(c.id, "keyboard_confirmed")
    s.trace("user", "Corrected text", f"“{c.text}” (original kept in audit)")
    await push(s)
    return s.snapshot()


@app.post("/api/{sid}/confirm")
async def confirm(sid: str, body: ConfirmBody):
    s = get_session(sid)
    if body.method not in ALLOWED_CONFIRMATION_METHODS:
        raise HTTPException(400, "not an explicit confirmation method")
    ev = _guard(lambda: s.confirm(body.method))
    m = s.message
    try:
        token = authorize_speech(m, s.audit, ev.id)
    except PolicyViolation as exc:
        s.trace("policy", "Speech blocked", str(exc), level="error")
        s.log("policy", "speech_blocked", {"reason": str(exc)})
        await push(s)
        raise HTTPException(403, str(exc))
    s.log("policy", "speech_authorized", {"confirmation_event_id": ev.id, "text": m.confirmed_text})
    s.trace("policy", "Speech authorized", f"exact confirmed text · {body.method.replace('_', ' ')}")
    await push(s)
    latency = None
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(f"{service_url('tts')}/speak", json={
                "message_id": m.id, "text": m.confirmed_text, "token": token,
                "voice": s.settings["voice"], "speed": s.settings["speech_rate"]})
        r.raise_for_status()
        s.audio[m.id] = r.content
        latency = int(r.headers.get("X-Latency-Ms", 0))
        s.trace("tts", "Spoke confirmed message", f"“{m.confirmed_text}”", latency_ms=latency)
    except Exception as exc:
        s.trace("tts", "TTS unavailable — browser voice used", type(exc).__name__, level="warn")
    s.mark_spoken(latency)
    msg_id = m.id
    await push(s)
    return {"message_id": msg_id, "audio_url": f"/api/{sid}/audio/{msg_id}" if msg_id in s.audio else None,
            "text": s.spoken[-1]["text"], "state": s.snapshot()}


@app.get("/api/{sid}/audio/{message_id}")
async def audio(sid: str, message_id: str):
    s = get_session(sid)
    if message_id not in s.audio:
        raise HTTPException(404)
    return Response(s.audio[message_id], media_type="audio/wav")


@app.post("/api/{sid}/reject")
async def reject(sid: str):
    s = get_session(sid)
    s.reject_all()
    s.trace("user", "None of these", "all candidates rejected — nothing will be spoken")
    await coordinate(s)
    await push(s)
    return s.snapshot()


@app.post("/api/{sid}/undo")
async def undo(sid: str):
    s = get_session(sid)
    s.undo()
    s.trace("user", "Undo", "")
    await push(s)
    return s.snapshot()


@app.post("/api/{sid}/new")
async def new(sid: str):
    s = get_session(sid)
    s.new_message()
    await push(s)
    return s.snapshot()


@app.post("/api/{sid}/settings")
async def settings(sid: str, body: dict):
    s = get_session(sid)
    for k in ("available_inputs", "gaze_calibrated", "use_llm", "voice", "speech_rate"):
        if k in body:
            s.settings[k] = body[k]
    await push(s)
    return s.snapshot()


@app.post("/api/{sid}/safety/attempt_unconfirmed_speech")
async def attempt_unconfirmed(sid: str):
    """Demo: a coordinator tries to speak its own guess. The policy gate must refuse."""
    s = get_session(sid)
    m = s.message
    try:
        authorize_speech(m, s.audit, m.confirmation_event_id or "fabricated-event")
        outcome = "allowed"
    except PolicyViolation as exc:
        outcome = f"blocked: {exc}"
    s.trace("policy", "Unconfirmed speech attempt", outcome, level="error" if outcome.startswith("blocked") else "warn")
    s.log("policy", "safety_probe", {"outcome": outcome})
    await push(s)
    return {"outcome": outcome}


@app.get("/api/{sid}/audit")
async def audit(sid: str):
    return [e.model_dump() for e in get_session(sid).audit]


@app.get("/api/{sid}/state")
async def state(sid: str):
    return get_session(sid).snapshot()


# ---- EEG proxy (single origin for the browser) -------------------------------

@app.get("/api/eeg/{path:path}")
async def eeg_proxy(path: str, model_name: str | None = None):
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.get(f"{service_url('eeg')}/{path}", params={"model_name": model_name} if model_name else None)
    return Response(r.content, status_code=r.status_code, media_type="application/json")


# ---- websocket ------------------------------------------------------------------

@app.websocket("/ws/{sid}")
async def ws(websocket: WebSocket, sid: str):
    await websocket.accept()
    s = get_session(sid)
    sockets.setdefault(sid, set()).add(websocket)
    await websocket.send_json({"type": "state", "state": s.snapshot()})
    try:
        while True:
            await websocket.receive_text()  # keepalive pings
    except WebSocketDisconnect:
        sockets[sid].discard(websocket)
