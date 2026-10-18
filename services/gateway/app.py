"""Gateway: session state, specialist routing, coordinator calls, policy-gated speech.

The frontend talks only to this service (REST + one WebSocket per session).
Every optional specialist may be down; the phrase/spelling board keeps working.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time

import httpx
from fastapi import File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel

from services.gateway.state import Session, TransitionError
from shared.confirmation import PolicyViolation, authorize_speech, phrase_preload_token, sign
from shared.provenance import MODEL_LEDGER, ROOT, service_url
from shared.schemas import ALLOWED_CONFIRMATION_METHODS, Candidate, Evidence, ToolCall
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

_last_ready: dict[str, tuple[float, dict]] = {}
BUSY_GRACE_S = 120   # a ready service that is slow to answer is busy with a clip (the lip model can take ~30 s), not offline


async def _probe(client: httpx.AsyncClient, name: str) -> dict:
    try:
        r = await client.get(f"{service_url(name)}/health", timeout=1.5)
        h = r.json()
        if h.get("state") == "ready":
            _last_ready[name] = (time.time(), h)
        return h
    except httpx.TimeoutException:
        seen = _last_ready.get(name)
        if seen and time.time() - seen[0] < BUSY_GRACE_S:
            return {**seen[1], "busy": True}
        return {"service": name, "state": "unavailable", "error": "Timeout"}
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

# what each agent decided, passed to the coordinator with its readings (voice: services/audio, lips: services/vsr)
EVIDENCE_DETAIL = ("decision", "reason", "transcript", "margin", "mode", "enroll_best_dist", "enroll_max_dist", "enroll_negative")

async def coordinate(s: Session):
    m = s.message
    req = {
        "message_id": m.id,
        "candidates": [{"id": c.id, "text": c.text, "modality": c.modality, "score": c.score,
                        "supporting_modalities": c.supporting_modalities, "origin": c.origin} for c in m.candidates],
        "evidence": [{"modality": e.modality, "status": e.status, "quality_flags": e.quality_flags,
                      "top2_margin": e.quality.get("top2_margin"), "n_hypotheses": len(e.hypotheses),
                      "detail": {k: e.quality[k] for k in EVIDENCE_DETAIL if e.quality.get(k) is not None}}
                     for e in (s.evidence[i] for i in m.evidence_ids)],
        "available_inputs": s.settings["available_inputs"],
        "gaze_calibrated": s.settings["gaze_calibrated"],
        "person_activity": s.settings.get("person_activity", []),
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
    plan = validate(deterministic_plan(pr), pr)
    accepted = _accepted_reading(s)
    if accepted and not LLM_FOR_ACCEPTED and plan.valid and plan.name == "ask_confirmation" \
            and plan.arguments.get("candidate_ids") == [accepted.id]:
        # nothing for the model to resolve: act on the rules' plan now instead of waiting ~0.3 s for Nemotron
        apply_plan(s, plan)
        s.trace("coordinator", "Accepted reading: rules decide, Nemotron not needed",
                f"“{accepted.text}” · {' + '.join(accepted.supporting_modalities)}")
        return
    apply_plan(s, plan, quiet=True)
    s.trace("coordinator", "Nemotron is planning the next question", f"{len(m.candidates)} candidate(s) · {len(m.evidence_ids)} evidence event(s)", level="pending")
    task = asyncio.create_task(_refine_with_model(s, req, _plan_version(s)))
    _background.add(task)
    task.add_done_callback(_background.discard)


_background: set[asyncio.Task] = set()

# Nemotron is for ambiguous and unknown readings. When the specialists have already settled the reading (the voice
# agent accepted it, or lips and voice agree on it) the rules decide that case exactly as Nemotron is asked to, so the
# person does not wait for the model and the GPU does not run it. UNSPOKEN_LLM_FOR_ACCEPTED=1 restores the model call.
LLM_FOR_ACCEPTED = os.environ.get("UNSPOKEN_LLM_FOR_ACCEPTED", "0") != "0"


def _accepted_reading(s: Session):
    """The message's only candidate, if the specialists accepted it: every evidence event is `proposed` (no
    ambiguity, no lip/voice disagreement) and the voice agent supports it. A lip-only match still goes to the model."""
    m = s.message
    if m.status not in ("proposed", "clarifying") or len(m.candidates) != 1:
        return None
    c = m.candidates[0]
    evs = [s.evidence[e] for e in m.evidence_ids]
    if c.origin != "evidence" or not evs or any(e.status != "proposed" for e in evs):
        return None
    return c if "audio" in c.supporting_modalities else None


def _schedule(coro):
    task = asyncio.create_task(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)


async def _auto_speak(s: Session, message_id: str, candidate_id: str):
    """Demo auto-speak: say the coordinator's final reading, unless the person moved on or said No meanwhile."""
    m = s.message
    if m.id != message_id or m.status not in ("proposed", "ambiguous", "clarifying"):
        return
    ev = s.auto_confirm(candidate_id)
    try:
        await _speak(s, ev, demo_auto=True)
    except HTTPException:
        pass  # blocked: the policy gate already traced why


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
        if s.last_plan and _plan_version(s) == plan_version:
            apply_plan(s, s.last_plan)
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
    accepted = _accepted_reading(s)
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
    elif call.name == "propose_correction":
        # the corrected sentence becomes a candidate of its own, shown first; the readings stay in the audit
        src = [c for c in m.candidates if c.id in call.arguments["source_candidate_ids"]]
        c = Candidate(text=" ".join(str(call.arguments["text"]).split()), origin="correction", modality=src[0].modality,
                      evidence_event_id=src[0].evidence_event_id, source_mode=src[0].source_mode,
                      supporting_modalities=sorted({md for x in src for md in x.supporting_modalities}))
        m.candidates.append(c)
        s.shown_candidate_ids = [c.id, src[0].id]
        if m.status in ("proposed", "ambiguous"):
            m.status = "clarifying"
        s.log("coordinator", "correction_proposed", {"candidate_id": c.id, "text": c.text,
                                                       "from": [{"id": x.id, "text": x.text} for x in src]})
        s.trace("coordinator", f"{who}: propose_correction", f"“{src[0].text}” → “{c.text}” — {call.rationale}",
                latency_ms=call.latency_ms, data={"call": call.model_dump()})
    elif call.name == "ignore_input":
        # noise, not a message: drop the readings so they neither show nor pile up with the next clip
        s.new_message()
        s.last_plan = call
        s.trace("coordinator", f"{who}: ignore_input", f"{call.arguments.get('reason')} — {call.rationale}",
                latency_ms=call.latency_ms, data={"call": call.model_dump()})
    elif call.name == "switch_input_method":
        s.trace("coordinator", f"{who}: switch_input_method → {call.arguments.get('method')}", call.rationale,
                latency_ms=call.latency_ms, data={"call": call.model_dump()})
    # Demo auto-speak covers accepted readings only. Ambiguous readings, lip-only matches and corrections always wait
    # for the person's Yes/No.
    if call.name == "ask_confirmation" and s.settings["auto_speak"] and accepted and s.shown_candidate_ids == [accepted.id]:
        s.auto_speak_candidate_id = accepted.id
        _schedule(_auto_speak(s, m.id, accepted.id))


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


async def _infer(s: Session, service: str, field: str, filename: str, data: bytes, content_type: str, extra: dict | None = None) -> Evidence:
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{service_url(service)}/infer", files={field: (filename, data, content_type)},
                              data={"session_id": s.id, **(extra or {})})
    if r.status_code != 200:
        raise HTTPException(502, f"{service} error: {r.text[:200]}")
    return Evidence(**r.json())


async def _unavailable(s: Session, service: str, exc: Exception):
    s.timeline = [t for t in s.timeline if t["level"] != "pending"]
    s.trace(service, f"{service} unavailable", f"{type(exc).__name__}. The board is still available.", level="error")
    await push(s)
    raise HTTPException(503, f"{service} unavailable")


async def _apply_evidence(s: Session, service: str, ev: Evidence, label: str) -> dict:
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


async def _forward_capture(s: Session, service: str, field: str, upload: UploadFile, label: str, extra: dict | None = None):
    data = await upload.read()
    s.trace(service, f"{label} received", f"{len(data) / 1024:.0f} KB · processing", level="pending")
    await push(s)
    try:
        ev = await _infer(s, service, field, upload.filename or field, data, upload.content_type or "application/octet-stream", extra)
    except HTTPException:
        raise
    except Exception as exc:
        await _unavailable(s, service, exc)
    return await _apply_evidence(s, service, ev, label)


@app.post("/api/{sid}/capture/lip")
async def capture_lip(sid: str, video: UploadFile = File(...), moving_ms: int | None = Form(None)):
    # moving_ms: how long the lips moved in the clip, measured by the browser on the lip landmarks
    return await _forward_capture(get_session(sid), "vsr", "video", video, "Mouth clip",
                                  None if moving_ms is None else {"moving_ms": str(moving_ms)})


# Streamed mouth clips. While the person mouths, the browser sends the recording in 250 ms pieces over the
# session WebSocket; they are passed on to the lip agent at once, which decodes them and finds the face as they
# arrive. When the mouth has been still for the pause, the browser asks for the reading here; the upload and most
# of the preparation are already done. A stream that broke (a piece lost, the agent restarted) answers 409 and
# the browser sends the whole clip the usual way.
_lip_streams: dict[str, dict] = {}
LIP_STREAM_TTL_S = 60
_vsr_client: httpx.AsyncClient | None = None


def _lip_key(sid: str, stream_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", f"{sid}.{stream_id}")[:120]


def _lip_stream(key: str) -> dict:
    return _lip_streams.setdefault(key, {"n": 0, "broken": False, "t": time.time(), "cv": asyncio.Condition()})


async def _lip_chunk(sid: str, header: dict, payload: bytes):
    global _vsr_client
    now = time.time()
    for k in [k for k, v in _lip_streams.items() if now - v["t"] > LIP_STREAM_TTL_S]:
        _lip_streams.pop(k)
    key = _lip_key(sid, str(header.get("id", "")))
    st = _lip_stream(key)
    _vsr_client = _vsr_client or httpx.AsyncClient(timeout=10)
    if header.get("discard"):
        _lip_streams.pop(key, None)
        try:
            await _vsr_client.post(f"{service_url('vsr')}/stream/{key}/discard")
        except Exception:
            pass
        return
    if st["broken"] or header.get("seq") != st["n"]:
        st["broken"] = True
    else:
        try:
            r = await _vsr_client.post(f"{service_url('vsr')}/stream/{key}/chunk", content=payload,
                                       params={"mime": str(header.get("mime", "video/webm"))})
            st["broken"] = r.status_code != 200
        except Exception:
            st["broken"] = True
    st["n"] += 1
    st["t"] = now
    async with st["cv"]:
        st["cv"].notify_all()


class LipStreamFinish(BaseModel):
    stream_id: str
    chunks: int
    moving_ms: int | None = None   # lip-landmark movement time, as for a whole-clip upload


@app.post("/api/{sid}/capture/lip/stream")
async def capture_lip_stream(sid: str, req: LipStreamFinish):
    s = get_session(sid)
    key = _lip_key(sid, req.stream_id)
    st = _lip_stream(key)  # the last pieces may still be on their way over the WebSocket
    try:
        async with st["cv"]:
            await asyncio.wait_for(st["cv"].wait_for(lambda: st["n"] >= req.chunks or st["broken"]), 5)
    except asyncio.TimeoutError:
        st["broken"] = True
    _lip_streams.pop(key, None)
    if st["broken"]:
        try:
            await _vsr_client.post(f"{service_url('vsr')}/stream/{key}/discard")
        except Exception:
            pass
        raise HTTPException(409, "stream incomplete: send the whole clip")
    s.trace("vsr", "Mouth clip received", f"streamed in {req.chunks} pieces · processing", level="pending")
    await push(s)
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{service_url('vsr')}/stream/{key}/finish",
                                  data={"session_id": s.id, **({} if req.moving_ms is None else {"moving_ms": str(req.moving_ms)})})
    except Exception as exc:
        await _unavailable(s, "vsr", exc)
    if r.status_code == 404:
        s.timeline = [t for t in s.timeline if t["level"] != "pending"]
        raise HTTPException(409, "stream lost by the lip agent: send the whole clip")
    if r.status_code != 200:
        raise HTTPException(502, f"vsr error: {r.text[:200]}")
    return await _apply_evidence(s, "vsr", Evidence(**r.json()), "Mouth clip")


@app.post("/api/{sid}/capture/audio")
async def capture_audio(sid: str, audio: UploadFile = File(...), enhance: bool = Form(True)):
    return await _forward_capture(get_session(sid), "audio", "audio", audio, "Voice clip", {"enhance": str(enhance).lower()})


# Early voice decode. The browser cuts an utterance after 700 ms of silence; at 350 ms it sends the clip here
# and the voice agent starts decoding it *without* touching the session. If the person speaks again, the browser
# discards it and keeps recording (the utterance is never split differently than before). At 700 ms it commits,
# and the result, usually already finished, is applied exactly like a normal capture (same noise screening).
# The decode and the upload overlap the silence the person waits through anyway, instead of following it.
_early: dict[str, dict[str, asyncio.Task]] = {}
EARLY_MAX = 4  # per session; older ones are dropped


@app.post("/api/{sid}/capture/audio/early")
async def capture_audio_early(sid: str, audio: UploadFile = File(...)):
    s = get_session(sid)
    data = await audio.read()
    pending = _early.setdefault(sid, {})
    while len(pending) >= EARLY_MAX:
        pending.pop(next(iter(pending))).cancel()
    early_id = f"early_{time.time_ns():x}"
    pending[early_id] = asyncio.create_task(_infer(s, "audio", "audio", audio.filename or "clip.wav", data, audio.content_type or "audio/wav"))
    return {"early_id": early_id}


class EarlyBody(BaseModel):
    early_id: str


@app.post("/api/{sid}/capture/audio/commit")
async def capture_audio_commit(sid: str, body: EarlyBody):
    s = get_session(sid)
    task = _early.get(sid, {}).pop(body.early_id, None)
    if task is None:
        raise HTTPException(404, "unknown or expired early clip")
    if not task.done():
        s.trace("audio", "Voice clip received", "decoding (started during the pause)", level="pending")
        await push(s)
    try:
        ev = await task
    except HTTPException:
        raise
    except Exception as exc:
        await _unavailable(s, "audio", exc)
    return await _apply_evidence(s, "audio", ev, "Voice clip")


@app.post("/api/{sid}/capture/audio/discard")
async def capture_audio_discard(sid: str, body: EarlyBody):
    task = _early.get(sid, {}).pop(body.early_id, None)
    if task:
        task.cancel()
    return {"discarded": task is not None}


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
    return await _speak(s, ev)


async def _speak(s: Session, ev, demo_auto: bool = False) -> dict:
    sid, m = s.id, s.message
    try:
        token = authorize_speech(m, s.audit, ev.id, demo_auto=demo_auto)
    except PolicyViolation as exc:
        s.trace("policy", "Speech blocked", str(exc), level="error")
        s.log("policy", "speech_blocked", {"reason": str(exc)})
        await push(s)
        raise HTTPException(403, str(exc))
    s.log("policy", "speech_authorized", {"confirmation_event_id": ev.id, "text": m.confirmed_text})
    s.trace("policy", "Speech authorized", "auto-speak (demo) · no confirmation asked" if demo_auto
            else f"exact confirmed text · {m.confirmation_method.replace('_', ' ')}")
    await push(s)
    latency = None
    pre = _phrase_audio.get((m.confirmed_text, s.settings["voice"], s.settings["speech_rate"]))
    if pre:
        # the exact confirmed text is a pre-rendered phrase: the browser already holds this audio
        s.audio[m.id] = pre[1]
        s.trace("tts", "Spoke automatically" if demo_auto else "Spoke confirmed message",
                f"“{m.confirmed_text}” · pre-rendered phrase audio", latency_ms=0)
    else:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(f"{service_url('tts')}/speak", json={
                    "message_id": m.id, "text": m.confirmed_text, "token": token,
                    "voice": s.settings["voice"], "speed": s.settings["speech_rate"]})
            r.raise_for_status()
            s.audio[m.id] = r.content
            latency = int(r.headers.get("X-Latency-Ms", 0))
            s.trace("tts", "Spoke automatically" if demo_auto else "Spoke confirmed message", f"“{m.confirmed_text}”", latency_ms=latency)
        except Exception as exc:
            s.trace("tts", "TTS unavailable — browser voice used", type(exc).__name__, level="warn")
    s.mark_spoken(latency)
    if pre:
        s.spoken[-1]["phrase_audio_id"] = pre[0]
    msg_id = m.id
    await push(s)
    return {"message_id": msg_id, "audio_url": f"/api/{sid}/audio/{msg_id}" if msg_id in s.audio else None,
            "phrase_audio_id": pre[0] if pre else None, "text": s.spoken[-1]["text"], "state": s.snapshot()}


# ---- pre-rendered phrase audio ------------------------------------------------
# The fixed phrases (the voice agent's list, the person's recorded lip phrases, the board and Help) are rendered
# once and preloaded by the browser, so speaking one of them skips Kokoro and the audio download. Rendering is
# not speaking: the browser plays a phrase only when the gateway reports it spoken, which happens only after
# authorize_speech passed for exactly that text. Anything else (edited, spelled, corrected) still goes to Kokoro.
_phrase_audio: dict[tuple[str, str, float], tuple[str, bytes]] = {}
_phrase_audio_by_id: dict[str, bytes] = {}
_phrase_lock = asyncio.Lock()
PRELOAD_MAX, PRELOAD_TEXT_MAX = 64, 80


async def _fixed_phrases(client: httpx.AsyncClient) -> list[str]:
    texts = []
    try:
        import yaml
        texts += list(yaml.safe_load((ROOT / "asr_eval" / "configs" / "eval.yaml").read_text())["phrases"].values())
    except Exception:
        pass
    try:
        r = await client.get(f"{service_url('vsr')}/enroll", timeout=3)
        texts += [p["phrase"] for p in r.json().get("phrases", []) if not p["phrase"].startswith("[unknown]")]
    except Exception:
        pass
    return texts


class PreloadRequest(BaseModel):
    texts: list[str] = []


@app.post("/api/{sid}/phrase_audio")
async def phrase_audio_manifest(sid: str, req: PreloadRequest):
    """Render the fixed phrases for this session's voice and rate; returns their ids for the browser to fetch."""
    s = get_session(sid)
    voice, speed = s.settings["voice"], s.settings["speech_rate"]
    out = []
    async with _phrase_lock, httpx.AsyncClient(timeout=30) as client:
        texts = await _fixed_phrases(client) + [t for t in req.texts if len(t) <= PRELOAD_TEXT_MAX]
        for text in list(dict.fromkeys(" ".join(t.split()) for t in texts if t.strip()))[:PRELOAD_MAX]:
            key = (text, voice, speed)
            if key not in _phrase_audio:
                pid = sign("phrase-audio", f"{text}|{voice}|{speed}")[:16]
                try:
                    r = await client.post(f"{service_url('tts')}/speak", json={
                        "message_id": f"preload-{pid}", "text": text, "token": phrase_preload_token(pid, text),
                        "voice": voice, "speed": speed})
                    r.raise_for_status()
                except Exception:
                    continue
                _phrase_audio[key] = (pid, r.content)
                _phrase_audio_by_id[pid] = r.content
            out.append({"id": _phrase_audio[key][0], "text": text})
    s.log("gateway", "phrase_audio_preloaded", {"count": len(out), "voice": voice, "speed": speed})
    return {"phrases": out}


@app.get("/api/phrase_audio/{pid}")
async def phrase_audio(pid: str):
    if pid not in _phrase_audio_by_id:
        raise HTTPException(404)
    return Response(_phrase_audio_by_id[pid], media_type="audio/wav", headers={"Cache-Control": "private, max-age=86400"})


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
    for k in ("available_inputs", "gaze_calibrated", "person_activity", "use_llm", "voice", "speech_rate"):
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
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data:  # a piece of a streamed mouth clip: 2-byte header length, JSON header, then the bytes
                n = int.from_bytes(data[:2], "big")
                await _lip_chunk(sid, json.loads(data[2:2 + n]), data[2 + n:])
            # text messages are keepalive pings
    except WebSocketDisconnect:
        pass
    finally:
        sockets[sid].discard(websocket)
