"""Coordination agent: chooses *how to ask*, never *what the patient means*.

Receives structured evidence summaries (never raw video/EEG), calls a small
instruction model (NVIDIA Nemotron 3.5 Lightning by default) with narrowly typed
tools, validates every call outside the model, and falls back to a
deterministic policy when the model is slow, unavailable or invalid.

Backends (UNSPOKEN_LLM_BACKEND):
  ollama  -> native /api/chat (local, Metal)            [default]
  openai  -> any OpenAI-compatible server: vLLM on Brev, mlx_lm.server, llama.cpp
"""

from __future__ import annotations

import json
import os
import time

import httpx
from pydantic import BaseModel, Field

from shared.schemas import ToolCall
from shared.service import make_app

BACKEND = os.environ.get("UNSPOKEN_LLM_BACKEND", "ollama")
BASE_URL = os.environ.get("UNSPOKEN_LLM_BASE_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("UNSPOKEN_LLM_MODEL", "nemotron-3.5-lightning:30b-a3b")
TIMEOUT_S = float(os.environ.get("UNSPOKEN_LLM_TIMEOUT", 20))
MAX_RATIONALE = 160

app = make_app("unspoken-coordinator")

SYSTEM_PROMPT = """You are the coordination agent inside Unspoken, an assistive communication system for people who cannot speak.
Specialist models (lip reading, weak-speech ASR, gaze, recorded-EEG replay) produced candidate messages. You decide the next
interaction step by calling exactly ONE tool. Hard rules:
- You never write, rephrase, complete or translate the patient's message. You only reference candidate ids that exist.
- Prefer the smallest useful question: if one candidate clearly dominates, ask to confirm just it; if 2-4 are plausible, offer them.
- If quality flags show no face, no mouth motion, no speech or very short input, request a retake or switch input method.
- If lip and audio candidates disagree, offer both so the person chooses; do not pick for them.
- If the person rejected all candidates before ("none_of_these" in history), switch to the spelling or phrase board.
- Choose an input method the person currently has available. Gaze is only usable if gaze calibration is good.
- The rationale is a short note to the person (max 20 words) about WHY you ask, never a guess of their meaning.
Respond with one tool call and no other text."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ask_confirmation",
            "description": "Show 1-4 existing candidates and ask the person to choose one or reject all, using an accessible input method.",
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 4},
                    "input_method": {"type": "string", "enum": ["gaze", "blink", "button", "keyboard"]},
                    "rationale": {"type": "string"},
                },
                "required": ["candidate_ids", "input_method", "rationale"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_retake",
            "description": "Ask the person to try the same modality again because the capture quality was inadequate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "modality": {"type": "string", "enum": ["lip", "audio"]},
                    "reason": {"type": "string", "enum": ["no_face", "low_mouth_motion", "too_short", "no_speech_detected", "low_signal", "occlusion"]},
                    "rationale": {"type": "string"},
                },
                "required": ["modality", "reason", "rationale"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "switch_input_method",
            "description": "Move the person to a different way of expressing themselves.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {"type": "string", "enum": ["spelling", "board", "gaze", "blink", "button", "lip", "audio"]},
                    "rationale": {"type": "string"},
                },
                "required": ["method", "rationale"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "speak_confirmed_text",
            "description": "Request speech for a message the person already confirmed. Requires the confirmation event id.",
            "parameters": {
                "type": "object",
                "properties": {"confirmation_event_id": {"type": "string"}, "rationale": {"type": "string"}},
                "required": ["confirmation_event_id", "rationale"],
            },
        },
    },
]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


class CandidateSummary(BaseModel):
    id: str
    text: str
    modality: str
    score: float | None = None
    supporting_modalities: list[str] = Field(default_factory=list)


class EvidenceSummary(BaseModel):
    modality: str
    status: str
    quality_flags: list[str] = Field(default_factory=list)
    top2_margin: float | None = None
    n_hypotheses: int = 0


class PlanRequest(BaseModel):
    message_id: str
    candidates: list[CandidateSummary]
    evidence: list[EvidenceSummary]
    available_inputs: list[str]
    gaze_calibrated: bool = False
    history: list[str] = Field(default_factory=list)
    use_llm: bool = True


def validate(call: ToolCall, req: PlanRequest) -> ToolCall:
    """Every model-proposed call is checked here, outside the model."""
    errors: list[str] = []
    a = call.arguments
    ids = {c.id for c in req.candidates}
    if call.name not in TOOL_NAMES:
        errors.append(f"unknown tool {call.name}")
    if call.name == "ask_confirmation":
        cids = a.get("candidate_ids") or []
        if not cids or not isinstance(cids, list):
            errors.append("no candidate ids")
        unknown = [c for c in cids if c not in ids]
        if unknown:
            errors.append(f"invented candidate ids {unknown}")
        if len(cids) > 4:
            errors.append("too many candidates")
        method = a.get("input_method")
        if method not in req.available_inputs:
            errors.append(f"input method {method!r} not available")
        if method == "gaze" and not req.gaze_calibrated:
            errors.append("gaze not calibrated")
    elif call.name == "switch_input_method":
        if a.get("method") not in ({"spelling", "board", "lip", "audio"} | set(req.available_inputs)):
            errors.append(f"method {a.get('method')!r} not available")
    elif call.name == "request_retake":
        if a.get("modality") not in ("lip", "audio"):
            errors.append("retake only for lip/audio")
    # no forced consensus: when lips and voice disagree, the person sees both
    if call.name == "ask_confirmation" and not errors:
        cids = set(a.get("candidate_ids") or [])
        live = {c.modality for c in req.candidates if c.modality in ("lip", "audio")}
        offered = {c.modality for c in req.candidates if c.id in cids} | {
            m for c in req.candidates if c.id in cids for m in c.supporting_modalities}
        if len(live) > 1 and not live <= offered:
            errors.append(f"modalities disagree; offer a reading from each of {sorted(live)}")
    # history-aware product rules, kept in code rather than in the prompt
    if call.name == "request_retake" and req.history.count("retake") >= 2:
        errors.append("two retakes already failed; offer another input method instead")
    if call.name == "ask_confirmation" and "none_of_these" in req.history[-3:]:
        errors.append("person already rejected these readings; switch to spelling or the board")
    if call.name == "speak_confirmed_text":
        # The coordinator never holds a confirmation; the gateway policy gate re-checks anyway.
        errors.append("speak requested before a user confirmation event exists")
    rationale = str(call.rationale or a.get("rationale", ""))[:MAX_RATIONALE]
    call.rationale = rationale
    call.valid = not errors
    call.validation_errors = errors
    return call


def deterministic_plan(req: PlanRequest, why: str = "") -> ToolCall:
    """Safe rule-based policy; also the baseline in the evaluation ablation."""
    flags = {f for e in req.evidence for f in e.quality_flags}
    method = "gaze" if req.gaze_calibrated and "gaze" in req.available_inputs else (
        "button" if "button" in req.available_inputs else req.available_inputs[0])
    if "none_of_these" in req.history[-3:]:
        return ToolCall(name="switch_input_method", arguments={"method": "spelling"},
                        rationale="None of the guesses fit. Let's spell it.", by="deterministic_fallback")
    if not req.candidates:
        reason = next((f for f in ("no_face", "low_mouth_motion", "no_speech_detected", "too_short", "low_signal", "occlusion") if f in flags), "too_short")
        modality = "audio" if reason in ("no_speech_detected", "low_signal") else "lip"
        if req.history.count("retake") >= 2:
            return ToolCall(name="switch_input_method", arguments={"method": "board"},
                            rationale="Capture keeps failing. The phrase board is ready.", by="deterministic_fallback")
        return ToolCall(name="request_retake", arguments={"modality": modality, "reason": reason},
                        rationale=f"I could not read that ({reason.replace('_', ' ')}). Please try once more.",
                        by="deterministic_fallback")
    ranked = sorted(req.candidates, key=lambda c: (len(c.supporting_modalities), c.score or -1e9), reverse=True)
    margins = [e.top2_margin for e in req.evidence if e.top2_margin is not None]
    clear = len(ranked) == 1 or (margins and min(margins) > 3.0 and len(ranked[0].supporting_modalities) >= 1)
    chosen = ranked[:1] if clear else ranked[:3]
    for mod in ("lip", "audio"):  # keep one reading from every disagreeing modality
        if any(c.modality == mod for c in ranked) and not any(mod in {c.modality, *c.supporting_modalities} for c in chosen):
            best = next(c for c in ranked if c.modality == mod)
            chosen = (chosen[:-1] if len(chosen) >= 3 else chosen) + [best]
    note = "One clear reading. Is this what you mean?" if clear else "Several readings look similar. Which one, if any?"
    return ToolCall(name="ask_confirmation",
                    arguments={"candidate_ids": [c.id for c in chosen], "input_method": method},
                    rationale=note + (f" ({why})" if why else ""), by="deterministic_fallback")


def _user_payload(req: PlanRequest) -> str:
    return json.dumps({
        "candidates": [c.model_dump() for c in req.candidates],
        "evidence": [e.model_dump() for e in req.evidence],
        "available_inputs": req.available_inputs,
        "gaze_calibrated": req.gaze_calibrated,
        "history": req.history[-6:],
    })


async def _call_llm(req: PlanRequest) -> tuple[str, dict, dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": _user_payload(req)}]
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        if BACKEND == "ollama":
            r = await client.post(f"{BASE_URL}/api/chat", json={
                "model": MODEL, "messages": messages, "tools": TOOLS, "stream": False, "think": False,
                "options": {"temperature": 0, "num_predict": 200}, "keep_alive": "30m",
            })
            r.raise_for_status()
            body = r.json()
            calls = body["message"].get("tool_calls") or []
            if not calls:
                raise ValueError(f"model returned no tool call: {body['message'].get('content', '')[:120]!r}")
            fn = calls[0]["function"]
            args = fn["arguments"] if isinstance(fn["arguments"], dict) else json.loads(fn["arguments"])
            return fn["name"], args, {"eval_count": body.get("eval_count"), "total_s": (body.get("total_duration") or 0) / 1e9}
        r = await client.post(f"{BASE_URL}/v1/chat/completions", json={
            "model": MODEL, "messages": messages, "tools": TOOLS, "tool_choice": "auto",
            "temperature": 0, "max_tokens": 200, "chat_template_kwargs": {"enable_thinking": False},
        })
        r.raise_for_status()
        body = r.json()
        calls = body["choices"][0]["message"].get("tool_calls") or []
        if not calls:
            raise ValueError("model returned no tool call")
        fn = calls[0]["function"]
        return fn["name"], json.loads(fn["arguments"]), {"usage": body.get("usage")}


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            if BACKEND == "ollama":
                tags = (await client.get(f"{BASE_URL}/api/tags")).json()
                ok = any(m["name"] == MODEL for m in tags.get("models", []))
            else:
                ok = (await client.get(f"{BASE_URL}/v1/models")).status_code == 200
        state = "ready" if ok else "unavailable"
    except Exception:
        state = "unavailable"
    return {"service": "coordinator", "state": state, "backend": BACKEND, "model": MODEL, "base_url": BASE_URL,
            "fallback": "deterministic policy always available"}


@app.post("/plan")
async def plan(req: PlanRequest) -> ToolCall:
    t0 = time.time()
    if not req.use_llm:
        call = deterministic_plan(req)
        call.latency_ms = int((time.time() - t0) * 1000)
        return validate(call, req)
    try:
        name, args, meta = await _call_llm(req)
        rationale = str(args.pop("rationale", ""))
        call = ToolCall(name=name, arguments=args, rationale=rationale, raw=meta, by="llm")
        call.latency_ms = int((time.time() - t0) * 1000)
        call = validate(call, req)
        if call.valid:
            return call
        fallback = deterministic_plan(req, why="model call rejected")
        fallback.raw = {"rejected_llm_call": call.model_dump()}
    except Exception as exc:
        fallback = deterministic_plan(req, why="model unavailable")
        fallback.raw = {"llm_error": f"{type(exc).__name__}: {exc}"[:300]}
    fallback.latency_ms = int((time.time() - t0) * 1000)
    return validate(fallback, req)
