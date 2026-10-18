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
import re
import time
from pathlib import Path
from difflib import SequenceMatcher

import httpx
from pydantic import BaseModel, Field

from shared.schemas import ToolCall
from shared.service import make_app

BACKEND = os.environ.get("UNSPOKEN_LLM_BACKEND", "ollama")
BASE_URL = os.environ.get("UNSPOKEN_LLM_BASE_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("UNSPOKEN_LLM_MODEL", "nemotron-3.5-lightning:30b-a3b")
TIMEOUT_S = float(os.environ.get("UNSPOKEN_LLM_TIMEOUT", 20))
# Compact mode (OpenAI-compatible backends): the model answers with one short JSON object constrained by a schema
# ({"a":"ask","ids":[1],"m":"button"}) instead of a tool call with a rationale, and sees candidates numbered 1..n
# instead of their ids. About 15 output tokens instead of ~80. Code maps the answer back onto the same tools and
# ids before validation, and writes the note to the person from a template.
COMPACT = os.environ.get("UNSPOKEN_COORD_COMPACT", "0") != "0"
# Optional validator rules (off by default; measured in docs/RESULTS.md §5):
# - NEGATION: a propose_correction must keep its source reading's negation ("I don't need water" never becomes
#   "I need water"), using the voice agent's own negation check (asr_eval/harness/text.py, configured words).
# - IGNORE_HISTORY: ignore_input is rejected after two failed tries or a "none of these" in the last 3 turns, and
#   when a specialist accepted a lip/audio reading with no quality flags.
VALIDATOR_NEGATION = os.environ.get("UNSPOKEN_VALIDATOR_NEGATION", "0") != "0"
VALIDATOR_IGNORE_HISTORY = os.environ.get("UNSPOKEN_VALIDATOR_IGNORE_HISTORY", "0") != "0"
MAX_RATIONALE = 160

app = make_app("unspoken-coordinator")

SYSTEM_PROMPT = """You are the coordination agent inside Unspoken, an assistive communication system for people who cannot speak.
Specialist models (lip reading, weak-speech ASR, gaze, recorded-EEG replay) produced candidate messages. You decide the next
interaction step by calling exactly ONE tool. Hard rules:
- Lip reading and weak speech often get a word or two wrong ("i wore my family" for "i want my family", "i need wader"
  for "i need water"). When the best reading is close to a short, sensible thing a patient would say, call
  propose_correction with that sentence: keep the person's words, only fix the misread ones, never add new meaning.
  If the reading already makes sense, do not correct it. The person still confirms before anything is spoken.
- Otherwise you only reference candidate ids that exist.
- Each evidence item's detail says how sure that agent was; use it, not the raw scores alone. Scores of different
  agents are on different scales and must not be compared with each other.
  Voice: decision "accepted" passed its calibrated gates (a strong reading); "unknown" with reason "small_margin"
  means two phrases came close; transcript is what it actually heard (if the transcript is not the phrase, be wary).
  Lips: mode "enrollment" matched the person's own recordings; enroll_best_dist is the distance to the closest one
  (lower is closer; at or above enroll_max_dist it is not a phrase). A reading close to the cut-off is weak; one well
  under it is the person saying a phrase they recorded themselves, never noise, even with no voice.
  A reading supported by both lips and voice is much stronger than either alone.
- Prefer the smallest useful question: if one candidate clearly dominates, ask to confirm just it; if 2-4 are plausible, offer them.
- Capture is always on, so most clips are not messages: background sound, the person breathing, yawning or talking to
  someone else. If the readings look like noise, call ignore_input; staying silent is better than asking about noise.
  Signs of noise: a single modality with weak scores, fragments or filler words, lip and audio readings that share no words
  and are both weak, or text that is not something a person would say to be understood.
- Only request a retake when the person clearly tried to say something and the capture failed.
- If lip and audio candidates disagree, offer both so the person chooses; do not pick for them.
- If the person rejected all candidates before ("none_of_these" in history), switch to the spelling or phrase board.
- Choose an input method the person currently has available. Gaze is only usable if gaze calibration is good.
- person_activity says what the person is doing right now. While they are mouthing or using their voice
  ("mouthing", "voice"), their eyes wander and a look is not a choice: ask with blink or a button, not gaze.
  When person_activity is empty they are still and looking, and gaze is the easiest way for them to answer.
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
            "name": "propose_correction",
            "description": "Offer a corrected version of a misread candidate (same words, misread ones fixed) for the person to confirm.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_candidate_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
                    "text": {"type": "string", "description": "The corrected sentence, at most 12 words."},
                    "input_method": {"type": "string", "enum": ["gaze", "blink", "button", "keyboard"]},
                    "rationale": {"type": "string"},
                },
                "required": ["source_candidate_ids", "text", "input_method", "rationale"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ignore_input",
            "description": "Treat these readings as noise, not a message: nothing is shown and the person is not asked anything.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "enum": ["noise", "not_speech", "fragment", "unclear"]},
                    "rationale": {"type": "string"},
                },
                "required": ["reason", "rationale"],
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

COMPACT_PROMPT = """You are the coordination agent inside Unspoken, an assistive communication system for people who cannot speak.
Specialist models (lip reading, weak-speech ASR, gaze, recorded-EEG replay) produced numbered candidate messages. You choose the
next interaction step. Rules:
- Usually answer "ask". Use "fix" only when the best reading is NOT a sensible sentence but is one or two words away from a
  short thing a patient would say ("i wore my family" for "i want my family", "i need wader" for "i need water"): keep the
  person's words, only fix the misread ones, never add new meaning. Never fix a reading that already makes sense, never
  fix when lip and audio disagree (ask about both instead), and never fix after the person rejected readings.
- Only use candidate numbers that exist.
- Prefer the smallest useful question: if one candidate clearly dominates, ask about just it; if 2-4 are plausible, offer them.
- Capture is always on, so most clips are not messages: background sound, breathing, yawning or talking to someone else.
  If the readings look like noise, answer "ignore"; staying silent is better than asking about noise. Signs of noise: a single
  modality with weak scores, fragments or filler words, lip and audio readings that share no words and are both weak, or text
  that is not something a person would say to be understood.
- Only ask for a retake when the person clearly tried to say something and the capture failed.
- If lip and audio candidates disagree, offer both so the person chooses; do not pick for them.
- If the person rejected all candidates before ("none_of_these" in history), switch to spelling or the board.
- Use an input method the person has (inputs). Gaze only if gaze_ok is true.
Answer with one JSON object:
{"a":"ask","ids":[1],"m":"button"}            show candidates (1-4) and ask; m: gaze|blink|button|keyboard
{"a":"fix","ids":[1],"text":"...","m":"button"}  offer the corrected sentence of a misread candidate (max 12 words)
{"a":"retake","mod":"lip","why":"no_face"}      mod: lip|audio; why: no_face|low_mouth_motion|too_short|no_speech_detected|low_signal|occlusion
{"a":"switch","to":"spelling"}                  to: spelling|board|gaze|blink|button|lip|audio
{"a":"ignore","why":"noise"}                    why: noise|not_speech|fragment|unclear"""

_INPUTS = ["gaze", "blink", "button", "keyboard"]
COMPACT_SCHEMA = {"anyOf": [
    {"type": "object", "properties": {"a": {"enum": ["ask"]}, "ids": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 4},
                                      "m": {"enum": _INPUTS}}, "required": ["a", "ids", "m"], "additionalProperties": False},
    {"type": "object", "properties": {"a": {"enum": ["fix"]}, "ids": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 3},
                                      "text": {"type": "string", "maxLength": 120}, "m": {"enum": _INPUTS}},
     "required": ["a", "ids", "text", "m"], "additionalProperties": False},
    {"type": "object", "properties": {"a": {"enum": ["retake"]}, "mod": {"enum": ["lip", "audio"]},
                                      "why": {"enum": ["no_face", "low_mouth_motion", "too_short", "no_speech_detected", "low_signal", "occlusion"]}},
     "required": ["a", "mod", "why"], "additionalProperties": False},
    {"type": "object", "properties": {"a": {"enum": ["switch"]}, "to": {"enum": ["spelling", "board", "gaze", "blink", "button", "lip", "audio"]}},
     "required": ["a", "to"], "additionalProperties": False},
    {"type": "object", "properties": {"a": {"enum": ["ignore"]}, "why": {"enum": ["noise", "not_speech", "fragment", "unclear"]}},
     "required": ["a", "why"], "additionalProperties": False},
]}
COMPACT_NOTES = {  # the note to the person, written by code in compact mode
    "ask_confirmation": "Is this what you mean?",
    "propose_correction": "That reading looked slightly off. Is this what you mean?",
    "request_retake": "I could not read that. Please try once more.",
    "switch_input_method": "Let's try another way.",
    "ignore_input": "Looked like noise, not a message.",
}


class CandidateSummary(BaseModel):
    id: str
    text: str
    modality: str
    score: float | None = None
    supporting_modalities: list[str] = Field(default_factory=list)
    origin: str = "evidence"


class EvidenceSummary(BaseModel):
    modality: str
    status: str
    quality_flags: list[str] = Field(default_factory=list)
    top2_margin: float | None = None
    n_hypotheses: int = 0
    # how sure the agent itself was: voice decision/reason/transcript, lip mode and match distance (see prompt)
    detail: dict = Field(default_factory=dict)


class PlanRequest(BaseModel):
    message_id: str
    candidates: list[CandidateSummary]
    evidence: list[EvidenceSummary]
    available_inputs: list[str]
    gaze_calibrated: bool = False
    person_activity: list[str] = Field(default_factory=list)   # "mouthing", "voice": happening right now
    history: list[str] = Field(default_factory=list)
    use_llm: bool = True


# a lip match at or under this fraction of the cut-off is clear: held-out takes of a recorded phrase sat at
# 0.10-0.18 (median 0.13) against a 0.24 cut-off, strangers' clips at 0.247 and above
CLEAR_MATCH = 0.8
# ...or it is within the cut-off and ahead of every other phrase by this much (held-out takes of the long sentence
# led the next phrase by 0.19-0.25; takes of short look-alike phrases by as little as 0.05)
CLEAR_GAP = 0.08


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
        if method == "gaze" and req.person_activity:
            errors.append(f"person is {' and '.join(req.person_activity)} now, so a look is not a choice: ask with blink or a button")
    elif call.name == "switch_input_method":
        if a.get("method") not in ({"spelling", "board", "lip", "audio"} | set(req.available_inputs)):
            errors.append(f"method {a.get('method')!r} not available")
    elif call.name == "request_retake":
        if a.get("modality") not in ("lip", "audio"):
            errors.append("retake only for lip/audio")
    elif call.name == "propose_correction":
        errors += correction_errors(a, req)
        if VALIDATOR_NEGATION:
            errors += negation_errors(a, req)
    elif call.name == "ignore_input":
        # a reading two modalities agree on, or one from an explicit input, is a real message
        if any(len(set(c.supporting_modalities)) >= 2 for c in req.candidates):
            errors.append("lip and audio agree on a reading; it is not noise")
        if any(c.modality not in ("lip", "audio") for c in req.candidates):
            errors.append("only lip/audio readings can be ignored as noise")
        if VALIDATOR_IGNORE_HISTORY:
            errors += ignore_history_errors(req)
        # a clear match to the person's own recording is a message: the lip agent only reads a clip whose lips
        # really moved, and a match well inside the cut-off is not chance (strangers' clips stay above it)
        for e in req.evidence:
            d = e.detail
            best, cut, gap = d.get("enroll_best_dist"), d.get("enroll_max_dist"), d.get("enroll_gap")
            if e.modality == "lip" and d.get("mode") == "enrollment" and best is not None and cut and best <= cut \
                    and (best <= CLEAR_MATCH * cut or (gap is not None and gap >= CLEAR_GAP)):
                errors.append(f"a clear match to the person's own recording (distance {best:.2f} of {cut:.2f}"
                              f"{'' if gap is None else f', ahead of the next phrase by {gap:.2f}'}) is a message, not noise")
            if e.modality == "audio" and d.get("decision") == "accepted":
                errors.append("the voice agent accepted this reading; it is not noise")
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


_NEGATION_WORDS: set[str] | None = None


def _negated(text: str) -> bool:
    """The voice agent's negation check: asr_eval's normalisation (don't -> do not) and its configured words."""
    global _NEGATION_WORDS
    from asr_eval.harness.text import has_negation, normalize
    if _NEGATION_WORDS is None:
        try:
            import yaml
            cfg = yaml.safe_load((Path(__file__).resolve().parents[2] / "asr_eval" / "configs" / "eval.yaml").read_text())
            _NEGATION_WORDS = set(cfg["decision"]["negation_words"])
        except Exception:
            _NEGATION_WORDS = {"not", "no", "never"}
    return has_negation(normalize(text), _NEGATION_WORDS)


def negation_errors(a: dict, req: PlanRequest) -> list[str]:
    src = [c for c in req.candidates if c.id in (a.get("source_candidate_ids") or [])]
    fixed = _negated(str(a.get("text", "")))
    return ["correction adds or removes a negation"] if any(_negated(c.text) != fixed for c in src) else []


def ignore_history_errors(req: PlanRequest) -> list[str]:
    errors = []
    if req.history.count("retake") >= 2:
        errors.append("two tries already failed; offer another input method, not silence")
    if "none_of_these" in req.history[-3:]:
        errors.append("the person rejected the readings; switch input method, not silence")
    accepted = {e.modality for e in req.evidence if e.modality in ("lip", "audio") and e.status == "proposed" and not e.quality_flags}
    if any(c.modality in accepted for c in req.candidates):
        errors.append("a specialist accepted this reading with no quality flags; it is not noise")
    return errors


CORRECTION_MAX_WORDS = 12
WORD_MATCH = 0.6   # a corrected word "comes from" the reading if it is this similar to one of its words


def _words(t: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", t.lower()).split()


def correction_errors(a: dict, req: PlanRequest) -> list[str]:
    """A correction may fix misread words, never say something else: it stays anchored to what was read."""
    errors: list[str] = []
    by_id = {c.id: c for c in req.candidates}
    src = [by_id[i] for i in (a.get("source_candidate_ids") or []) if i in by_id]
    if not src or len(src) != len(a.get("source_candidate_ids") or []):
        errors.append("correction must name existing source candidates")
        return errors
    if any(c.modality not in ("lip", "audio") or c.origin != "evidence" for c in src):
        errors.append("only lip/audio readings are corrected; the person's own choices are not")
    words = _words(str(a.get("text", "")))
    if not words or len(words) > CORRECTION_MAX_WORDS:
        errors.append("correction must be 1-12 words")
        return errors
    read = {w for c in src for w in _words(c.text)}
    if any(" ".join(words) == " ".join(_words(c.text)) for c in src):
        errors.append("correction repeats the reading; ask_confirmation instead")
    kept = sum(1 for w in words if any(SequenceMatcher(None, w, r).ratio() >= WORD_MATCH for r in read))
    if kept < max(1, round(0.5 * len(words))):
        errors.append(f"correction strays from what was read ({kept}/{len(words)} words anchored)")
    if min(abs(len(words) - len(_words(c.text))) for c in src) > 2:
        errors.append("correction changes the length too much")
    method = a.get("input_method")
    if method not in req.available_inputs:
        errors.append(f"input method {method!r} not available")
    if method == "gaze" and not req.gaze_calibrated:
        errors.append("gaze not calibrated")
    if method == "gaze" and req.person_activity:
        errors.append(f"person is {' and '.join(req.person_activity)} now, so a look is not a choice: ask with blink or a button")
    return errors


def deterministic_plan(req: PlanRequest, why: str = "") -> ToolCall:
    """Safe rule-based policy; also the baseline in the evaluation ablation."""
    flags = {f for e in req.evidence for f in e.quality_flags}
    # gaze when the person is still and looking; while they mouth or use their voice, a blink or a button
    if req.gaze_calibrated and "gaze" in req.available_inputs and not req.person_activity:
        method = "gaze"
    else:
        method = next((m for m in ("blink", "button") if m in req.available_inputs), req.available_inputs[0])
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
        "person_activity": req.person_activity,
        "history": req.history[-6:],
    })


def _compact_payload(req: PlanRequest) -> str:
    def cand(n: int, c: CandidateSummary) -> dict:
        d = {"n": n, "text": c.text, "mod": c.modality}
        if c.score is not None:
            d["score"] = round(c.score, 2)
        if [m for m in c.supporting_modalities if m != c.modality]:
            d["also"] = [m for m in c.supporting_modalities if m != c.modality]
        if c.origin != "evidence":
            d["origin"] = c.origin
        return d

    def ev(e: EvidenceSummary) -> dict:
        d = {"mod": e.modality, "status": e.status, "n": e.n_hypotheses}
        if e.quality_flags:
            d["flags"] = e.quality_flags
        if e.top2_margin is not None:
            d["margin"] = round(e.top2_margin, 2)
        return d

    return json.dumps({"cands": [cand(i + 1, c) for i, c in enumerate(req.candidates)], "evidence": [ev(e) for e in req.evidence],
                       "inputs": req.available_inputs, "gaze_ok": req.gaze_calibrated, "history": req.history[-3:]},
                      separators=(",", ":"))


def _from_compact(o: dict, req: PlanRequest) -> tuple[str, dict]:
    """The compact answer as the tool call it stands for; a number with no candidate stays visibly invalid."""
    ids = [req.candidates[i - 1].id if isinstance(i, int) and 1 <= i <= len(req.candidates) else f"no_candidate_{i}"
           for i in o.get("ids") or []]
    a = o.get("a")
    if a == "ask":
        return "ask_confirmation", {"candidate_ids": ids, "input_method": o.get("m")}
    if a == "fix":
        return "propose_correction", {"source_candidate_ids": ids, "text": o.get("text", ""), "input_method": o.get("m")}
    if a == "retake":
        return "request_retake", {"modality": o.get("mod"), "reason": o.get("why")}
    if a == "switch":
        return "switch_input_method", {"method": o.get("to")}
    if a == "ignore":
        return "ignore_input", {"reason": o.get("why")}
    return str(a), {}


async def _call_compact(req: PlanRequest) -> tuple[str, dict, dict]:
    messages = [{"role": "system", "content": COMPACT_PROMPT}, {"role": "user", "content": _compact_payload(req)}]
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        r = await client.post(f"{BASE_URL}/v1/chat/completions", json={
            "model": MODEL, "messages": messages, "temperature": 0, "max_tokens": 60,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "step", "schema": COMPACT_SCHEMA}},
        })
        r.raise_for_status()
        body = r.json()
    name, args = _from_compact(json.loads(body["choices"][0]["message"]["content"]), req)
    args["rationale"] = COMPACT_NOTES.get(name, "")
    return name, args, {"usage": body.get("usage"), "compact": True}


async def _call_llm(req: PlanRequest) -> tuple[str, dict, dict]:
    if COMPACT and BACKEND != "ollama":
        return await _call_compact(req)
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
