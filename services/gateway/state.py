"""Deterministic session state machine.

    collecting -> proposed | ambiguous -> clarifying -> selected -> confirmed -> spoken
                                   \\-> rejected (none of these) -> collecting

Transitions, candidate creation and confirmation are code. The coordinator can
only choose which existing candidates to show and which input method to use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from shared.schemas import AuditEvent, Candidate, Evidence, Message, ToolCall, new_id, now_ms


class TransitionError(Exception):
    pass


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9' ]", "", text.lower()).strip()


def word_overlap(a: str, b: str) -> float:
    wa, wb = set(norm(a).split()), set(norm(b).split())
    return len(wa & wb) / max(1, len(wa | wb))


@dataclass
class Session:
    id: str
    message: Message = field(default_factory=Message)
    evidence: dict[str, Evidence] = field(default_factory=dict)
    audit: list[AuditEvent] = field(default_factory=list)
    spoken: list[dict] = field(default_factory=list)
    timeline: list[dict] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    last_plan: ToolCall | None = None
    shown_candidate_ids: list[str] = field(default_factory=list)
    settings: dict = field(default_factory=lambda: {
        "available_inputs": ["button", "keyboard"],
        "gaze_calibrated": False,
        "use_llm": True,
        "voice": "af_heart",
        "speech_rate": 1.0,
    })
    audio: dict[str, bytes] = field(default_factory=dict)

    # ---- logging --------------------------------------------------------
    def log(self, actor, kind, data=None, message_id=None) -> AuditEvent:
        ev = AuditEvent(session_id=self.id, actor=actor, kind=kind, data=data or {}, message_id=message_id or self.message.id)
        self.audit.append(ev)
        return ev

    def trace(self, agent: str, title: str, detail: str = "", latency_ms: int | None = None, level: str = "info", data: dict | None = None):
        self.timeline.append({"id": new_id("tl"), "ts": now_ms(), "agent": agent, "title": title, "detail": detail,
                              "latency_ms": latency_ms, "level": level, "data": data or {}})
        self.timeline = self.timeline[-120:]

    # ---- evidence -> candidates ------------------------------------------
    def add_evidence(self, ev: Evidence) -> list[Candidate]:
        m = self.message
        if m.status in ("confirmed", "spoken", "rejected"):
            self.new_message()
            m = self.message
        if ev.source_mode == "prerecorded":
            m.source_mode = "prerecorded"
        self.evidence[ev.event_id] = ev
        m.evidence_ids.append(ev.event_id)
        self.log("specialist", "evidence_received", {"modality": ev.modality, "event_id": ev.event_id,
                                                     "raw_hypotheses": [h.model_dump() for h in ev.hypotheses],
                                                     "quality_flags": ev.quality_flags})
        added = []
        for h in ev.hypotheses:
            text = h.text if ev.modality != "eeg_replay" else f"Recorded EEG replay result: {h.text}"
            existing = next((c for c in m.candidates if norm(c.text) == norm(text)), None)
            if existing:
                if ev.modality not in existing.supporting_modalities:
                    existing.supporting_modalities.append(ev.modality)
                continue
            c = Candidate(text=text, origin="eeg_replay" if ev.modality == "eeg_replay" else "evidence",
                          evidence_event_id=ev.event_id, hypothesis_id=h.id, modality=ev.modality,
                          source_mode=ev.source_mode, score=h.score, supporting_modalities=[ev.modality])
            m.candidates.append(c)
            added.append(c)
        if m.candidates:
            m.status = "ambiguous" if self.is_ambiguous() else "proposed"
        return added

    def is_ambiguous(self) -> bool:
        evs = [self.evidence[e] for e in self.message.evidence_ids]
        if any(e.status in ("ambiguous", "unknown") for e in evs):
            return True
        tops = {e.modality: e.hypotheses[0].text for e in evs if e.hypotheses}
        if "lip" in tops and "audio" in tops and norm(tops["lip"]) != norm(tops["audio"]):
            return True  # no forced consensus across modalities
        return len(self.message.candidates) > 1

    def agreement(self) -> dict | None:
        evs = [self.evidence[e] for e in self.message.evidence_ids]
        tops = {e.modality: e.hypotheses[0].text for e in evs if e.hypotheses}
        if "lip" in tops and "audio" in tops:
            return {"lip": tops["lip"], "audio": tops["audio"], "word_overlap": round(word_overlap(tops["lip"], tops["audio"]), 2)}
        return None

    # ---- user actions -----------------------------------------------------
    def add_user_candidate(self, text: str, origin: str, modality: str = "board") -> Candidate:
        text = " ".join(text.split())
        if not text:
            raise TransitionError("empty text")
        m = self.message
        if m.status in ("confirmed", "spoken"):
            self.new_message()
            m = self.message
        c = Candidate(text=text, origin=origin, modality=modality, supporting_modalities=[modality])
        m.candidates.append(c)
        self.log("user", "user_text_entered", {"candidate_id": c.id, "text": text, "origin": origin})
        return c

    def select(self, candidate_id: str, method: str) -> Candidate:
        m = self.message
        c = next((c for c in m.candidates if c.id == candidate_id), None)
        if not c:
            raise TransitionError("unknown candidate")
        if m.status in ("confirmed", "spoken"):
            raise TransitionError("message already confirmed")
        m.selected_candidate_id = c.id
        m.status = "selected"
        self.log("user", "user_selected", {"candidate_id": c.id, "text": c.text, "method": method})
        return c

    def edit(self, candidate_id: str, text: str) -> Candidate:
        original = next((c for c in self.message.candidates if c.id == candidate_id), None)
        if not original:
            raise TransitionError("unknown candidate")
        c = self.add_user_candidate(text, "user_edit", original.modality)
        c.evidence_event_id = original.evidence_event_id
        c.source_mode = original.source_mode
        self.log("user", "user_edited", {"from_candidate": candidate_id, "from_text": original.text, "to_text": c.text})
        return c

    def confirm(self, method: str, actor: str = "user") -> AuditEvent:
        m = self.message
        if m.status != "selected" or not m.selected_candidate_id:
            raise TransitionError("select a message before confirming")
        c = next(c for c in m.candidates if c.id == m.selected_candidate_id)
        if c.source_mode == "prerecorded":
            actor = "operator"
        ev = self.log(actor, "user_confirmed", {"candidate_id": c.id, "text": c.text, "method": method})
        m.confirmed_text = c.text
        m.confirmation_method = method  # validated by the policy gate
        m.confirmation_event_id = ev.id
        m.status = "confirmed"
        return ev

    def reject_all(self, reason: str = "none_of_these"):
        m = self.message
        self.log("user", "user_rejected_all", {"reason": reason, "candidate_ids": [c.id for c in m.candidates]})
        self.history.append("none_of_these")
        m.selected_candidate_id = None
        m.status = "rejected"

    def undo(self):
        m = self.message
        if m.status == "selected":
            m.selected_candidate_id = None
            m.status = "ambiguous" if self.is_ambiguous() else ("proposed" if m.candidates else "collecting")
            self.log("user", "user_undo", {"undid": "selection"})
        elif m.status in ("proposed", "ambiguous", "clarifying", "rejected"):
            self.log("user", "user_undo", {"undid": "message", "candidates": [c.text for c in m.candidates]})
            self.new_message()
        elif m.status == "spoken" and self.spoken:
            last = self.spoken.pop()
            self.log("user", "user_undo", {"undid": "spoken_from_history", "text": last["text"]})

    def new_message(self):
        self.message = Message(session_id=self.id)
        self.shown_candidate_ids = []
        self.last_plan = None
        self.history = []  # retake / rejection history is per message

    def mark_spoken(self, latency_ms: int | None):
        m = self.message
        m.status = "spoken"
        m.spoken_ms = now_ms()
        c = next(c for c in m.candidates if c.id == m.selected_candidate_id)
        self.spoken.append({
            "message_id": m.id, "text": m.confirmed_text, "method": m.confirmation_method, "ts": m.spoken_ms,
            "modality": c.modality, "origin": c.origin, "source_mode": c.source_mode,
            "supporting_modalities": c.supporting_modalities, "tts_latency_ms": latency_ms,
            "raw": [{"modality": self.evidence[e].modality, "hypotheses": [h.text for h in self.evidence[e].hypotheses]}
                    for e in m.evidence_ids],
        })
        self.history.append("spoken")

    def snapshot(self) -> dict:
        m = self.message
        return {
            "session_id": self.id,
            "message": m.model_dump(),
            "evidence": [self.evidence[e].model_dump() for e in m.evidence_ids],
            "agreement": self.agreement(),
            "plan": self.last_plan.model_dump() if self.last_plan else None,
            "shown_candidate_ids": self.shown_candidate_ids,
            "spoken": self.spoken[-30:],
            "timeline": self.timeline[-60:],
            "settings": self.settings,
            "audit_count": len(self.audit),
        }
