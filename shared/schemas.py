"""Strict typed messages shared by every Unspoken service.

Mirrors plan.md §4. `score` values are raw model scores (log-likelihoods,
word probabilities, classifier margins) and are never presented as calibrated
probabilities.
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field

Modality = Literal["lip", "audio", "gaze", "gesture", "board", "eeg_replay"]
SourceMode = Literal["live", "prerecorded"]
EvidenceStatus = Literal["proposed", "ambiguous", "confirmed", "rejected", "unknown"]

QualityFlag = Literal[
    "no_face",
    "partial_face",
    "occlusion",
    "low_mouth_motion",
    "too_short",
    "too_long",
    "low_signal",
    "no_speech_detected",
    "low_word_confidence",
    "enhancer_disagrees",
    "out_of_distribution",
    "calibration_poor",
    "recorded_data",
]

ConfirmationMethod = Literal[
    "gaze_dwell_confirmed",
    "deliberate_blink_confirmed",
    "accessible_button_confirmed",
    "keyboard_confirmed",
    "other_explicit_user_confirmed",
]

ALLOWED_CONFIRMATION_METHODS: frozenset[str] = frozenset(ConfirmationMethod.__args__)  # type: ignore[attr-defined]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now_ms() -> int:
    return int(time.time() * 1000)


class Provenance(BaseModel):
    service: str
    checkpoint: str
    preprocess: str
    device: str = "cpu"
    latency_ms: int | None = None
    extra: dict = Field(default_factory=dict)


class Hypothesis(BaseModel):
    id: str = Field(default_factory=lambda: new_id("hyp"))
    text: str
    score: float
    score_kind: str = "raw"
    # optional per-word evidence, e.g. Whisper word probabilities
    words: list[dict] | None = None


class Evidence(BaseModel):
    """One specialist agent's output for one capture."""

    event_id: str = Field(default_factory=lambda: new_id("evt"))
    session_id: str = "demo-session"
    modality: Modality
    source_mode: SourceMode = "live"
    capture_time_ms: int = Field(default_factory=now_ms)
    source_ref: str = ""
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    quality_flags: list[QualityFlag] = Field(default_factory=list)
    quality: dict = Field(default_factory=dict)
    provenance: Provenance
    status: EvidenceStatus = "proposed"


class Candidate(BaseModel):
    """A message the user may pick. Text always originates from evidence or the user."""

    id: str = Field(default_factory=lambda: new_id("cand"))
    text: str
    origin: Literal["evidence", "user_edit", "board", "spelling", "eeg_replay"]
    evidence_event_id: str | None = None
    hypothesis_id: str | None = None
    modality: Modality
    source_mode: SourceMode = "live"
    score: float | None = None
    supporting_modalities: list[Modality] = Field(default_factory=list)


MessageStatus = Literal[
    "collecting", "proposed", "ambiguous", "clarifying", "selected", "confirmed", "spoken", "rejected"
]


class Message(BaseModel):
    id: str = Field(default_factory=lambda: new_id("msg"))
    session_id: str = "demo-session"
    status: MessageStatus = "collecting"
    evidence_ids: list[str] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    selected_candidate_id: str | None = None
    confirmed_text: str | None = None
    confirmation_method: ConfirmationMethod | None = None
    confirmation_event_id: str | None = None
    source_mode: SourceMode = "live"
    created_ms: int = Field(default_factory=now_ms)
    spoken_ms: int | None = None


class AuditEvent(BaseModel):
    id: str = Field(default_factory=lambda: new_id("aud"))
    ts_ms: int = Field(default_factory=now_ms)
    session_id: str = "demo-session"
    actor: Literal["user", "coordinator", "policy", "specialist", "system", "operator"]
    kind: str
    message_id: str | None = None
    data: dict = Field(default_factory=dict)


# ---- Coordinator tool calls (plan.md §4) ---------------------------------

InputMethod = Literal["gaze", "blink", "button", "keyboard", "spelling", "board"]


class ToolCall(BaseModel):
    name: Literal[
        "propose_candidates",
        "ask_confirmation",
        "switch_input_method",
        "request_retake",
        "append_confirmed_text",
        "speak_confirmed_text",
        "reset_or_undo",
        "ignore_input",
    ]
    arguments: dict
    rationale: str = ""
    raw: dict | None = None
    by: Literal["llm", "deterministic_fallback"] = "llm"
    latency_ms: int | None = None
    valid: bool = True
    validation_errors: list[str] = Field(default_factory=list)
