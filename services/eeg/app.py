"""Brain-signal research replay: recorded EEG, newly run P300 inference.

Every response is labelled source_mode=prerecorded. No live brain input exists.
"""

from __future__ import annotations

import time

from fastapi import HTTPException
from pydantic import BaseModel

from services.eeg import p300
from shared.schemas import Evidence, Hypothesis, Provenance
from shared.service import Readiness, make_app

app = make_app("unspoken-eeg")
LABEL = "EEG RESEARCH REPLAY — previously recorded signal, newly run inference"


def _warm():
    subs = p300.available_subjects()
    if not subs:
        raise RuntimeError("no BNCI2014-008 files in data/bnci2014_008 (run scripts/setup_models.sh)")
    p300.train_and_evaluate(subs[0])
    return subs


model = Readiness("eeg", _warm)


@app.on_event("startup")
def _startup():
    model.start()


@app.get("/health")
def health():
    return {"service": "eeg", **model.health(), "label": LABEL, "dataset": "BNCI2014-008 (CC BY-NC-ND 4.0)",
            "checkpoint": "xDAWN+TangentSpace+LR trained per participant on calibration runs"}


@app.get("/subjects")
def subjects():
    out = []
    for s in p300.available_subjects():
        rec = p300.load(s)
        out.append({"subject": s, **rec.meta})
    return {"label": LABEL, "subjects": out}


@app.get("/subjects/{subject}")
def subject_summary(subject: str):
    if subject not in p300.available_subjects():
        raise HTTPException(404, "unknown subject")
    return {"label": LABEL, **p300.summary(subject)}


@app.get("/subjects/{subject}/trial/{char}")
def trial(subject: str, char: int, model_name: str = "xdawn_riemann_lr"):
    if subject not in p300.available_subjects() or not 0 <= char < 35:
        raise HTTPException(404, "unknown subject/char")
    return {"label": LABEL, **p300.trial_payload(subject, char, model_name)}


class EvidenceRequest(BaseModel):
    subject: str
    chars: list[int]
    model_name: str = "xdawn_riemann_lr"
    session_id: str = "demo-session"


@app.post("/evidence")
def evidence(req: EvidenceRequest):
    """Package decoded selections as typed evidence for the shared communication UI."""
    t0 = time.time()
    preds = [p300.trial_payload(req.subject, c, req.model_name)["prediction"] for c in req.chars]
    text = "".join(p["char"] for p in preds).replace("_", " ")
    return Evidence(
        session_id=req.session_id, modality="eeg_replay", source_mode="prerecorded",
        source_ref=f"BNCI2014-008/{req.subject}.mat#chars{req.chars}",
        hypotheses=[Hypothesis(text=text, score=min(p["margin"] for p in preds),
                               score_kind="min row/col evidence margin, uncalibrated")],
        quality_flags=["recorded_data"], quality={"per_char": preds, "label": LABEL},
        provenance=Provenance(service="eeg", checkpoint=req.model_name, preprocess="1-20 Hz bandpass, -0.1..0.7 s epochs, baseline",
                              latency_ms=int((time.time() - t0) * 1000)),
    )
