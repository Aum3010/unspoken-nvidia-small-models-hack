"""Visual speech specialist: silent mouth video -> n-best raw VSR hypotheses.

Wraps Chaplin's Auto-AVSR pipeline (third_party/chaplin, LRS3_V_WER19.1) but
exposes the *raw* beam-search n-best list. Chaplin's LLM "correction" step is
deliberately not used: the coordinator may ask questions, never rewrite words.
"""

from __future__ import annotations

import base64
import os
import sys
import time

import cv2
import numpy as np
import torch
from fastapi import File, Form, UploadFile

from shared.provenance import ROOT
from shared.schemas import Evidence, Hypothesis, Provenance
from shared.service import Readiness, decode_video_rgb, make_app, pick_device

CHAPLIN = ROOT / "third_party" / "chaplin"
sys.path.insert(0, str(CHAPLIN))

MODEL_FPS = 25.0
NBEST = 5
BEAM = int(os.environ.get("UNSPOKEN_VSR_BEAM", 12))
# Mean absolute frame-to-frame change of the 88x88 mouth crop below which we
# flag "low_mouth_motion" (a still face makes Auto-AVSR hallucinate fluent text).
MOTION_THRESHOLD = float(os.environ.get("UNSPOKEN_VSR_MOTION_THRESHOLD", 2.2))

app = make_app("unspoken-vsr")


def _device() -> torch.device:
    return torch.device(pick_device("UNSPOKEN_VSR_DEVICE"))


def _load():
    from pipelines.data.transforms import VideoTransform
    from pipelines.detectors.mediapipe.detector import LandmarksDetector
    from pipelines.detectors.mediapipe.video_process import VideoProcess
    from pipelines.model import AVSR

    models = CHAPLIN / "benchmarks" / "LRS3"
    dev = _device()
    avsr = AVSR(
        "video",
        str(models / "models/LRS3_V_WER19.1/model.pth"),
        str(models / "models/LRS3_V_WER19.1/model.json"),
        str(models / "language_models/lm_en_subword/model.pth"),
        str(models / "language_models/lm_en_subword/model.json"),
        penalty=0.0, ctc_weight=0.1, lm_weight=0.3, beam_size=BEAM, device=dev,
    )
    n_params = sum(p.numel() for p in avsr.model.parameters())
    # warm-up so the first real request is not 2x slower
    with torch.no_grad():
        avsr.beam_search(avsr.model.encode(torch.zeros(1, 12, 88, 88).to(dev)))
    return {
        "avsr": avsr,
        "detector": LandmarksDetector(),
        "process": VideoProcess(convert_gray=True),
        "transform": VideoTransform(speed_rate=1),
        "device": str(dev),
        "params": n_params,
    }


model = Readiness("vsr", _load)


@app.on_event("startup")
def _startup():
    model.start()


@app.get("/health")
def health():
    h = model.health()
    if model.state == "ready":
        h.update(device=model.value["device"], params=model.value["params"])
    return {"service": "vsr", **h, "checkpoint": "Auto-AVSR LRS3_V_WER19.1 (Chaplin)", "beam": BEAM}


def _thumbs(crops: np.ndarray, n: int = 8) -> list[str]:
    idx = np.linspace(0, len(crops) - 1, min(n, len(crops))).astype(int)
    out = []
    for i in idx:
        ok, buf = cv2.imencode(".jpg", crops[i], [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            out.append("data:image/jpeg;base64," + base64.b64encode(buf).decode())
    return out


def _nbest_texts(avsr, hyps) -> list[Hypothesis]:
    seen, out = set(), []
    for h in hyps:
        d = h.asdict()
        tokens = [avsr.token_list[int(i)] for i in d["yseq"][1:-1]]
        text = "".join(tokens).replace("▁", " ").replace("<eos>", "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(Hypothesis(
            text=text,
            score=round(float(d["score"]), 3),
            score_kind="beam log-likelihood (CTC+attention+LM), uncalibrated",
            words=[{"word": w} for w in text.split()],
        ))
        if len(out) >= NBEST:
            break
    return out


@app.post("/infer")
def infer(video: UploadFile = File(...), session_id: str = Form("demo-session"), source_ref: str = Form("")):
    t0 = time.time()
    m = model.get()
    data = video.file.read()
    frames, duration = decode_video_rgb(data, MODEL_FPS)
    flags: list[str] = []
    quality: dict = {"duration_s": round(duration, 2), "frames": int(len(frames))}
    prov = Provenance(
        service="vsr", checkpoint="Auto-AVSR LRS3_V_WER19.1", device=m["device"],
        preprocess="pyav-25fps -> mediapipe face det -> mean-face similarity warp -> 96px mouth ROI -> centercrop 88",
    )

    if len(frames) < 0.6 * MODEL_FPS:
        flags.append("too_short")
        return _evidence(session_id, source_ref, [], flags, quality, prov, t0)
    if duration > 12:
        flags.append("too_long")

    with model.lock:
        landmarks = m["detector"].detect(frames, m["detector"].full_range_detector)
        if all(lm is None for lm in landmarks):
            landmarks = m["detector"].detect(frames, m["detector"].short_range_detector)
        detected = sum(lm is not None for lm in landmarks) / len(landmarks)
        quality["face_detected_ratio"] = round(detected, 3)
        if detected == 0:
            flags.append("no_face")
            return _evidence(session_id, source_ref, [], flags, quality, prov, t0)
        if detected < 0.9:
            flags.append("partial_face")

        try:
            crops = m["process"](frames, landmarks)
        except Exception as exc:
            flags.append("occlusion")
            quality["crop_error"] = str(exc)
            return _evidence(session_id, source_ref, [], flags, quality, prov, t0)

        x = m["transform"](torch.tensor(crops))
        center = crops[:, 4:92, 4:92].astype(np.float32)
        motion = float(np.abs(np.diff(center, axis=0)).mean()) if len(center) > 1 else 0.0
        quality["mouth_motion"] = round(motion, 3)
        quality["mouth_motion_threshold"] = MOTION_THRESHOLD
        if motion < MOTION_THRESHOLD:
            flags.append("low_mouth_motion")

        t_inf = time.time()
        with torch.no_grad():
            enc = m["avsr"].model.encode(x.to(m["avsr"].device))
            hyps = m["avsr"].beam_search(enc)
        quality["decode_ms"] = int((time.time() - t_inf) * 1000)

    hypotheses = _nbest_texts(m["avsr"], hyps)
    if len(hypotheses) >= 2:
        quality["top2_margin"] = round(hypotheses[0].score - hypotheses[1].score, 3)
    quality["mouth_thumbs"] = _thumbs(crops)
    return _evidence(session_id, source_ref, hypotheses, flags, quality, prov, t0)


def _evidence(session_id, source_ref, hyps, flags, quality, prov, t0) -> Evidence:
    prov.latency_ms = int((time.time() - t0) * 1000)
    status = "proposed" if hyps and not {"no_face", "low_mouth_motion"} & set(flags) else "unknown"
    if hyps and len(hyps) > 1 and quality.get("top2_margin", 99) < 1.5:
        status = "ambiguous"
    return Evidence(
        session_id=session_id, modality="lip", source_mode="live", source_ref=source_ref or "browser-camera-clip",
        hypotheses=hyps, quality_flags=flags, quality=quality, provenance=prov, status=status,
    )
