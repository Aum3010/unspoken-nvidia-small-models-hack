"""Visual speech specialist: silent mouth video -> n-best raw VSR hypotheses.

Wraps Chaplin's Auto-AVSR pipeline (third_party/chaplin, LRS3_V_WER19.1) but
exposes the *raw* beam-search n-best list. Chaplin's LLM "correction" step is
deliberately not used: the coordinator may ask questions, never rewrite words.
"""

from __future__ import annotations

import base64
import os

# before numpy loads: OpenBLAS threads contend with the models sharing the CPU and made enrollment matching
# 10-30 s a clip (0.1 s single-threaded); run_all.sh sets the same for every service
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import File, Form, HTTPException, Request, UploadFile

from shared.provenance import ROOT
from shared.schemas import Evidence, Hypothesis, Provenance
from shared.service import Readiness, decode_video_rgb, make_app, pick_device
from services.vsr.enroll import EnrollmentStore
from services.vsr.phrases import score_phrases, tokenize

CHAPLIN = ROOT / "third_party" / "chaplin"
sys.path.insert(0, str(CHAPLIN))

MODEL_FPS = 25.0
NBEST = 5
BEAM = int(os.environ.get("UNSPOKEN_VSR_BEAM", 12))
# Mean absolute frame-to-frame change of the 88x88 mouth crop below which we
# flag "low_mouth_motion" (a still face makes Auto-AVSR hallucinate fluent text).
MOTION_THRESHOLD = float(os.environ.get("UNSPOKEN_VSR_MOTION_THRESHOLD", 2.2))
# Optional closed phrase list: its best matches are added as candidates (see services/vsr/phrases.py).
PHRASES_PATH = os.environ.get("UNSPOKEN_VSR_PHRASES", "")
PHRASE_TOP = 3
# Optional fine-tuned decoder + CTC head (scripts/finetune_vsr_grid.py). The GRID one only knows GRID words.
DECODER_PATH = os.environ.get("UNSPOKEN_VSR_DECODER", "")
# Free-text beam search (any sentence) runs only when a clip matches none of the recorded phrases (and no
# phrase list is set). Off by default once phrases are recorded (UNSPOKEN_VSR_FREE_TEXT=1 turns it on): an
# unmatched clip then returns no reading at once, instead of a 20-30 s free-text guess that holds up every
# clip behind it under always-on capture.
FREE_TEXT = os.environ.get("UNSPOKEN_VSR_FREE_TEXT", "0") != "0"
# Personal enrollment: a clip farther than this (length-normalised DTW over cosine distance) from every
# recording is not one of the person's phrases. On the first 10 recordings, a held-out take of the same
# phrase was 0.13-0.28 from its others, another recorded phrase >= 0.22, an unrelated GRID clip ~0.33.
ENROLL_MAX_DIST = float(os.environ.get("UNSPOKEN_ENROLL_MAX_DIST", 0.30))
# Enrollment scores are -100 x distance, so the shared margin rules apply: the closest phrase is offered
# alone when it beats the next by > 3 (0.03 distance; held-out takes beat other phrases by >= 0.048).
ENROLL_SCALE = 100.0

app = make_app("unspoken-vsr")
enrolled = EnrollmentStore()
SUGGESTIONS = Path(__file__).with_name("patient_phrases.txt")
NEGATIVES = Path(__file__).with_name("negative_phrases.txt")
# Recordings of near-miss sentences ("I don't need water") are stored under this prefix. When a clip's
# closest recording is one of them, the clip is not one of the person's phrases: no reading is offered.
NEG_PREFIX = "[unknown] "


def _lines(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]


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
    if DECODER_PATH:
        ck = torch.load(DECODER_PATH, map_location="cpu")
        avsr.model.decoder.load_state_dict(ck["decoder"])
        avsr.model.ctc.load_state_dict(ck["ctc"])
    phrases = []
    if PHRASES_PATH:
        piece_to_id = {p: i for i, p in enumerate(avsr.token_list)}
        lines = [l.strip() for l in open(PHRASES_PATH) if l.strip() and not l.lstrip().startswith("#")]
        phrases = [(l, tokenize(l, piece_to_id)) for l in lines]
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
        "phrases": phrases,
    }


model = Readiness("vsr", _load)


@app.on_event("startup")
def _startup():
    model.start()


@app.get("/health")
def health():
    h = model.health()
    if model.state == "ready":
        h.update(device=model.value["device"], params=model.value["params"], phrase_list=len(model.value["phrases"]),
                 enrolled_phrases=len(enrolled.phrases()))
    return {"service": "vsr", **h, "checkpoint": "Auto-AVSR LRS3_V_WER19.1 (Chaplin)", "beam": BEAM, "free_text": FREE_TEXT,
            "decoder": DECODER_PATH or "original"}


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


def _encode(m, data: bytes, flags: list[str], quality: dict):
    """Clip bytes -> (encoder features, mouth crops), or (None, None) with a quality flag. Caller holds model.lock."""
    try:
        frames, duration = decode_video_rgb(data, MODEL_FPS)
    except Exception as exc:  # empty or corrupt upload
        flags.append("too_short")
        quality["decode_error"] = type(exc).__name__
        return None, None
    return _encode_frames(m, frames, duration, None, flags, quality)


def _encode_frames(m, frames: np.ndarray, duration: float, landmarks: list | None, flags: list[str], quality: dict):
    """Decoded frames (and their face landmarks, when a streamed clip already found them) -> as `_encode`."""
    quality.update(duration_s=round(duration, 2), frames=int(len(frames)))
    if len(frames) < 0.6 * MODEL_FPS:
        flags.append("too_short")
        return None, None
    if duration > 12:
        flags.append("too_long")
    if landmarks is None:
        landmarks = m["detector"].detect(frames, m["detector"].full_range_detector)
    if all(lm is None for lm in landmarks):
        landmarks = m["detector"].detect(frames, m["detector"].short_range_detector)
    detected = sum(lm is not None for lm in landmarks) / len(landmarks)
    quality["face_detected_ratio"] = round(detected, 3)
    if detected == 0:
        flags.append("no_face")
        return None, None
    if detected < 0.9:
        flags.append("partial_face")
    try:
        crops = m["process"](frames, landmarks)
    except Exception as exc:
        flags.append("occlusion")
        quality["crop_error"] = str(exc)
        return None, None
    x = m["transform"](torch.tensor(crops))
    center = crops[:, 4:92, 4:92].astype(np.float32)
    motion = float(np.abs(np.diff(center, axis=0)).mean()) if len(center) > 1 else 0.0
    quality["mouth_motion"] = round(motion, 3)
    quality["mouth_motion_threshold"] = MOTION_THRESHOLD
    if motion < MOTION_THRESHOLD:
        flags.append("low_mouth_motion")
    with torch.no_grad():
        enc = m["avsr"].model.encode(x.to(m["avsr"].device))
    return enc, crops


@app.post("/infer")
def infer(video: UploadFile = File(...), session_id: str = Form("demo-session"), source_ref: str = Form("")):
    data = video.file.read()
    return _read_clip(lambda m, flags, quality: _encode(m, data, flags, quality), session_id, source_ref)


def _read_clip(encode, session_id: str, source_ref: str, quality: dict | None = None) -> Evidence:
    t0 = time.time()
    m = model.get()
    flags: list[str] = []
    quality = quality if quality is not None else {}
    prov = Provenance(
        service="vsr", checkpoint="Auto-AVSR LRS3_V_WER19.1", device=m["device"],
        preprocess="pyav-25fps -> mediapipe face det -> mean-face similarity warp -> 96px mouth ROI -> centercrop 88",
    )

    with model.lock:
        enc, crops = encode(m, flags, quality)
        if enc is None:
            return _evidence(session_id, source_ref, [], flags, quality, prov, t0)

        # 1. the person's own recordings (fast: DTW over the encoder features)
        personal = []
        if enrolled.phrases():
            t_en = time.time()
            matches = enrolled.match(enc.cpu().numpy(), len(enrolled.phrases()))
            quality["enroll_best_dist"] = round(matches[0][1], 4) if matches else None
            quality["enroll_max_dist"] = ENROLL_MAX_DIST
            if matches and matches[0][0].startswith(NEG_PREFIX) and matches[0][1] <= ENROLL_MAX_DIST:
                # closest to a recorded near-miss: deliberately not a phrase, and no free-text guess either
                quality["enroll_negative"] = matches[0][0][len(NEG_PREFIX):]
                quality["mode"] = "negative"
                quality["enroll_ms"] = int((time.time() - t_en) * 1000)
                quality["mouth_thumbs"] = _thumbs(crops)
                return _evidence(session_id, source_ref, [], flags, quality, prov, t0)
            matches = [(p, d) for p, d in matches if not p.startswith(NEG_PREFIX)][:PHRASE_TOP]
            for phrase, dist in matches:
                if dist <= ENROLL_MAX_DIST:
                    personal.append(Hypothesis(text=phrase, score=round(-ENROLL_SCALE * dist, 2),
                                               score_kind="personal enrollment match (-100 x DTW distance)",
                                               words=[{"word": w} for w in phrase.split()]))
            quality["enroll_ms"] = int((time.time() - t_en) * 1000)

        # 2. only when the clip matches none of them: the phrase list, then free text. Their scores are on
        #    other scales, so they are never mixed into a list of enrollment matches (the policy ranks by score).
        phrase_hyps, hyps = [], []
        if not personal and m["phrases"]:
            t_ph = time.time()
            with torch.no_grad():
                scores = score_phrases(m["avsr"].model, enc, [t for _, t in m["phrases"]])
            for i in scores.argsort(descending=True)[:PHRASE_TOP].tolist():
                phrase_hyps.append(Hypothesis(
                    text=m["phrases"][i][0], score=round(float(scores[i]), 3),
                    score_kind="phrase-list log-likelihood (decoder+CTC), uncalibrated",
                    words=[{"word": w} for w in m["phrases"][i][0].split()]))
            quality["phrase_ms"] = int((time.time() - t_ph) * 1000)
        if not personal and not phrase_hyps and (FREE_TEXT or not enrolled.phrases()):
            t_inf = time.time()
            with torch.no_grad():
                hyps = m["avsr"].beam_search(enc)
            quality["decode_ms"] = int((time.time() - t_inf) * 1000)
        quality["mode"] = "enrollment" if personal else "phrase_list" if phrase_hyps else "free_text" if hyps else "none"

    hypotheses, seen = [], set()
    for h in personal + phrase_hyps + _nbest_texts(m["avsr"], hyps):
        if h.text.lower() not in seen:
            seen.add(h.text.lower())
            hypotheses.append(h)
    if len(hypotheses) >= 2:
        quality["top2_margin"] = round(hypotheses[0].score - hypotheses[1].score, 3)
    quality["mouth_thumbs"] = _thumbs(crops)
    return _evidence(session_id, source_ref, hypotheses, flags, quality, prov, t0)


# ---- streamed clips ---------------------------------------------------------
# The browser sends the clip in 250 ms pieces while the person is still mouthing. Each stream is decoded as it
# arrives, and each frame goes through face detection as soon as it is final, so after the pause only cropping,
# the encoder and matching remain. The frames are the ones the whole-clip decode gives: the same decoder reads
# the same bytes, frames are picked on the same 25 fps grid (a grid point is final once a later frame exists),
# and detection is per frame. Cropping (it smooths landmarks over neighbouring frames) runs on the whole clip at
# the end, exactly as for an uploaded clip.

class _Feed:
    """A byte stream that grows as chunks arrive; a read waits for more bytes until the stream is closed."""

    def __init__(self):
        self.buf, self.pos, self.closed = bytearray(), 0, False
        self.cv = threading.Condition()

    def push(self, data: bytes):
        with self.cv:
            self.buf += data
            self.cv.notify_all()

    def close(self):
        with self.cv:
            self.closed = True
            self.cv.notify_all()

    def read(self, n: int = -1) -> bytes:
        with self.cv:
            while self.pos >= len(self.buf) and not self.closed:
                self.cv.wait()
            end = len(self.buf) if n is None or n < 0 else min(len(self.buf), self.pos + n)
            out = bytes(self.buf[self.pos:end])
            self.pos = end
            return out


_stream_detector: dict = {}
_stream_detect_lock = threading.Lock()


def _detector():
    """Face detection for streamed clips; its own instance (detection is per frame), one stream at a time."""
    if "d" not in _stream_detector:
        from pipelines.detectors.mediapipe.detector import LandmarksDetector
        _stream_detector["d"] = LandmarksDetector()
    return _stream_detector["d"]


class _ClipStream:
    def __init__(self, mime: str):
        self.feed, self.mime, self.touched = _Feed(), mime, time.time()
        self.frames: list[np.ndarray] = []
        self.times: list[float] = []
        self.grid: list[int] = []      # source frame index of each 25 fps grid point
        self.landmarks: list = []
        self.error: str | None = None
        self.done = threading.Event()
        self.chunks = 0
        threading.Thread(target=self._run, daemon=True).start()

    def _detect_until(self, t_end: float | None):
        """Grid points before t_end (all of them when None) get their frame and face detection."""
        times = np.asarray(self.times) - self.times[0]
        points = np.arange(0.0, times[-1] + 1.0 / MODEL_FPS if t_end is None else t_end, 1.0 / MODEL_FPS)[len(self.grid):]
        if not len(points):
            return
        idx = np.clip(np.searchsorted(times, points, side="right") - 1, 0, len(self.frames) - 1)
        d = _detector()
        with _stream_detect_lock:
            self.landmarks += d.detect([self.frames[i] for i in idx], d.full_range_detector)
        self.grid += idx.tolist()

    def _run(self):
        import av
        try:
            fmt = "mp4" if "mp4" in self.mime else "matroska"
            with av.open(self.feed, format=fmt, options={"probesize": "32", "analyzeduration": "0"}) as c:
                for frame in c.decode(c.streams.video[0]):
                    if frame.pts is None:
                        continue
                    self.times.append(float(frame.pts * frame.time_base))
                    self.frames.append(frame.to_ndarray(format="rgb24"))
                    self._detect_until(self.times[-1] - self.times[0])  # grid points before this frame are final
            if self.frames:
                self._detect_until(None)
        except Exception as exc:
            self.error = type(exc).__name__
        finally:
            self.done.set()

    def encode(self, m, flags: list[str], quality: dict):
        if self.error or not self.frames:
            flags.append("too_short")
            quality["decode_error"] = self.error or "no frames"
            return None, None
        times = np.asarray(self.times) - self.times[0]
        frames = np.stack([self.frames[i] for i in self.grid])
        return _encode_frames(m, frames, float(times[-1]) + 1.0 / MODEL_FPS, list(self.landmarks), flags, quality)


_streams: dict[str, _ClipStream] = {}
STREAM_TTL_S = 60


@app.post("/stream/{stream_id}/chunk")
async def stream_chunk(stream_id: str, request: Request, mime: str = "video/webm"):
    data = await request.body()
    for k in [k for k, v in _streams.items() if time.time() - v.touched > STREAM_TTL_S]:
        _streams.pop(k).feed.close()
    st = _streams.get(stream_id)
    if st is None:
        st = _streams[stream_id] = _ClipStream(mime)
    st.feed.push(data)
    st.chunks += 1
    st.touched = time.time()
    return {"chunks": st.chunks}


@app.post("/stream/{stream_id}/finish")
def stream_finish(stream_id: str, session_id: str = Form("demo-session"), source_ref: str = Form("")):
    st = _streams.pop(stream_id, None)
    if st is None:
        raise HTTPException(404, "unknown stream")
    t0 = time.time()
    st.feed.close()
    st.done.wait(30)
    quality = {"streamed": True, "stream_chunks": st.chunks, "stream_tail_ms": int((time.time() - t0) * 1000)}
    return _read_clip(st.encode, session_id, source_ref, quality)


@app.post("/stream/{stream_id}/discard")
def stream_discard(stream_id: str):
    st = _streams.pop(stream_id, None)
    if st:
        st.feed.close()
    return {"discarded": st is not None}


# ---- personal enrollment ----------------------------------------------------

@app.get("/enroll")
def enroll_list():
    return {"phrases": enrolled.phrases(), "suggestions": _lines(SUGGESTIONS), "negatives": _lines(NEGATIVES), "negative_prefix": NEG_PREFIX}


@app.post("/enroll")
def enroll_add(video: UploadFile = File(...), phrase: str = Form(...)):
    if not phrase.strip():
        raise HTTPException(400, "empty phrase")
    m = model.get()
    flags: list[str] = []
    quality: dict = {}
    with model.lock:
        enc, _ = _encode(m, video.file.read(), flags, quality)
    bad = {"too_short", "no_face", "occlusion", "low_mouth_motion"} & set(flags)
    if enc is None or bad:
        raise HTTPException(422, f"recording not saved: {', '.join(sorted(bad or flags)).replace('_', ' ')}")
    count = enrolled.add(phrase, enc.cpu().numpy())
    return {"phrase": " ".join(phrase.split()), "count": count, "quality": quality, "phrases": enrolled.phrases()}


@app.delete("/enroll")
def enroll_remove(phrase: str):
    enrolled.remove(phrase)
    return {"phrases": enrolled.phrases()}


def _evidence(session_id, source_ref, hyps, flags, quality, prov, t0) -> Evidence:
    prov.latency_ms = int((time.time() - t0) * 1000)
    status = "proposed" if hyps and not {"no_face", "low_mouth_motion"} & set(flags) else "unknown"
    if hyps and len(hyps) > 1 and quality.get("top2_margin", 99) < 1.5:
        status = "ambiguous"
    return Evidence(
        session_id=session_id, modality="lip", source_mode="live", source_ref=source_ref or "browser-camera-clip",
        hypotheses=hyps, quality_flags=flags, quality=quality, provenance=prov, status=status,
    )
