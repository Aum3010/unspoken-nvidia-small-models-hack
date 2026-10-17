"""Live test bench: speak into your local browser, decide on the GPU box.

    # on the GPU box (or the laptop, CPU)
    python serve.py --config configs/eval.yaml --model nemotron-streaming-0.6b --method forced_sum --port 8800
    # on the laptop
    brev port-forward <instance> --port 8800:8800      # then open http://localhost:8800

Each clip goes through the same pipeline as run_eval.py (Silero gate, model, rankers,
decide()) with the thresholds frozen in results/thresholds.json, so what you see live is
what the evaluation measured. "Save" writes the last clip as a labelled recording to
data/own/ (consent required), feeding the next evaluation run.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from harness import audio  # noqa: E402
from harness.config import load_config  # noqa: E402
from harness.decide import decide_arrays, explain, features  # noqa: E402
from harness.models import make  # noqa: E402
from harness.pipeline import analyze, task_forms  # noqa: E402
from harness.vad import Vad, stats  # noqa: E402

NOTICE = ("Clips are saved on the evaluation GPU machine (data/own/) and used only to evaluate this prototype. "
          "Get a verbal OK from anyone you record, and use pseudonyms, not names.")
OWN_NAME = re.compile(r"^[a-z0-9]+_[a-z]+_[a-z0-9_]+_\d+\.wav$")


def json_safe(obj):
    """JSON has no infinity: a threshold of -inf ("no minimum") becomes null."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def decode(data: bytes) -> np.ndarray:
    """Any browser container (webm/opus, ogg, wav, mp4) -> mono float32 at 16 kHz."""
    import av

    out = []
    with av.open(io.BytesIO(data)) as c:
        res = av.AudioResampler(format="flt", layout="mono", rate=audio.SR)
        for frame in c.decode(c.streams.audio[0]):
            out += [f.to_ndarray().reshape(-1) for f in res.resample(frame)]
        out += [f.to_ndarray().reshape(-1) for f in res.resample(None)]
    return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)


class Bench:
    def __init__(self, cfg, model_name: str, method: str, point: str = "balanced"):
        self.cfg = cfg
        self.points = {"strict": cfg["decision"]["far_target"], **(cfg["decision"].get("operating_points") or {})}
        self.lock = threading.Lock()
        self.vad = Vad()
        self.vocab = {"phrases20": cfg.phrases}
        self.forms = task_forms(self.vocab)
        th_path = cfg.results / "thresholds.json"
        self.thresholds = json.loads(th_path.read_text()) if th_path.exists() else {}
        self.clips: OrderedDict[str, np.ndarray] = OrderedDict()
        self.model = None
        self.select(model_name, method, point)

    def methods_for(self, model_name: str) -> list[str]:
        scoring = self.cfg.model(model_name)["kind"] in ("hf_ctc", "nemo_transducer")
        return [m for m in self.cfg["decision"]["methods"] if m == "text" or scoring]

    def th_key(self, model_name: str, method: str, point: str) -> str:
        return f"{model_name}/{method}|phrases20" + ("" if point == "strict" else f"|{point}")

    def select(self, model_name: str, method: str, point: str = "balanced"):
        with self.lock:
            if self.model is None or self.model.name != model_name:
                if self.model is not None:
                    self.model.unload()
                self.model = None
                m = make(self.cfg.model(model_name), self.cfg)
                t = time.perf_counter()
                m.load()
                self.load_s = round(time.perf_counter() - t, 1)
                self.model = m
                self.model.transcribe(np.zeros(audio.SR, np.float32))  # warm-up
            if method not in self.methods_for(model_name):
                method = "text"
            self.method = method
            self.point = point if point in self.points else "strict"
            self.th = self.thresholds.get(self.th_key(model_name, method, self.point))

    def recognize(self, wav: np.ndarray) -> dict:
        t0 = time.perf_counter()
        probs = self.vad.probs(wav)
        vad_ms = (time.perf_counter() - t0) * 1000
        vs = stats(probs, self.cfg["vad"]["thresholds"], self.cfg["vad"]["min_speech_ms"])
        with self.lock:
            tr, asr_ms, outs = analyze(self.model, wav, ["phrases20"], self.forms)
        _, ph, free, score_ms = outs[0]
        t1 = time.perf_counter()
        row = {"path": "live", "eval_task": "phrases20", "transcript": tr["text"], "phrase_logp": ph, "free_logp": free,
               "label_type": "live", "phrase_id": "", **vs}
        f = features(pd.DataFrame([row]), self.vocab, self.method, set(self.cfg["decision"]["negation_words"]))
        r = f.iloc[0].to_dict()
        if self.th:
            decision = str(decide_arrays(f, self.th["vad_t"], self.th["abs_t"], self.th["margin_t"])[0])
            reason = explain(r, self.th)
        else:
            decision, reason = "untuned", "no thresholds for this model/method/sensitivity yet (its evaluation has not finished)"
        from harness.decide import rank_scores

        ranked = rank_scores(r, self.vocab["phrases20"], self.method) if r.get("best") is not None else []
        decide_ms = (time.perf_counter() - t1) * 1000
        return {
            "decision": decision if decision in ("unknown", "no_speech", "untuned") else "candidate",
            "phrase_id": decision if decision not in ("unknown", "no_speech", "untuned") else None,
            "phrase_text": self.cfg.phrases.get(decision), "reason": reason,
            "top3": [{"id": p, "text": self.cfg.phrases[p], "score": round(float(s), 4)} for p, s in ranked[:3]],
            "margin": None if not np.isfinite(r.get("margin", np.nan)) else round(float(min(r["margin"], 1e6)), 4),
            "negation_mismatch": bool(r.get("neg_mismatch")), "transcript": tr["text"],
            "signals": {k: v for k, v in tr.items() if k not in ("text", "nbest")}, "vad": vs,
            "duration_s": round(len(wav) / audio.SR, 2),
            "timings_ms": {"vad": round(vad_ms, 1), "asr": round(asr_ms, 1), "score": round(score_ms, 1) if score_ms else None,
                           "decide": round(decide_ms, 1)},
            "model": self.model.name, "method": self.method, "point": self.point, "thresholds": json_safe(self.th),
        }


class Select(BaseModel):
    model: str
    method: str
    point: str = "balanced"


class Save(BaseModel):
    clip_id: str
    speaker: str
    condition: str
    label: str


class Delete(BaseModel):
    name: str


def build_app(bench: Bench):
    app = FastAPI(title="unspoken-asr-bench")

    @app.get("/")
    def index():
        return FileResponse(ROOT / "static" / "index.html")

    @app.get("/api/config")
    def config():
        import torch

        cfg = bench.cfg
        return {
            "phrases": cfg.phrases, "hard_negatives": cfg.hard_negatives,
            "conditions": cfg["own"]["conditions"], "negatives": list(cfg["own"]["negatives"]),
            "models": [{"name": m["name"], "kind": m["kind"], "methods": bench.methods_for(m["name"]),
                        "tuned": [x for x in bench.methods_for(m["name"]) if bench.th_key(m["name"], x, bench.point) in bench.thresholds]}
                       for m in cfg.enabled_models()],
            "model": bench.model.name, "method": bench.method, "point": bench.point,
            "points": {k: v for k, v in bench.points.items()}, "thresholds": json_safe(bench.th), "load_s": bench.load_s,
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU", "notice": NOTICE,
            "negative_prompts": {"silence": "Stay silent, then release", "breathing": "Breathe audibly", "cough": "Cough or clear your throat",
                                 "unrelated": "Say any unrelated sentence"},
            "thresholds_note": next((v.get("fallback") for k, v in bench.thresholds.items() if k.endswith("|phrases20") and v.get("fallback")), None),
        }

    @app.post("/api/select")
    def select(body: Select):
        try:
            bench.select(body.model, body.method, body.point)
        except Exception as exc:
            raise HTTPException(500, f"{type(exc).__name__}: {exc}")
        return config()

    @app.post("/api/recognize")
    async def recognize(audio_file: UploadFile = File(..., alias="audio")):
        t0 = time.perf_counter()
        data = await audio_file.read()
        try:
            wav = decode(data)
        except Exception as exc:
            raise HTTPException(400, f"could not decode audio: {exc}")
        decode_ms = (time.perf_counter() - t0) * 1000
        out = bench.recognize(wav)
        clip_id = uuid.uuid4().hex[:10]
        bench.clips[clip_id] = wav
        while len(bench.clips) > 50:
            bench.clips.popitem(last=False)
        out["clip_id"] = clip_id
        out["timings_ms"]["decode"] = round(decode_ms, 1)
        out["timings_ms"]["server_total"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    @app.post("/api/save")
    def save(body: Save):
        cfg = bench.cfg
        if not re.fullmatch(r"[a-z0-9]+", body.speaker):
            raise HTTPException(400, "speaker must be a pseudonym of lowercase letters/digits")
        if body.condition not in cfg["own"]["conditions"]:
            raise HTTPException(400, "unknown condition")
        valid = set(cfg.phrases) | {f"neg_{k}" for k in cfg.hard_negatives} | {f"neg_{k}" for k in cfg["own"]["negatives"]}
        if body.label not in valid:
            raise HTTPException(400, "unknown label")
        wav = bench.clips.get(body.clip_id)
        if wav is None:
            raise HTTPException(404, "clip expired; record it again")
        out = cfg.data / "own"
        out.mkdir(parents=True, exist_ok=True)
        log = out / "speakers.csv"  # who recorded when (pseudonyms only)
        if not (log.exists() and any(r and r[0] == body.speaker for r in csv.reader(log.open()))):
            with log.open("a", newline="") as fh:
                csv.writer(fh).writerow([body.speaker, time.strftime("%Y-%m-%d %H:%M"), "web"])
        take = 1
        while (out / f"{body.speaker}_{body.condition}_{body.label}_{take}.wav").exists():
            take += 1
        path = out / f"{body.speaker}_{body.condition}_{body.label}_{take}.wav"
        audio.save(path, wav)
        n = sum(1 for _ in out.glob(f"{body.speaker}_*.wav"))
        return {"saved": path.name, "speaker_total": n}

    @app.post("/api/delete")
    def delete(body: Delete):
        """Undo a save (Redo in the recorder). Only names the recorder itself produces are accepted."""
        if not OWN_NAME.fullmatch(body.name):
            raise HTTPException(400, "not a recording name")
        path = bench.cfg.data / "own" / body.name
        if not path.exists():
            raise HTTPException(404, "already gone")
        path.unlink()
        return {"deleted": body.name}

    @app.get("/api/progress")
    def progress(speaker: str, condition: str):
        """Saved clips per label for one speaker + condition, so a session can resume."""
        if not re.fullmatch(r"[a-z0-9]+", speaker) or condition not in bench.cfg["own"]["conditions"]:
            raise HTTPException(400, "bad speaker or condition")
        counts: dict[str, int] = {}
        for f in (bench.cfg.data / "own").glob(f"{speaker}_{condition}_*.wav"):
            label = f.stem[len(f"{speaker}_{condition}_"):].rsplit("_", 1)[0]
            counts[label] = counts.get(label, 0) + 1
        return {"counts": counts, "total": sum(counts.values())}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--model", default="nemotron-streaming-0.6b")
    ap.add_argument("--method", default="forced_sum")
    ap.add_argument("--point", default="balanced", help="operating point: strict (headline) or a name from decision.operating_points")
    ap.add_argument("--host", default="127.0.0.1", help="keep 127.0.0.1 and reach it through brev port-forward")
    ap.add_argument("--port", type=int, default=8800)
    a = ap.parse_args()
    import uvicorn

    bench = Bench(load_config(a.config), a.model, a.method, a.point)
    print(f"ready: {bench.model.name}/{bench.method} [{bench.point}] on http://{a.host}:{a.port}", flush=True)
    uvicorn.run(build_app(bench), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
