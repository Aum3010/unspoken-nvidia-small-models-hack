"""Live test bench: speak into your local browser, decide on the GPU box.

    # on the GPU box (or the laptop, CPU)
    python serve.py --config configs/eval.yaml --model nemotron-streaming-0.6b --method forced_llr --port 8800
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
import re
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from harness import audio  # noqa: E402
from harness.config import load_config  # noqa: E402
from harness.live import Bench, decode, json_safe  # noqa: E402

NOTICE = ("Clips are saved on the evaluation GPU machine (data/own/) and used only to evaluate this prototype. "
          "Get a verbal OK from anyone you record, and use pseudonyms, not names.")
OWN_NAME = re.compile(r"^[a-z0-9]+_[a-z]+_[a-z0-9_]+_\d+\.wav$")


class Select(BaseModel):
    model: str
    method: str
    point: str = "balanced"
    language: str = "en"


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
                        "tuned": [x for x in bench.methods_for(m["name"]) if bench.th_key(m["name"], x, bench.point) in bench.thresholds],
                        "multilingual": bench.supports_language(m["name"])}
                       for m in cfg.enabled_models()],
            "model": bench.model.name, "method": bench.method, "point": bench.point,
            "points": {k: v for k, v in bench.points.items()}, "thresholds": json_safe(bench.th), "load_s": bench.load_s,
            "languages": [{"code": k, "name": v.get("name", k), "review_status": v.get("review_status", "")} for k, v in bench.languages.items()],
            "language": bench.lang, "language_note": bench.language_note,
            "phrases_native": {p: (t[0] if isinstance(t, list) else t) for p, t in bench.vocab["phrases20"].items()},
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU", "notice": NOTICE,
            "negative_prompts": {"silence": "(stay silent, then release)", "breathing": "(breathe in and out audibly)", "cough": "(cough or clear your throat)",
                                 "unrelated": "Say any unrelated sentence"},
            "thresholds_note": next((v.get("fallback") for k, v in bench.thresholds.items() if k.endswith("|phrases20") and v.get("fallback")), None),
        }

    @app.post("/api/select")
    def select(body: Select):
        try:
            bench.select(body.model, body.method, body.point, body.language)
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
        if bench.lang != "en":
            raise HTTPException(400, "recordings are English-only for now (the evaluation is English); switch to English to record")
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
    ap.add_argument("--method", default="forced_llr")
    ap.add_argument("--point", default="balanced", help="operating point: strict (headline) or a name from decision.operating_points")
    ap.add_argument("--language", default="en", help="patient language code from `languages` (multilingual models only)")
    ap.add_argument("--host", default="127.0.0.1", help="keep 127.0.0.1 and reach it through brev port-forward")
    ap.add_argument("--port", type=int, default=8800)
    a = ap.parse_args()
    import uvicorn

    bench = Bench(load_config(a.config), a.model, a.method, a.point, a.language)
    print(f"ready: {bench.model.name}/{bench.method} [{bench.point}] on http://{a.host}:{a.port}", flush=True)
    uvicorn.run(build_app(bench), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
