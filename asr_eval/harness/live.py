"""One live clip through the evaluated pipeline: Silero gate, model, rankers, decide().

Shared by serve.py (the live test bench) and the app's voice agent (services/audio), so both make
exactly the decisions the evaluation measured, with the thresholds frozen in results/thresholds.json.
"""

from __future__ import annotations

import io
import json
import threading
import time
from collections import OrderedDict

import numpy as np
import pandas as pd

from harness import audio
from harness.decide import decide_arrays, explain, features
from harness.models import make
from harness.pipeline import analyze, task_forms
from harness.vad import Vad, stats


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
    def __init__(self, cfg, model_name: str, method: str, point: str = "balanced", language: str = "en"):
        self.cfg = cfg
        langs = cfg.get("languages") or {}
        self.languages = {k: v for k, v in langs.items() if k != "default"} or {"en": {"name": "English", "locale": "en-US"}}
        self.lang = "en"
        self.points = {"strict": cfg["decision"]["far_target"], **(cfg["decision"].get("operating_points") or {})}
        self.lock = threading.Lock()
        self.vad = Vad()
        self.vocab = {"phrases20": cfg.phrases}
        self.forms = task_forms(self.vocab)
        self.language_note = None
        th_path = cfg.results / "thresholds.json"
        self.thresholds = json.loads(th_path.read_text()) if th_path.exists() else {}
        self.clips: OrderedDict[str, np.ndarray] = OrderedDict()
        self.model = None
        self.select(model_name, method, point, language)

    def methods_for(self, model_name: str) -> list[str]:
        spec = self.cfg.model(model_name)
        scoring = spec["kind"] == "hf_ctc" or (spec["kind"] == "nemo_transducer" and spec.get("forced_scoring", True))
        return [m for m in self.cfg["decision"]["methods"] if m == "text" or scoring]

    def supports_language(self, model_name: str) -> bool:
        spec = self.cfg.model(model_name)
        return spec["kind"] == "nemo_transducer" and bool(spec.get("target_lang"))

    def negation_words(self) -> set[str]:
        return set(self.languages.get(self.lang, {}).get("negation_words") or self.cfg["decision"]["negation_words"])

    def th_key(self, model_name: str, method: str, point: str) -> str:
        return f"{model_name}/{method}|phrases20" + ("" if point == "strict" else f"|{point}")

    def select(self, model_name: str, method: str, point: str = "balanced", language: str = "en"):
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
            # language: the patient speaks it; the answer is always the English phrase (phrase id is the shared key)
            if language not in self.languages or (language != "en" and not self.supports_language(model_name)):
                language = "en"
            if self.supports_language(model_name):
                self.model.set_language(self.languages[language]["locale"])
            self.lang = language
            phrases = self.cfg.phrases if language == "en" else self.languages[language]["phrases"]
            self.vocab = {"phrases20": phrases}
            self.forms = task_forms(self.vocab, language)
            self.language_note = None if language == "en" else (
                f"Listening in {self.languages[language]['name']}: thresholds were tuned on English recordings and are untested "
                f"for this language; translations are {self.languages[language].get('review_status', 'unreviewed')}.")
            self.th = self.thresholds.get(self.th_key(model_name, method, self.point))

    def recognize(self, wav: np.ndarray, assume_speech: bool = False) -> dict:
        """assume_speech: another sensor (the lips moving) already shows the person is speaking, so the VAD gate
        is passed. Silero VAD reads a whisper as no speech (no voicing); every other gate still applies."""
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
        f = features(pd.DataFrame([row]), self.vocab, self.method, self.negation_words(), self.lang)
        vad_passed_by_lips = False
        if assume_speech and self.th and not f[f"vad_speech_s@{self.th['vad_t']}"].iloc[0] > 0:
            f[f"vad_speech_s@{self.th['vad_t']}"] = 1e-3
            vad_passed_by_lips = True
        r = f.iloc[0].to_dict()
        if self.th:
            decision = str(decide_arrays(f, self.th["vad_t"], self.th["abs_t"], self.th["margin_t"])[0])
            reason = explain(r, self.th)
        else:
            decision, reason = "untuned", "no thresholds for this model/method/sensitivity yet (its evaluation has not finished)"
        from harness.decide import rank_scores

        ranked = rank_scores(r, self.vocab["phrases20"], self.method, self.lang) if r.get("best") is not None else []
        voc = self.vocab["phrases20"]

        def native(p):
            return voc[p][0] if isinstance(voc[p], list) else voc[p]
        decide_ms = (time.perf_counter() - t1) * 1000
        return {
            "decision": decision if decision in ("unknown", "no_speech", "untuned") else "candidate",
            "phrase_id": decision if decision not in ("unknown", "no_speech", "untuned") else None,
            "phrase_text": self.cfg.phrases.get(decision), "reason": reason,
            "top3": [{"id": p, "text": self.cfg.phrases[p], "native": native(p), "score": round(float(s), 4)} for p, s in ranked[:3]],
            "language": self.lang, "phrase_native": native(decision) if decision in voc else None,
            "margin": None if not np.isfinite(r.get("margin", np.nan)) else round(float(min(r["margin"], 1e6)), 4),
            "negation_mismatch": bool(r.get("neg_mismatch")), "transcript": tr["text"], "vad_passed_by_lips": vad_passed_by_lips,
            "signals": {k: v for k, v in tr.items() if k not in ("text", "nbest")}, "vad": vs,
            "duration_s": round(len(wav) / audio.SR, 2),
            "timings_ms": {"vad": round(vad_ms, 1), "asr": round(asr_ms, 1), "score": round(score_ms, 1) if score_ms else None,
                           "decide": round(decide_ms, 1)},
            "model": self.model.name, "method": self.method, "point": self.point, "thresholds": json_safe(self.th),
        }
