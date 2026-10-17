"""One clip through one model: transcript, forced phrase scores, timings.

Shared by run_eval.py (batch evaluation) and serve.py (live UI), so the live
demo makes exactly the decisions the evaluation measured.
"""

from __future__ import annotations

import time

from harness.decide import alternatives, surface_forms


def task_forms(vocabs: dict[str, dict], lang: str = "en") -> dict[str, list[tuple[str, str]]]:
    """(phrase_id, surface form) pairs per task; a phrase's score is the max over its forms and alternatives."""
    return {task: list(dict.fromkeys((pid, f) for pid, txt in v.items() for a in alternatives(txt) for f in surface_forms(a, lang)))
            for task, v in vocabs.items()}


def _sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def analyze(model, x, tasks: list[str], forms: dict[str, list[tuple[str, str]]]):
    """Returns (transcript dict, asr_ms, [(task, phrase_logp | None, free_logp | None, score_ms | None)])."""
    _sync()
    t0 = time.perf_counter()
    tr = model.transcribe(x)
    _sync()
    asr_ms = (time.perf_counter() - t0) * 1000
    outs = []
    for task in tasks:
        if not model.supports_scoring:
            outs.append((task, None, None, None))
            continue
        _sync()
        t1 = time.perf_counter()
        texts = [f for _, f in forms[task]] + [tr["text"]]
        sc = model.score_phrases(x, texts)
        ph: dict[str, list] = {}
        for (pid, _), s in zip(forms[task], sc[:-1]):
            if pid not in ph or s[0] > ph[pid][0]:
                ph[pid] = [round(s[0], 4), s[1]]
        free = [round(sc[-1][0], 4), sc[-1][1]]
        _sync()
        outs.append((task, ph, free, (time.perf_counter() - t1) * 1000))
    return tr, asr_ms, outs
