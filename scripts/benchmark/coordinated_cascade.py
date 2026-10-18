"""E7: agents coordinating on one utterance: voice and lips propose, the coordinator decides, EEG confirms when asked.

Every number an agent returns is a recorded output, not simulated:
- voice (Nemotron Speech 0.6B) and lips (Auto-AVSR) phrase scores for 200 held-out GRID utterances in 7 audio
  conditions (E1, `evaluations/results/av_fusion/`). Temperatures and thresholds are fitted on the other speaker.
- EEG confirmation replays held-out Study F session-3 selections decided by the E6 stopping policy
  (`neuro_comm_student/episodes_labeller.csv`): its outcome (right, wrong, abstain) and its EEG time.

The one composite step is the pairing: a GRID utterance is confirmed by an ALS participant's recorded selection. The
EEG agent's error rate is its 36-way rate, although the confirmation offers only 2-5 options; this is conservative.

Decision per utterance (the coordinator sees the voice, lips and fused top-5):
  commit  speak a candidate now
  confirm show k of the candidates plus "none of these" and let the EEG agent select (~11 s)
  abstain say nothing

EEG mapping: a right EEG selection picks the true phrase if shown, otherwise "none" (no phrase spoken). A wrong one picks
one of the other shown options (or "none") uniformly. An EEG abstention speaks nothing.

Reward (as neuro_comm): +1 right, -1 wrong, -0.2 nothing spoken, -0.01 per second.

Policies: voice / lips / fused alone (E1 rule: commit above a 2% risk threshold), always confirm, a cascade (commit
above the fused 2% risk threshold, otherwise confirm the top k, k fitted on the other speaker), and an LLM coordinator
(Nemotron 30B on vLLM, or any OpenAI-compatible model) given the same top-5 and costs.

Usage: coordinated_cascade.py [--llm-url http://localhost:8001/v1 --llm-model big --llm-name nemotron30b]
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
_s = importlib.util.spec_from_file_location("avf", ROOT / "scripts" / "benchmark" / "av_fusion_grid.py")
avf = importlib.util.module_from_spec(_s)
_s.loader.exec_module(avf)

OUT = ROOT / "evaluations" / "results" / "coordinated_cascade"
EEG_TEST = ROOT / "evaluations" / "results" / "neuro_comm_student" / "episodes_labeller.csv"
R_OK, P_WRONG, P_NONE, C_SEC = 1.0, 1.0, 0.2, 0.01
TOPK = 5


def load():
    cand = {c["id"]: c for c in avf.load("cands")["clips"]}
    lip = {r["id"]: r for r in avf.load("lips")["rows"]}
    vo = avf.load("voice")
    voi = {r["id"]: r for r in vo["rows"]}
    ids = [i for i in cand if lip.get(i, {}).get("scores") and i in voi]
    y = np.array([cand[i]["cands"].index(cand[i]["truth"]) for i in ids])
    spk = np.array([cand[i]["speaker"] for i in ids])
    L = np.array([lip[i]["scores"] for i in ids])
    V = {c: np.array([voi[i]["scores"][c] for i in ids]) for c in vo["conditions"]}
    ms = {"voice": np.array([voi[i]["ms"]["clean"] for i in ids]), "lips": np.array([lip[i]["ms"] for i in ids])}
    return ids, cand, y, spk, L, V, ms, vo["conditions"]


def eeg_stats(df: pd.DataFrame) -> dict:
    return {"p_right": float(df.correct.mean()), "p_wrong": float(df.incorrect_commit.mean()),
            "p_abstain": float(df.abstained.mean()), "sec": float(df.latency_s.mean())}


def expected_confirm(p_shown: float, k: int, e: dict) -> float:
    """Expected reward of confirming, given the fused mass on the k shown options (used only to fit the cascade)."""
    # right EEG: truth shown -> +1, else none; wrong EEG: picks one of the other k options (k of k+1 are phrases)
    right = p_shown * R_OK - (1 - p_shown) * P_NONE
    wrong_phrase = k / (k + 1)
    wrong = wrong_phrase * -P_WRONG + (1 - wrong_phrase) * -P_NONE
    return e["p_right"] * right + e["p_wrong"] * wrong + e["p_abstain"] * -P_NONE - C_SEC * e["sec"]


def play(action: str, choice, shown, truth: int, eeg_row, rng) -> dict:
    """Outcome of one decision. eeg_row: the recorded EEG selection replayed for a confirm."""
    if action == "commit":
        return {"spoken": choice, "right": choice == truth, "eeg_s": 0.0, "confirmed": False}
    if action == "abstain":
        return {"spoken": None, "right": False, "eeg_s": 0.0, "confirmed": False}
    opts = list(shown) + [None]
    if eeg_row.correct:
        pick = truth if truth in shown else None
    elif eeg_row.incorrect_commit:
        right_opt = truth if truth in shown else None
        others = [o for o in opts if o != right_opt]
        pick = others[rng.integers(len(others))]
    else:
        pick = None
    return {"spoken": pick, "right": pick is not None and pick == truth, "eeg_s": float(eeg_row.latency_s), "confirmed": True}


def reward(o: dict) -> float:
    r = R_OK if o["right"] else (-P_WRONG if o["spoken"] is not None else -P_NONE)
    return r - C_SEC * o["eeg_s"]


def prompt(cands: list[str], pv, pl, pf) -> tuple[str, list[int]]:
    order = list(np.argsort(-pf)[:TOPK])
    rows = "\n".join(f"{n + 1}. \"{cands[j]}\"  voice {pv[j]:.2f}  lips {pl[j]:.2f}  combined {pf[j]:.2f}"
                     for n, j in enumerate(order))
    return rows, order


SYSTEM = ("You coordinate a communication aid for a person who cannot speak reliably. A voice agent and a lip-reading "
          "agent each scored the same short utterance against candidate phrases; their probabilities are calibrated, and "
          "'combined' merges them. You decide what happens next. Options:\n"
          "- commit: speak one candidate now.\n"
          "- confirm: show 2 to 4 candidates (plus 'none of these') on the patient's screen; the patient selects one "
          "with a brain-computer interface. This takes about 10 seconds, and the EEG selection itself is right about 62% "
          "of the time, wrong about 3.5%, and gives no answer about 35%.\n"
          "- abstain: speak nothing.\n"
          "Scoring: +1 for the right phrase spoken, -1 for a wrong phrase spoken, -0.2 when nothing is spoken, -0.01 per "
          "second of waiting. Background noise may make the voice agent unreliable; lip reading is unaffected by noise.\n"
          "Reply with JSON only: {\"action\": \"commit\", \"option\": n} or {\"action\": \"confirm\", \"options\": [n, ...]} "
          "or {\"action\": \"abstain\"}.")


async def ask_llm(url, model, items, conc=32):
    import httpx

    sem = asyncio.Semaphore(conc)
    out = [None] * len(items)

    async def one(i, user, client):
        async with sem:
            t0 = time.perf_counter()
            try:
                r = await client.post(f"{url}/chat/completions", json={
                    "model": model, "temperature": 0, "max_tokens": 60,
                    "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                    "chat_template_kwargs": {"enable_thinking": False}}, timeout=120)
                txt = r.json()["choices"][0]["message"]["content"] or ""
            except Exception as e:  # noqa: BLE001
                txt = f"ERROR {e}"
            out[i] = (txt, time.perf_counter() - t0)

    async with httpx.AsyncClient() as client:
        await asyncio.gather(*(one(i, u, client) for i, u in enumerate(items)))
    return out


def parse(txt: str, order: list[int]):
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        d = json.loads(m.group(0))
        a = d["action"]
        if a == "commit":
            return "commit", order[int(d["option"]) - 1], None, True
        if a == "confirm":
            opts = [order[int(n) - 1] for n in d["options"]][:4]
            return ("confirm", None, opts, True) if opts else ("abstain", None, None, False)
        if a == "abstain":
            return "abstain", None, None, True
    except Exception:  # noqa: BLE001
        pass
    return "abstain", None, None, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-url")
    ap.add_argument("--llm-model", default="big")
    ap.add_argument("--llm-name", default="nemotron30b")
    ap.add_argument("--risk", type=float, default=0.02)
    a = ap.parse_args()

    ids, cand, y, spk, L, V, ms, conds = load()
    eeg = pd.read_csv(EEG_TEST).sort_values("episode_id").reset_index(drop=True)
    # cascade fitting uses EEG rates from the calibration sessions (sessions 1-2), never session 3
    cal_path = ROOT / "evaluations" / "results" / "neuro_comm_student" / "eeg_calibration_rates.json"
    e_cal = json.loads(cal_path.read_text())
    rows, llm_items = [], []
    for test in sorted(set(spk)):
        cal, tst = spk != test, np.where(spk == test)[0]
        Vc = np.concatenate([V[c][cal] for c in conds])
        Lc, yc = np.concatenate([L[cal]] * len(conds)), np.concatenate([y[cal]] * len(conds))
        tv, tl = avf._fit_t(Vc, yc), avf._fit_t(L[cal], y[cal])
        Fc = avf._logsm(avf._logsm(Vc / tv) + avf._logsm(Lc / tl))
        tau = {"voice": avf._tau(avf._logsm(Vc / tv), yc, a.risk), "lips": avf._tau(avf._logsm(L[cal] / tl), y[cal], a.risk),
               "fused": avf._tau(Fc, yc, a.risk)}
        # cascade: commit at the same 2% risk threshold as fused (E1); below it, confirm the top k. k is fitted on the
        # calibration speaker by expected reward of the confirm branch
        pf_c = np.exp(Fc)
        unsure = pf_c.max(1) < tau["fused"]
        ranks = np.argsort(-pf_c, axis=1)
        k_cas = max((2, 3, 4), key=lambda k: np.where((ranks[unsure, :k] == yc[unsure, None]).any(1),
                                                       expected_confirm(1.0, k, e_cal), expected_confirm(0.0, k, e_cal)).mean())
        t_cas = tau["fused"]
        for ci, c in enumerate(conds):
            for i in tst:
                pv = np.exp(avf._logsm(V[c][i] / tv))
                pl = np.exp(avf._logsm(L[i] / tl))
                pf = np.exp(avf._logsm(np.log(pv) + np.log(pl)))
                ep = {"clip": ids[i], "speaker": test, "condition": c, "truth": int(y[i]),
                      "eeg": eeg.iloc[(i * len(conds) + ci) % len(eeg)], "rng": np.random.default_rng(i * 31 + ci),
                      "audio_ms": float(max(ms["voice"][i], ms["lips"][i])),
                      "agent_ms": {"voice": float(ms["voice"][i]), "lips": float(ms["lips"][i])}}
                order = list(np.argsort(-pf))
                dec = {
                    "voice": ("commit", int(pv.argmax()), None) if pv.max() >= tau["voice"] else ("abstain", None, None),
                    "lips": ("commit", int(pl.argmax()), None) if pl.max() >= tau["lips"] else ("abstain", None, None),
                    "fused": ("commit", int(pf.argmax()), None) if pf.max() >= tau["fused"] else ("abstain", None, None),
                    "always_confirm": ("confirm", None, order[:3]),
                    "cascade": ("commit", int(pf.argmax()), None) if pf.max() >= t_cas else ("confirm", None, order[:k_cas]),
                }
                for name, (act, ch, shown) in dec.items():
                    o = play(act, ch, shown or [], ep["truth"], ep["eeg"], np.random.default_rng(i * 31 + ci))
                    rows.append({"policy": name, "clip": ep["clip"], "speaker": test, "condition": c, "action": act, **o,
                                 "reward": reward(o), "latency_s": ep["agent_ms"].get(name, ep["audio_ms"]) / 1000 + o["eeg_s"], "model_s": 0.0,
                                 "valid": True, "fit": f"t={t_cas:.3f},k={k_cas}" if name == "cascade" else ""})
                if a.llm_url:
                    txt, order5 = prompt(cand[ids[i]]["cands"], pv, pl, pf)
                    llm_items.append((ep, order5, f"Candidates:\n{txt}\nDecide."))
    if a.llm_url:
        res = asyncio.run(ask_llm(a.llm_url, a.llm_model, [u for *_, u in llm_items]))
        for (ep, order5, _), (txt, dt) in zip(llm_items, res):
            act, ch, shown, valid = parse(txt, order5)
            ci = conds.index(ep["condition"])
            o = play(act, ch, shown or [], ep["truth"], ep["eeg"], np.random.default_rng(ids.index(ep["clip"]) * 31 + ci))
            rows.append({"policy": a.llm_name, "clip": ep["clip"], "speaker": ep["speaker"], "condition": ep["condition"],
                         "action": act, **o, "reward": reward(o), "latency_s": ep["audio_ms"] / 1000 + dt + o["eeg_s"],
                         "model_s": dt, "valid": valid, "fit": txt[:120]})
    df = pd.DataFrame(rows)
    df["wrong"] = df.spoken.notna() & ~df.right
    df["nothing"] = df.spoken.isna()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"_{a.llm_name}" if a.llm_url else ""
    df.drop(columns=[]).to_csv(OUT / f"episodes{tag}.csv", index=False)
    g = df.groupby(["policy", "condition"])
    by = g.agg(n=("right", "size"), right=("right", "mean"), wrong=("wrong", "sum"), nothing=("nothing", "mean"),
               confirmed=("confirmed", "mean"), latency_p50=("latency_s", "median"), latency_mean=("latency_s", "mean"),
               reward=("reward", "mean"), invalid=("valid", lambda v: int((~v).sum()))).reset_index()
    by.to_csv(OUT / f"by_condition{tag}.csv", index=False)
    tot = df.groupby("policy").agg(n=("right", "size"), right=("right", "mean"), wrong=("wrong", "sum"),
                                   nothing=("nothing", "mean"), confirmed=("confirmed", "mean"),
                                   latency_p50=("latency_s", "median"), latency_mean=("latency_s", "mean"),
                                   model_s_mean=("model_s", "mean"), reward=("reward", "mean"),
                                   invalid=("valid", lambda v: int((~v).sum()))).reset_index()
    tot.to_csv(OUT / f"overall{tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print(tot.round(3).to_string(index=False))
    piv = by.pivot(index="condition", columns="policy", values="right").reindex(conds)
    print("\nright spoken, by condition\n" + piv.round(3).to_string())
    print("\nwrong spoken, by condition\n" + by.pivot(index="condition", columns="policy", values="wrong").reindex(conds).to_string())


if __name__ == "__main__":
    main()
