"""Train a small Nemotron to make the neuro_comm stopping decision, using only the environment's own reward.

The 30B coordinator never abstains on this task and waits too long (RESULTS.md section 7). This script asks whether a
4B Nemotron, fine-tuned on verifier-selected trajectories from the *training* sessions, does better on the held-out
session.

Action space. At each repetition the student chooses one of three words: `more` (request_more_eeg), `commit`
(commit_character with the current top candidate) or `abstain`. It acts through the same ReplaySession tools the Gym
server exposes, so the reward, verifier and timing are the environment's, unchanged. Choosing a character other than the
top candidate is not offered: it is almost never right, and every baseline commits the top candidate too.

Training data (best-of-N with the verifier). For each calibration episode (Study F sessions 1-2, 600 selections) every
simple trajectory is replayed: request k repetitions, then commit or abstain, for every k. The environment's verifier
scores each one, and the highest-reward trajectory (earliest on ties) gives one labelled example per step:
`more` before the stop, then `commit` or `abstain`. The session-3 evaluation episodes are never read here.

Decision at test time. No sampling: the student scores the three words after the prompt and takes the most likely,
so the base model and the fine-tuned model are compared on identical prompts.

Value labels (second attempt). Hindsight best-of-N commits at the first repetition where the top candidate happens to
be right, which taught the student to stop too early. `value_data` instead labels every repetition of every calibration
episode with the action of highest *expected* reward under models fitted on the calibration sessions (ValueModel); the
labeller itself is also evaluated as a policy, so the LLM's contribution is visible.

Stages:
  data        hindsight labels: data/student/train.jsonl from calibration episodes  (CPU)
  value_data  expected-reward labels: data/student/train_value.jsonl                (CPU)
  train  LoRA on the student, loss on the action word only                       (GPU)
  eval   base and fine-tuned student on session 3, through the Gym replay       (GPU)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.eeg.study_f import replay  # noqa: E402
from services.eeg.study_f.policies import LocalEnv  # noqa: E402

EVD = ROOT / "data" / "evidence" / "study_f"
WORK = ROOT / "data" / "student"
OUT = ROOT / "evaluations" / "results" / "neuro_comm_student"
MODEL = "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"
ACTIONS = ["more", "commit", "abstain"]


def system_text(cfg: replay.RewardConfig) -> str:
    return ("detailed thinking off\n"
            "You control a P300 speller for a person with ALS. EEG evidence arrives in repetitions; after each one you see "
            "the posterior over the 36 characters from a fixed decoder. You cannot see the intended character. Reply with "
            "one word: more (wait for another repetition), commit (select the top candidate), or abstain (give up on this "
            f"selection). Scoring: correct commit +{cfg.r_correct:g}, wrong commit -{cfg.p_wrong:g}, abstain "
            f"-{cfg.p_abstain:g}, each second of EEG -{cfg.c_sec:g} (one repetition is about 2.5 s).")


def state_text(o: dict) -> str:
    top = ", ".join(f"{c['character']} {c['posterior']:.3f}" for c in o["top_candidates"])
    return (f"Repetition {o['repetition']}, {o['repetitions_remaining']} left"
            f"{' (evidence exhausted: more is not possible)' if o['evidence_exhausted'] else ''}. "
            f"Top candidates: {top}. Entropy {o['entropy_bits']:.2f} bits. EEG time so far {o['decision_latency_s_so_far']:.1f} s. "
            "Action:")


def trajectories(ev, truth: int, cfg):
    """Every 'k more, then commit/abstain' trajectory, replayed and scored by the environment."""
    best = None
    for k in range(ev.n_avail):
        for final in ("commit", "abstain"):
            env = LocalEnv(ev, cfg)
            obs = [env.call("get_state")["observation"]]
            for _ in range(k):
                obs.append(env.call("request_more_eeg")["observation"])
            if final == "commit":
                env.call("commit_character", {"character": obs[-1]["top_candidate"]})
            else:
                env.call("abstain")
            r = env.session.outcome(truth)["reward"]
            if best is None or r > best[0] + 1e-9:
                best = (r, obs, final)
    return best


def data():
    cfg = replay.RewardConfig()
    eps = replay.load_episodes(EVD, roles=("calibration",))
    truth = replay.load_ground_truth(EVD)
    rows, finals = [], {"commit": 0, "abstain": 0}
    for sid, ev in eps.items():
        r, obs, final = trajectories(ev, truth[sid], cfg)
        finals[final] += 1
        for i, o in enumerate(obs):
            rows.append({"episode_id": sid, "participant_id": ev.participant_id, "state": state_text(o),
                         "action": final if i == len(obs) - 1 else "more"})
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "train.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    print(f"{len(eps)} calibration episodes -> {len(rows)} steps; final actions {finals}; "
          f"step labels { {a: sum(x['action'] == a for x in rows) for a in ACTIONS} }")


FEATS = ["repetition", "repetitions_remaining", "top_posterior", "second_posterior", "entropy_bits"]


def _x(o: dict):
    import numpy as np

    p1, p2 = max(o["top_posterior"], 1e-6), max(o["second_posterior"], 1e-6)
    return np.array([o[k] for k in FEATS] + [np.log(p1 / p2), np.log(p1 / (1 - p1 + 1e-6))], float)


class ValueModel:
    """Expected-reward labels fitted on calibration sessions only (no hindsight about the episode being labelled).

    P(top candidate is right | state) by logistic regression; the value of waiting by fitted value iteration with
    gradient boosting on the observed next state. Action = argmax of commit 2P-1, abstain -p_abstain, more E[V'] - cost.
    """

    def fit(self, states, cfg, iters: int = 6):
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import LogisticRegression

        X = np.stack([_x(o) for o, *_ in states])
        right = np.array([ok for _, ok, _, _ in states])
        nxt = np.array([n for *_, n, _ in states])
        dt = np.array([d for *_, d in states])
        self.clf = LogisticRegression(C=1.0, max_iter=2000).fit(X, right)
        qc = 2 * self.clf.predict_proba(X)[:, 1] - 1
        has = nxt >= 0
        v = np.maximum(qc, -cfg.p_abstain)
        for _ in range(iters):
            y = v[nxt[has]] - cfg.c_sec * dt[has] - cfg.c_turn
            self.reg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, min_samples_leaf=40,
                                                     random_state=0).fit(X[has], y)
            qm = np.full(len(X), -np.inf)
            qm[has] = self.reg.predict(X[has])
            v = np.maximum(np.maximum(qc, -cfg.p_abstain), qm)
        self.cfg = cfg
        return self

    def action(self, o: dict) -> str:
        x = _x(o)[None]
        q = [-float("inf") if o["evidence_exhausted"] else float(self.reg.predict(x)[0]),
             2 * float(self.clf.predict_proba(x)[0, 1]) - 1, -self.cfg.p_abstain]
        return ACTIONS[max(range(3), key=q.__getitem__)]

    def run(self, env) -> dict:
        obs = env.call("get_state")["observation"]
        n = 0
        while not env.done:
            a = self.action(obs)
            n += 1
            if a == "more":
                obs = env.call("request_more_eeg")["observation"]
            elif a == "commit":
                env.call("commit_character", {"character": obs["top_candidate"]})
            else:
                env.call("abstain")
        return {"policy": "value_labeller", "model_calls": 0, "model_latency_s": 0.0, "escalated": False}


def _all_states(eps, truth, cfg):
    """Every repetition of every calibration episode, with (right?, next index, seconds to next)."""
    states = []
    for sid, ev in eps.items():
        env = LocalEnv(ev, cfg)
        path = [env.call("get_state")["observation"]]
        while not path[-1]["evidence_exhausted"]:
            path.append(env.call("request_more_eeg")["observation"])
        base = len(states)
        for i, o in enumerate(path):
            last = i == len(path) - 1
            states.append((o, o["top_candidate"] == ev.chars[truth[sid]], -1 if last else base + i + 1,
                           0.0 if last else path[i + 1]["acquisition_s_so_far"] - o["acquisition_s_so_far"]))
    return states


def value_data():
    import pickle

    cfg = replay.RewardConfig()
    eps = replay.load_episodes(EVD, roles=("calibration",))
    truth = replay.load_ground_truth(EVD)
    states = _all_states(eps, truth, cfg)
    vm = ValueModel().fit(states, cfg)
    rows = [{"state": state_text(o), "action": vm.action(o)} for o, *_ in states]
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "train_value.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    (WORK / "value_model.pkl").write_bytes(pickle.dumps(vm))
    print(f"{len(eps)} calibration episodes -> {len(rows)} states; labels "
          f"{ {a: sum(x['action'] == a for x in rows) for a in ACTIONS} }")


def _load(adapter: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


def _prompt(tok, cfg, state: str) -> str:
    return tok.apply_chat_template([{"role": "system", "content": system_text(cfg)}, {"role": "user", "content": state}],
                                   tokenize=False, add_generation_prompt=True)


def _action_ids(tok) -> list[int]:
    return [tok(a, add_special_tokens=False).input_ids[0] for a in ACTIONS]


def train(epochs: int, lr: float, batch: int, rank: int, data_file: str, adapter: str):
    import torch
    from peft import LoraConfig, get_peft_model

    cfg = replay.RewardConfig()
    tok, model = _load(None)
    model.train()
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    rows = [json.loads(l) for l in (WORK / data_file).read_text().splitlines()]
    ids = _action_ids(tok)
    assert len(set(ids)) == 3, f"action words share a first token: {ids}"
    # no class weights: the labels are hindsight-optimal (commit exactly when the top candidate is right), so plain
    # cross-entropy learns P(oracle action | state) and the argmax becomes a stop rule; up-weighting commits would
    # teach it to guess
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rng = random.Random(0)
    t0, step = time.time(), 0
    for ep in range(epochs):
        rng.shuffle(rows)
        for i in range(0, len(rows), batch):
            b = rows[i:i + batch]
            enc = tok([_prompt(tok, cfg, r["state"]) for r in b], return_tensors="pt", padding=True).to("cuda")
            logits = model(**enc).logits[:, -1, :][:, ids].float()      # restricted to the three action words
            y = torch.tensor([ACTIONS.index(r["action"]) for r in b], device="cuda")
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1
            if step % 50 == 0:
                print(f"epoch {ep} step {step} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
    model.save_pretrained(WORK / adapter)
    print(f"saved {WORK / adapter} after {step} steps, {time.time() - t0:.0f}s")


class StudentPolicy:
    def __init__(self, tok, model, cfg, name):
        self.tok, self.model, self.cfg, self.name = tok, model, cfg, name
        self.ids = _action_ids(tok)

    def decide(self, obs: dict) -> tuple[str, float]:
        import torch

        enc = self.tok(_prompt(self.tok, self.cfg, state_text(obs)), return_tensors="pt").to("cuda")
        t0 = time.perf_counter()
        with torch.no_grad():
            s = self.model(**enc).logits[0, -1, self.ids].float()
        torch.cuda.synchronize()
        if obs["evidence_exhausted"]:
            s[0] = -1e9                                           # more is not possible
        return ACTIONS[int(s.argmax())], time.perf_counter() - t0

    def run(self, env) -> dict:
        obs = env.call("get_state")["observation"]
        calls, lat = 0, 0.0
        while not env.done:
            a, dt = self.decide(obs)
            calls += 1
            lat += dt
            if a == "more":
                obs = env.call("request_more_eeg")["observation"]
            elif a == "commit":
                env.call("commit_character", {"character": obs["top_candidate"]})
            else:
                env.call("abstain")
        return {"policy": self.name, "model_calls": calls, "model_latency_s": lat, "escalated": True}


def evaluate(which: list[str], tag: str = ""):
    import importlib.util

    import pandas as pd

    spec = importlib.util.spec_from_file_location("nce", ROOT / "scripts" / "benchmark" / "run_neuro_comm_eval.py")
    nce = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nce)
    cfg = replay.RewardConfig()
    eps = replay.load_episodes(EVD, roles=("evaluation",))
    truth = replay.load_ground_truth(EVD)
    rows = []
    import pickle

    for w in which:
        if w == "labeller":
            pol, model = pickle.loads((WORK / "value_model.pkl").read_bytes()), None
        else:
            tok, model = _load({"finetuned": str(WORK / "adapter"), "value": str(WORK / "adapter_value")}.get(w))
            pol = StudentPolicy(tok, model, cfg, f"nano4b_{w}")
        for sid in sorted(eps):
            r = nce.run_episode(eps[sid], truth[sid], pol, cfg)
            r.pop("events", None)
            rows.append(r)
        print(f"{w} done", flush=True)
        del model
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"episodes{tag}.csv", index=False)
    per = nce.summarize(df, 0.0)
    per.to_csv(OUT / f"per_participant{tag}.csv", index=False)
    agg = nce.aggregate(per)
    agg.to_csv(OUT / f"aggregate{tag}.csv", index=False)
    print(agg[["policy", "mean_reward", "accuracy", "incorrect_commits", "abstentions", "mean_eeg_latency_s",
               "mean_model_latency_s"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    sub.add_parser("data")
    sub.add_parser("value_data")
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--data", default="train.jsonl")
    t.add_argument("--adapter", default="adapter")
    e = sub.add_parser("eval")
    e.add_argument("--which", nargs="+", default=["base", "finetuned"])
    e.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.stage == "data":
        data()
    elif a.stage == "value_data":
        value_data()
    elif a.stage == "train":
        train(a.epochs, a.lr, a.batch, a.rank, a.data, a.adapter)
    else:
        evaluate(a.which, a.tag)
