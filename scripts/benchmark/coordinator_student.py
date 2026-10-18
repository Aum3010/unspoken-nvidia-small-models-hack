"""E7b: Nemotron Nano 4B as the E7 coordinator, LoRA-trained on one speaker, tested on the other (both folds).

Labels are the safe rule's decisions on the calibration speaker (commit above the fused 2% risk threshold, otherwise
confirm the top k with EEG). The student reads the same text the 30B coordinator read (voice, lips and combined top-5)
and scores the three action words, as in neuro_comm_student.py. Question: can a 4B replace the 30B as coordinator and
keep the rule's safety, at a fraction of the latency?

Stages:
  prep   (Mac)  data/coord_student/{s1,s2}_{train,test}.jsonl from E1 scores; fold = test speaker
  run    (GPU)  per fold: LoRA on the train file, score the test file -> {fold}_pred.jsonl
  eval   (Mac)  replay predictions through coordinated_cascade.play; compare with E7 policies
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "data" / "coord_student"
MODEL = "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"
ACTIONS = ["commit", "confirm", "abstain"]
SYSTEM_SUFFIX = "\nReply with one word: commit (speak option 1), confirm (show the top options for EEG selection), or abstain."


def _cc():
    sys.path.insert(0, str(ROOT))
    s = importlib.util.spec_from_file_location("cc", ROOT / "scripts" / "benchmark" / "coordinated_cascade.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def system_text(cc) -> str:
    return "detailed thinking off\n" + cc.SYSTEM.split("Reply with JSON only")[0].rstrip() + SYSTEM_SUFFIX


def episodes():
    """Per fold, every (clip, condition) with the 30B's prompt, the rule's action and what replay needs."""
    import numpy as np

    cc = _cc()
    avf = cc.avf
    ids, cand, y, spk, L, V, ms, conds = cc.load()
    e_cal = json.loads((ROOT / "evaluations" / "results" / "neuro_comm_student" / "eeg_calibration_rates.json").read_text())
    out = {}
    for test in sorted(set(spk)):
        cal = spk != test
        Vc = np.concatenate([V[c][cal] for c in conds])
        Lc, yc = np.concatenate([L[cal]] * len(conds)), np.concatenate([y[cal]] * len(conds))
        tv, tl = avf._fit_t(Vc, yc), avf._fit_t(L[cal], y[cal])
        Fc = avf._logsm(avf._logsm(Vc / tv) + avf._logsm(Lc / tl))
        tau = avf._tau(Fc, yc, 0.02)
        pf_c = np.exp(Fc)
        unsure = pf_c.max(1) < tau
        ranks = np.argsort(-pf_c, axis=1)
        k = max((2, 3, 4), key=lambda k: np.where((ranks[unsure, :k] == yc[unsure, None]).any(1),
                                                   cc.expected_confirm(1.0, k, e_cal), cc.expected_confirm(0.0, k, e_cal)).mean())
        rows = []
        for ci, c in enumerate(conds):
            for i in range(len(ids)):
                pv = np.exp(avf._logsm(V[c][i] / tv))
                pl = np.exp(avf._logsm(L[i] / tl))
                pf = np.exp(avf._logsm(np.log(pv) + np.log(pl)))
                txt, order = cc.prompt(cand[ids[i]]["cands"], pv, pl, pf)
                rows.append({"i": i, "clip": ids[i], "speaker": str(spk[i]), "condition": c, "ci": ci, "truth": int(y[i]),
                             "order": [int(o) for o in order], "k": int(k), "state": f"Candidates:\n{txt}\nDecide.",
                             "label": "commit" if pf.max() >= tau else "confirm"})
        out[test] = ([r for r in rows if r["speaker"] != test], [r for r in rows if r["speaker"] == test])
    return out, cc


def prep():
    eps, cc = episodes()
    WORK.mkdir(parents=True, exist_ok=True)
    for fold, (tr, te) in eps.items():
        for name, rows in (("train", tr), ("test", te)):
            (WORK / f"{fold}_{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        print(fold, "train", len(tr), {a: sum(r["label"] == a for r in tr) for a in ACTIONS}, "test", len(te), "k", tr[0]["k"])
    (WORK / "system.txt").write_text(system_text(cc))


def run(epochs: int, lr: float, batch: int, rank: int):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    system = (WORK / "system.txt").read_text()
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = [tok(a, add_special_tokens=False).input_ids[0] for a in ACTIONS]
    assert len(set(ids)) == 3, ids
    prompt = lambda s: tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": s}],
                                               tokenize=False, add_generation_prompt=True)
    for fold in ("s1", "s2"):
        tr = [json.loads(l) for l in (WORK / f"{fold}_train.jsonl").read_text().splitlines()]
        te = [json.loads(l) for l in (WORK / f"{fold}_test.jsonl").read_text().splitlines()]
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
        model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        # zero-shot scores first (adapter disabled), then train, then fine-tuned scores
        preds = {"base": [], "tuned": []}

        def score(tag):
            model.eval()
            for r in te:
                enc = tok(prompt(r["state"]), return_tensors="pt").to("cuda")
                t0 = time.perf_counter()
                with torch.no_grad():
                    s = model(**enc).logits[0, -1, ids].float()
                torch.cuda.synchronize()
                preds[tag].append({"clip": r["clip"], "condition": r["condition"], "action": ACTIONS[int(s.argmax())],
                                   "ms": (time.perf_counter() - t0) * 1000})

        with model.disable_adapter():
            score("base")
        model.train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
        rng = random.Random(0)
        t0, step = time.time(), 0
        for _ in range(epochs):
            rng.shuffle(tr)
            for i in range(0, len(tr), batch):
                b = tr[i:i + batch]
                enc = tok([prompt(r["state"]) for r in b], return_tensors="pt", padding=True).to("cuda")
                logits = model(**enc).logits[:, -1, :][:, ids].float()
                y = torch.tensor([ACTIONS.index(r["label"]) for r in b], device="cuda")
                loss = torch.nn.functional.cross_entropy(logits, y)
                loss.backward()
                opt.step()
                opt.zero_grad()
                step += 1
                if step % 20 == 0:
                    print(f"{fold} step {step} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
        score("tuned")
        for tag, p in preds.items():
            (WORK / f"{fold}_pred_{tag}.jsonl").write_text("\n".join(json.dumps(x) for x in p) + "\n")
        print(f"{fold}: trained {step} steps in {time.time() - t0:.0f}s", flush=True)
        del model, opt
        torch.cuda.empty_cache()


def evaluate():
    import numpy as np
    import pandas as pd

    cc = _cc()
    out = cc.OUT
    eeg = pd.read_csv(cc.EEG_TEST).sort_values("episode_id").reset_index(drop=True)
    ids, *_ , ms, conds = cc.load()
    rows = []
    for fold in ("s1", "s2"):
        te = {(r["clip"], r["condition"]): r for r in map(json.loads, (WORK / f"{fold}_test.jsonl").read_text().splitlines())}
        for tag in ("base", "tuned"):
            for p in map(json.loads, (WORK / f"{fold}_pred_{tag}.jsonl").read_text().splitlines()):
                r = te[(p["clip"], p["condition"])]
                act, shown = p["action"], (r["order"][:r["k"]] if p["action"] == "confirm" else [])
                ep_eeg = eeg.iloc[(r["i"] * len(conds) + r["ci"]) % len(eeg)]
                o = cc.play(act, r["order"][0], shown, r["truth"], ep_eeg, np.random.default_rng(r["i"] * 31 + r["ci"]))
                audio = max(ms["voice"][r["i"]], ms["lips"][r["i"]]) / 1000
                rows.append({"policy": f"nano4b_{tag}", "clip": r["clip"], "speaker": fold, "condition": r["condition"],
                             "action": act, **o, "reward": cc.reward(o), "latency_s": audio + p["ms"] / 1000 + o["eeg_s"],
                             "model_s": p["ms"] / 1000, "valid": True, "fit": ""})
    new = pd.DataFrame(rows)
    old = pd.read_csv(out / "episodes_nemotron30b.csv")
    df = pd.concat([old[old.policy.isin(["fused", "cascade", "nemotron30b", "lips"])], new], ignore_index=True)
    df["wrong"] = df.spoken.notna() & ~df.right
    df["nothing"] = df.spoken.isna()
    new.to_csv(out / "episodes_nano4b.csv", index=False)
    tot = df.groupby("policy").agg(n=("right", "size"), right=("right", "mean"), wrong=("wrong", "sum"),
                                   nothing=("nothing", "mean"), confirmed=("confirmed", "mean"),
                                   latency_mean=("latency_s", "mean"), model_ms_p50=("model_s", lambda v: 1000 * v.median()),
                                   reward=("reward", "mean")).reset_index()
    tot.to_csv(out / "overall_nano4b.csv", index=False)
    pd.set_option("display.width", 200)
    print(tot.round(3).to_string(index=False))
    rule = df[df.policy == "cascade"].set_index(["clip", "condition"]).sort_index()
    tuned = df[df.policy == "nano4b_tuned"].set_index(["clip", "condition"]).sort_index()
    print("agreement with rule:", round(float((rule.action == tuned.action).mean()), 4))
    rng = np.random.default_rng(0)
    for col in ("right", "wrong"):
        x = (tuned.groupby("clip")[col].sum() - rule.groupby("clip")[col].sum()).to_numpy()
        bs = [x[rng.integers(0, len(x), len(x))].sum() for _ in range(5000)]
        print(f"nano4b_tuned - cascade {col}: {int(x.sum()):+d}, 95% CI {np.percentile(bs, [2.5, 97.5]).round(0)}")
    by = df.groupby(["policy", "condition"]).right.mean().unstack(0).reindex(conds)
    print(by.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    sub.add_parser("prep")
    r = sub.add_parser("run")
    r.add_argument("--epochs", type=int, default=2)
    r.add_argument("--lr", type=float, default=2e-4)
    r.add_argument("--batch", type=int, default=16)
    r.add_argument("--rank", type=int, default=16)
    sub.add_parser("eval")
    a = ap.parse_args()
    {"prep": prep, "eval": evaluate}.get(a.stage, lambda: run(a.epochs, a.lr, a.batch, a.rank))()
