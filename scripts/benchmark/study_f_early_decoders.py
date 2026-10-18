"""Experiment 2: regularized ERP decoders for early evidence, selected on calibration sessions only.

    /home/claude/venv-eeg/bin/python scripts/benchmark/study_f_early_decoders.py cache      # epochs -> data/processed/study_f_epochs (once)
    /home/claude/venv-eeg/bin/python scripts/benchmark/study_f_early_decoders.py select     # out-of-fold calibration comparison
    /home/claude/venv-eeg/bin/python scripts/benchmark/study_f_early_decoders.py finalize   # fit selected decoder, score session 3

Same preprocessing as the frozen evidence (services/eeg/study_f/evidence.py). Candidates are small regularized linear ERP
models; the existing two decoders are scored with the same criterion from their frozen out-of-fold calibration scores.

Selection criterion (pre-registered, calibration only): mean log posterior of the target character over flash prefixes
1..36 (about the first four repetitions, <= 9.5 s), alpha 1, Gaussian LLR fitted on out-of-fold calibration scores;
averaged per participant, then over participants. A new decoder is adopted only if it beats the best existing decoder on
that criterion. Session 3 is scored only in `finalize`, after `selection.json` is written, and only for the adopted model.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.eeg.study_f import evidence as ev  # noqa: E402
from services.eeg.study_f import flash_stopping as fs  # noqa: E402

AUD = ROOT / "data/processed/audit"
RAW = ROOT / "data/raw/bigp3bci"
CACHE = ROOT / "data/processed/study_f_epochs"
EVD = ROOT / "data/evidence/study_f"
OUT_EVD = ROOT / "data/evidence/study_f_early"
RES = ROOT / "evaluations/results/eeg_fast_selection"
EARLY_K = 36
ROLES = ("calibration", "evaluation", "evaluation_secondary")


# ---- candidates ---------------------------------------------------------------------------------------------------------

def _flat(X):
    return X.reshape(len(X), -1)


def _compact(X):
    """0.1-0.6 s window, 32 Hz (pairwise mean of the 64 Hz samples)."""
    w = X[:, :, 6:38]
    return w.reshape(len(w), w.shape[1], -1, 2).mean(-1).reshape(len(w), -1)


class XdawnSLDA:
    """xDAWN spatial filters (4 per class) -> filtered time courses -> shrinkage LDA."""

    def fit(self, X, y):
        from pyriemann.spatialfilters import Xdawn
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        self.xd = Xdawn(nfilter=4, estimator="lwf").fit(X, y)
        self.lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(_flat(self.xd.transform(X)), y)
        return self

    def decision_function(self, X):
        return self.lda.decision_function(_flat(self.xd.transform(X)))


class CompactSLDA:
    def fit(self, X, y):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        self.lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(_compact(X), y)
        return self

    def decision_function(self, X):
        return self.lda.decision_function(_compact(X))


class EnsembleLdaRiemann:
    """Mean of the two existing decoders' scores, each z-scored on its training fold."""

    def fit(self, X, y):
        self.ms = [ev.make_decoder("shrinkage_lda").fit(X, y), ev.make_decoder("xdawn_riemann_lr").fit(X, y)]
        s = [m.decision_function(X) for m in self.ms]
        self.mu, self.sd = [v.mean() for v in s], [v.std() for v in s]
        return self

    def decision_function(self, X):
        return np.mean([(m.decision_function(X) - mu) / sd for m, mu, sd in zip(self.ms, self.mu, self.sd)], 0)


CANDIDATES = {"xdawn_slda": XdawnSLDA, "compact_slda": CompactSLDA, "ens_lda_riemann": EnsembleLdaRiemann}


# ---- cache --------------------------------------------------------------------------------------------------------------

def tables():
    return {"splits": pd.read_csv(AUD / "splits_session_3way.csv"), "recordings": pd.read_csv(AUD / "recordings.csv"),
            "selections": pd.read_csv(AUD / "selections.csv"), "events": pd.read_csv(AUD / "stimulus_events.csv"),
            "labels": pd.read_csv(AUD / "verifier_flash_labels.csv")}


def cache():
    CACHE.mkdir(parents=True, exist_ok=True)
    tabs = tables()
    lab = tabs["labels"].set_index(["selection_id", "flash_index"])["is_target_flash"]
    for pid in sorted(tabs["recordings"].participant_id.unique()):
        if (CACHE / f"{pid}.npz").exists():
            continue
        t0 = time.time()
        split = tabs["splits"].query("participant_id == @pid and role in @ROLES")
        rec = tabs["recordings"].set_index("run_id")
        sel = tabs["selections"].query("participant_id == @pid").set_index("selection_id")
        e_all = tabs["events"][tabs["events"].selection_id.isin(sel.index)].copy()
        e_all["run_id"] = sel.loc[e_all.selection_id, "run_id"].values
        X, meta = [], []
        for run_id, role in zip(split.run_id, split.role):
            e = e_all[e_all.run_id == run_id]
            X.append(ev.epoch(ev.bandpass(ev.load_eeg(RAW / rec.loc[run_id, "path"])), e.onset_sample.values))
            m = e[["selection_id", "flash_index", "repetition_index", "onset_sample", "onset_s", "chars_shown"]].copy()
            m["role"], m["session"] = role, int(sel.loc[e.selection_id.iloc[0], "session"])
            meta.append(m)
        M = pd.concat(meta, ignore_index=True)
        cal = (M.role == "calibration").values
        y = np.full(len(M), -1)
        y[cal] = lab.loc[list(zip(M.selection_id[cal], M.flash_index[cal]))].values   # labels only for calibration rows
        np.savez(CACHE / f"{pid}.npz", X=np.concatenate(X).astype(np.float32), y=y)
        M.to_csv(CACHE / f"{pid}_meta.csv.gz", index=False)
        print(pid, len(M), f"{time.time() - t0:.0f}s", flush=True)


def load(pid):
    z = np.load(CACHE / f"{pid}.npz")
    return z["X"], z["y"], pd.read_csv(CACHE / f"{pid}_meta.csv.gz")


# ---- criterion ----------------------------------------------------------------------------------------------------------

def early_logpost(meta_cal: pd.DataFrame, llr: np.ndarray, targets: dict, k=EARLY_K) -> float:
    m = meta_cal.assign(llr=llr, decoder="x", participant_id="p")
    sels = fs.load_selections(m, "x")
    v = []
    for s in sels:
        L = s.evidence()[:k]
        z = L - L.max(1, keepdims=True)
        lp = z[:, targets[s.selection_id]] - np.log(np.exp(z).sum(1))
        v.append(lp.mean())
    return float(np.mean(v))


def oof_scores(make, X, y, sess):
    s = np.full(len(y), np.nan)
    for held in (1, 2):
        tr, te = sess != held, sess == held
        s[te] = make().fit(X[tr], y[tr]).decision_function(X[te])
    return s


def auc(s, y):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


def select():
    gt = pd.read_csv(EVD / "ground_truth.csv")
    targets = {str(a): int(b) - 1 for a, b in zip(gt.selection_id, gt.target_index)}
    frozen = pd.read_csv(EVD / "flash_evidence.csv.gz")
    frozen = frozen[frozen.role == "calibration"]
    rows = []
    oof_store = {}
    for pid in sorted(frozen.participant_id.unique()):
        X, y, M = load(pid)
        cal = (M.role == "calibration").values
        Mc, Xc, yc = M[cal].reset_index(drop=True), X[cal], y[cal]
        sess = Mc.session.values
        for dec in ev.DECODERS:          # existing decoders: frozen out-of-fold scores, same criterion
            f = frozen[(frozen.participant_id == pid) & (frozen.decoder == dec)].set_index(["selection_id", "flash_index"])
            s = f.loc[list(zip(Mc.selection_id, Mc.flash_index)), "score"].to_numpy(float)
            g = ev.GaussianLLR().fit(s, yc)
            rows.append({"participant_id": pid, "decoder": dec, "existing": True, "oof_auc": auc(s, yc),
                         "early_logpost": early_logpost(Mc, g(s), targets)})
        for name, make in CANDIDATES.items():
            t0 = time.time()
            s = oof_scores(make, Xc, yc, sess)
            g = ev.GaussianLLR().fit(s, yc)
            oof_store[(pid, name)] = s
            rows.append({"participant_id": pid, "decoder": name, "existing": False, "oof_auc": auc(s, yc),
                         "early_logpost": early_logpost(Mc, g(s), targets), "fit_s": time.time() - t0})
        print(pid, flush=True)
    df = pd.DataFrame(rows)
    RES.mkdir(parents=True, exist_ok=True)
    df.to_csv(RES / "early_decoder_selection_calibration.csv", index=False)
    summ = df.groupby(["decoder", "existing"])[["oof_auc", "early_logpost"]].mean().reset_index().sort_values("early_logpost", ascending=False)
    best_existing = summ[summ.existing].iloc[0]
    best_new = summ[~summ.existing].iloc[0]
    adopted = bool(best_new.early_logpost > best_existing.early_logpost)
    sel = {"criterion": f"mean log p(target) over flash prefixes 1..{EARLY_K}, out-of-fold calibration, per-participant mean",
           "summary": summ.to_dict("records"), "best_existing": best_existing.decoder, "best_new": best_new.decoder,
           "adopted": adopted, "decoder_for_session3": best_new.decoder if adopted else None,
           "chance_logpost": float(np.log(1 / 36))}
    (RES / "early_decoder_selection.json").write_text(json.dumps(sel, indent=1, default=float))
    print(summ.to_string(index=False))
    print("adopted:", adopted, best_new.decoder)


def finalize():
    sel = json.loads((RES / "early_decoder_selection.json").read_text())
    name = sel["decoder_for_session3"]
    if not name:
        print("no new decoder adopted; nothing to score on session 3")
        return
    OUT_EVD.mkdir(parents=True, exist_ok=True)
    flash, timing = [], {}
    for pid in sorted(p.stem for p in CACHE.glob("F_*.npz")):
        X, y, M = load(pid)
        cal = (M.role == "calibration").values
        sess = M.session.values
        s = np.full(len(M), np.nan)
        oof = oof_scores(CANDIDATES[name], X[cal], y[cal], sess[cal])
        model = CANDIDATES[name]().fit(X[cal], y[cal])
        s[~cal] = model.decision_function(X[~cal])
        s[cal] = oof
        g = ev.GaussianLLR().fit(oof, y[cal])
        out = M[["selection_id", "flash_index", "repetition_index", "onset_sample", "onset_s", "chars_shown", "role", "session"]].copy()
        out["decoder"], out["score"], out["llr"], out["participant_id"] = name, s, g(s), pid
        flash.append(out)
        # CPU scoring cost of one epoch (single-flash streaming), selected and existing decoders
        if not timing:
            for dn, mk in ((name, None), ("shrinkage_lda", "shrinkage_lda"), ("xdawn_riemann_lr", "xdawn_riemann_lr")):
                m = model if mk is None else ev.make_decoder(mk).fit(X[cal], y[cal])
                x1 = X[~cal][:1]
                for _ in range(5):
                    m.decision_function(x1)
                t0 = time.perf_counter()
                for i in range(200):
                    m.decision_function(X[~cal][i:i + 1])
                timing[dn] = (time.perf_counter() - t0) / 200 * 1e3
        print(pid, flush=True)
    flash = pd.concat(flash, ignore_index=True)
    flash.to_csv(OUT_EVD / f"flash_evidence_{name}.csv.gz", index=False, float_format="%.6g")
    blocks = ev.rep_blocks(flash)
    ids = sorted(blocks)
    np.savez_compressed(OUT_EVD / f"rep_evidence_{name}.npz", selection_id=np.array(ids),
                        delta=np.stack([blocks[i]["delta"] for i in ids]),
                        last_onset_s=np.stack([blocks[i]["last_onset_s"] for i in ids]),
                        n_flashes=np.stack([blocks[i]["n_flashes"] for i in ids]),
                        first_onset_s=np.array([blocks[i]["first_onset_s"] for i in ids]))
    (RES / "decoder_cpu_ms_per_epoch.json").write_text(json.dumps(timing, indent=1))
    print("wrote", OUT_EVD, timing)


if __name__ == "__main__":
    {"cache": cache, "select": select, "finalize": finalize}[sys.argv[1]]()
