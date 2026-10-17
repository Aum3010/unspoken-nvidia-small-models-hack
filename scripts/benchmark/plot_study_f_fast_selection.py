"""Figures for evaluations/results/eeg_fast_selection (session 3 Static test, held out).

    /home/claude/venv-eeg/bin/python scripts/benchmark/plot_study_f_fast_selection.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "evaluations/results/eeg_fast_selection"
DECS = ["xdawn_riemann_lr", "shrinkage_lda", "ens_lda_riemann"]
BASE = "baseline_rep_bayes_tau0.99_commit"
FAM_STYLE = {"sweep_rep_alpha1_commit": ("repetition, commit on exhaustion (original family)", "#888888", "--"),
             "sweep_flash_alpha1_commit": ("flash, commit on exhaustion", "#4477aa", "-"),
             "sweep_rep_participant_abstain": ("repetition, participant alpha, abstain", "#cc8844", "--"),
             "sweep_flash_participant_abstain": ("flash, participant alpha, abstain", "#228833", "-")}
RISK_MARK = {0.0: "o", 0.02: "s", 0.05: "^", 0.1: "D"}


def pareto(sweep, agg):
    fig, axes = plt.subplots(len(DECS), 2, figsize=(12, 4.2 * len(DECS)))
    for row, dec in zip(axes, DECS):
        s = sweep[(sweep.decoder == dec) & (sweep.role == "evaluation")]
        a = agg[(agg.decoder == dec) & (agg.role == "evaluation")]
        for fam, (lab, col, ls) in FAM_STYLE.items():
            f = s[s.family == fam].sort_values("lat_median")
            row[0].plot(f.lat_median, f.wrong, ls, color=col, marker=".", label=lab)
            row[1].plot(f.lat_median, f.accuracy, ls, color=col, marker=".", label=lab)
        for fam, col in (("flash_participant", "#228833"), ("flash_global", "#aa3377"), ("flash_hybrid", "#66ccee"), ("rep_participant", "#cc8844")):
            for _, r in a[a.family == fam].iterrows():
                for ax, y in ((row[0], r.wrong), (row[1], r.accuracy)):
                    ax.scatter(r.lat_median, y, marker=RISK_MARK[r.risk_target], s=70, facecolor="none", edgecolor=col, lw=1.6)
        b = a[a.policy == BASE].iloc[0]
        for ax, y in ((row[0], b.wrong), (row[1], b.accuracy)):
            ax.scatter(b.lat_median, y, marker="*", s=220, color="k", zorder=5, label="original baseline (bayes tau 0.99)")
        row[0].set(xlabel="median decision latency per attempted selection (s)", ylabel="incorrect commits per 120 attempted",
                   title=f"{dec}: errors vs latency")
        row[1].set(xlabel="median decision latency per attempted selection (s)", ylabel="accuracy (correct / attempted)",
                   title=f"{dec}: accuracy vs latency")
        for ax in row:
            ax.grid(alpha=0.3)
    h = [plt.Line2D([], [], color=c, ls=l, marker=".", label=lab) for lab, c, l in FAM_STYLE.values()]
    h += [plt.Line2D([], [], color="k", marker="*", ls="", ms=12, label="original baseline")]
    h += [plt.Line2D([], [], color="#555", marker=m, ls="", mfc="none", label=f"calibration-chosen, risk target {r:g}") for r, m in RISK_MARK.items()]
    h += [plt.Line2D([], [], color=c, marker="o", ls="", mfc="none", label=f) for f, c in
          (("flash participant", "#228833"), ("flash global", "#aa3377"), ("flash hybrid", "#66ccee"), ("rep participant", "#cc8844"))]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle("Session 3 Static (held out), 120 selections. Lines: one tau swept on session 3 (descriptive only). "
                 "Markers: thresholds chosen on sessions 1-2.", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(R / "pareto_latency_errors_accuracy.png", dpi=120)
    plt.close(fig)


def prefix(pc, oracle):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, dec in zip(axes, ["xdawn_riemann_lr", "ens_lda_riemann"]):
        p = pc[(pc.decoder == dec) & (pc.role == "evaluation")]
        for pid, g in p.groupby("participant_id"):
            ax.plot(g.t_dec, g.accuracy, lw=1, label=pid)
        m = p.groupby("t_dec").accuracy.mean()
        ax.plot(m.index, m.values, "k", lw=2.5, label="mean")
        ax.axhline(1 / 36, color="grey", ls=":", label="chance")
        ax.set(xlabel="decision time after flash k (s, recorded onset + 0.7 s)", title=f"{dec}: argmax accuracy at every flash prefix")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("fraction of selections whose argmax is the target")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(R / "prefix_accuracy_per_participant.png", dpi=120)
    plt.close(fig)


def cdf(sel, oracle):
    fig, ax = plt.subplots(figsize=(8, 5))
    t = np.linspace(0, 32, 400)
    e = sel[sel.role == "evaluation"]
    for dec, pol, lab, col, ls in (("xdawn_riemann_lr", BASE, "original baseline (xdawn, tau 0.99, commit)", "k", "-"),
                                   ("xdawn_riemann_lr", "flash_participant_risk0.02", "xdawn, flash participant, risk 2%", "#4477aa", "-"),
                                   ("ens_lda_riemann", "flash_participant_risk0.02", "ensemble, flash participant, risk 2%", "#228833", "-")):
        g = e[(e.decoder == dec) & (e.policy == pol)]
        ax.plot(t, [(g.correct & (g.t_dec <= x)).mean() for x in t], ls, color=col, label=f"{lab}: correct commits")
        ax.plot(t, [(g.wrong & (g.t_dec <= x)).mean() for x in t], ":", color=col, label=f"{lab}: incorrect commits")
    o = oracle[(oracle.role == "evaluation") & (oracle.decoder == "ens_lda_riemann")]
    ax.plot(t, [(o.stable_correct_t <= x).mean() for x in t], color="#aa3377", ls="--",
            label="ensemble: argmax already stays correct (information reference, not a policy)")
    for x in (1, 2, 3, 5):
        ax.axvline(x, color="grey", lw=0.6, alpha=0.5)
    ax.set(xlabel="decision time (s)", ylabel="fraction of the 120 attempted selections", title="When are characters committed? (session 3)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(R / "latency_cdf.png", dpi=120)
    plt.close(fig)


def per_participant(per):
    e = per[per.role == "evaluation"]
    pols = [("xdawn_riemann_lr", BASE, "baseline"), ("xdawn_riemann_lr", "flash_participant_risk0.02", "xdawn flash p 2%"),
            ("ens_lda_riemann", "flash_participant_risk0.02", "ensemble flash p 2%")]
    pids = sorted(e.participant_id.unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    w = 0.27
    for i, (dec, pol, lab) in enumerate(pols):
        g = e[(e.decoder == dec) & (e.policy == pol)].set_index("participant_id").loc[pids]
        x = np.arange(len(pids)) + (i - 1) * w
        axes[0].bar(x, g.ccpm, w, label=lab)
        axes[1].bar(x, g.wrong, w, label=lab)
        axes[2].bar(x, g.abstained, w, label=lab)
    for ax, t in zip(axes, ("correct characters per minute", "incorrect commits (of 12)", "abstentions / yields (of 12)")):
        ax.set_xticks(range(len(pids)), pids, rotation=45)
        ax.set_title(t)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(R / "per_participant.png", dpi=120)
    plt.close(fig)


def main():
    sweep = pd.read_csv(R / "descriptive_frontier_sweep.csv")
    agg = pd.read_csv(R / "aggregate.csv")
    sel = pd.read_csv(R / "per_selection.csv.gz")
    oracle = pd.read_csv(R / "oracle_information_times.csv")
    pc = pd.read_csv(R / "prefix_accuracy_per_participant.csv")
    per = pd.read_csv(R / "per_participant.csv")
    pareto(sweep, agg)
    prefix(pc, oracle)
    cdf(sel, oracle)
    per_participant(per)
    print("figures in", R)


if __name__ == "__main__":
    main()
