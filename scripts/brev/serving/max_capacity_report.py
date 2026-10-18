"""Experiment D's report: serving/max_capacity.json, max_capacity.md and max_capacity.png from the ramps
max_capacity.sh wrote (max_capacity_parts/ramp_{shared,dedicated}.json) and the watchdog's gpu_trace.csv.
Missing parts are reported as missing, never filled in.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "evaluations" / "results" / "serving"
P = OUT / "max_capacity_parts"
SLO_P95_MS = 500
MODES = ("shared", "dedicated")


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def trace() -> list[dict]:
    try:
        with open(OUT / "gpu_trace.csv") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def peaks(rows: list[dict], t0: float, t1: float) -> dict:
    sel = [r for r in rows if r.get("time") and t0 <= float(r["time"]) <= t1 + 2]
    def mx(k):
        vals = [float(r[k]) for r in sel if r.get(k) not in (None, "", "[N/A]")]
        return max(vals) if vals else None
    return {"mem_mib": mx("mem_used_mib"), "temp_c": mx("temp_c"), "power_w": mx("power_w"), "util_pct": mx("util_pct"), "samples": len(sel)}


def mode_summary(mode: str, rows: list[dict]) -> dict:
    ramp, relaunch = load(P / f"ramp_{mode}.json"), load(P / f"relaunch_{mode}.json")
    safety = (P / f"stop_{mode}.txt").read_text().strip() if (P / f"stop_{mode}.txt").exists() else None
    if not ramp:
        return {"mode": mode, "missing": f"ramp_{mode}.json not written", "safety_stop": safety,
                "relaunch": {k: (relaunch or {}).get(k) for k in ("ok", "ready_s", "argv")} if relaunch else None}
    levels = []
    for lv in ramp["levels"]:
        levels.append({k: lv.get(k) for k in ("level", "n", "latency_ms", "ttft_ms", "req_per_s", "out_tok_per_s", "valid_rate",
                                             "expected_valid_rate", "valid_regressions", "same_as_first", "error_rate", "errors",
                                             "timeouts", "stopped_early", "error_examples")}
                      | {"peak": peaks(rows, lv["t_start"], lv["t_end"])})
    ok = [lv for lv in levels if (lv["error_rate"] or 0) <= 0.02 and not lv["stopped_early"]]
    slo = None
    for lv in levels:   # the highest level up to which every level met the SLO
        if lv in ok and (lv["latency_ms"]["p95"] or 1e9) < SLO_P95_MS:
            slo = lv["level"]
        else:
            break
    stop = ramp.get("stop")
    if safety:
        reason = f"{safety} (level {stop['level'] if stop else ramp.get('stopped_before', '?')})"
    elif stop:
        reason = f"level {stop['level']}: {stop['reason']}"
    elif ramp.get("failures"):
        reason = f"bench failed: {ramp['failures']}"
    else:
        reason = f"none: the ramp completed at level {levels[-1]['level']} (top level allowed by max-num-seqs)" if levels else "no levels ran"
    mns = ramp.get("max_num_seqs")
    return {"mode": mode, "server_args": ramp.get("server_args"), "max_num_seqs": mns,
            "relaunch": {k: (relaunch or {}).get(k) for k in ("ok", "ready_s")} if relaunch else None,
            "reference": ramp.get("reference"), "levels": levels,
            "max_concurrency_p95_under_500ms": slo, "peak_req_per_s": max((lv["req_per_s"] or 0) for lv in ok) if ok else None,
            "peak_out_tok_per_s": max((lv["out_tok_per_s"] or 0) for lv in ok) if ok else None,
            "limit_level": (stop or {}).get("level"), "stop_reason": reason, "safety_stop": safety,
            "valid_regressions_total": sum(lv["valid_regressions"] or 0 for lv in levels),
            "peak_mem_mib": max((lv["peak"]["mem_mib"] or 0) for lv in levels) if levels else None,
            "peak_temp_c": max((lv["peak"]["temp_c"] or 0) for lv in levels) if levels else None,
            "peak_power_w": max((lv["peak"]["power_w"] or 0) for lv in levels) if levels else None}


def plot(modes: list[dict], path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"shared": "#2a78d6", "dedicated": "#eb6834"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    ticks = set()
    for m in modes:
        lv = [x for x in m.get("levels", []) if x["latency_ms"]["p50"] is not None]
        if not lv:
            continue
        xs = [x["level"] for x in lv]
        ticks |= set(xs)
        c = colors[m["mode"]]
        a.plot(xs, [x["latency_ms"]["p50"] for x in lv], "-o", color=c, lw=2, ms=5, label=f"{m['mode']} p50")
        a.plot(xs, [x["latency_ms"]["p95"] for x in lv], "--o", color=c, lw=2, ms=5, label=f"{m['mode']} p95")
        b.plot(xs, [x["req_per_s"] for x in lv], "-o", color=c, lw=2, ms=5, label=f"{m['mode']} (max-num-seqs {m['max_num_seqs']})")
    a.axhline(SLO_P95_MS, color="#52514e", lw=1, ls=":")
    a.text(1, SLO_P95_MS * 1.08, "500 ms", color="#52514e", fontsize=8)
    a.set(xscale="log", yscale="log", xlabel="concurrent requests", ylabel="latency, ms", title="Coordinator latency vs concurrency")
    b.set(xscale="log", xlabel="concurrent requests", ylabel="requests / s", title="Throughput vs concurrency")
    b.set_ylim(bottom=0)
    for ax in (a, b):
        ax.set_xticks(sorted(ticks), [str(t) for t in sorted(ticks)])
        ax.grid(alpha=.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)


def main():
    rows = trace()
    modes = [mode_summary(m, rows) for m in MODES]
    res = {"experiment": "D: max-capacity ramp of the vLLM coordinator", "slo": f"p95 < {SLO_P95_MS} ms",
           "requests": "same as experiment A: NeMo Gym eval tasks through the coordinator's prompt, tools and validator, streamed",
           "levels": "1 2 4 (below the ramp, for the SLO point) then 8 16 32 64 128, up to 2 x max-num-seqs; 3 warm-up + max(20, 2 x level)",
           "stop_rules": {"safety (restore at once)": "CUDA error / illegal memory / OOM / Traceback in the vLLM log; GPU memory > 92 GiB; "
                                                    "temperature > 87 C; (shared) an agent's /health not ready twice in a row (2 s apart)",
                          "performance (end this mode's ramp)": "> 2% errors+timeouts at a level; p95 > 5 s"},
           "modes": modes, "restore": load(OUT / "restore_D.json"), "gpu_trace": "serving/gpu_trace.csv"}
    OUT.joinpath("max_capacity.json").write_text(json.dumps(res, indent=2))

    md = ["# D. Max-capacity ramp: vLLM coordinator (Nemotron 3.5 Lightning NVFP4)", "",
          "Same requests as A (coordinator prompt, tools and validator; temperature 0, thinking off, max_tokens 200, streamed). "
          f"Levels {res['levels']}. A watchdog sampled the GPU every 2 s (`gpu_trace.csv`).", "",
          "Kernels: Marlin weight-only FP4 + Triton attention (no native FP4/FlashInfer); numbers are conservative.", ""]
    for m in modes:
        md += [f"## {m['mode'].upper()}" + (f" (`{m.get('server_args')}`)" if m.get("server_args") else ""), ""]
        if "missing" in m:
            md += [f"Missing: {m['missing']}. Safety stop: {m.get('safety_stop')}. Relaunch: {m.get('relaunch')}.", ""]
            continue
        md += ["| Conc. | n | latency p50 / p95 / p99 ms | TTFT p50 / p95 / p99 ms | req/s | out tok/s | valid (expected) | regressions | errors | peak GiB / °C / W |",
               "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for lv in m["levels"]:
            L, T, pk = lv["latency_ms"], lv["ttft_ms"], lv["peak"]
            gib = f"{pk['mem_mib'] / 1024:.1f}" if pk["mem_mib"] else "-"
            md.append(f"| {lv['level']} | {lv['n']}{' (stopped)' if lv['stopped_early'] else ''} | {L['p50']} / {L['p95']} / {L['p99']} | "
                      f"{T['p50']} / {T['p95']} / {T['p99']} | {lv['req_per_s']} | {lv['out_tok_per_s']} | "
                      f"{100 * (lv['valid_rate'] or 0):.0f}% ({100 * (lv['expected_valid_rate'] or 0):.0f}%) | {lv['valid_regressions']} | "
                      f"{100 * (lv['error_rate'] or 0):.1f}% | {gib} / {pk['temp_c'] or '-'} / {pk['power_w'] or '-'} |")
        md += ["", f"- Max concurrency with p95 < {SLO_P95_MS} ms: **{m['max_concurrency_p95_under_500ms'] or 'none of the levels'}**; "
               f"peak throughput **{m['peak_req_per_s']} req/s** ({m['peak_out_tok_per_s']} output tok/s).",
               f"- Stop: {m['stop_reason']}.",
               f"- Peak GPU: {(m['peak_mem_mib'] or 0) / 1024:.1f} GiB, {m['peak_temp_c']} °C, {m['peak_power_w']} W.",
               f"- Reference (each task alone): valid {m['reference']}."]
        if m["valid_regressions_total"]:
            md.append(f"- **WARNING: {m['valid_regressions_total']} tool call(s) that are valid when sent alone were INVALID under load.**")
        else:
            md.append("- Valid-call rate under load equals the same tasks' rate when sent alone (0 regressions). It is below 100% because "
                      "Nemotron alone fails the validator on some task types (the app then falls back to rules).")
        md.append("")
    r = res["restore"] or {}
    md += [f"Restore: {'OK' if r.get('ok') else 'FAILED or missing'} ({r.get('safety_tests')}, all ready {r.get('all_ready')}, "
           f"vLLM default args {r.get('vllm_default_args')}).", "", "![latency and throughput](max_capacity.png)"]
    OUT.joinpath("max_capacity.md").write_text("\n".join(md) + "\n")
    if any(m.get("levels") for m in modes):
        plot(modes, OUT / "max_capacity.png")
    print("\n".join(md))


if __name__ == "__main__":
    main()
