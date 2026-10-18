"""Experiment C's report: serving/isolation.json + isolation.md from the parts isolation_experiment.sh wrote.

Baselines come from existing files (not re-run): latency_brev_final2.json (all-MPS, MTP off) and
coordinator_heldout_brev.json (held-out, guarded 0.985). Missing parts are reported as missing, never filled in.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "evaluations" / "results"
OUT = RES / "serving"
P = OUT / "isolation_parts"
TOL_TASKS = 1   # held-out "unchanged" = within one task's worth of reward


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def text(p: Path) -> str | None:
    return p.read_text().strip() if p.exists() else None


def main():
    base_lat, base_ho = load(RES / "latency_brev_final2.json"), load(RES / "coordinator_heldout_brev.json")
    lat, ho = load(OUT / "latency_isolated_mtp.json"), load(OUT / "heldout_isolated_mtp.json")
    fault, mps, relaunch = load(P / "fault.json"), load(P / "mps_check.json"), load(P / "relaunch.json")
    res = {"experiment": "C: vLLM isolated from MPS + MTP (num_speculative_tokens 1); small models under MPS",
           "compute_mode": (load(P / "compute_mode.json") or {}).get("compute_mode"),
           "vllm_relaunch": {k: relaunch.get(k) for k in ("ok", "ready_s", "env_changes", "argv")} if relaunch else None,
           "mps_check": mps, "restore": load(OUT / "restore_C.json"), "spec_decode_counters": text(P / "spec_decode.txt"),
           "vllm_log_after_kill": text(P / "vllm_after_kill.txt"), "agent_log_errors_after_kill": text(P / "agents_after_kill.txt")}

    rows = []
    if lat and base_lat:
        for label, path in (("coordinator alone", ("alone", "coordinator")), ("coordinator, every agent busy", ("together", "coordinator")),
                            ("voice, every agent busy", ("together", "audio")), ("lips, every agent busy", ("together", "vsr")),
                            ("Kokoro, every agent busy", ("together", "tts")), ("voice alone", ("alone", "audio")),
                            ("lips alone", ("alone", "vsr")), ("one message end to end", ("end_to_end", "total"))):
            a, b = base_lat[path[0]].get(path[1], {}), lat[path[0]].get(path[1], {})
            rows.append({"what": label, "all_mps": [a.get("p50"), a.get("p95")], "isolated_mtp": [b.get("p50"), b.get("p95")],
                         "p50_change_pct": round(100 * (b["p50"] - a["p50"]) / a["p50"], 1) if a.get("p50") and b.get("p50") else None})
        res["latency"] = {"baseline_file": "latency_brev_final2.json (n 20)", "file": "serving/latency_isolated_mtp.json (n 10)", "rows": rows,
                          "gpu_all_mps": base_lat["together"].get("_gpu"), "gpu_isolated": lat["together"].get("_gpu")}
    else:
        res["latency"] = {"missing": "latency_isolated_mtp.json not written (see .run/logs/serving_C.log)"}

    if ho and base_ho:
        g, b = ho["policies"]["guarded"], base_ho["policies"]["guarded"]
        tol = TOL_TASKS / ho["n_tasks"]
        res["heldout"] = {"n_tasks": ho["n_tasks"], "baseline": b["reward_mean"], "isolated_mtp": g["reward_mean"],
                          "unsafe": g["unsafe_rate"], "valid": g["valid_rate"], "llm_share": g.get("llm_share"),
                          "p50_ms": g["latency_ms_median"], "baseline_p50_ms": b["latency_ms_median"],
                          "unchanged": g["reward_mean"] >= b["reward_mean"] - tol and g["unsafe_rate"] == 0,
                          "tolerance": f"one task ({tol:.3f})",
                          "by_category_diff": {k: [b["by_category"].get(k), v] for k, v in g["by_category"].items() if v != b["by_category"].get(k)}}
    else:
        res["heldout"] = {"missing": "heldout_isolated_mtp.json not written"}
    res["fault_test"] = fault or {"missing": "fault test did not run"}

    # recommendation, from the measurements only
    contained = bool(fault and fault.get("pass"))
    small_rows = [r for r in rows if r["what"] in ("voice, every agent busy", "lips, every agent busy", "Kokoro, every agent busy")]
    small_worse = [r for r in small_rows if (r["p50_change_pct"] or 0) > 20]
    coord = next((r for r in rows if r["what"] == "coordinator, every agent busy"), None)
    mtp_ok = bool(res["heldout"].get("unchanged")) and coord is not None and (coord["p50_change_pct"] or 0) < 0
    if not relaunch or not relaunch.get("ok"):
        rec, why = "keep all-MPS", "the isolated vLLM did not start, so nothing was measured"
    elif not contained:
        rec, why = "keep all-MPS", "killing vLLM still disturbed the other agents, so isolation did not contain the fault"
    elif small_worse:
        rec, why = "keep all-MPS (isolation only with a second GPU or MIG)", (
            "the fault was contained, but outside MPS vLLM time-slices with the small models: "
            + ", ".join(f"{r['what']} {r['p50_change_pct']:+.0f}% p50" for r in small_worse))
    elif mtp_ok:
        rec, why = "isolate + MTP", ("fault contained, the small models kept their latency, the coordinator is faster and the held-out "
                                     "reward is unchanged. One run each; MTP's earlier crash is not explained, so keep the flag off "
                                     "until it survives a longer soak")
    else:
        rec, why = "isolate vLLM (MTP off)", "fault contained and the small models kept their latency; MTP did not show a clear win here"
    res["recommendation"] = {"choice": rec, "why": why, "defaults_changed": False}
    OUT.joinpath("isolation.json").write_text(json.dumps(res, indent=2))

    md = ["# C. vLLM outside MPS + MTP: fault isolation", "",
          f"Compute mode: {res['compute_mode']}. vLLM restarted from its own running command line with `CUDA_MPS_PIPE_DIRECTORY`/"
          "`CUDA_MPS_LOG_DIRECTORY` at an empty temp dir (own CUDA context, not an MPS client) and "
          "`--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":1}'`; voice, lips and Kokoro stayed MPS clients.", ""]
    if mps:
        md += [f"MPS check: EngineCore pid {mps['engine_pid']}, MPS clients {mps['mps_client_pids']}, nvidia-smi `{mps['nvidia_smi_engine_row']}` "
               f"-> {'still an MPS client' if mps['engine_is_mps_client'] else 'not an MPS client'}.", ""]
    if rows:
        md += ["## Latency (p50 / p95 ms)", "", "| | all-MPS, MTP off (`latency_brev_final2.json`) | vLLM isolated + MTP | p50 change |", "|---|---:|---:|---:|"]
        md += [f"| {r['what']} | {r['all_mps'][0]} / {r['all_mps'][1]} | {r['isolated_mtp'][0]} / {r['isolated_mtp'][1]} | "
               f"{'' if r['p50_change_pct'] is None else f'{r['p50_change_pct']:+.0f}%'} |" for r in rows]
        md += ["", f"n = 10 per row here vs 20 in the baseline; run-to-run variation is about ±10% (SERVING.md).", ""]
    h = res["heldout"]
    if "missing" not in h:
        md += ["## NeMo Gym held-out (guarded = deployed coordinator)", "",
               f"{h['n_tasks']} tasks: **{h['isolated_mtp']:.3f}** vs {h['baseline']:.3f} (`coordinator_heldout_brev.json`), unsafe {h['unsafe']:.2f}, "
               f"Nemotron decided {h['llm_share']}, p50 {h['p50_ms']} vs {h['baseline_p50_ms']} ms. "
               f"{'Unchanged' if h['unchanged'] else 'Lower'} (tolerance {h['tolerance']}). Categories that moved: {h['by_category_diff'] or 'none'}.", ""]
    if fault and "checks" in fault:
        md += ["## Fault test: `kill -9` of the vLLM EngineCore", "", "| Check | Result | Seconds after the kill | Detail |", "|---|---|---:|---|"]
        md += [f"| {k} | {'PASS' if v['ok'] else 'FAIL'} | {v['s_after_kill']} | {v['detail']} |" for k, v in fault["checks"].items()]
        md += ["", f"Small-model processes alive afterwards: {fault['small_model_pids_alive']}. Overall: **{'PASS' if fault['pass'] else 'FAIL'}**.",
               "", "A `kill -9` is a process death, not the CUDA fault (`cudaErrorLaunchFailure`) MTP once hit; under MPS such a fault "
               "reaches the shared server. This test shows the others survive vLLM dying; it does not reproduce that fault.", ""]
    r = res["restore"] or {}
    md += ["## Recommendation", "", f"**{rec}**: {why}. Defaults were not changed.", "",
           f"Restore: {'OK' if r.get('ok') else 'FAILED or missing'} ({r.get('safety_tests')}, all ready: {r.get('all_ready')}, MTP on: {r.get('mtp_on')}, "
           f"vLLM default args: {r.get('vllm_default_args')})."]
    OUT.joinpath("isolation.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
