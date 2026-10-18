"""evaluations/results/serving/SUMMARY.md from the four experiments' result files (A, B, C, D) and final_check.json.

    python scripts/brev/serving/build_summary.py [--messages-per-hour 20] [--coord-fraction 0.05] [--headroom 1.0]

Rooms per GPU is an ESTIMATE for the coordinator only: rooms = capacity x headroom / (messages per patient-hour x
coordinator fraction / 3600). Capacity = req/s at the highest level whose p95 < 500 ms (and, as an upper bound,
the peak req/s). The coordinator fraction 2/40 is the share of messages Nemotron planned in the ad-hoc voice bench of
the same run as latency_brev_final2.json (TECHNICAL_VERBOSE §4); the conservative variant assumes every message needs it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "evaluations" / "results" / "serving"
KERNELS = "Kernels: Marlin weight-only FP4 + Triton attention (no native FP4/FlashInfer); numbers are conservative."


def load(name: str):
    try:
        return json.loads((OUT / name).read_text())
    except Exception:
        return None


def ms(d: dict, k: str = "p50"):
    return "-" if not d or d.get(k) is None else f"{d[k]:.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--messages-per-hour", type=float, default=20)
    ap.add_argument("--coord-fraction", type=float, default=2 / 40)
    ap.add_argument("--headroom", type=float, default=1.0, help="share of the measured capacity to plan on (1.0 = all of it)")
    args = ap.parse_args()
    A, B, C, D, F = load("concurrency.json"), load("panel_check.json"), load("isolation.json"), load("max_capacity.json"), load("final_check.json")
    md = ["# Serving experiments around the vLLM coordinator (Nemotron 3.5 Lightning NVFP4)", "",
          "Brev `marginal-sapphire-whale`, RTX PRO 6000 Blackwell 96 GB, branch `serving-experiments` (from `demo-final`). "
          "Client on the box, warm. Baselines from existing files, not re-run: `latency_brev_final2.json` (latency), "
          "`coordinator_heldout_brev.json` (held-out 0.985). Every number below is in a JSON file next to this one.", "",
          KERNELS, ""]

    # ---- A
    md += ["## A. Concurrency on the live vLLM (`--max-num-seqs 8`, every agent loaded, MTP off)", ""]
    if A:
        md += ["| Concurrent | p50 / p95 / p99 latency ms | p50 / p95 TTFT ms | req/s | output tok/s | valid call | same answer as at 1 | errors |",
               "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for lv in A["levels"]:
            L, T = lv["latency_ms"], lv["ttft_ms"]
            same = "-" if lv.get("same_as_first") is None else f"{100 * lv['same_as_first']:.0f}%"
            md.append(f"| {lv['level']} | {ms(L)} / {ms(L, 'p95')} / {ms(L, 'p99')} | {ms(T)} / {ms(T, 'p95')} | {lv['req_per_s']} | "
                      f"{lv['out_tok_per_s']} | {100 * lv['valid_rate']:.0f}% | {same} | {lv['errors'] + lv['timeouts']} |")
        md += ["", f"- {A['levels'][0]['n']} measured requests per level after 3 warm-up; the same {A['n_tasks']}-task NeMo Gym "
               "sequence at every level; 0 prefix-cache hits (as documented: 4,224-token blocks, ~1,700-token prompt).",
               "- Throughput stops growing at 8 concurrent requests: that is the server's `--max-num-seqs 8`, not the GPU (see D). "
               "Level 16 only queues.",
               "- Valid call = parses and passes the coordinator's validator. 85% at every level: the same tasks fail alone "
               "(Nemotron re-asks after a 'none of these'); in the app they fall back to the rules. Load did not change validity.",
               "- A first run with unshuffled tasks (eval.jsonl is sorted by category) gave 70% at level 4 for that reason; it was replaced.", ""]
    else:
        md += ["Missing: concurrency.json.", ""]

    # ---- D
    md += ["## D. Max-capacity ramp", ""]
    rooms_rows = []
    if D and not D.get("skipped"):
        md += ["| Mode | vLLM | max concurrency at p95 < 500 ms | its req/s | peak req/s (output tok/s) | stop | peak GPU |", "|---|---|---:|---:|---:|---|---|"]
        for m in D["modes"]:
            if "missing" in m:
                md.append(f"| {m['mode']} | - | - | - | - | {m['missing']}; {m.get('safety_stop')} | - |")
                continue
            slo_lv = next((lv for lv in m["levels"] if lv["level"] == m["max_concurrency_p95_under_500ms"]), None)
            util = (m.get("server_args") or "").split("--gpu-memory-utilization")[-1].split()[0] if "--gpu-memory-utilization" in (m.get("server_args") or "") else "?"
            md.append(f"| {m['mode']} | max-num-seqs {m['max_num_seqs']}, util {util} | {m['max_concurrency_p95_under_500ms'] or 'none'} | "
                      f"{slo_lv['req_per_s'] if slo_lv else '-'} | {m['peak_req_per_s']} ({m['peak_out_tok_per_s']}) | {m['stop_reason']} | "
                      f"{(m['peak_mem_mib'] or 0) / 1024:.1f} GiB, {m['peak_temp_c']} °C, {m['peak_power_w']} W |")
            rooms_rows.append((m["mode"], slo_lv["req_per_s"] if slo_lv else None, m["peak_req_per_s"]))
            if m["valid_regressions_total"]:
                md.append(f"| | **WARNING: {m['valid_regressions_total']} call(s) valid alone were invalid under load** | | | | | |")
        md += ["", "Per-level tables (p50/p95/p99 latency and TTFT, req/s, tok/s, valid rate, errors, peak memory/temperature/power): "
               "`max_capacity.md`; plot `max_capacity.png`; GPU every 2 s: `gpu_trace.csv`. Levels 1, 2, 4 were added below the "
               "8→128 ramp because p95 already passes 500 ms at 4 concurrent requests in A.", ""]
    else:
        md += [f"Not run: {(D or {}).get('reason', 'max_capacity.json missing')}.", ""]

    # ---- rooms
    lam = args.messages_per_hour / 3600
    md += ["## Rooms per GPU (ESTIMATE, coordinator only)", "",
           f"Assumptions (flags of `build_summary.py`): {args.messages_per_hour:g} messages per patient per hour; coordinator fraction "
           f"{args.coord_fraction:.3f} (2 of 40 messages needed Nemotron in the voice bench of the `latency_brev_final2.json` run; accepted "
           f"readings are decided by rules) and a conservative 1.0; headroom {args.headroom:g} of the measured capacity; arrivals spread "
           "evenly. Capacity = closed-loop req/s at the highest concurrency with p95 < 500 ms; 'peak' ignores latency.", "",
           "| Mode | capacity (req/s) | rooms, fraction 2/40 | rooms, fraction 1 |", "|---|---:|---:|---:|"]
    if not rooms_rows and A:
        rooms_rows.append(("shared (from A only)", next((lv["req_per_s"] for lv in reversed(A["levels"]) if lv["latency_ms"]["p95"] < 500), None), None))
    for mode, slo_rps, peak in rooms_rows:
        for label, cap in (("at p95 < 500 ms", slo_rps), ("peak", peak)):
            if cap:
                md.append(f"| {mode}, {label} | {cap} | {cap * args.headroom / (lam * args.coord_fraction):,.0f} | {cap * args.headroom / lam:,.0f} |")
            else:
                md.append(f"| {mode}, {label} | none | - | - |")
    md += ["", "- These count only the coordinator. In shared mode the voice and lip agents (one replica each, no batching) and the "
           "person's 0.7–0.9 s pauses bound a room long before vLLM does; those were not load-tested here.",
           "- A room's burst (several messages within a second) waits behind others at high counts; the p95 < 500 ms row is the one to plan on.", ""]

    # ---- C
    md += ["## C. vLLM outside MPS + MTP (fault isolation)", ""]
    if C and not C.get("skipped"):
        f, h, rec = C.get("fault_test") or {}, C.get("heldout") or {}, C.get("recommendation") or {}
        md.append(f"- vLLM had its own CUDA context (not an MPS client: {not (C.get('mps_check') or {}).get('engine_is_mps_client', True)}); "
                  f"compute mode {C.get('compute_mode')}.")
        if "checks" in f:
            md.append(f"- **Containment: {'PASS' if f.get('pass') else 'FAIL'}** after `kill -9` of the EngineCore: " +
                      "; ".join(f"{k} {'pass' if v['ok'] else 'FAIL'} ({v['s_after_kill']} s)" for k, v in f["checks"].items()) +
                      ". A process kill, not the CUDA fault MTP once hit.")
        rows = {r["what"]: r for r in (C.get("latency") or {}).get("rows", [])}
        def row(k):
            r = rows.get(k)
            return f"{r['all_mps'][0]} → {r['isolated_mtp'][0]} ms ({r['p50_change_pct']:+.0f}%)" if r else "-"
        md.append(f"- Latency vs all-MPS (p50, n 10 vs 20): coordinator alone {row('coordinator alone')}, with every agent busy "
                  f"{row('coordinator, every agent busy')}; voice {row('voice, every agent busy')}; Kokoro {row('Kokoro, every agent busy')}; "
                  f"lips {row('lips, every agent busy')}; one message end to end {row('one message end to end')}.")
        sd = C.get("spec_decode_counters") or ""
        try:
            nums = [float(line.split()[-1]) for line in sd.splitlines()]
            md.append(f"- MTP acceptance: {nums[1] / nums[0]:.1%} of draft tokens.")
        except Exception:
            pass
        if "isolated_mtp" in h:
            md.append(f"- MTP held-out reward (guarded, {h['n_tasks']} tasks): **{h['isolated_mtp']:.3f}** vs 0.985; unsafe {h['unsafe']:.2f}; "
                      f"within one task: {h['unchanged']}; moved: {h['by_category_diff'] or 'nothing'}.")
        md += [f"- **Recommendation: {rec.get('choice')}.** {rec.get('why')}. Defaults unchanged.", ""]
    else:
        md += [f"Not run: {(C or {}).get('reason', 'isolation.json missing')}.", ""]

    # ---- B
    md += ["## B. Serving metrics card (`UNSPOKEN_SERVING_PANEL`, default 0)", ""]
    if B:
        md += [f"- **Works: {'yes' if B['works'] else 'NO'}.** Flag off → {B['flag_off']['http']}; flag on → {B['flag_on']['http']} "
               f"{json.dumps(B['flag_on']['body'])}; vLLM unreachable → {json.dumps(B['vllm_down']['body'])}, `/api/health` {B['vllm_down']['health_http']}.",
               f"- Smoke through a flag-on gateway (accepted voice, ambiguous + No, Yes): {'pass' if B['smoke_ok'] else 'FAIL'}; "
               f"safety tests: {B['safety_tests']}; card unit tests: {B['panel_unit_tests']}.",
               "- The running app has the flag off (the route is a 404 and the card is hidden).", ""]
    else:
        md += ["Missing: panel_check.json.", ""]

    # ---- final
    md += ["## Final state", ""]
    if F:
        md += [f"- **{'Restored to defaults' if F['ok'] else 'NOT at defaults'}** ({F['date']} UTC): /api/health all ready {F['all_ready']}; "
               f"safety tests {F['safety_tests']}; MPS on {F['mps_on']}, MTP on {F['mtp_on']}; vLLM `{F['vllm_args']}` (default: {F['vllm_default_args']}); "
               f"non-default flags: {F['non_default_flags'] or 'none'}; smoke through :8000: {'pass' if F['smoke_ok'] else 'FAIL'}.",
               "- The brief's '25 safety tests' is now 49 (`evaluations/safety_tests`, all of them run each time).", ""]
    else:
        md += ["Missing: final_check.json.", ""]
    for name in ("restore_C.json", "restore_D.json"):
        r = load(name)
        if r:
            md.append(f"- `{name}`: {'OK' if r['ok'] else 'FAILED'} in {r['seconds']} s ({r['safety_tests']}).")
    md += ["", "## Reproduce", "", "```bash",
           "# laptop (WSL, where `brev` is): ship the branch, then run on the box",
           "bash scripts/brev/sync_app.sh marginal-sapphire-whale",
           "# box, ~/unspoken (each under nohup, as below)",
           "export PYTHONPATH=. UNSPOKEN_LLM_BACKEND=openai UNSPOKEN_LLM_BASE_URL=http://127.0.0.1:8001 UNSPOKEN_LLM_MODEL=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
           "nohup /workspace/venv-app/bin/python scripts/brev/serving/bench_concurrency.py --levels 1 4 8 16 > .run/logs/serving_A.log 2>&1 < /dev/null &   # A",
           "nohup bash scripts/brev/serving/panel_live_check.sh   > .run/logs/serving_B.log 2>&1 < /dev/null &   # B (test gateways on 8010-8012)",
           "nohup bash scripts/brev/serving/isolation_experiment.sh > .run/logs/serving_C.log 2>&1 < /dev/null &  # C (waits for 10 idle min, restores)",
           "nohup bash scripts/brev/serving/max_capacity.sh      > .run/logs/serving_D.log 2>&1 < /dev/null &   # D (after C; waits, restores)",
           "bash scripts/brev/serving/final_check.sh                                                             # defaults, health, tests, smoke",
           "# laptop: copy the results back and write this file",
           "bash scripts/brev/serving/fetch_results.sh marginal-sapphire-whale && python scripts/brev/serving/build_summary.py",
           "```", "", "## Files (`evaluations/results/serving/`)", "",
           "- A: `concurrency.json`, `concurrency.md`, `concurrency.png`",
           "- B: `panel_check.json`; code: `services/gateway/app.py` (`/api/serving_metrics`), `web/src/glass/GlassApp.tsx` (`ServingCard`), "
           "tests `scripts/brev/serving/test_serving_panel.py`",
           "- C: `isolation.json`, `isolation.md`, `latency_isolated_mtp.json`, `heldout_isolated_mtp.json`, `isolation_parts/`, `restore_C.json`",
           "- D: `max_capacity.json`, `max_capacity.md`, `max_capacity.png`, `gpu_trace.csv`, `max_capacity_parts/`, `restore_D.json`",
           "- Final: `final_check.json`; scripts: `scripts/brev/serving/`"]
    (OUT / "SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
