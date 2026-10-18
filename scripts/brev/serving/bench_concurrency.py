"""Concurrency on the live vLLM coordinator (Nemotron 3.5 Lightning): real coordinator requests, several at once.

    cd ~/unspoken && PYTHONPATH=. /workspace/venv-app/bin/python scripts/brev/serving/bench_concurrency.py \
        --levels 1 4 8 16 --out evaluations/results/serving/concurrency

Requests are the coordinator's own: its SYSTEM_PROMPT, TOOLS and user payload (imported from
services/coordinator/app.py, not copied), built from the NeMo Gym eval tasks, with the production parameters
(temperature 0, thinking off, tool_choice auto, max_tokens 200 as in `_call_llm`). Each request streams
(stream=True) so the client sees its time to first token. A tool call counts as valid when it parses and passes the
coordinator's own validator (`validate`), with the validator flags at their defaults.

Per level: 3 warm-up requests, then N measured requests dispatched through a pool of `level` workers.
`--measured auto` = max(20, 2 x level) (the max-capacity ramp). /metrics is scraped at the start and at the end.
`--stop-file`: if that file appears (the max-capacity watchdog writes it), the level stops early and is marked.
Writes <out>.json and <out>.md (and <out>.png unless --no-plot). Read-only: it never restarts anything.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import services.coordinator.app as coord  # noqa: E402  (prompt builder, tools, validator: imported, never copied)
from shared.schemas import ToolCall  # noqa: E402

TASKS = ROOT / "evaluations" / "nemo_gym" / "unspoken_communication" / "data" / "eval.jsonl"
VLLM = os.environ.get("UNSPOKEN_LLM_BASE_URL", "http://127.0.0.1:8001")
MODEL = os.environ.get("UNSPOKEN_LLM_MODEL", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4")
MAX_TOKENS = 200   # production value, services/coordinator/app.py `_call_llm`
TIMEOUT_S = 30.0
COUNTERS = ["prompt_tokens_total", "generation_tokens_total", "request_success_total", "prefix_cache_queries_total",
            "prefix_cache_hits_total", "num_preemptions_total", "time_to_first_token_seconds_sum",
            "time_to_first_token_seconds_count", "e2e_request_latency_seconds_sum", "e2e_request_latency_seconds_count",
            "spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total", "spec_decode_num_accepted_tokens_total"]


def load_requests() -> list[coord.PlanRequest]:
    return [coord.PlanRequest(**json.loads(line)["plan_request"]) for line in TASKS.read_text().splitlines() if line.strip()]


def body(req: coord.PlanRequest) -> dict:
    messages = [{"role": "system", "content": coord.SYSTEM_PROMPT}, {"role": "user", "content": coord._user_payload(req)}]
    return {"model": MODEL, "messages": messages, "tools": coord.TOOLS, "tool_choice": "auto", "temperature": 0,
            "max_tokens": MAX_TOKENS, "chat_template_kwargs": {"enable_thinking": False}, "stream": True,
            "stream_options": {"include_usage": True}}


async def one(c: httpx.AsyncClient, req: coord.PlanRequest) -> dict:
    """One streamed coordinator request: latency, TTFT, output tokens, and whether its tool call is valid."""
    t0 = time.perf_counter()
    ttft, tokens, calls, row = None, None, {}, {}
    try:
        async with asyncio.timeout(TIMEOUT_S):
            async with c.stream("POST", f"{VLLM}/v1/chat/completions", json=body(req)) as r:
                if r.status_code != 200:
                    await r.aread()
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
                async for line in r.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    if chunk.get("usage"):
                        tokens = chunk["usage"].get("completion_tokens")
                    for ch in chunk.get("choices") or []:
                        d = ch.get("delta") or {}
                        if ttft is None and (d.get("content") or d.get("reasoning_content") or d.get("reasoning") or d.get("tool_calls")):
                            ttft = (time.perf_counter() - t0) * 1000
                        for tc in d.get("tool_calls") or []:
                            acc = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                            fn = tc.get("function") or {}
                            acc["name"] += fn.get("name") or ""
                            acc["arguments"] += fn.get("arguments") or ""
    except TimeoutError:
        return {"ok": False, "timeout": True, "error": f"timeout {TIMEOUT_S:.0f} s", "latency_ms": (time.perf_counter() - t0) * 1000}
    except Exception as exc:
        return {"ok": False, "timeout": False, "error": f"{type(exc).__name__}: {exc}"[:200], "latency_ms": (time.perf_counter() - t0) * 1000}
    row = {"ok": True, "timeout": False, "latency_ms": (time.perf_counter() - t0) * 1000, "ttft_ms": ttft, "out_tokens": tokens}
    if not calls:
        return row | {"valid": False, "invalid_reason": "no tool call"}
    fn = calls[min(calls)]
    try:
        args = json.loads(fn["arguments"] or "{}")
    except json.JSONDecodeError:
        return row | {"valid": False, "tool": fn["name"], "invalid_reason": "arguments do not parse"}
    rationale = str(args.pop("rationale", ""))
    call = coord.validate(ToolCall(name=fn["name"], arguments=args, rationale=rationale, by="llm"), req)
    return row | {"valid": call.valid, "tool": fn["name"], "invalid_reason": "; ".join(call.validation_errors)[:160] or None}


def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(round(q * (len(s) - 1))))], 1)


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["ok"]]
    lat = [r["latency_ms"] for r in ok]
    ttft = [r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None]
    toks = [r["out_tokens"] for r in ok if r.get("out_tokens") is not None]
    out = {"n": len(rows), "ok": len(ok), "errors": sum(not r["ok"] and not r["timeout"] for r in rows),
           "timeouts": sum(r["timeout"] for r in rows)}
    out["error_rate"] = round((out["errors"] + out["timeouts"]) / len(rows), 4) if rows else None
    out["valid_rate"] = round(sum(bool(r.get("valid")) for r in ok) / len(ok), 4) if ok else None
    for name, xs in (("latency_ms", lat), ("ttft_ms", ttft)):
        out[name] = {"p50": pct(xs, .5), "p95": pct(xs, .95), "p99": pct(xs, .99),
                     "mean": round(statistics.fmean(xs), 1) if xs else None, "max": round(max(xs), 1) if xs else None}
    out["out_tokens_mean"] = round(statistics.fmean(toks), 1) if toks else None
    out["out_tokens_sum"] = sum(toks)
    return out


async def run_level(c: httpx.AsyncClient, reqs: list[coord.PlanRequest], level: int, n: int, offset: int,
                    stop_file: Path | None = None) -> dict:
    stopped = lambda: stop_file is not None and stop_file.exists()  # noqa: E731
    warm = [reqs[(offset + i) % len(reqs)] for i in range(3)]
    await asyncio.gather(*[one(c, r) for r in warm])
    queue = list(range(n))
    rows: list[dict | None] = [None] * n

    async def worker():
        while queue and not stopped():
            i = queue.pop(0)
            req = reqs[(offset + 3 + i) % len(reqs)]
            rows[i] = await one(c, req) | {"task": req.message_id}

    t0 = time.perf_counter()
    await asyncio.gather(*[worker() for _ in range(level)])
    wall = time.perf_counter() - t0
    done = [r for r in rows if r is not None]
    s = summarize(done)
    s |= {"level": level, "measured_requested": n, "wall_s": round(wall, 2),
          "req_per_s": round(len(done) / wall, 2) if wall else None,
          "out_tok_per_s": round(s["out_tokens_sum"] / wall, 1) if wall else None,
          "stopped_early": len(done) < n, "invalid_examples": [r for r in done if r["ok"] and not r.get("valid")][:5],
          "error_examples": [r["error"] for r in done if not r["ok"]][:5],
          "rows": [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()} for r in done]}
    return s


async def scrape(c: httpx.AsyncClient) -> dict:
    """vLLM's counters and gauges (no histogram buckets), summed over label sets."""
    try:
        text = (await c.get(f"{VLLM}/metrics", timeout=5)).text
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith("vllm:") or "_bucket" in line or "_created" in line:
            continue
        name, _, val = line.rpartition(" ")
        key = name.split("{")[0][5:]
        try:
            out[key] = out.get(key, 0.0) + float(val)
        except ValueError:
            pass
    return out


def delta(a: dict, b: dict) -> dict:
    d = {k: round(b[k] - a.get(k, 0.0), 4) for k in COUNTERS if k in b}
    if d.get("time_to_first_token_seconds_count"):
        d["server_ttft_mean_ms"] = round(1000 * d["time_to_first_token_seconds_sum"] / d["time_to_first_token_seconds_count"], 1)
    if d.get("prefix_cache_queries_total"):
        d["prefix_cache_hit_rate"] = round(d.get("prefix_cache_hits_total", 0) / d["prefix_cache_queries_total"], 4)
    if d.get("spec_decode_num_draft_tokens_total"):
        d["spec_decode_acceptance"] = round(d.get("spec_decode_num_accepted_tokens_total", 0) / d["spec_decode_num_draft_tokens_total"], 4)
    return d


def markdown(res: dict) -> str:
    lines = [f"# vLLM coordinator under concurrency ({res['date']})", "",
             f"Server: `{res['server_args']}`. Requests: {res['n_tasks']} NeMo Gym eval tasks through the coordinator's own prompt, "
             f"tools and validator; temperature 0, thinking off, tool_choice auto, max_tokens {MAX_TOKENS}, streamed, {TIMEOUT_S:.0f} s timeout. "
             "Client on the box (no network).", "",
             "| Concurrency | n | p50 / p95 / p99 latency ms | p50 / p95 TTFT ms | req/s | output tok/s | tokens/req | valid call | errors+timeouts |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for lv in res["levels"]:
        L, T = lv["latency_ms"], lv["ttft_ms"]
        lines.append(f"| {lv['level']} | {lv['n']}{' (stopped)' if lv['stopped_early'] else ''} | {L['p50']} / {L['p95']} / {L['p99']} | "
                     f"{T['p50']} / {T['p95']} | {lv['req_per_s']} | {lv['out_tok_per_s']} | {lv['out_tokens_mean']} | "
                     f"{100 * (lv['valid_rate'] or 0):.0f}% | {lv['errors'] + lv['timeouts']} |")
    m = res.get("metrics_delta") or {}
    lines += ["", f"/metrics over the whole run: {json.dumps(m)}", "",
              "- TTFT is measured by the client: the time to the first streamed token (text, reasoning or tool-call delta). "
              "The tool-call parser can hold tokens back, so it is an upper bound; `server_ttft_mean_ms` above is vLLM's own.",
              "- A valid call parses and passes `services/coordinator/app.py::validate` (flags at defaults). An invalid call is "
              "not an error: in the app it falls back to the rules."]
    if res.get("max_num_seqs"):
        lines.append(f"- The server runs `--max-num-seqs {res['max_num_seqs']}`: levels above it queue in vLLM. That is this "
                     "server's cap, not the GPU's limit (see max_capacity.md).")
    return "\n".join(lines) + "\n"


def plot(res: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lv = [x for x in res["levels"] if x["ok"]]
    xs = [x["level"] for x in lv]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    for key, color, label in (("p50", "#2a78d6", "p50"), ("p95", "#eb6834", "p95")):
        a.plot(xs, [x["latency_ms"][key] for x in lv], "-o", color=color, lw=2, ms=6, label=f"latency {label}")
    a.plot(xs, [x["ttft_ms"]["p50"] for x in lv], "--o", color="#1baf7a", lw=2, ms=6, label="TTFT p50")
    a.set(xscale="log", xlabel="concurrent requests", ylabel="ms", title="Coordinator latency")
    b.plot(xs, [x["req_per_s"] for x in lv], "-o", color="#2a78d6", lw=2, ms=6)
    b.set(xscale="log", xlabel="concurrent requests", ylabel="requests / s", title="Throughput")
    for ax in (a, b):
        ax.set_xticks(xs, [str(x) for x in xs])
        ax.grid(alpha=.25)
        ax.spines[["top", "right"]].set_visible(False)
    a.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--measured", default="20", help="requests per level, or 'auto' = max(20, 2 x level)")
    ap.add_argument("--out", default=str(ROOT / "evaluations" / "results" / "serving" / "concurrency"), help="path without extension")
    ap.add_argument("--stop-file", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stop = Path(args.stop_file) if args.stop_file else None
    reqs = load_requests()
    limits = httpx.Limits(max_connections=max(args.levels) + 8, max_keepalive_connections=max(args.levels) + 8)
    res = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "label": args.label, "n_tasks": len(reqs), "levels": []}
    try:
        import subprocess
        cmd = subprocess.run(["pgrep", "-af", "[v]llm serve"], capture_output=True, text=True).stdout.strip().splitlines()
        res["server_args"] = cmd[0].split(" serve ", 1)[-1] if cmd else "unknown"
        if "--max-num-seqs" in res["server_args"]:
            res["max_num_seqs"] = int(res["server_args"].split("--max-num-seqs")[1].split()[0])
    except Exception as exc:
        res["server_args"] = f"unknown ({exc})"
    async with httpx.AsyncClient(timeout=TIMEOUT_S + 5, limits=limits) as c:
        res["metrics_start"] = await scrape(c)
        offset = 0
        for level in args.levels:
            if stop and stop.exists():
                res["stopped_before"] = level
                break
            n = max(20, 2 * level) if args.measured == "auto" else int(args.measured)
            print(f"level {level}: 3 warm-up + {n} measured", flush=True)
            try:
                lv = await run_level(c, reqs, level, n, offset, stop)
            except Exception as exc:  # keep partial results, never invent numbers
                res.setdefault("failures", []).append({"level": level, "error": f"{type(exc).__name__}: {exc}"[:300]})
                print(f"  level {level} failed: {exc}", flush=True)
                break
            offset += n + 3
            res["levels"].append(lv)
            print(f"  p50/p95 {lv['latency_ms']['p50']}/{lv['latency_ms']['p95']} ms  ttft p50 {lv['ttft_ms']['p50']}  "
                  f"{lv['req_per_s']} req/s  valid {lv['valid_rate']}  err {lv['error_rate']}", flush=True)
            out.with_suffix(".json").write_text(json.dumps(res, indent=1))  # partial results survive a crash
        res["metrics_end"] = await scrape(c)
    if "error" not in res["metrics_start"] and "error" not in res["metrics_end"]:
        res["metrics_delta"] = delta(res["metrics_start"], res["metrics_end"])
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    out.with_suffix(".md").write_text(markdown(res))
    if not args.no_plot and res["levels"]:
        plot(res, out.with_suffix(".png"))
    print(markdown(res))


if __name__ == "__main__":
    asyncio.run(main())
