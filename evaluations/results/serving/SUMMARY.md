# Serving experiments around the vLLM coordinator (Nemotron 3.5 Lightning NVFP4)

Brev `marginal-sapphire-whale`, RTX PRO 6000 Blackwell 96 GB, branch `serving-experiments` (from `demo-final`). Client on the box, warm. Baselines from existing files, not re-run: `latency_brev_final2.json` (latency), `coordinator_heldout_brev.json` (held-out 0.985). Every number below is in a JSON file next to this one.

Kernels: Marlin weight-only FP4 + Triton attention (no native FP4/FlashInfer); numbers are conservative.

## A. Concurrency on the live vLLM (`--max-num-seqs 8`, every agent loaded, MTP off)

| Concurrent | p50 / p95 / p99 latency ms | p50 / p95 TTFT ms | req/s | output tok/s | valid call | same answer as at 1 | errors |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 281 / 310 / 336 | 90 / 92 | 3.64 | 248.7 | 85% | - | 0 |
| 4 | 542 / 628 / 636 | 160 / 240 | 7.49 | 511.7 | 85% | 90% | 0 |
| 8 | 856 / 921 / 938 | 398 / 441 | 9.36 | 636.3 | 85% | 100% | 0 |
| 16 | 1291 / 1781 / 1816 | 954 / 1263 | 9.42 | 644.0 | 85% | 100% | 0 |

- 20 measured requests per level after 3 warm-up; the same 72-task NeMo Gym sequence at every level; 0 prefix-cache hits (as documented: 4,224-token blocks, ~1,700-token prompt).
- Throughput stops growing at 8 concurrent requests: that is the server's `--max-num-seqs 8`, not the GPU (see D). Level 16 only queues.
- Valid call = parses and passes the coordinator's validator. 85% at every level: the same tasks fail alone (Nemotron re-asks after a 'none of these'); in the app they fall back to the rules. Load did not change validity.
- A first run with unshuffled tasks (eval.jsonl is sorted by category) gave 70% at level 4 for that reason; it was replaced.

## D. Max-capacity ramp

| Mode | vLLM | max concurrency at p95 < 500 ms | its req/s | peak req/s (output tok/s) | stop | peak GPU |
|---|---|---:|---:|---:|---|---|
| shared | max-num-seqs 64, util 0.55 | 2 | 5.37 | 16.25 (1065.4) | level 64: p95 6377 ms > 5 s | 63.0 GiB, 64.0 °C, 502.35 W |
| dedicated | max-num-seqs 128, util 0.85 | 2 | 5.36 | 16.19 (1064.4) | level 64: p95 6260 ms > 5 s | 82.6 GiB, 64.0 °C, 501.15 W |

Per-level tables (p50/p95/p99 latency and TTFT, req/s, tok/s, valid rate, errors, peak memory/temperature/power): `max_capacity.md`; plot `max_capacity.png`; GPU every 2 s: `gpu_trace.csv`. Levels 1, 2, 4 were added below the 8→128 ramp because p95 already passes 500 ms at 4 concurrent requests in A.
- Shared and dedicated are the same within noise: in shared mode the voice, lip and TTS agents were loaded but idle, so both measure vLLM alone. KV memory was not the limit: 64 requests x ~1.9k tokens is ~120k tokens against a 3.76M-token KV cache already at util 0.55 (vllm.log), so the extra memory of the dedicated mode went unused; latency grows with the batch (per-step compute). Level 128 did not run: both ramps stopped at 64 on p95.
- Valid-call regressions under load (valid alone, invalid under load): shared 0, dedicated 0; no safety stop fired (no CUDA error, OOM or Traceback; memory and temperature under the limits).

## Rooms per GPU (ESTIMATE, coordinator only)

Assumptions (flags of `build_summary.py`): 20 messages per patient per hour; coordinator fraction 0.050 (2 of 40 messages needed Nemotron in the voice bench of the `latency_brev_final2.json` run; accepted readings are decided by rules) and a conservative 1.0; headroom 1 of the measured capacity; arrivals spread evenly. Capacity = closed-loop req/s at the highest concurrency with p95 < 500 ms; 'peak' ignores latency.

| Mode | capacity (req/s) | rooms, fraction 2/40 | rooms, fraction 1 |
|---|---:|---:|---:|
| shared, at p95 < 500 ms | 5.37 | 19,332 | 967 |
| shared, peak | 16.25 | 58,500 | 2,925 |
| dedicated, at p95 < 500 ms | 5.36 | 19,296 | 965 |
| dedicated, peak | 16.19 | 58,284 | 2,914 |

- These count only the coordinator. In shared mode the voice and lip agents (one replica each, no batching) and the person's 0.7–0.9 s pauses bound a room long before vLLM does; those were not load-tested here.
- A room's burst (several messages within a second) waits behind others at high counts; the p95 < 500 ms row is the one to plan on.

## C. vLLM outside MPS + MTP (fault isolation)

- vLLM had its own CUDA context (not an MPS client: True); compute mode Default.
- **Containment: PASS** after `kill -9` of the EngineCore: voice_health_ready pass (0.1 s); lips_health_ready pass (0.11 s); tts_health_ready pass (0.11 s); accepted_voice_auto_spoken pass (0.45 s); ambiguous_gets_rules_question pass (0.45 s). A process kill, not the CUDA fault MTP once hit.
- Latency vs all-MPS (p50/p95, n 10 vs 20): coordinator alone 333/338 → 232/268 ms (-30% p50), with every agent busy 340/366 → 262/379 ms (-23% p50); voice 78/106 → 104/266 ms (+33% p50); Kokoro 32/56 → 53/80 ms (+66% p50); lips 332/349 → 308/343 ms (-7% p50); one message end to end 309/322 → 285/292 ms (-8% p50).
- MTP acceptance: 89.8% of draft tokens.
- MTP held-out reward (guarded, 44 tasks): **0.970** vs 0.985; unsafe 0.00; within one task: True; moved: {'misread': [1.0, 0.889]}.
- **Recommendation: keep all-MPS (isolation only with a second GPU or MIG).** The fault was contained, but outside MPS vLLM time-slices with the small models: voice, every agent busy +33% p50, Kokoro, every agent busy +66% p50. Defaults unchanged.

## B. Serving metrics card (`UNSPOKEN_SERVING_PANEL`, default 0)

- **Works: yes.** Flag off → 404; flag on → 200 {"state": "ready", "running": 0, "waiting": 0, "ttft_p50_ms": 107.5, "ttft_window": "since start", "kv_cache_usage_pct": 0.0, "prefix_cache_hit_rate": 0.0, "spec_decode_acceptance": null}; vLLM unreachable → {"state": "unavailable", "error": "ConnectError"}, `/api/health` 200.
- Smoke through a flag-on gateway (accepted voice, ambiguous + No, Yes): pass; safety tests: 49 passed in 2.52s; card unit tests: 4 passed in 0.28s.
- The running app has the flag off (the route is a 404 and the card is hidden).

## Final state

- **Restored to defaults** (2026-10-18 09:36:39 UTC): /api/health all ready True; safety tests 49 passed in 2.51s; MPS on True, MTP on False; vLLM `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 --host 127.0.0.1 --port 8001 --max-model-len 16384 --gpu-memory-utilization 0.55 --attention-backend TRITON_ATTN --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 --enable-prefix-caching --max-num-seqs 8` (default: True); non-default flags: none; smoke through :8000: pass.
- The brief's '25 safety tests' is now 49 (`evaluations/safety_tests`, all of them run each time).

- `restore_C.json`: OK in 43 s (49 passed in 2.52s).
- `restore_D.json`: OK in 47 s (49 passed in 2.54s).

## Reproduce

```bash
# laptop (WSL, where `brev` is): ship the branch, then run on the box
bash scripts/brev/sync_app.sh marginal-sapphire-whale
# box, ~/unspoken (each under nohup, as below)
export PYTHONPATH=. UNSPOKEN_LLM_BACKEND=openai UNSPOKEN_LLM_BASE_URL=http://127.0.0.1:8001 UNSPOKEN_LLM_MODEL=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
nohup /workspace/venv-app/bin/python scripts/brev/serving/bench_concurrency.py --levels 1 4 8 16 > .run/logs/serving_A.log 2>&1 < /dev/null &   # A
nohup bash scripts/brev/serving/panel_live_check.sh   > .run/logs/serving_B.log 2>&1 < /dev/null &   # B (test gateways on 8010-8012)
nohup bash scripts/brev/serving/isolation_experiment.sh > .run/logs/serving_C.log 2>&1 < /dev/null &  # C (waits for 10 idle min, restores)
nohup bash scripts/brev/serving/max_capacity.sh      > .run/logs/serving_D.log 2>&1 < /dev/null &   # D (after C; waits, restores)
bash scripts/brev/serving/final_check.sh                                                             # defaults, health, tests, smoke
# laptop: copy the results back and write this file
bash scripts/brev/serving/fetch_results.sh marginal-sapphire-whale && python scripts/brev/serving/build_summary.py
```

## Files (`evaluations/results/serving/`)

- A: `concurrency.json`, `concurrency.md`, `concurrency.png`
- B: `panel_check.json`; code: `services/gateway/app.py` (`/api/serving_metrics`), `web/src/glass/GlassApp.tsx` (`ServingCard`), tests `scripts/brev/serving/test_serving_panel.py`
- C: `isolation.json`, `isolation.md`, `latency_isolated_mtp.json`, `heldout_isolated_mtp.json`, `isolation_parts/`, `restore_C.json`
- D: `max_capacity.json`, `max_capacity.md`, `max_capacity.png`, `gpu_trace.csv`, `max_capacity_parts/`, `restore_D.json`
- Final: `final_check.json`; scripts: `scripts/brev/serving/`
