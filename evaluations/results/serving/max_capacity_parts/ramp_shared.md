# vLLM coordinator under concurrency (2026-10-18 09:31:18)

Server: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 --host 127.0.0.1 --port 8001 --max-model-len 16384 --gpu-memory-utilization 0.55 --attention-backend TRITON_ATTN --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 --enable-prefix-caching --max-num-seqs 64`. Requests: 72 NeMo Gym eval tasks through the coordinator's own prompt, tools and validator; temperature 0, thinking off, tool_choice auto, max_tokens 200, streamed, 30 s timeout. Client on the box (no network).

| Concurrency | n | p50 / p95 / p99 latency ms | p50 / p95 TTFT ms | req/s | output tok/s | tokens/req | valid call | same as first level | errors+timeouts |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 278.5 / 292.3 / 333.6 | 90.2 / 92.5 | 3.64 | 250.9 | 69.0 | 85% | 100% of 20 | 0 |
| 2 | 20 | 372.3 / 416.9 / 441.2 | 97.3 / 147.2 | 5.37 | 366.3 | 68.2 | 85% | 100% of 20 | 0 |
| 4 | 20 | 530.7 / 646.3 / 647.4 | 190.1 / 240.3 | 7.57 | 518.8 | 68.5 | 85% | 95% of 20 | 0 |
| 8 | 20 | 831.8 / 1077.9 / 1087.5 | 356.5 / 437.8 | 9.37 | 634.5 | 67.8 | 85% | 90% of 20 | 0 |
| 16 | 32 | 1266.3 / 1555.6 / 1557.1 | 805.3 / 822.6 | 12.42 | 815.6 | 65.7 | 91% | 100% of 32 | 0 |
| 32 | 64 | 2065.2 / 3202.0 / 3619.2 | 1524.0 / 1586.7 | 14.3 | 936.9 | 65.5 | 92% | 94% of 64 | 0 |
| 64 | 128 | 3724.6 / 6376.6 / 6692.4 | 1762.9 / 3115.6 | 16.25 | 1065.4 | 65.6 | 92% | 96% of 128 | 0 |

/metrics over the whole run: {"prompt_tokens_total": 686538.0, "generation_tokens_total": 26413.0, "request_success_total": 398.0, "prefix_cache_queries_total": 686538.0, "prefix_cache_hits_total": 0.0, "num_preemptions_total": 0.0, "time_to_first_token_seconds_sum": 178.045, "time_to_first_token_seconds_count": 398.0, "e2e_request_latency_seconds_sum": 724.8474, "e2e_request_latency_seconds_count": 398.0, "server_ttft_mean_ms": 447.3, "prefix_cache_hit_rate": 0.0}

- TTFT is measured by the client: the time to the first streamed token (text, reasoning or tool-call delta). The tool-call parser can hold tokens back, so it is an upper bound; `server_ttft_mean_ms` above is vLLM's own.
- A valid call parses and passes `services/coordinator/app.py::validate` (flags at defaults). An invalid call is not an error: in the app it falls back to the rules. Every level sends the same tasks, so the first level's valid rate is the reference: Nemotron alone fails the validator on some task types (e.g. it asks again after a 'none of these'; NeMo Gym: the deployed coordinator decides 86% of held-out tasks itself).
- `same as first level`: requests whose tool call (name + arguments) equals the first level's for the same task.
- The server runs `--max-num-seqs 64`: levels above it queue in vLLM. That is this server's cap, not the GPU's limit (see max_capacity.md).
