# vLLM coordinator under concurrency (2026-10-18 09:33:55)

Server: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 --host 127.0.0.1 --port 8001 --max-model-len 16384 --gpu-memory-utilization 0.85 --attention-backend TRITON_ATTN --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 --enable-prefix-caching --max-num-seqs 128`. Requests: 72 NeMo Gym eval tasks through the coordinator's own prompt, tools and validator; temperature 0, thinking off, tool_choice auto, max_tokens 200, streamed, 30 s timeout. Client on the box (no network).

| Concurrency | n | p50 / p95 / p99 latency ms | p50 / p95 TTFT ms | req/s | output tok/s | tokens/req | valid call | same as first level | errors+timeouts |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 278.1 / 333.5 / 382.3 | 88.6 / 90.9 | 3.57 | 254.2 | 71.1 | 85% | 95% of 20 | 0 |
| 2 | 20 | 372.2 / 437.5 / 439.7 | 97.2 / 147.1 | 5.36 | 365.8 | 68.2 | 85% | 95% of 20 | 0 |
| 4 | 20 | 522.8 / 624.8 / 682.0 | 193.7 / 239.5 | 7.51 | 525.1 | 69.9 | 85% | 95% of 20 | 0 |
| 8 | 20 | 817.8 / 1029.4 / 1120.5 | 401.4 / 438.3 | 9.32 | 641.5 | 68.8 | 85% | 95% of 20 | 0 |
| 16 | 32 | 1265.7 / 1513.3 / 1900.7 | 810.3 / 827.5 | 12.17 | 813.9 | 66.9 | 91% | 91% of 32 | 0 |
| 32 | 64 | 2056.1 / 3519.9 / 3580.6 | 1539.2 / 1603.4 | 14.27 | 937.1 | 65.7 | 92% | 94% of 64 | 0 |
| 64 | 128 | 3731.9 / 6259.5 / 6955.9 | 1771.3 / 3136.2 | 16.19 | 1064.4 | 65.7 | 92% | 94% of 128 | 0 |

/metrics over the whole run: {"prompt_tokens_total": 684707.0, "generation_tokens_total": 26570.0, "request_success_total": 397.0, "prefix_cache_queries_total": 684707.0, "prefix_cache_hits_total": 0.0, "num_preemptions_total": 0.0, "time_to_first_token_seconds_sum": 178.2665, "time_to_first_token_seconds_count": 397.0, "e2e_request_latency_seconds_sum": 727.3988, "e2e_request_latency_seconds_count": 397.0, "server_ttft_mean_ms": 449.0, "prefix_cache_hit_rate": 0.0}

- TTFT is measured by the client: the time to the first streamed token (text, reasoning or tool-call delta). The tool-call parser can hold tokens back, so it is an upper bound; `server_ttft_mean_ms` above is vLLM's own.
- A valid call parses and passes `services/coordinator/app.py::validate` (flags at defaults). An invalid call is not an error: in the app it falls back to the rules. Every level sends the same tasks, so the first level's valid rate is the reference: Nemotron alone fails the validator on some task types (e.g. it asks again after a 'none of these'; NeMo Gym: the deployed coordinator decides 86% of held-out tasks itself).
- `same as first level`: requests whose tool call (name + arguments) equals the first level's for the same task.
- The server runs `--max-num-seqs 128`: levels above it queue in vLLM. That is this server's cap, not the GPU's limit (see max_capacity.md).
