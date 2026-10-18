# vLLM coordinator under concurrency (2026-10-18 09:08:19)

Server: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 --host 127.0.0.1 --port 8001 --max-model-len 16384 --gpu-memory-utilization 0.55 --attention-backend TRITON_ATTN --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 --enable-prefix-caching --max-num-seqs 8`. Requests: 72 NeMo Gym eval tasks through the coordinator's own prompt, tools and validator; temperature 0, thinking off, tool_choice auto, max_tokens 200, streamed, 30 s timeout. Client on the box (no network).

| Concurrency | n | p50 / p95 / p99 latency ms | p50 / p95 TTFT ms | req/s | output tok/s | tokens/req | valid call | same as first level | errors+timeouts |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 280.6 / 309.8 / 335.6 | 90.3 / 92.2 | 3.64 | 248.7 | 68.4 | 85% | - | 0 |
| 4 | 20 | 541.8 / 628.2 / 636.2 | 159.8 / 240.0 | 7.49 | 511.7 | 68.3 | 85% | 90% of 20 | 0 |
| 8 | 20 | 855.5 / 920.9 / 938.4 | 398.4 / 440.6 | 9.36 | 636.3 | 68.0 | 85% | 100% of 20 | 0 |
| 16 | 20 | 1290.9 / 1780.6 / 1815.6 | 954.4 / 1263.3 | 9.42 | 644.0 | 68.4 | 85% | 100% of 20 | 0 |

/metrics over the whole run: {"prompt_tokens_total": 159900.0, "generation_tokens_total": 6330.0, "request_success_total": 92.0, "prefix_cache_queries_total": 159900.0, "prefix_cache_hits_total": 0.0, "num_preemptions_total": 0.0, "time_to_first_token_seconds_sum": 22.4895, "time_to_first_token_seconds_count": 92.0, "e2e_request_latency_seconds_sum": 62.5372, "e2e_request_latency_seconds_count": 92.0, "server_ttft_mean_ms": 244.5, "prefix_cache_hit_rate": 0.0}

- TTFT is measured by the client: the time to the first streamed token (text, reasoning or tool-call delta). The tool-call parser can hold tokens back, so it is an upper bound; `server_ttft_mean_ms` above is vLLM's own.
- A valid call parses and passes `services/coordinator/app.py::validate` (flags at defaults). An invalid call is not an error: in the app it falls back to the rules. Every level sends the same tasks, so the first level's valid rate is the reference: Nemotron alone fails the validator on some task types (e.g. it asks again after a 'none of these'; NeMo Gym: the deployed coordinator decides 86% of held-out tasks itself).
- `same as first level`: requests whose tool call (name + arguments) equals the first level's for the same task.
- The server runs `--max-num-seqs 8`: levels above it queue in vLLM. That is this server's cap, not the GPU's limit (see max_capacity.md).
