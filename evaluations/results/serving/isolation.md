# C. vLLM outside MPS + MTP: fault isolation

Compute mode: Default. vLLM restarted from its own running command line with `CUDA_MPS_PIPE_DIRECTORY`/`CUDA_MPS_LOG_DIRECTORY` at an empty temp dir (own CUDA context, not an MPS client) and `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`; voice, lips and Kokoro stayed MPS clients.

MPS check: EngineCore pid 513577, MPS clients ['508312', '508311', '508313'], nvidia-smi `| 0 N/A N/A 513577 C VLLM::EngineCore 53502MiB |` -> not an MPS client.

## Latency (p50 / p95 ms)

| | all-MPS, MTP off (`latency_brev_final2.json`) | vLLM isolated + MTP | p50 change |
|---|---:|---:|---:|
| coordinator alone | 333 / 338 | 232 / 268 | -30% |
| coordinator, every agent busy | 340 / 366 | 262 / 379 | -23% |
| voice, every agent busy | 78 / 106 | 104 / 266 | +33% |
| lips, every agent busy | 332 / 349 | 308 / 343 | -7% |
| Kokoro, every agent busy | 32 / 56 | 53 / 80 | +66% |
| voice alone | 59 / 66 | 63 / 68 | +7% |
| lips alone | 265 / 275 | 270 / 299 | +2% |
| one message end to end | 309 / 322 | 285 / 292 | -8% |

n = 10 per row here vs 20 in the baseline; run-to-run variation is about ±10% (SERVING.md).

## NeMo Gym held-out (guarded = deployed coordinator)

44 tasks: **0.970** vs 0.985 (`coordinator_heldout_brev.json`), unsafe 0.00, Nemotron decided 0.864, p50 227 vs 278 ms. Unchanged (tolerance one task (0.023)). Categories that moved: {'misread': [1.0, 0.889]}.

## Fault test: `kill -9` of the vLLM EngineCore

| Check | Result | Seconds after the kill | Detail |
|---|---|---:|---|
| voice_health_ready | PASS | 0.1 | ready |
| lips_health_ready | PASS | 0.11 | ready |
| tts_health_ready | PASS | 0.11 | ready |
| accepted_voice_auto_spoken | PASS | 0.45 | status=proposed spoken='I need water' by=demo_auto_spoken 85 ms |
| ambiguous_gets_rules_question | PASS | 0.45 | status=ambiguous question on screen after 87 ms: ask_confirmation ['cand_1559450f75b2', 'cand_5835e904090a', 'cand_f05ccbae9d6b']; final plan by=deterministic_fallback spoken=0 |

Small-model processes alive afterwards: {'508312': True, '508313': True, '508311': True}. Overall: **PASS**.

A `kill -9` is a process death, not the CUDA fault (`cudaErrorLaunchFailure`) MTP once hit; under MPS such a fault reaches the shared server. This test shows the others survive vLLM dying; it does not reproduce that fault.

## Recommendation

**keep all-MPS (isolation only with a second GPU or MIG)**: the fault was contained, but outside MPS vLLM time-slices with the small models: voice, every agent busy +33% p50, Kokoro, every agent busy +66% p50. Defaults were not changed.

Restore: OK (49 passed in 2.52s, all ready: True, MTP on: False, vLLM default args: True).
