# Serving: latency with every model running, and how to make it faster

Measured 2026-10-18 on the Brev instance (RTX PRO 6000 Blackwell 96 GB, 16 vCPU), all six services and vLLM
running, warm, batch 1, client on the same box (no laptop network in these numbers). Reproduce:

```bash
cd ~/unspoken && PYTHONPATH=. /workspace/venv-app/bin/python scripts/brev/bench_latency.py --n 20
# -> evaluations/results/latency_brev.json
```

Inputs: Kokoro speaks the 20 voice phrases (and those clips feed the voice agent); 12 GRID videos (3 s, 25 fps)
feed the lip agent; one BNCI letter feeds the EEG agent; a fixed two-candidate question feeds the coordinator.

## What runs where

| Process | Model | GPU memory |
|---|---|---:|
| vLLM (`VLLM::EngineCore`) | Nemotron 3.5 Lightning 30B-A3B NVFP4 (Marlin weight-only FP4, Triton attention) | 53.5 GB (weights 17.8 GB + KV cache, `--gpu-memory-utilization 0.55`) |
| services/audio (NeMo venv) | Nemotron Speech Streaming en 0.6B + Silero VAD | 9.0 GB |
| services/vsr | Auto-AVSR + subword LM (beam 12) | 2.2 GB |
| services/tts | Kokoro-82M | 1.4 GB |
| services/eeg, gateway, coordinator client | CPU only | — |
| asr_eval test bench (`serve.py`, not part of the app) | a second copy of Nemotron Speech | 5.6 GB |

## Results

**Each agent alone, then all of them at once** (one closed request loop per agent, running concurrently):

| Agent | Alone p50 / p95 ms | All together p50 / p95 ms | Slowdown (p50) |
|---|---:|---:|---:|
| Voice: Nemotron Speech + forced phrase scoring | 55 / 58 | 304 / 415 | 5.5× |
| Lips: Auto-AVSR, 3 s clip, beam 12 | 501 / 542 | 543 / **2760** | 1.1× (tail 5×) |
| Coordinator: Nemotron 3.5 on vLLM, one tool call | 241 / 251 | 304 / 357 | 1.3× |
| Speech out: Kokoro, one phrase | 23 / 25 | 78 / 92 | 3.4× |
| EEG replay: one letter (CPU) | 107 / 109 | 109 / 111 | 1.0× |

GPU while everything ran: **50% mean utilisation** (99% peak), 71.7 GB used.

**One message end to end, through the gateway** (voice + lip clip sent together → evidence → the coordinator's
question → select → confirm → policy gate → speech audio fetched):

| Stage | p50 ms | p95 ms |
|---|---:|---:|
| Both clips → evidence (lip reading dominates) | 639 | 694 |
| Nemotron's tool call applied (the deterministic plan is already on screen) | 376 | 408 |
| Confirm → policy gate → Kokoro audio | 36 | 37 |
| **Total** | **1032** | **1103** |

What the person waits for on top of this: the voice detector's 700 ms of silence before it cuts a clip, the clip
upload over the laptop ↔ Brev tunnel, and the time to look and confirm.

## Why everything slows down together

The GPU averaged 50% busy, so it is not out of compute. The cost is **sharing, not capacity**:

- **Five processes, five CUDA contexts.** Without NVIDIA MPS, kernels from different processes do not run at the
  same time: the GPU time-slices between contexts. The small, latency-bound models (voice 55 ms, Kokoro 23 ms) wait
  behind vLLM's and the lip model's kernels, which is why they slow down the most (5.5× and 3.4×).
- **Lip reading is the long pole.** Batch-1 beam search (beam 12) with a CTC prefix scorer issues many small kernels
  and does a lot of Python work per step; its tail grows to 2.8 s when it loses the GPU to the others.
- **A duplicate model.** The asr_eval test bench keeps a second Nemotron Speech resident and competes for the GPU.
- **The coordinator is not native FP4 here.** vLLM falls back to Marlin weight-only FP4 and Triton attention
  because FlashInfer's kernels need `nvcc`, which this image lacks (see [BREV.md](BREV.md) §5).

## How to serve it faster (in order of payoff)

1. **Turn on NVIDIA MPS** so the processes' kernels run concurrently instead of time-slicing. It needs no code
   changes and targets exactly the 5.5× / 3.4× slowdowns above:
   ```bash
   sudo nvidia-smi -c EXCLUSIVE_PROCESS            # optional, recommended with MPS
   nvidia-cuda-mps-control -d                       # start the MPS daemon, then (re)start the services
   # optional: cap each client's share of SMs, e.g. vLLM 60%, the small agents 20% each
   CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=60 vllm serve ...
   ```
   Re-run `bench_latency.py` before and after; keep it only if the "together" column improves.
2. **Remove the duplicate.** Stop the asr_eval bench while the app runs (`pkill -f "asr_eval.*serve.py"` on the
   instance), or point the bench at the app's voice service. Frees 5.6 GB and one GPU context.
3. **Isolate the LLM from the small models.** Either a 2-GPU instance (vLLM on GPU 0, every agent on GPU 1 via
   `CUDA_VISIBLE_DEVICES`), or MIG partitions on this card if the provider enables them (check
   `nvidia-smi mig -lgip`): hard isolation removes the interference entirely.
4. **Shorten the lip agent's path.**
   - Lower the beam (`UNSPOKEN_VSR_BEAM=6`, default 12) after re-checking accuracy on GRID (`scripts/run_evals.sh`).
   - Run it under bf16 autocast.
   - Send the browser's mouth crops instead of full frames, so the server skips its own MediaPipe face detection.
   - Start decoding while the clip is still being recorded.
5. **Make the coordinator native FP4.** Install a CUDA toolkit that matches the driver (CUDA 13.0) so FlashInfer
   can compile. Then drop `LLM_ATTN=TRITON_ATTN` and `VLLM_USE_FLASHINFER_SAMPLER=0` in `start_app.sh`, and let vLLM
   pick its FlashInfer/CUTLASS NVFP4 kernels. Also try:
   - speculative decoding (the model's MTP head, if vLLM supports it for this checkpoint);
   - a smaller `max_tokens` (one tool call needs far fewer than 200).

   Prefix caching is already on, so the system prompt and the tool definitions are reused.
6. **Cut the waiting the person feels, not just the compute.**
   - Nemotron Speech is a cache-aware streaming model: decode while the person speaks, so the phrase is ready when
     they stop.
   - With streaming, the voice detector's 700 ms silence hang-over can drop to about 300–400 ms.
   - The gateway already shows the deterministic plan instantly and lets Nemotron refine it.
7. **Fewer processes for the small models.** Voice, Kokoro and the lip model could share one process with a CUDA
   stream each. That is the code-level version of what MPS does from outside; do it only if MPS is not available.
8. **Many users at once.**
   - vLLM already batches the coordinator.
   - The other agents serve one request at a time (a model lock per service). For several patients, put them behind
     a batching server (NVIDIA Triton Inference Server with dynamic batching) or run one replica per GPU slice.

Keep the GPU memory budget in mind when changing `LLM_MEM`. The agents and vLLM's KV cache share the card, and vLLM
checks for free memory only when it starts.
