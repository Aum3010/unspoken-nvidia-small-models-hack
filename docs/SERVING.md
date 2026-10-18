# Serving: latency with every model running, and how to make it faster

Measured 2026-10-18 on the Brev instance (RTX PRO 6000 Blackwell 96 GB, 16 vCPU), all six services and vLLM
running, warm, batch 1, client on the same box (no laptop network in these numbers). Reproduce:

```bash
cd ~/unspoken && PYTHONPATH=. /workspace/venv-app/bin/python scripts/brev/bench_latency.py --n 20 --out evaluations/results/latency_brev_<label>.json
```

Inputs: Kokoro speaks the 20 voice phrases (and those clips feed the voice agent); 12 GRID videos (3 s, 25 fps)
feed the lip agent; one BNCI letter feeds the EEG agent; a fixed two-candidate question feeds the coordinator.
Run-to-run variation is about ±10% (the lip agent alone measured 501–619 ms p50 across runs).

## What runs where

| Process | Model | GPU memory |
|---|---|---:|
| vLLM (`VLLM::EngineCore`) | Nemotron 3.5 Lightning 30B-A3B NVFP4 (Marlin weight-only FP4, Triton attention) | 53.6 GB (weights 17.8 GB + KV cache, `--gpu-memory-utilization 0.55`) |
| services/audio (NeMo venv) | Nemotron Speech Streaming en 0.6B + Silero VAD | 6.6–8.8 GB |
| services/vsr | Auto-AVSR + subword LM (beam 12) | 2.2 GB |
| services/tts | Kokoro-82M | 1.4 GB |
| `nvidia-cuda-mps-server` | NVIDIA MPS (on by default) | 0.06 GB |
| services/eeg, gateway, coordinator client | CPU only | — |

The asr_eval test bench (`asr_eval/serve.py`) loaded a second copy of Nemotron Speech (5.6 GB). It is stopped:
the app uses only Nemotron Speech Streaming **English 0.6B**, in `services/audio`.

## Results

Three runs, each with every agent alone, then all of them at once (one closed request loop per agent,
concurrently), then one message end to end:

| | 1. as first deployed | 2. duplicate model stopped | 3. + NVIDIA MPS |
|---|---:|---:|---:|
| **All agents at once, p50 / p95 ms** | | | |
| Voice: Nemotron Speech + forced phrase scoring | 304 / 415 | 289 / 377 | **70 / 103** |
| Lips: Auto-AVSR, 3 s clip, beam 12 | 543 / 2760 | 543 / 2653 | 570 / **725** |
| Coordinator: Nemotron 3.5 on vLLM, one tool call | 304 / 357 | 293 / 360 | 263 / 281 |
| Speech out: Kokoro, one phrase | 78 / 92 | 82 / 97 | **32 / 49** |
| EEG replay: one letter (CPU) | 109 / 111 | 111 / 119 | 115 / 131 |
| GPU busy (mean) / time for the whole run | 50% / 16.0 s | 47% / 16.1 s | 61% / 11.8 s |
| GPU memory, peak | 71.7 GB | 64.4 GB | 63.1 GB |
| **Each agent alone, p50 ms** (voice / lips / coordinator / Kokoro) | 55 / 501 / 241 / 23 | 56 / 508 / 231 / 24 | 60 / 535 / 248 / 23 |
| **One message end to end, p50 / p95 ms** | 1032 / 1103 | 1030 / 1327 | **967 / 1086** |

- **MPS removes most of the interference.** With every agent busy, the voice agent drops from 289 to 70 ms p50,
  Kokoro from 82 to 32 ms, and the lip agent's tail from 2.7 to 0.7 s. The same work finishes in 11.8 s instead of 16.1 s.
- **The cost of MPS** is about 5% on an agent that runs alone (voice 56 → 60 ms). MPS does not change any model's
  arithmetic, so accuracy is unaffected.
- **Stopping the duplicate model** freed about 7 GB of GPU memory (peak 71.7 → 64.4 GB). It barely changed latency, because that copy sat idle.

**One message, stage by stage** (MPS on): both clips → evidence 564 ms (lip reading is the long pole) · Nemotron's
tool call applied 365 ms (the deterministic plan is already on screen) · confirm → policy gate → Kokoro audio 36 ms.

**The voice result after the person stops speaking** (run 4, MPS on; the utterance is cut after 700 ms of silence):

| From the end of the 700 ms silence until the voice evidence is in the session | p50 ms | p95 ms |
|---|---:|---:|
| Clip sent at that moment (before) | 102 | 110 |
| **Early decode**: clip sent at 350 ms of silence, committed at 700 ms | **2** | **3** |

Measured on the box. From the laptop, the clip's upload over the tunnel is also hidden, so the gain is larger.

## How to serve it faster: what was done, and what was decided

| # | Option | Status |
|---|---|---|
| 1 | **NVIDIA MPS** | **Done, on by default** (`start_app.sh`, `UNSPOKEN_MPS=0` to turn it off). Numbers above |
| 2 | **Stop the duplicate Nemotron Speech** (asr_eval bench) | **Done.** The app uses only Nemotron Speech English 0.6B, in `services/audio` |
| 3 | Isolate the LLM from the small models: 2 GPUs, or MIG on this card if the provider enables it (`nvidia-smi mig -lgip`) | **Deferred: do it last.** With MPS, interference is mostly gone; isolation would remove the rest |
| 4 | Faster lip agent | **Tested, nothing adopted** (no accuracy-neutral speed-up; details below) |
| 5 | Native FP4 for the coordinator | **Not recommended now** (see below) |
| 6 | Cut the waiting the person feels | **Done: early voice decode**, with no change to any decision (details below). The 700 ms end-of-utterance pause is kept |
| 7 | One process for the small models | Not needed: MPS gives the concurrency without code changes |
| 8 | Many users at once | vLLM already batches the coordinator; the other agents would need a batching server (Triton) or one replica per GPU slice |

### 4. The lip agent: speed vs accuracy (`evaluations/modality_baselines/vsr_speed_accuracy.py`)

132 GRID clips, 4 from each of 33 unseen speakers. Face detection and the mouth crop are done once per clip,
so every setting sees identical inputs.

| Setting | WER | Best-of-5 WER | CER | Same top-1 as now | Decode p50 |
|---|---:|---:|---:|---:|---:|
| **Beam 12, fp32 (current)** | **69.4%** | **60.9%** | **47.3%** | 132/132 | 282 ms |
| Beam 12, fp16 autocast | 69.4% | 60.7% | 47.3% | 132/132 | 295 ms |
| Beam 6, fp32 | 71.1% | 64.5% | 48.1% | 87/132 | 231 ms |
| Beam 6, fp16 autocast | 71.3% | 64.7% | 48.4% | 85/132 | 248 ms |

- **Beam 6** saves about 50 ms but loses accuracy (WER +1.7, best-of-5 WER +3.6 points). Rejected, because
  accuracy comes first.
- **fp16** gives the same answers but is not faster: the encoder takes 10 ms, and the rest is beam search, which
  half precision does not speed up. Not adopted.
- **bf16** cannot run: Chaplin's ESPnet attention asks numpy for the dtype's limits, and numpy has no bfloat16.
  It would need an upstream patch.
- **Where the time goes:** preparing the clip (video decode, server-side MediaPipe face detection, mouth crop)
  takes 250 ms. Decoding takes 282 ms, almost all beam search.
- **Sending the browser's mouth crops** instead of video would cut part of those 250 ms. But it changes the
  model's input preprocessing, so it needs its own accuracy test first. Not done.
- **Decoding while recording** is not possible with this model: its encoder needs the whole clip.

### 5. Native FP4 for the coordinator: not now

Today vLLM serves the NVFP4 checkpoint with **Marlin weight-only FP4**: weights in 4 bit, activations and
arithmetic in 16 bit. Native FP4 (FlashInfer/CUTLASS kernels) would also run activations in 4 bit. It needs:
- a CUDA toolkit that matches the driver (13.0) installed on the image;
- FlashInfer JIT builds;
- a re-run of the coordinator evaluation (NeMo Gym), because activation quantization can change its tool calls.

Reasons to wait:
- **It is the least accurate of the two options**, and accuracy comes first.
- **The coordinator is not the bottleneck.** 263 ms with everything busy, and the person already sees the
  deterministic plan while Nemotron answers.
- **It saves the most on a GPU the LLM has to itself.** Do it, if at all, after option 3, and keep it only if the
  NeMo Gym scores do not drop.

### 6. Early voice decode (`services/gateway` `capture/audio/early|commit|discard`, `web/src/glass/sensors.ts`)

The browser still ends an utterance only after 700 ms of silence, so pauses inside a phrase do not split it.
1. At 350 ms of silence it sends the clip, and the voice agent starts decoding it without touching the session.
2. If the person speaks again, the clip is discarded and recording continues.
3. At 700 ms the browser commits, and the finished result is applied exactly like a normal capture.

The committed clip has 350 ms of trailing silence instead of 700 ms. On the 304 real recordings
(`asr_eval/scripts/trailing_silence_check.py`), **no decision changed** (0/304):
- 169/234 phrases accepted correctly, 1 wrong, 1/70 false accepts on negatives, both ways.
- 266 of the 304 recordings have less than 700 ms after the speech anyway, so the thresholds were tuned on short tails.

Not done, because it would change accuracy:
- **A shorter pause than 700 ms** would split slow or dysarthric phrases.
- **Streaming the encoder while the person speaks** would save at most the encoder's share of about 60 ms.

Keep the GPU memory budget in mind when changing `LLM_MEM`. The agents and vLLM's KV cache share the card, and
vLLM checks for free memory only when it starts.
