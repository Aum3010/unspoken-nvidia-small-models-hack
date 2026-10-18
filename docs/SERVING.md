# Serving: latency with every model running, and how to make it faster

Measured 2026-10-18 on the Brev instance (RTX PRO 6000 Blackwell 96 GB, 16 vCPU), all six services and vLLM
running, warm, batch 1, client on the same box (no laptop network in these numbers). Reproduce:

```bash
cd ~/unspoken && PYTHONPATH=. /workspace/venv-app/bin/python scripts/brev/bench_latency.py --n 20 --out evaluations/results/latency_brev_<label>.json
```

Inputs: Kokoro speaks the 20 voice phrases (and those clips feed the voice agent); 12 GRID videos (3 s, 25 fps)
feed the lip agent; one BNCI letter feeds the EEG agent; a fixed two-candidate question feeds the coordinator.
Run-to-run variation is about ±10% (the lip agent alone measured 501–619 ms p50 across runs).

## Current: speedups on top of `ui` (local branch `ui-local`)

All measured on Brev with MPS on (`latency_brev_final2.json`, coordinator files in `evaluations/results/`). Accuracy
was checked for each one before it was kept.

| Change | Speed | Accuracy check | Kept |
|---|---|---|---|
| Accepted readings decided by rules, without Nemotron | accepted voice clip → spoken **413 → 105 ms** p50 right after the change, 71–73 ms in later runs (²) | rules ask exactly what Nemotron is asked to; Nemotron still runs for ambiguous/unknown | yes |
| Auto-speak only for accepted readings | — | `test_auto_speak.py`: ambiguous, lip-only, disagreement and corrections wait for Yes/No | yes |
| Batched DTW (enrollment matching) | lip agent alone **391 → 265 ms** | distances equal to 3e-8; 0 of 270 top-3 changes | yes |
| Pre-rendered phrase audio, preloaded by the page | skips Kokoro on the gateway: about 2 ms (cold 73 vs warm 71 ms); the page's preload request fails with HTTP 500 (TECHNICAL_VERBOSE §10), so the page still downloads the audio | the gate still decides; other text still goes to Kokoro | yes (bug open) |
| Streamed lip clips (250 ms pieces over the WebSocket) | lip reading after the clip ends **898 → 407 ms** (laptop), 477 → 104 ms (box) | identical readings 12/12 (box) and 8/8 (laptop) | yes |
| Compact coordinator JSON | 272 → 84 ms | worse on ambiguous readings and after a rejection | **no** (`UNSPOKEN_COORD_COMPACT=1`) |
| MTP speculative decoding | 272 → 229 ms | NeMo Gym unchanged | **no**: one CUDA fault under full load took every GPU process down under MPS (`LLM_MTP=1`) |
| Prefix caching (`--mamba-cache-mode all`) | — | — | **no**: cannot hit; the cache block is 4,224 tokens and the prompt about 1,700 |

(²) Measured with an ad-hoc voice-only bench on the box (40 Kokoro-rendered phrase clips), printed, not saved to a file: 413 / 459 ms before the change (Nemotron planned every message); 106 / 120 ms with accepted readings decided by rules (commit `51df483`); 105 / 115 ms after auto-speak was limited to accepted readings (`09c44c1`); 71 / 89 ms in the final run (same run as `latency_brev_final2.json`); on `demo-final`, 73 / 84 ms with the gateway's phrase cache cold and 71 / 80 ms warm. So the later 105 → 71 ms drop is **not** from pre-rendered phrase audio (worth about 2 ms), and its cause was not isolated.

**What a person waits for now:**
- an accepted voice message: about 0.85–0.9 s, of which 0.7 s is the deliberate pause;
- a lip message: about 1.3 s to the question, of which 0.9 s is the deliberate pause.

Not done:
- **Early lip send:** the early clip would be a different input, and verifying it needs the person's own videos.
- **Face-crop upload:** the upload is already hidden by streaming.
- **Another region:** 120 ms round trip; that is a deployment choice.

## Earlier: branch `ui` (enrollment lip reading, auto-speak, MPS + early voice decode)

Measured with an auto-speak-aware copy of the benchmark (the gateway speaks without a confirm step). Files:
`latency_brev_ui_mps0.json` and `latency_brev_ui_mps1.json`; `latency_brev_ui_as_pushed.json` is `ui` as first pushed, before
MPS and early decode were added.

| p50 / p95 ms | MPS off | MPS on |
|---|---:|---:|
| Voice, every agent busy | 244 / 375 | **75 / 118** |
| Kokoro, every agent busy | 79 / 93 | **30 / 62** |
| Coordinator, every agent busy | 352 / 425 | 336 / 342 |
| Lips (enrollment matcher), every agent busy | 436 / 468 | 404 / 464 |
| Alone: voice / lips / coordinator / Kokoro | 60 / 409 / 314 / 23 | 61 / 391 / 330 / 22 |
| One message end to end (voice + lip clip → auto-spoken) | 434 / 469 | 435 / 474 |
| Voice result after the 700 ms pause, clip sent then | 337 / 377 | 109 / 119 |
| Voice result after the 700 ms pause, **early decode** | **3 / 22** | **2 / 2** |

**From the laptop through the SSH tunnel:**

| Measurement | Result |
|---|---|
| Round trip | ≈120 ms |
| Upload speed | ≈1.75 MB/s |
| Lip clip upload (1.2 MB at 2.5 Mbps) | **0.68 s** (34 ms of it is server work) |
| Voice result after the pause, clip sent then | 211 ms p50 |
| Voice result after the pause, early decode | **64 ms** p50 |

**Changes on `ui` that affect speed:**
- **Lip agent:** about 0.4 s per clip: 250 ms decode, face detection and crop; 10 ms encoder; 127 ms DTW over the
  90 enrolled recordings. Free-text beam search is off once phrases are enrolled.
- **Coordinator:** 309–330 ms alone, up from 231 ms. The prompt is 1,784 tokens with two more tools, the output is
  73–86 tokens, and vLLM's prefix cache hits 0% (the model's Mamba layers are the likely reason).
- **Auto-speak** waits for Nemotron's plan, so the coordinator is now on every message's path to speech.

**What a person waits for:**

| Step | Voice message | Lip message |
|---|---:|---:|
| Pause before the clip is cut | 0.7 s | 0.9 s |
| Upload and agent | (hidden by early decode) | ≈0.68 s + ≈0.4 s |
| Coordinator | ≈0.33 s | ≈0.33 s |
| Audio back | ≈0.15–0.3 s | ≈0.15–0.3 s |
| **About** | **1.2–1.4 s** | **≈2.5 s** |

The biggest remaining costs:
- the lip video upload: a lower bitrate needs an accuracy check first;
- the coordinator's output length and its missing prefix cache.

The sections below are the runs and decisions that led here, made before the `ui` changes (free-text lip
reading, a confirm step).

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

- **MPS removes most of the interference** (run 2 → run 3, nothing else changed). With every agent busy, the voice agent drops from 289 to 70 ms p50,
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
