---
name: apple-silicon-to-cuda
description: Port Apple Silicon code (PyTorch MPS, MLX, mlx-whisper, mlx-audio, mlx-lm, Ollama/Metal, CoreML) to NVIDIA CUDA on Linux, e.g. an NVIDIA Brev GPU. Use when moving a Mac-built service, model or script to a CUDA box, when code calls torch.device("mps") / mlx / mlx_whisper / mlx_audio / ollama, or when a Mac-only dependency fails to install on Linux.
---

# Porting Apple Silicon code to CUDA

Goal: the same behaviour on an NVIDIA GPU, measured, not assumed. Work in this order: **inventory → map → port → verify → measure**. Never port and benchmark in the same step.

## 1. Inventory: find every Mac-only touch point

```bash
grep -rnE "mps|mlx|metal|coreml|ollama|MPS|torch\.backends\.mps|mlx_whisper|mlx_audio|mlx_lm|PYTORCH_ENABLE_MPS_FALLBACK" --include=*.py --include=*.sh --include=*.txt --include=*.toml .
grep -rnE "brew |/opt/homebrew|\.dylib|darwin|sysctl|afplay|say " --include=*.sh --include=*.py .
```

For each hit, write down: the file, what runs there (model, library), the precision/quantisation, and how its output is consumed. Model output formats (scores, timestamps, casing) are part of the contract.

## 2. Map: Apple stack to CUDA stack

| Apple Silicon | CUDA replacement | Notes |
|---|---|---|
| `torch.device("mps")` | `torch.device("cuda")` | Select with one helper: `"cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"`. Honour an env override (e.g. `APP_DEVICE`) |
| `torch.mps.synchronize()` / `empty_cache()` | `torch.cuda.synchronize()` / `empty_cache()` | Always synchronize before reading a timer |
| `PYTORCH_ENABLE_MPS_FALLBACK=1` | remove | CUDA has no op gaps to fall back from |
| MPS patches in third-party code (e.g. a CTC scorer forced to `x.device`) | usually unneeded | Upstream code normally targets CUDA. Keep a patch only if it's device-agnostic |
| `mlx_whisper` | `faster-whisper` (CTranslate2), or `transformers` Whisper | faster-whisper: `WhisperModel(repo, device="cuda", compute_type="float16")`. `word_timestamps=True` gives words with probabilities. Map `avg_logprob` / `no_speech_prob` from segments |
| `mlx_audio` Kokoro TTS | `kokoro` (`KPipeline(lang_code="a")`), PyTorch | Same hexgrad/Kokoro-82M weights; outputs 24 kHz |
| `mlx-lm` / Ollama (Metal) LLM | vLLM (OpenAI-compatible server) | `vllm serve <hf-repo> --enable-auto-tool-choice --tool-call-parser …`. Clients switch to `/v1/chat/completions`. Set `--gpu-memory-utilization` well below 0.9 when other models share the GPU |
| MLX 4-bit (`mlx-community/*-4bit`) | NVFP4 (Blackwell native, Hopper via weight-only kernels), FP8 (Hopper/Ada), AWQ/GPTQ | Different checkpoint repo; re-check quality and parameter accounting |
| CoreML models | ONNX Runtime CUDA / TensorRT, or the original PyTorch weights | Re-export from the source model, not from the CoreML file |
| `afplay` / `say` / AVFoundation | `soundfile` + browser playback, `ffmpeg` | Servers have no audio device |
| Homebrew paths, `.dylib` | apt packages, `.so` | `ffmpeg`, `espeak-ng`, `libsndfile1`, `python3-venv`, `python3-dev` |

MLX has an experimental Linux CUDA backend. Libraries built on it (mlx-whisper, mlx-audio) are not guaranteed to work there. Prefer native CUDA libraries unless you have verified the MLX path on the target GPU.

## 3. Environment on the CUDA box

1. **Match the PyTorch wheel to the GPU.** Blackwell (RTX PRO 6000, B200, sm_120/sm_100) needs CUDA 12.8+ wheels: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128`. Old wheels fail with "no kernel image is available". Hopper and Ada also work with cu128.
2. **Pin torch, then install the rest against it.** Otherwise NeMo/vLLM may replace it: `pip freeze | grep -E "^torch(audio)?==" > constraints.txt; pip install -r requirements.txt -c constraints.txt`. Write `pip freeze > requirements.lock`.
3. **Use separate venvs for stacks with conflicting pins** (vLLM, NeMo, mediapipe/`numpy<2`). Services run as separate processes anyway; give each launcher its own interpreter.
4. **CTranslate2 (faster-whisper) needs cuDNN 9.** torch's wheel ships it under `site-packages/nvidia/cudnn/lib`. If CTranslate2 can't find it, preload it before importing:
   ```python
   import ctypes, glob, site
   for sp in site.getsitepackages():
       for lib in glob.glob(f"{sp}/nvidia/cudnn/lib/libcudnn*.so.9"):
           ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
   ```
5. **Keep caches and data on the persistent disk** (`HF_HOME=/workspace/hf`). Long jobs go under `nohup … > log 2>&1 < /dev/null &` or tmux, so they survive SSH.
6. **Fork vs spawn.** Linux defaults to `fork`. Don't fork after CUDA is initialised; use `multiprocessing.get_context("spawn")` or start workers before touching the GPU. One model per subprocess also gives clean per-model memory numbers.

## 4. Port, one component at a time

- **Keep the interface, swap the backend.** Wrap each model behind the same function signature it had on the Mac (e.g. `transcribe(wav) -> {"text", "words", ...}`), so callers don't change.
- **Precision:**
  - fp16/bf16 on CUDA (bf16 needs Ampere or newer);
  - compute softmax/log-softmax and losses in fp32: `logits.float()`;
  - `torch.autocast("cuda", dtype=torch.bfloat16)` for NeMo-style models.
  - MPS lacked float64; CUDA has it, but don't let float64 creep into hot paths.
- **Determinism:** turn off feature dither, padding tricks and sampling. Set `temperature=0` and `condition_on_previous_text=False` for Whisper.
- **Library defaults differ.** faster-whisper skips segments it thinks are silent (`no_speech_threshold`); mlx-whisper's defaults may not. Make silence handling explicit, and record the signals instead of trusting hidden skips.
- **Memory reporting:** `torch.cuda.max_memory_allocated()` misses non-PyTorch runtimes (CTranslate2, TensorRT). Also record device-level use via `torch.cuda.mem_get_info()` or `nvidia-smi`.
- **Parameter counting:** quantised tensors are packed differently.
  - MLX: `U32` + `.scales`, 32/bits values per element.
  - NVFP4: `U8` = 2 weights per byte, plus `F8_E4M3` block scales that are *not* parameters.
  - Count logical parameters, or a budget ledger will undercount by about 2×.

## 5. Verify: prove it's the same, then prove it works

1. **Smoke test on CPU on the laptop first** (same code, `device="cpu"`, tiny model such as `whisper-tiny`). GPU time is for code that already runs.
2. **Golden-output check.** Run 10–20 fixed inputs through the Mac version and the CUDA version and compare the outputs. Expect small numeric drift; flag changed transcripts or rankings.
3. **Sanity metric on a public set.** For ASR: LibriSpeech test-clean WER should be in the low single digits. If it's off by 10×, the wiring is wrong (sample rate, channels, casing, normaliser), not the model.
4. **Edge inputs:** silence, sub-frame clips (< 32 ms crash some VADs: pad them), very long clips, clipped audio. One bad clip must be logged, not crash the run.

## 6. Measure, on the target GPU

- Batch 1, warm (discard the first ~5 runs), `torch.cuda.synchronize()` around every timed stage. Report p50/p95/max and RTFx.
- Memory: peak allocator and device-level.
- Cold load time separately.
- Measure on the GPU you'll deploy on. Memory transfers across GPUs; latency doesn't.
- If something else will share the GPU later (e.g. an LLM), cap its memory (`--gpu-memory-utilization`) and re-measure under load.

## Checklist

- [ ] Inventory grep is clean, or every remaining hit is a guarded fallback
- [ ] Device selection goes through one helper with an env override
- [ ] torch wheel matches the GPU architecture; versions pinned in `requirements.lock`
- [ ] Each replaced model passes the golden-output check and a public sanity metric
- [ ] Silence/edge inputs handled explicitly; per-item errors logged, not fatal
- [ ] Latency and memory measured on the target GPU, synchronized, warm
- [ ] Docs updated: what changed, why, and any measured differences

## In this repo

| Mac component | CUDA replacement used | Where |
|---|---|---|
| `services/audio` (mlx-whisper base) | faster-whisper / Parakeet / Nemotron via `asr_eval/harness/models.py` | Evaluated in `asr_eval/`; the winner replaces it |
| `services/tts` (mlx-audio Kokoro) | `kokoro` KPipeline on CUDA | `asr_eval/scripts/make_tts.py` |
| `services/vsr` (torch MPS + Chaplin patch) | torch CUDA (`UNSPOKEN_VSR_DEVICE=cuda`) | `services/vsr/app.py` |
| Coordinator (Ollama / MLX) | vLLM, OpenAI API (`UNSPOKEN_LLM_BACKEND=openai`) | `scripts/serve_vllm_brev.sh` |
| `scripts/verify_weights.py` (MLX packing) | needs NVFP4 unpacking before reporting Brev checkpoints | `scripts/verify_weights.py` |

Verified on an RTX PRO 6000 Blackwell (driver 580): torch 2.11.0+cu128, NeMo 3.0.0, faster-whisper 1.2.1 / CTranslate2 4.8.3, transformers 5.20.1. See `docs/BREV.md` for the box itself.
