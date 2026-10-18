# Working with NVIDIA Brev

Brev is where GPU work runs: the **whole Unspoken backend** (gateway, lip reading, voice, EEG replay, TTS, and the Nemotron coordinator on vLLM; see §4b), and the **audio-only ASR evaluation** in `asr_eval/`. The web UI stays on the laptop and reaches the backend through an SSH tunnel. This page covers the CLI, choosing an instance, the day-to-day loops, and the problems we already hit.

## 1. Install the CLI

| Your machine | Install |
|---|---|
| macOS | `brew install brevdev/homebrew-brev/brev` |
| Linux | `bash asr_eval/scripts/brev/install_cli_wsl.sh` (official binary, checksum-verified, no sudo) |
| Windows | No native build exists (releases ship only darwin/linux). Use **WSL Ubuntu**: `wsl -d Ubuntu -- bash asr_eval/scripts/brev/install_cli_wsl.sh` |

On Windows, run every `brev …` command inside WSL, for example `wsl -d Ubuntu -- bash -lc "brev ls"`. The CLI ends up in `~/.local/bin/brev` inside WSL.

## 2. Log in

```bash
brev login            # opens the browser, or:
brev login --token <token-from-console>
brev ls               # should print your org and instances
```

- Treat API keys and tokens like passwords. Never paste them into chats, issues, commits or shared docs.
- If one leaks, revoke it in the Brev console and create a new one.
- The CLI stores its credentials under `~/.brev/`.

## 3. Pick an instance

```bash
brev search gpu --min-vram 24 --sort price --wide   # FEATURES: S = stoppable, R = rebootable, P = flex ports
brev search gpu --min-vram 80 --sort price --wide
```

Prices seen on 2026-10-17 (they change; re-run `brev search`):

| Type | GPU | $/hr | vCPU / RAM | Disk | Stoppable |
|---|---|---:|---|---|---|
| `hyperstack_A6000` | A6000 48 GB (Ampere) | 0.60 | 28 / 58 GB | 100 GB | no |
| `massedcompute_L40S` | L40S 48 GB (Ada) | 1.06 | 12 / 72 GB | 625 GB | no |
| `g6.xlarge` (AWS) | L4 24 GB | 0.97 | 4 / 16 GB | flexible | yes |
| `massedcompute_A100_sxm4_80G` | A100 80 GB | 1.66 | 14 / 100 GB | 625 GB | no |
| `massedcompute_RTXPro6000` | RTX PRO 6000 96 GB (Blackwell) | 2.63 | 16 / 192 GB | 750 GB | no |
| `hyperstack_H100` | H100 80 GB | 3.00 | 28 / 180 GB | 850 GB | no |

**How to choose:**
- **Audio models only:** any 24–48 GB GPU is enough. The candidates have 0.6–1.6B parameters and need a few GB.
  - Memory numbers transfer between GPUs; latency does not.
  - A slower card gives a conservative latency number.
  - Many vCPUs help, because building degraded copies and VAD are CPU work.
- **Full stack with the Nemotron 3.5 Lightning coordinator:** Blackwell (RTX PRO 6000 / B200) runs the NVFP4 checkpoint natively. Hopper (H100) runs it through weight-only fallback kernels.
- **Non-stoppable instances keep billing until deleted.** Copy results off before deleting.

```bash
brev create unspoken-asr --type hyperstack_A6000    # billing starts now
brev delete unspoken-asr                            # billing stops; disk is gone
```

### The instance we use

| | |
|---|---|
| Name | `marginal-sapphire-whale` (created in the console 2026-10-17) |
| Type | `massedcompute_RTXPro6000`, $2.63/hr, **not stoppable** |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB, driver 580.126.09 |
| Host | 16 vCPU, 141 GB RAM, 665 GB free disk, Ubuntu with Python 3.10.12, user `shadeform`, passwordless sudo |
| Layout | `/workspace/venv-asr` (NeMo venv: asr_eval and the app's voice agent), `/workspace/venv-app` (app agents, torch cu128), `/workspace/venv-vllm` (vLLM), `/workspace/hf` (HF cache), `/workspace/asr_eval_data` (data, symlinked as `~/asr_eval/data`), `~/asr_eval` (bench code), `~/unspoken` (app backend) |
| Recordings | `/workspace/asr_eval_data/own/`: 307 clips from 3 speakers (17 MB), backed up to the laptop in `asr_eval/data/own/` on 2026-10-17 |
| Running | The whole app backend from branch `ui` (§4b: vLLM + 6 services under NVIDIA MPS, about 63 GB of GPU memory). The asr_eval live test bench is **stopped** (it held a second copy of Nemotron Speech); restart it with `serve_ui.sh` only when recording |
| Status | **Kept running** for more recording sessions. Billing continues at $2.63/hr until deleted (about $16 by the end of 2026-10-17). Copy recordings home before deleting |

## 4. The loop

Everything goes through `brev exec` (run a command) and `brev copy` (move files). Both use SSH under the hood. Long jobs run with `nohup`, so they survive the SSH session closing.

```bash
# 0. develop and test on your laptop first (CPU, free)
cd asr_eval && python run_eval.py --config configs/local_smoke.yaml --limit 6

# 1. commit, then ship the committed harness + helper scripts
bash asr_eval/scripts/brev/sync.sh marginal-sapphire-whale

# 2. one-time: environment, requirements.lock, datasets, Kokoro sanity clips (~20-40 min)
brev exec marginal-sapphire-whale 'cd ~/asr_eval && nohup bash scripts/brev_prepare.sh > prepare.log 2>&1 < /dev/null &'

# 3. quick end-to-end pass (10 clips per group), then the full run (+ leave-one-speaker-out with `all`)
brev exec marginal-sapphire-whale 'bash ~/run_stage.sh quick'
brev exec marginal-sapphire-whale 'bash ~/run_stage.sh all'

# 4. watch, or block until done
brev exec marginal-sapphire-whale 'bash ~/status.sh'
brev exec marginal-sapphire-whale 'bash ~/wait_for.sh ~/asr_eval/run.log run_eval.py 90'

# 5. bring results home (summary, plots, failures, thresholds, LOSO, env, requirements.lock) and back up recordings
bash asr_eval/scripts/brev/fetch_results.sh marginal-sapphire-whale
brev copy marginal-sapphire-whale:/workspace/asr_eval_data/own/ asr_eval/data/own/   # gitignored; voice data stays out of git

# 6. delete the instance when finished (after step 5!)
brev delete marginal-sapphire-whale
```

### Live test UI: browser on your laptop, models on Brev

`asr_eval/serve.py` serves a push-to-talk page and runs each clip through the evaluation pipeline: Silero gate → model → rankers → `decide()` with the thresholds frozen in `results/thresholds.json`. The page shows:
- the decision (candidate / unknown / no speech) and **which gate decided it**;
- the top 3 phrases with scores, and the transcript;
- the VAD and Whisper signals;
- server and round-trip latency.

The **Record** tab is a guided recorder:
1. Enter a pseudonym and pick a condition.
2. Prompts (the 20 phrases, optionally the negatives) appear one at a time in random order.
3. Hold, speak, release: the clip is saved under that prompt in `data/own/` and the next prompt appears.
4. **R** redoes the last clip, **S** skips one, and a session resumes where that speaker and condition left off.

The **Test** tab shows full decisions without saving anything. A **Sensitivity** selector switches between the strict (headline) and balanced operating points.

```bash
brev exec marginal-sapphire-whale 'bash ~/serve_ui.sh parakeet-ctc-0.6b ctc_sum'   # (re)start; any model/method from the config
brev port-forward marginal-sapphire-whale --port 8800:8800                         # laptop (WSL on Windows)
# open http://localhost:8800 in Chrome/Edge on the laptop
```

- The server listens on `127.0.0.1` on the instance and is reachable only through the SSH tunnel. Don't expose it publicly.
- Browsers allow the microphone only on `localhost` or HTTPS, and the tunnel provides `localhost`.
- On Windows, WSL forwards `localhost` to Windows, so the Windows browser works.
- `brev port-forward` returns immediately and leaves a persistent SSH connection holding the tunnel.
- Restart `serve_ui.sh` after an evaluation run so it loads the new thresholds.
- Round-trip latency in the UI includes your network path to the instance (Kansas City for ours); server time is listed separately.

### Helper scripts (`asr_eval/scripts/brev/`)

| Script | Runs on | Does |
|---|---|---|
| `install_cli_wsl.sh` | laptop (WSL/Linux) | Installs the Brev CLI binary after checking its SHA-256 |
| `sync.sh <inst>` | laptop | `git archive HEAD asr_eval`, copies it and the helpers up, unpacks over `~/asr_eval` |
| `fetch_results.sh <inst> [--raw]` | laptop | Copies results back. Per-clip CSVs only with `--raw`, because they contain volunteers' transcripts |
| `serve_ui.sh [model] [method] [point]` | instance | (Re)starts the live test bench on 127.0.0.1:8800 (default: Nemotron en, `forced_llr`, balanced) |
| `run_stage.sh quick\|full\|all\|phases X` | instance | Starts `run_eval.py` under `nohup` (`all` = every phase + `scripts/loso.py`), logging to `~/asr_eval/run.log` |
| `status.sh` | instance | Log tails, data sizes, running jobs, GPU use |
| `wait_for.sh LOG PATTERN MIN` | instance | Blocks until a job ends, then prints its log tail |
| `mirror_speed.sh` | instance | Measures OpenSLR mirror speeds |

Other useful commands: `brev shell <inst>` (interactive shell), `brev port-forward <inst> --port 8888:8888` (e.g. Jupyter), `brev open <inst>` (VS Code/Cursor).

## 4b. The whole app backend on Brev

The UI runs on the laptop (`cd web && npm run dev` → http://localhost:8800). Every agent runs on the instance, bound to `127.0.0.1`, and the laptop reaches only the gateway, through `brev port-forward`.

| Service | Mac | Brev (CUDA) |
|---|---|---|
| gateway :8000, eeg :8103 | CPU | CPU, unchanged |
| audio :8102 | Nemotron Speech Streaming en 0.6B + Silero VAD (NeMo, CPU/MPS) | the same, CUDA bf16, run from the NeMo venv `/workspace/venv-asr` (asr_eval's decision code and frozen thresholds) |
| tts :8104 | mlx-audio Kokoro-82M | `kokoro` KPipeline on CUDA (`UNSPOKEN_TTS_BACKEND`) |
| vsr :8101 | torch MPS | torch CUDA (`pick_device`, `UNSPOKEN_VSR_DEVICE`); matches the person's enrolled phrases in `data/enroll/` (copied separately, see below) |
| coordinator :8105 | Ollama | vLLM 0.31 on 127.0.0.1:8001 serving `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` (`--gpu-memory-utilization 0.55`, so the agents fit beside it under MPS). On this RTX PRO 6000 (sm_120) vLLM picks Marlin weight-only FP4 for the GEMMs/MoE, Triton attention and the PyTorch sampler (see §5) |

```bash
# laptop: commit first, then ship services/shared/scripts to ~/unspoken
bash scripts/brev/sync_app.sh marginal-sapphire-whale
# instance, once (~15-30 min): uv venvs /workspace/venv-app (torch cu128) + /workspace/venv-vllm, checkpoints, EEG decoders
brev exec marginal-sapphire-whale 'cd ~/unspoken && nohup bash scripts/brev/setup_app.sh > .run/setup.log 2>&1 < /dev/null &'
# laptop -> instance: the lip agent's enrollment data is not part of sync_app.sh
git archive --format=tar.gz -o /tmp/enroll.tgz HEAD:data enroll && brev copy /tmp/enroll.tgz marginal-sapphire-whale:/home/shadeform/enroll.tgz
brev exec marginal-sapphire-whale 'cd ~/unspoken && rm -rf data/enroll && tar -xzf ~/enroll.tgz -C data'
# instance: (re)start everything detached (NVIDIA MPS on; UNSPOKEN_MPS=0 for off); `status` shows each /health, GPU memory, MPS
brev exec marginal-sapphire-whale 'bash ~/unspoken/scripts/brev/start_app.sh'
brev exec marginal-sapphire-whale 'bash ~/unspoken/scripts/brev/start_app.sh status'
# laptop: stop any local gateway (port 8000), then tunnel and open the UI
bash scripts/brev/connect.sh marginal-sapphire-whale
```

- **Privacy:** mouth video and voice clips leave the laptop for the instance (over SSH). They are processed in memory and not stored. The gateway reports `host: "NVIDIA Brev"`, and on `demo-final` the UI shows it: the camera chip says "Camera · clips to NVIDIA Brev" and calibration says clips are processed on the GPU server (on branch `ui` it still says "on device": tell the person there).
- **Ports:** the asr_eval bench also uses 8800 on the instance. The app's UI uses the laptop's 8800, so don't forward the bench to 8800 while the app UI is open.
- **NVIDIA MPS** is on by default: `start_app.sh` starts the daemon before any GPU process, and `UNSPOKEN_MPS=0` stops it. It is detected by full command line (`pgrep -f`), since `pgrep -x` cannot match the 15-character-truncated process name.
- **Early voice decode:** the gateway's `/capture/audio/early|commit|discard` routes let the voice decode run during the end-of-utterance pause.
- Measured 2026-10-18 on branch `ui` (MPS on, asr_eval bench stopped): the whole stack peaks at about 63 GB of the 96 GB. One message end to end took 0.44 s p50 on the box on branch `ui` (`latency_brev_ui_mps1.json`), and **0.31 s** after the speedups (`latency_brev_final2.json`, 309 ms p50); the voice agent takes 75 ms with every model busy; the voice result is ready 64 ms after the pause from the laptop ([SERVING.md](SERVING.md)). Every detail: [TECHNICAL_VERBOSE.md](TECHNICAL_VERBOSE.md).
- **Latency:** `PYTHONPATH=. /workspace/venv-app/bin/python scripts/brev/bench_latency.py` (on the instance) measures each agent alone, all at once, and one message end to end. Results and how to serve faster: [SERVING.md](SERVING.md).
- Logs: `~/unspoken/.run/logs/<service>.log`; pinned versions: `~/unspoken/.run/requirements-cuda.lock`, `requirements-vllm.lock`.

## 5. Problems we hit, and the fixes

| Symptom | Cause | Fix |
|---|---|---|
| vLLM: `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist` | FlashInfer JIT-compiles its attention and sampling kernels on first use; the image has no CUDA toolkit (and the driver is CUDA 13.0) | `--attention-backend TRITON_ATTN` and `VLLM_USE_FLASHINFER_SAMPLER=0` (both in `start_app.sh`); no compiler needed |
| `brev port-forward` returns but `localhost:8000` refuses for a while | The forward rides brev's long-lived SSH control master and appears late; it then outlives the command | Wait, or use `scripts/brev/connect.sh` (dedicated SSH, up at once, Ctrl-C closes it) |
| `mne==1.13.2` vs `numpy==1.26.4` unsatisfiable | New mne needs numpy ≥ 2.1; mediapipe 0.10.21 and pyriemann 0.7 need numpy < 2 | `mne==1.11.0` in `requirements-cuda.txt` |
| `brew: command not found` on Windows | Brev has no Windows build | Install the Linux binary in WSL (`install_cli_wsl.sh`) |
| `syntax error near unexpected token` from `brev exec` | Quotes are re-parsed by PowerShell/Git Bash → WSL → SSH | Put logic in a script on the instance and `brev exec '<inst>' 'bash ~/script.sh'` |
| Git Bash turns `/tmp/...` into `C:/...` when calling `wsl` | MSYS path conversion | Call `wsl` from PowerShell, or set `MSYS_NO_PATHCONV=1` |
| MUSAN download stuck at ~0.3 MB/s | Main OpenSLR server throttles | `fetch_data.py` uses the ELDA mirror first (~10 MB/s measured). Check with `mirror_speed.sh` |
| Background job dies when `brev exec` returns | Job tied to the SSH session | `nohup … > log 2>&1 < /dev/null &` |
| `python3 -m venv` fails | Ubuntu lacks `python3-venv` | `setup_brev.sh` installs it with `sudo -n apt-get` |
| No `/workspace` | Provider image differs | `setup_brev.sh` creates it with `sudo -n`, falling back to `~/workspace` |
| Old PyTorch errors with "no kernel image" on Blackwell | Needs CUDA 12.8+ builds | `setup_brev.sh` installs torch from the `cu128` index |
| Kokoro import fails on Windows ("filename or extension is too long") | spaCy DLL path exceeds Windows MAX_PATH | Generate TTS on Brev. Locally use `make_tts.py --engine sapi` for plumbing tests only |
| `NvvmSupportError: libNVVM cannot be found` from NeMo's transducer loss | NeMo's RNNT/TDT losses are Numba CUDA kernels; pip PyTorch ships no libNVVM, and the pip NVVM wheel wasn't picked up | Use NeMo's pure-PyTorch losses (`pytorch`, `tdt_pytorch`). RNNT is fast enough (13–21 ms for 66 phrases); TDT is not (3.5–7 s) |
| `ValueError: lengths must be contiguous` | `expand()` gives stride-0 tensors; NeMo's CUDA losses need contiguous memory | `.contiguous()` on length and target tensors |
| `CUDA error: an illegal memory access` that then breaks every later clip | B × T × U × V joint tensor over 2³¹ elements (Nemotron 3.5's 13k-token vocabulary × a long clip) overflows int32 indexing | Score phrase forms in chunks capped at 2·10⁸ joint elements; long texts alone |
| `prompt_indices must be provided when concat mode is enabled` | Nemotron 3.5 is prompt-conditioned | Pass `prompt_indices` (the locale's id from the model's `prompt_dictionary`) to `forward()` |
| Transcripts end in `<en-US>` | Nemotron 3.5 appends a language tag | Strip `<xx-XX>` tags before matching, scoring and WER |
| A helper "fails" (exit 1) although it worked | The WSL wrapper's `grep -v` exits 1 when it filters every line | Check the result, not the exit code |
| `pgrep -f PATTERN` never reports a job as finished | The waiting script's own command line contains PATTERN | Use the `[x]yz` trick: `'[s]cripts/loso.py'` matches the job, not itself |

## 6. Cost and privacy rules

- **Debug on the laptop** (`configs/local_smoke.yaml`). The instance should only ever run code that already passed locally.
- **Run in one sitting and delete afterwards** on non-stoppable types. Re-creating costs about 15–20 min of setup.
- **Volunteer recordings uploaded to `/workspace` are personal data.**
  - Consent text in `scripts/record.py` says they're stored on the evaluation GPU machine.
  - Don't keep raw per-clip outputs in git.
  - Deleting the instance deletes the disk; delete leftover copies elsewhere too.
- **Don't expose HTTP ports publicly** for anything that accepts audio without authentication.
