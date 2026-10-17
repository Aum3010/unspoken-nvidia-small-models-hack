# Working with NVIDIA Brev

Brev is where GPU work runs. Right now that is the **audio-only ASR evaluation** in `asr_eval/`. Nothing else in the app (coordinator, lip reading, EEG, TTS service) is started on Brev. This page covers the CLI, choosing an instance, the day-to-day loop, and the problems we already hit.

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
| Layout | `/workspace/venv-asr` (venv), `/workspace/hf` (HF cache), `/workspace/asr_eval_data` (data, symlinked as `~/asr_eval/data`), `~/asr_eval` (code) |

## 4. The loop

Everything goes through `brev exec` (run a command) and `brev copy` (move files). Both use SSH under the hood. Long jobs run with `nohup`, so they survive the SSH session closing.

```bash
# 0. develop and test on your laptop first (CPU, free)
cd asr_eval && python run_eval.py --config configs/local_smoke.yaml --limit 6

# 1. commit, then ship the committed harness + helper scripts
bash asr_eval/scripts/brev/sync.sh marginal-sapphire-whale

# 2. one-time: environment, requirements.lock, datasets, Kokoro sanity clips (~20-40 min)
brev exec marginal-sapphire-whale 'cd ~/asr_eval && nohup bash scripts/brev_prepare.sh > prepare.log 2>&1 < /dev/null &'

# 3. quick end-to-end pass (10 clips per group), then the full run
brev exec marginal-sapphire-whale 'bash ~/run_stage.sh quick'
brev exec marginal-sapphire-whale 'bash ~/run_stage.sh full'

# 4. watch, or block until done
brev exec marginal-sapphire-whale 'bash ~/status.sh'
brev exec marginal-sapphire-whale 'bash ~/wait_for.sh ~/asr_eval/run.log run_eval.py 90'

# 5. bring results home (summary, plots, failures, thresholds, env, requirements.lock)
bash asr_eval/scripts/brev/fetch_results.sh marginal-sapphire-whale

# 6. delete the instance when finished (after step 5!)
brev delete marginal-sapphire-whale
```

### Helper scripts (`asr_eval/scripts/brev/`)

| Script | Runs on | Does |
|---|---|---|
| `install_cli_wsl.sh` | laptop (WSL/Linux) | Installs the Brev CLI binary after checking its SHA-256 |
| `sync.sh <inst>` | laptop | `git archive HEAD asr_eval`, copies it and the helpers up, unpacks over `~/asr_eval` |
| `fetch_results.sh <inst> [--raw]` | laptop | Copies results back. Per-clip CSVs only with `--raw`, because they contain volunteers' transcripts |
| `run_stage.sh quick\|full\|phases X` | instance | Starts `run_eval.py` under `nohup`, logging to `~/asr_eval/run.log` |
| `status.sh` | instance | Log tails, data sizes, running jobs, GPU use |
| `wait_for.sh LOG PATTERN MIN` | instance | Blocks until a job ends, then prints its log tail |
| `mirror_speed.sh` | instance | Measures OpenSLR mirror speeds |

Other useful commands: `brev shell <inst>` (interactive shell), `brev port-forward <inst> --port 8888:8888` (e.g. Jupyter), `brev open <inst>` (VS Code/Cursor).

## 5. Problems we hit, and the fixes

| Symptom | Cause | Fix |
|---|---|---|
| `brew: command not found` on Windows | Brev has no Windows build | Install the Linux binary in WSL (`install_cli_wsl.sh`) |
| `syntax error near unexpected token` from `brev exec` | Quotes are re-parsed by PowerShell/Git Bash → WSL → SSH | Put logic in a script on the instance and `brev exec '<inst>' 'bash ~/script.sh'` |
| Git Bash turns `/tmp/...` into `C:/...` when calling `wsl` | MSYS path conversion | Call `wsl` from PowerShell, or set `MSYS_NO_PATHCONV=1` |
| MUSAN download stuck at ~0.3 MB/s | Main OpenSLR server throttles | `fetch_data.py` uses the ELDA mirror first (~10 MB/s measured). Check with `mirror_speed.sh` |
| Background job dies when `brev exec` returns | Job tied to the SSH session | `nohup … > log 2>&1 < /dev/null &` |
| `python3 -m venv` fails | Ubuntu lacks `python3-venv` | `setup_brev.sh` installs it with `sudo -n apt-get` |
| No `/workspace` | Provider image differs | `setup_brev.sh` creates it with `sudo -n`, falling back to `~/workspace` |
| Old PyTorch errors with "no kernel image" on Blackwell | Needs CUDA 12.8+ builds | `setup_brev.sh` installs torch from the `cu128` index |
| Kokoro import fails on Windows ("filename or extension is too long") | spaCy DLL path exceeds Windows MAX_PATH | Generate TTS on Brev. Locally use `make_tts.py --engine sapi` for plumbing tests only |

## 6. Cost and privacy rules

- **Debug on the laptop** (`configs/local_smoke.yaml`). The instance should only ever run code that already passed locally.
- **Run in one sitting and delete afterwards** on non-stoppable types. Re-creating costs about 15–20 min of setup.
- **Volunteer recordings uploaded to `/workspace` are personal data.**
  - Consent text in `scripts/record.py` says they're stored on the evaluation GPU machine.
  - Don't keep raw per-clip outputs in git.
  - Deleting the instance deletes the disk; delete leftover copies elsewhere too.
- **Don't expose HTTP ports publicly** for anything that accepts audio without authentication.
