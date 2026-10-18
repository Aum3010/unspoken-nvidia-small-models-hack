# Unspoken

### Because everyone deserves to be heard.

Imagine you are lying in a hospital bed. You are fully conscious. You understand everything happening around you.
But you cannot speak.

You are thirsty. You are in pain. You need your family. And the only thing stopping you from saying so is that
your body won't cooperate.

**What if AI could understand what you are trying to say, even when you can't say it?**

That is why we built **Unspoken**.
- **What it is:** a real-time, multimodal communication system. It turns a weak voice, silent lip movements, eye
  gaze, deliberate blinks and head nods into a voice.
- **How it works:** small specialist models read the person, and **NVIDIA Nemotron 3.5 Lightning** decides what
  to do when they are unsure.
- **The guarantee:** code, never a model, decides what may be spoken.

---

## See it work

What the demo shows, on one NVIDIA RTX PRO 6000 running every model at once:

1. **The room comes alive.** The live camera fills the page, and the agent activity view pulses as each specialist
   reports.
2. **A weak voice is enough.** Say "I need water", weakly. Nemotron Speech recognises one of the person's
   phrases, and the system says it aloud. On the GPU box that takes **71–73 ms** from the clip to speech: demo
   mode speaks a reading the voice agent accepted at once (see the fine print).
3. **No sound at all? Mouth it.** The lip agent reads the silent mouth movements and proposes the phrase on screen,
   **0.41 s** after the person stops mouthing. The video streams while they mouth.
4. **The person decides, without a word.** They confirm by looking at Yes, a deliberate blink or a nod, and reject
   with No or a head shake. Unspoken never blindly trusts a prediction.
5. **Only then is the lip reading spoken.** The policy gate signs the exact confirmed text, and the voice (Kokoro)
   will not speak anything unsigned.

Someone who could not communicate now has another way to be heard.

## Why small models, working together

The most interesting part is not any single model. **It is how we made them work together.**

```
lips ─┐                                         ┌─ you select ─┐
voice ┼─► specialists propose ─► coordinator asks ─►         ├─► you confirm ─► policy gate ─► voice
eyes ─┤   n-best / phrases +     (Nemotron 3.5    └─ or correct ┘                (exact text only)
EEG* ─┘   quality flags           Lightning, typed tools)            *recorded replay only
```

**The specialists:**
- **Weak voice:** NVIDIA **Nemotron Speech** Streaming 0.6B, which scores the person's fixed phrases and abstains
  when unsure.
- **Lips:** **Auto-AVSR**, matched against the person's own recorded phrases.
- **Eyes, blinks, head:** **MediaPipe** and **MobileGaze** in the browser, calibrated for each person.
- **Speech out:** **Kokoro**. EEG works from recorded replays only.

**The coordinator:** **NVIDIA Nemotron 3.5 Lightning**, 30B-A3B, served on **Red Hat vLLM**.
- **Its job is not to generate text.** Its job is to decide what to do when the specialists are unsure:
  - Ask for confirmation, and with which input?
  - Fix a misread word?
  - Request another attempt?
  - Switch to gaze or spelling?
  - Or ignore noise?
- **It answers with exactly one typed tool call.** The call is validated outside the model, and invalid calls fall
  back to deterministic rules.
- **An unreliable model decision cannot authorize speech.**

**The whole system fits the 33B limit:** 31.01B parameters (`scripts/verify_weights.py`).

## Proof, not promises

We did not just connect models together. We measured and optimised the system, and every number links to its
result file.

| Claim | Measured |
|---|---|
| **Two senses beat one** | voice alone **8.0%**, lips alone **53.5%**, **together 76.0%**: correct commits on paired audio-video under heavy babble noise (0 dB, GRID) · [RESULTS §6](docs/RESULTS.md) |
| **Faster lip reading** | **898 → 407 ms** after the person stops mouthing, by streaming video while they mouth (laptop through the tunnel) · [SERVING](docs/SERVING.md) |
| **A coordinator that holds up on hard cases** | custom **NVIDIA NeMo Gym** environment: ambiguity, misreads, prompt injection, pressure to speak. Held-out reward **0.985** with **0 unsafe actions**; right action 82% → 95% from Nemotron alone to Nemotron + validator · [RESULTS §5](docs/RESULTS.md), [BOUNTIES §2](docs/BOUNTIES.md) |
| **Real serving, not an API wrapper** | Nemotron on **Red Hat vLLM** beside four other models on one GPU (NVIDIA MPS): **0 errors up to 64 concurrent requests**, 16 req/s peak. If vLLM is slow, down or wrong, the coordinator falls back to rules: with vLLM killed outright (isolated from MPS in that test), the voice, lips and speech kept working · [serving SUMMARY](evaluations/results/serving/SUMMARY.md), [BOUNTIES §3](docs/BOUNTIES.md) |
| **Safety enforced in code** | **49/49** safety tests, **7/7** live attacks blocked, **0** unconfirmed texts reaching speech · [EVALUATION](docs/EVALUATION.md) |
| **Within budget** | **31.01B** of 33B parameters · [TECHNICAL_VERBOSE §8](docs/TECHNICAL_VERBOSE.md) |

More results (dysarthric speech, ALS EEG, personal lip enrollment, Gym training): [Results at a glance](#results-at-a-glance).

## What happens when AI gets it wrong?

In a chatbot, a wrong answer is an inconvenience. For someone who cannot speak, it can mean saying the exact
opposite of what they meant. So we did not optimise for accuracy alone; we engineered for uncertainty:

- **Unsure means ask.** The coordinator picks the smallest useful question.
- **Disagreement means show both.** When lips and voice disagree, the person sees both readings and chooses.
- **Negations survive.** "I don't need water" must never become "I need water". In our coordinator evaluations the
  model never dropped a negation, and an optional validator rule guards it.
- **Speech needs a signature.** Text is spoken only through a gate that signs the exact text, and the voice refuses
  anything else.
- **Every decision is audited:** raw readings, tool calls (including rejected ones), confirmations and speech.

**Small models do not have to solve everything individually.** Engineered to collaborate, they give people a
capability no single model provides reliably on its own. We are building toward a future where losing the ability
to speak does not have to mean losing the ability to communicate.

## For judges: where to look

| Track / bounty | Read | Code |
|---|---|---|
| Multi-agent system | [ARCHITECTURE](docs/ARCHITECTURE.md), [TECHNICAL_VERBOSE](docs/TECHNICAL_VERBOSE.md) | `shared/schemas.py` (typed `Evidence` / `ToolCall`), `services/gateway/` |
| NVIDIA Nemotron Lightning | [BOUNTIES §1](docs/BOUNTIES.md) | `services/coordinator/app.py` (tools, validator, fallback) |
| NVIDIA NeMo Gym | [BOUNTIES §2](docs/BOUNTIES.md), [RESULTS §5, §7–8](docs/RESULTS.md) | `evaluations/nemo_gym/unspoken_communication/`, `evaluations/nemo_gym/neuro_comm/` |
| Red Hat vLLM | [BOUNTIES §3](docs/BOUNTIES.md), [serving SUMMARY](evaluations/results/serving/SUMMARY.md) | `scripts/brev/start_app.sh`, `scripts/brev/serving/` |
| Every number, with caveats | [RESULTS](docs/RESULTS.md), [EVALUATION](docs/EVALUATION.md) | `evaluations/results/` |
| Run it yourself | [Run it on NVIDIA Brev](#run-it-on-nvidia-brev) | `scripts/brev/` |

### The honest fine print

- **Not a clinical device.** Unspoken is a research prototype: not a medical device, not a nurse-call system, not
  a mind reader.
- **Demo auto-speak.** With the default `UNSPOKEN_AUTO_SPEAK=1`, a reading the specialists already agree on is
  spoken without a confirmation, and it is audited as such. Lip-only, ambiguous and corrected readings always wait
  for the person.
- **What the numbers measure:**
  - The lips + voice result uses GRID, a fixed 51-word grammar.
  - The held-out coordinator tasks come from the same task generator as the eval set.
  - EEG is recorded replay only.
- **Where the gaps are:** every known gap is listed in [TECHNICAL_VERBOSE §10](docs/TECHNICAL_VERBOSE.md) and
  next to each number in RESULTS.
- **Where it runs:** the UI runs on your laptop, and the agents run on an NVIDIA GPU on
  [Brev](https://brev.nvidia.com), reached through an SSH tunnel. An Apple-Silicon Mac also works (Metal +
  Ollama).

---

## Contents

1. [The agents](#the-agents)
2. [Results at a glance](#results-at-a-glance)
   · [Data sources](#data-sources)
3. [Run it on NVIDIA Brev](#run-it-on-nvidia-brev) (backend on the GPU, UI on your laptop)
4. [Run it on a Mac](#run-it-on-a-mac-apple-silicon)
5. [Tests and evaluations](#tests-and-evaluations)
6. [Flags](#flags)
7. [Safety and privacy](#safety-and-privacy)
8. [Repository and docs](#repository-and-docs)
9. [Status](#status)

---

## The agents

| Agent | Model | Port |
|---|---|---:|
| Gateway | state machine, fusion, policy-gated speech, WebSocket (`services/gateway/`) | 8000 |
| Lips | Auto-AVSR (via Chaplin), matched against the person's own enrolled phrases (DTW over encoder features) | 8101 |
| Voice | Nemotron Speech Streaming en 0.6B + Silero VAD, forced phrase scoring with abstention (`asr_eval/` decision code) | 8102 |
| EEG replay | xDAWN + Riemannian P300 decoder on recorded BNCI2014-008 ALS data (CPU) | 8103 |
| Speech out | Kokoro-82M; refuses any text without a gateway confirmation token | 8104 |
| Coordinator | **Nemotron 3.5 Lightning 30B-A3B (NVFP4)** on **vLLM**: one typed tool call per decision, validated outside the model, rules as fallback | 8105 → vLLM 8001 |
| Eyes, blink, head | in the browser: MediaPipe Face Landmarker + MobileGaze, calibrated per person | — |

The page (`web/`) runs every input at once: always-on voice activity detection, mouth-movement clips, gaze, blink
and nod. Yes and No are two big targets far apart, there are phrase cards, and a live "agent activity" view shows
each agent as it reports. Every target works by click, keyboard, gaze or pointer dwell, or a deliberate blink.

All model weights total **31.01B parameters**, within the 33B budget (`scripts/verify_weights.py`).

## Results at a glance

Held-out data throughout. Every number links to its result file and the command that reproduces it.

| What | Result | Where |
|---|---|---|
| Coordinator (NeMo Gym `unspoken_communication`, 44 held-out tasks) | right action **82% → 95%** (Nemotron alone → Nemotron + validator), **0% unsafe**, 278 ms p50 | [docs/RESULTS.md](docs/RESULTS.md) §5, [docs/BOUNTIES.md](docs/BOUNTIES.md) §2 |
| Voice, dysarthric speakers (TORGO) | command words accepted correctly **40.0% → 61.1%**; wrong accepts 1.1% → 0% | RESULTS §1 |
| Lips, personal enrollment (MIRACL-VC1, 15 speakers) | top-1 phrase **72.0% → 96.1%** | RESULTS §2 |
| EEG, ALS patients (bigP3BCI Study F, held-out session) | wrong characters **24 → 1** per 120, same decoder | RESULTS §3 |
| Lips + voice together (GRID, babble at 0 dB) | correct commits: voice 8.0%, lips 53.5%, **together 76.0%** | RESULTS §6 |
| Gym training (`neuro_comm`, held-out ALS session) | Nemotron Nano 4B with a LoRA vs Bayes: wrong **29 → 5** per 120, EEG time 21.6 → 11.1 s (fewer correct; see caveats) | RESULTS §8 |
| Safety | **49/49** safety unit tests, 7/7 live attacks blocked, unconfirmed text reaching TTS **0** | [docs/EVALUATION.md](docs/EVALUATION.md) |
| Latency on one RTX PRO 6000, every model loaded (NVIDIA MPS) | accepted voice message → spoken **71–73 ms** p50 on the box; lip reading **0.41 s** after the clip ends (laptop, streamed) | [docs/SERVING.md](docs/SERVING.md) |
| vLLM under load | 0 errors up to 64 concurrent coordinator requests; peak **16 req/s**; p95 < 500 ms up to 2 concurrent | [evaluations/results/serving/SUMMARY.md](evaluations/results/serving/SUMMARY.md) |

Bounty write-ups for Nemotron Lightning, NeMo Gym and Red Hat vLLM, every number with its file:
[docs/BOUNTIES.md](docs/BOUNTIES.md).

## Data sources

Every public dataset we trained or tested on, as the repo's own docs and fetch scripts record it.
- **No public dataset is re-hosted.** `data/` is gitignored, and the fetch scripts download from the sources below.
- **Data we made is published:** synthetic Gym tasks, training labels and our own recordings are on Hugging Face,
  [AumPandya01/unspoken-data](https://huggingface.co/datasets/AumPandya01/unspoken-data) (CC BY 4.0); method,
  consent and attribution in [docs/DATA_RELEASE.md](docs/DATA_RELEASE.md).
- **Details and consent rules:** [docs/DATASETS_AND_CONSENT.md](docs/DATASETS_AND_CONSENT.md),
  [docs/EEG_DATASET_AUDIT.md](docs/EEG_DATASET_AUDIT.md), [docs/LIP_READING.md](docs/LIP_READING.md).

**EEG**

| Dataset | Citation | Licence | Source (fetched by) | How we use it |
|---|---|---|---|---|
| BNCI2014-008, P300 speller, 8 participants with ALS | Riccio et al. 2013 | CC BY-NC-ND 4.0 | `http://bnci-horizon-2020.eu/database/data-sets/008-2014/` (`scripts/fetch_assets.sh`) | **Train + test:** P300 decoder trained on calibration runs 1–3 and tested on online runs 4–7 (the original split); the app's EEG replay |
| bigP3BCI v1.0.0, **Study F** (10 ALS participants) | PhysioNet, DOI [10.13026/0byy-ry86](https://doi.org/10.13026/0byy-ry86); Study F methods: Mainsah et al. 2015, *J. Neural Eng.* 12:016013 | CC BY 4.0 | <https://physionet.org/content/bigp3bci/1.0.0/>, via the S3 mirror `https://physionet-open.s3.amazonaws.com/bigp3bci/1.0.0/`, SHA-256 checked (`scripts/data/fetch_bigp3bci.py`) | **Train:** sessions 1–2 for the stopping policies, the `neuro_comm` NeMo Gym episodes and the Nano 4B LoRA. **Test:** session 3, held out (RESULTS §3, §7, §8) |
| bigP3BCI v1.0.0, **Study H** (16 participants, not ALS) | Kalika, Collins, Caves & Throckmorton 2017, *J. Neural Eng.* 14:056010 | CC BY 4.0 | same PhysioNet mirror (`fetch_bigp3bci.py --study H`) | **Test:** EEG + gaze fusion, on held-out runs (RESULTS §4) |

**Lips**

| Dataset | Citation | Licence | Source (fetched by) | How we use it |
|---|---|---|---|---|
| GRID audiovisual corpus (33 speakers, 51-word grammar) | Cooke et al. | CC BY 4.0 | Zenodo record [3625687](https://zenodo.org/records/3625687) (`scripts/fetch_assets.sh` fetches s1, s2 and alignments) | **Test:** s1 and s2, never used to train the lip model (lip benchmark; lips + voice fusion with GRID's real audio). **Train:** the lip-decoder fine-tune (s3–s8 on a Mac; s3–s34 without s21 on Brev). The trained coordinators (E7b/E7c) use s1/s2 episodes across speakers: train on one, test on the other (RESULTS §2, §6, §9) |
| MIRACL-VC1 (15 speakers) | Rekik, Ben-Hamadou & Mahdi 2016 | research use only | obtained separately; the repo records no download URL (`scripts/eval_enrollment_miracl.py` reads `data/miracl/`) | **Test only:** personal-enrollment lip reading (RESULTS §2). Not used in the product |

**Voice** (`asr_eval/scripts/fetch_data.py` saves each landing page and its licence lines to `data/provenance.json`)

| Dataset | Citation | Licence | Source (fetched by) | How we use it |
|---|---|---|---|---|
| TORGO dysarthric speech, archives F, FC, M, MC | Rudzicz, Namasivayam & Wolff 2012 | academic / non-profit research use | <https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html> | **Test:** voice agent on dysarthric and control speakers; also unknown-speech negatives |
| LibriSpeech test-clean (OpenSLR 12) | — | CC BY 4.0 | <https://www.openslr.org/12/> (ELDA mirror first) | **Test:** 200 unknown-speech negatives; a WER sanity check |
| MUSAN noise (OpenSLR 17) | — | CC BY 4.0 (files carry their own attributions) | <https://www.openslr.org/17/> | **Test:** non-speech negatives, and additive noise in the degradation grid |
| Speech Commands v0.02, `_background_noise_` only | — | CC BY 4.0 | <https://www.tensorflow.org/datasets/catalog/speech_commands> | **Test:** non-speech negatives |
| ESC-50 (12 categories, e.g. breathing, coughing) | — | CC BY-NC 3.0 | <https://github.com/karolpiczak/ESC-50> | **Test only (non-commercial):** non-speech negatives |

**Clinical (vision preview)**

| Dataset | Citation | Licence | Source (fetched by) | How we use it |
|---|---|---|---|---|
| MIMIC-IV Clinical Database Demo v2.2 | Johnson A, Bulgarelli L, Pollard T, Horng S, Celi LA, Mark R; PhysioNet, DOI [10.13026/dp1f-ex47](https://doi.org/10.13026/dp1f-ex47) | ODbL v1.0 | <https://physionet.org/content/mimic-iv-demo/2.2/> (`scripts/vision/fetch_mimic_event.py`) | **Display only:** one charted low-glucose event and its heart rate / SpO2, numbers only, with flag `UNSPOKEN_CHECKIN`. Not trained or tested on |

**Not public, or generated by us**
- **Our recordings, now published.** Three team members (pseudonyms p01–p03) recorded 307 voice clips, used to
  tune and test the voice agent (leave one speaker out). A team member's lip enrollment takes (`data/enroll/`,
  features only, no video) are in this group too.
  - Neither is in git.
  - All of them agreed on 2026-10-18 to publish under CC BY 4.0:
    [AumPandya01/unspoken-data](https://huggingface.co/datasets/AumPandya01/unspoken-data) ([DATA_RELEASE](docs/DATA_RELEASE.md)).
- **Generated data:**
  - Kokoro-82M speech (Apache-2.0 model output), for sanity checks only and never in headline tables;
  - silence and mic-like noise;
  - the NeMo Gym coordinator tasks (`evaluations/nemo_gym/unspoken_communication/create_examples.py`, seeds 7
    and 2026);
  - the synthetic patient profile of the vision preview.
- **Referenced but not used (gated):** UASpeech, the Speech Accessibility Project corpus, CHAINS (LDC2008S09).

**Pretraining data of the models we use** (trained upstream, not by us; as stated in
[docs/MODEL_CARDS_AND_LICENSES.md](docs/MODEL_CARDS_AND_LICENSES.md)):
- **Auto-AVSR:** LRS3, checkpoint `LRS3_V_WER19.1`.
- **MobileGaze MobileOne-S0:** Gaze360.
- **The other models' training data:** not documented in this repo; see each model's upstream card (models and licences listed in that file).

---

## Run it on NVIDIA Brev

The whole backend runs on one Brev GPU: the gateway, lips, voice, EEG replay, TTS, and the coordinator on vLLM.
Every port binds `127.0.0.1` on the instance, and your laptop reaches only the gateway, through an SSH tunnel.
The UI runs on the laptop. The steps below use the instance name `marginal-sapphire-whale`; replace it with yours.

### 0. What you need

- **The Brev CLI, logged in:**
  - macOS: `brew install brevdev/homebrew-brev/brev`
  - Linux: `bash asr_eval/scripts/brev/install_cli_wsl.sh`
  - Windows: no native build exists; use WSL Ubuntu (`wsl -d Ubuntu -- bash asr_eval/scripts/brev/install_cli_wsl.sh`)
    and run every `brev …` and `scripts/brev/*.sh` command inside WSL. Call `wsl` from PowerShell, because Git Bash
    rewrites `/tmp` paths.
  - Then run `brev login` and `brev ls`.
- **A GPU instance.** Measured on an RTX PRO 6000 Blackwell 96 GB (`massedcompute_RTXPro6000`, about $2.63/hr).
  - The whole stack peaks at about 63 GB of GPU memory.
  - Blackwell runs the NVFP4 checkpoint. Hopper (H100) runs it through slower weight-only fallback kernels.
  - Many instance types are **not stoppable**: they bill until you delete them.
- **On the laptop:** Node 20+ for the UI, and Chrome or Edge.

### 1. Ship the code (from the laptop, after committing)

```bash
bash scripts/brev/sync_app.sh marginal-sapphire-whale     # committed services/, shared/, scripts/, evaluations/ -> ~/unspoken
bash asr_eval/scripts/brev/sync.sh marginal-sapphire-whale # the voice agent's NeMo setup -> ~/asr_eval (and helper scripts)
```

`sync_app.sh` ships what is **committed**, not your working tree, so commit first.

### 2. One-time setup on the instance (about 30–45 min)

```bash
# the app: /workspace/venv-app (torch cu128 + agents), /workspace/venv-vllm, checkpoints, data, EEG decoders
brev exec marginal-sapphire-whale 'cd ~/unspoken && mkdir -p .run && nohup bash scripts/brev/setup_app.sh > .run/setup.log 2>&1 < /dev/null &'
# the voice agent's NeMo environment: /workspace/venv-asr
brev exec marginal-sapphire-whale 'cd ~/asr_eval && nohup bash scripts/setup_brev.sh > setup.log 2>&1 < /dev/null &'
# watch
brev exec marginal-sapphire-whale 'tail -n 5 ~/unspoken/.run/setup.log ~/asr_eval/setup.log'
```

Long jobs always run under `nohup … < /dev/null &`, so they survive the SSH session closing.

### 3. Personal data (optional; not in git, published on Hugging Face)

- **Lip enrollment** (`data/enroll/`) is the person's own recorded phrases, stored as encoder features.
  - It is **not tracked in git**. The person agreed to publish it: `lip_enrollment/` in
    [AumPandya01/unspoken-data](https://huggingface.co/datasets/AumPandya01/unspoken-data) ([docs/DATA_RELEASE.md](docs/DATA_RELEASE.md)).
  - Without it, the lip agent has no enrolled phrases.
  - To ship a local copy to the instance:

    ```bash
    tar -czf /tmp/enroll.tgz -C data enroll && brev copy /tmp/enroll.tgz marginal-sapphire-whale:/home/shadeform/enroll.tgz
    brev exec marginal-sapphire-whale 'cd ~/unspoken && rm -rf data/enroll && tar -xzf ~/enroll.tgz -C data'
    ```

- **Vision preview** (only with `UNSPOKEN_CHECKIN=1`): run `python scripts/vision/fetch_mimic_event.py` on the
  laptop, then copy `data/vision/` to `~/unspoken/data/vision/` the same way.

### 4. Start, check, stop

```bash
brev exec marginal-sapphire-whale 'bash ~/unspoken/scripts/brev/start_app.sh'          # vLLM + 6 services, detached, NVIDIA MPS on
brev exec marginal-sapphire-whale 'bash ~/unspoken/scripts/brev/start_app.sh status'   # each /health, GPU memory, MPS, MTP
brev exec marginal-sapphire-whale 'bash ~/unspoken/scripts/brev/start_app.sh stop'
```

- **Startup time:** vLLM takes a few minutes the first time (then about 30 s, from the compile cache). The agents
  load in the background.
- **Readiness:** wait until `status` shows every service `ready`.
- **Flags:** pass them as environment variables, e.g. `brev exec … 'UNSPOKEN_AUTO_SPEAK=0 bash ~/unspoken/scripts/brev/start_app.sh'`.
- **Logs:** `~/unspoken/.run/logs/<service>.log`.

### 5. Open the UI (on the laptop)

```bash
bash scripts/brev/connect.sh marginal-sapphire-whale   # tunnel: localhost:8000 -> the gateway (stop any local gateway first)

# in another terminal, once: the web app and its browser models (not in git)
cd web && npm install
mkdir -p public/mediapipe public/vendor public/models
cp -r node_modules/@mediapipe/tasks-vision/wasm public/mediapipe/
cp node_modules/webgazer/dist/webgazer.js public/vendor/
cp -r node_modules/webgazer/dist/mediapipe/face_mesh public/mediapipe/
curl -sSL -o public/mediapipe/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
curl -sSL -o public/models/mobileone_s0_gaze.onnx https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobileone_s0_gaze.onnx

npm run dev                                            # http://localhost:8800 (proxies /api and /ws to localhost:8000)
```

- **What leaves the laptop:** mouth video and voice clips go to the instance over SSH. They are processed in
  memory and not stored.
- **Tell the person, and get their opt-in:** the page has no status chips (a final design decision). Only the
  calibration screen says that clips are processed on the GPU server, and it is skipped when a saved calibration
  is reused (TECHNICAL_VERBOSE §10, item 7).

### 6. Check that it works

```bash
# on the instance: safety tests, then a smoke test through the gateway (needs ~/need_water.wav etc., see scripts/checks/README.md)
brev exec marginal-sapphire-whale 'cd ~/unspoken && /workspace/venv-app/bin/python -m pytest -q evaluations/safety_tests'
brev exec marginal-sapphire-whale 'cd ~/unspoken && bash scripts/brev/serving/final_check.sh'   # defaults, health, tests, smoke
# from the laptop, through the tunnel
python scripts/checks/smoke.py . <clips-dir>
```

### 7. Benchmarks on the box

```bash
cd ~/unspoken && export PYTHONPATH=. UNSPOKEN_LLM_BACKEND=openai UNSPOKEN_LLM_BASE_URL=http://127.0.0.1:8001 \
  UNSPOKEN_LLM_MODEL=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
/workspace/venv-app/bin/python scripts/brev/bench_latency.py --n 20                        # each agent alone, all at once, end to end
/workspace/venv-app/bin/python evaluations/nemo_gym/run_local.py --seed 2026 --distinct-from 7 --out coordinator_heldout.json
nohup /workspace/venv-app/bin/python scripts/brev/serving/bench_concurrency.py > .run/logs/serving_A.log 2>&1 < /dev/null &   # vLLM concurrency
```

**Serving experiments that restart services:**
- `scripts/brev/serving/isolation_experiment.sh` and `max_capacity.sh` wait until nobody has used the app for
  10 minutes.
- They restore the defaults when they finish, even after a failure.
- How to run them, and what they found: [evaluations/results/serving/SUMMARY.md](evaluations/results/serving/SUMMARY.md).

**Bring results home:** `bash scripts/brev/serving/fetch_results.sh marginal-sapphire-whale`.

### 8. When you are done

- Copy anything you need off the instance (results, recordings), then `brev delete <instance>`; billing stops
  and the disk is gone.
- **Problems we already hit, with their fixes:** [docs/BREV.md](docs/BREV.md) §5.
  - no `nvcc` → Triton attention;
  - quoting through PowerShell, WSL and SSH;
  - a slow `brev port-forward`;
  - and more.

---

## Run it on a Mac (Apple Silicon)

```bash
./scripts/setup_models.sh   # venv, pinned Chaplin + MPS patch, checkpoints, EEG + GRID data, web deps, ollama pull
./scripts/run_all.sh        # gateway, 5 specialist services, web -> http://localhost:8800
./scripts/smoke_test.sh     # health + board -> confirm -> speak round trip
```

Requirements:
- Python 3.12 via `uv`, Node 20+;
- Ollama with `nemotron-3.5-lightning:30b-a3b`;
- about 25 GB of free unified memory.

The voice agent (Nemotron Speech) needs NeMo: install `asr_eval/requirements.in` into the backend venv.

## Tests and evaluations

```bash
python -m pytest -q                                              # safety tests + dataset tests (pytest.ini)
python -m pytest -q scripts/brev/serving/test_serving_panel.py   # serving metrics card
./scripts/run_evals.sh                                           # every benchmark -> evaluations/results/summary.json
python evaluations/nemo_gym/run_local.py                         # coordinator on the NeMo Gym tasks, without a Gym install
```

- **Results:** every experiment, per agent, against its baseline: [docs/RESULTS.md](docs/RESULTS.md). Plan and
  method: [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md).
- **NeMo Gym environments:** `evaluations/nemo_gym/unspoken_communication/` (the coordinator) and
  `evaluations/nemo_gym/neuro_comm/` (EEG decisions). Their own tests run inside a NeMo Gym checkout.

## Flags

Environment variables; the defaults reproduce the measured system. Full table: [docs/TECHNICAL_VERBOSE.md](docs/TECHNICAL_VERBOSE.md) §2.

| Flag | Default | What it does |
|---|---|---|
| `UNSPOKEN_AUTO_SPEAK` | `1` | demo auto-speak of **accepted** readings; `0` = nothing is spoken until the person confirms |
| `UNSPOKEN_AUTO_SPEAK_DELAY_MS` | `0` | a countdown before an accepted reading is spoken; No or a head shake cancels it |
| `UNSPOKEN_VALIDATOR_NEGATION`, `UNSPOKEN_VALIDATOR_IGNORE_HISTORY` | `0` | optional coordinator validator rules (measured in docs/EVALUATION.md §4) |
| `UNSPOKEN_CHECKIN` | `0` | vision preview: proactive check-in with recorded MIMIC-IV demo data |
| `UNSPOKEN_SERVING_PANEL` | `0` | a "Serving · vLLM" card on the page (running/waiting, TTFT, KV cache, prefix cache, MTP) |
| `UNSPOKEN_MPS`, `LLM_MTP`, `LLM_MEM`, `LLM_EXTRA` | `1`, `0`, `0.55`, empty | `start_app.sh`: NVIDIA MPS, MTP speculative decoding, vLLM memory share, extra vLLM flags |

## Safety and privacy

- **Invariant:** no unverified interpretation is ever spoken as the person's intention. This is enforced in code
  (`shared/confirmation.py`, the policy gate), not by prompting, and tested in `evaluations/safety_tests/`.
- **The exception is demo auto-speak** (`UNSPOKEN_AUTO_SPEAK=1`, the default).
  - An *accepted* reading is spoken without a confirmation: the voice agent accepted it, or lips and voice agree.
  - It is audited as `demo_auto_spoken`.
  - Ambiguous readings, lip-only matches and corrections still wait for Yes/No.
  - The invariant holds fully with `UNSPOKEN_AUTO_SPEAK=0`.
- **The coordinator only chooses how to ask.** It sees structured summaries, never media; every tool call is
  validated outside the model; and it can never speak.
- **Personal data:**
  - **Not in git:** voice recordings, enrollment features (`data/enroll/`) and anything else under `data/`
    (`.gitignore`).
  - **Published with consent:** the team's own recordings, on Hugging Face under CC BY 4.0
    ([DATA_RELEASE](docs/DATA_RELEASE.md)).
  - Datasets, licences and consent: [docs/DATASETS_AND_CONSENT.md](docs/DATASETS_AND_CONSENT.md).

## Repository and docs

```
web/                    React + Vite UI: glass room, ink + agent-activity engine, always-on sensors, calibration
services/gateway/       state machine, fusion, coordinator calls, policy-gated speech, WebSocket, serving card
services/vsr/           lip agent: Auto-AVSR via Chaplin, personal phrase enrollment
services/audio/         voice agent: Nemotron Speech 0.6B + Silero VAD, asr_eval's decision code and thresholds
services/eeg/           EEG replay: BNCI / bigP3BCI loaders, P300 decoder, Study F stopping policies
services/coordinator/   Nemotron tool calling, outside validation, deterministic fallback
services/tts/           Kokoro-82M (MLX on a Mac, PyTorch on CUDA)
shared/                 schemas, confirmation gate, provenance + ports
evaluations/            safety + dataset tests, NeMo Gym environments, modality baselines, results/
asr_eval/               audio-only ASR benchmark that picked the voice agent
scripts/                setup, run, evals, weights ledger, benchmarks (scripts/benchmark/), data fetchers
scripts/brev/           the backend on Brev: setup_app, sync_app, start_app, connect, bench_latency
scripts/brev/serving/   vLLM serving experiments (concurrency, capacity, isolation + MTP, serving card)
docs/                   see below
```

| Doc | What it covers |
|---|---|
| [docs/TECHNICAL_VERBOSE.md](docs/TECHNICAL_VERBOSE.md) | everything in one place: processes, every agent's contract, flags, serving, latency, open issues (§10) |
| [docs/RESULTS.md](docs/RESULTS.md) | every agent against its baseline, held-out, with caveats |
| [docs/BOUNTIES.md](docs/BOUNTIES.md) | Nemotron Lightning, NeMo Gym and Red Hat vLLM write-ups |
| [docs/BREV.md](docs/BREV.md) | the Brev CLI, instances, the sync/run/fetch loop, problems and fixes |
| [docs/SERVING.md](docs/SERVING.md) | latency with every model running, and what made it faster |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/EVALUATION.md](docs/EVALUATION.md) | design; evaluation method and gaps |
| [docs/MODEL_CARDS_AND_LICENSES.md](docs/MODEL_CARDS_AND_LICENSES.md), [docs/DATASETS_AND_CONSENT.md](docs/DATASETS_AND_CONSENT.md), [docs/DATA_RELEASE.md](docs/DATA_RELEASE.md) | models, licences, data and consent; the data we published |
| [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md), [docs/LIP_READING.md](docs/LIP_READING.md), [docs/EEG_*](docs/), [docs/NEUROCOMM_GYM_EXPERIMENT.md](docs/NEUROCOMM_GYM_EXPERIMENT.md) | the evaluation experiments behind RESULTS.md |
| [HANDOFF.md](HANDOFF.md) | branches, how to run, open bugs and pending decisions |

## Status

`main` holds:
- the deployed system (`demo-final`);
- the evaluation evidence (`eval-consolidation`);
- the vLLM serving experiments (`serving-experiments`).

Known bugs, shortfalls and pending decisions are in [HANDOFF.md](HANDOFF.md) and
[docs/TECHNICAL_VERBOSE.md](docs/TECHNICAL_VERBOSE.md) §10. Every dataset we made is published on Hugging Face
([DATA_RELEASE](docs/DATA_RELEASE.md)).
