# Unspoken

**Because everyone deserves to be heard.**

Picture someone in a hospital bed who is fully awake and understands everything around them, but cannot speak.
They are thirsty, or in pain, or they want their family, and nobody can tell.

Unspoken gives that person a way to talk again. It listens to a weak voice, reads silent lip movements, follows
the eyes, and notices a deliberate blink or a nod. Small models each read one of these signals. NVIDIA Nemotron
3.5 Lightning decides what to do when they are unsure. Plain code, never a model, decides what gets said out loud.

This is a research prototype, not a medical device.

## Quickstart

**On an NVIDIA Brev GPU, with the UI on your laptop.** On Windows, run the `brev` and `scripts/brev/*.sh` commands
inside WSL. Replace `<instance>` with your instance name.

```bash
# once: log in, send the code, set up the instance (30 to 45 minutes, runs in the background)
brev login
bash scripts/brev/sync_app.sh <instance>
bash asr_eval/scripts/brev/sync.sh <instance>
brev exec <instance> 'cd ~/unspoken && mkdir -p .run && nohup bash scripts/brev/setup_app.sh > .run/setup.log 2>&1 < /dev/null &'
brev exec <instance> 'cd ~/asr_eval && nohup bash scripts/setup_brev.sh > setup.log 2>&1 < /dev/null &'

# every time: start the backend, then wait until status shows every service ready
brev exec <instance> 'bash ~/unspoken/scripts/brev/start_app.sh'
brev exec <instance> 'bash ~/unspoken/scripts/brev/start_app.sh status'

# on the laptop: open the tunnel, then the UI at http://localhost:8800
bash scripts/brev/connect.sh <instance>
cd web && npm install && npm run dev

# when you are done
brev exec <instance> 'bash ~/unspoken/scripts/brev/start_app.sh stop'
```

**On an Apple Silicon Mac**, everything runs locally:

```bash
./scripts/setup_models.sh   # environment, checkpoints, data, web dependencies, Ollama model
./scripts/run_all.sh        # all services and the web page at http://localhost:8800
./scripts/smoke_test.sh     # a quick round trip: ask, confirm, speak
```

**Tests:** `python -m pytest -q`

More detail on running it is in [Running on Brev](#running-on-brev) below.

## What it looks like

1. The page shows the live camera. An activity view lights up as each model reports what it saw.
2. Say "I need water", even weakly. The voice model recognises it and Unspoken says it aloud, about 70 ms after
   the clip on the GPU.
3. If no sound comes out, mouth it. The lip reader proposes the phrase about 0.4 s after the person stops moving
   their lips.
4. The person confirms by looking at Yes, blinking on purpose or nodding. No, or a head shake, rejects it.
5. Only then is the lip reading spoken.

People can also record their own phrases at `/?record`, which makes lip reading much more accurate for them.

## How it works

```
lips  ─┐                                        ┌─ you pick ─┐
voice ─┼─► models propose ─► coordinator asks ─►            ├─► you confirm ─► policy gate ─► voice
eyes  ─┤   readings + quality  (Nemotron 3.5     └─ or fix ───┘                 (exact text only)
EEG*  ─┘                        Lightning)                              *recorded replays only
```

| Part | Model | Where |
|---|---|---|
| Voice | NVIDIA Nemotron Speech Streaming 0.6B, scoring the person's phrases and abstaining when unsure | `services/audio/` |
| Lips | Auto-AVSR, matched against the person's own recordings | `services/vsr/` |
| Eyes, blinks, nods | MediaPipe and MobileGaze, in the browser, calibrated per person | `web/` |
| EEG | a P300 decoder on recorded data, replay only | `services/eeg/` |
| Speech | Kokoro, which refuses any text that was not signed by the gate | `services/tts/` |
| Coordinator | **NVIDIA Nemotron 3.5 Lightning 30B-A3B** on **Red Hat vLLM** | `services/coordinator/` |

The coordinator does not write answers. It picks the next step when the models are unsure: ask about one reading
or offer a few, fix a misread word, ask for another try, switch to gaze or spelling, or ignore background noise.
It answers with one typed tool call, code checks that call, and anything invalid falls back to fixed rules. It
cannot make anything be spoken.

All the models together have 31.01B parameters, inside the 33B limit.

## Results

| What we measured | Result | Source |
|---|---|---|
| Lips and voice together, heavy background noise (GRID, babble at 0 dB) | voice alone 8.0%, lips alone 53.5%, **together 76.0%** correct | [RESULTS §6](docs/RESULTS.md) |
| Coordinator on held-out NeMo Gym tasks | reward **0.985**, **0 unsafe actions**; right action rises from 82% (Nemotron alone) to 95% with the validator | [RESULTS §5](docs/RESULTS.md) |
| Lip reading wait after the person stops mouthing | **898 ms down to 407 ms**, by streaming the video while they mouth | [SERVING](docs/SERVING.md) |
| Voice on dysarthric speech (TORGO) | correct accepts 40.0% to **61.1%**, wrong accepts 1.1% to 0% | [RESULTS §1](docs/RESULTS.md) |
| Lip reading with the person's own recordings (MIRACL-VC1, 15 speakers) | top phrase right **72.0% to 96.1%** | [RESULTS §2](docs/RESULTS.md) |
| EEG with ALS patients (held-out session) | wrong characters **24 to 1** per 120 | [RESULTS §3](docs/RESULTS.md) |
| Nemotron on vLLM, sharing one GPU with every other model | **0 errors up to 64 requests at once**, 16 requests per second at peak | [serving summary](evaluations/results/serving/SUMMARY.md) |
| Safety | 49 of 49 safety tests pass, 7 of 7 attacks blocked, no unconfirmed text ever reached speech | [EVALUATION](docs/EVALUATION.md) |

Each number comes with its caveats in RESULTS. In short:
- GRID is a fixed 51-word grammar.
- The held-out coordinator tasks come from the same generator as the eval set.
- EEG is replayed from recordings, not live.

## When the AI is wrong

A chatbot that gets something wrong is annoying. Here, a mistake could mean saying the opposite of what someone
meant, so the system is built around doubt:

- **If it is unsure, it asks.** The coordinator chooses the smallest question that settles it.
- **If lips and voice disagree, the person sees both** and picks.
- **"I don't need water" never turns into "I need water".** The model never dropped a negation in our tests, and an
  optional rule guards against it.
- **Nothing is spoken without a signature.** The gate signs the exact text, and the voice refuses anything else.
- **Everything is logged:** raw readings, tool calls (including rejected ones), confirmations and speech.

One exception, for demos: by default (`UNSPOKEN_AUTO_SPEAK=1`) a reading the models already agree on is spoken
without asking, and it is logged as such. Lip-only, unclear and corrected readings always wait for the person.
Set `UNSPOKEN_AUTO_SPEAK=0` to require a confirmation every time.

## For judges

| Track or bounty | Read | Code |
|---|---|---|
| Multi-agent system | [ARCHITECTURE](docs/ARCHITECTURE.md), [TECHNICAL_VERBOSE](docs/TECHNICAL_VERBOSE.md) | `shared/schemas.py`, `services/gateway/` |
| NVIDIA Nemotron Lightning | [BOUNTIES §1](docs/BOUNTIES.md) | `services/coordinator/app.py` |
| NVIDIA NeMo Gym | [BOUNTIES §2](docs/BOUNTIES.md), [RESULTS §5, §7, §8](docs/RESULTS.md) | `evaluations/nemo_gym/` |
| Red Hat vLLM | [BOUNTIES §3](docs/BOUNTIES.md), [serving summary](evaluations/results/serving/SUMMARY.md) | `scripts/brev/start_app.sh`, `scripts/brev/serving/` |

## Running on Brev

- **The machine.** Built and measured on an RTX PRO 6000 Blackwell with 96 GB (about $2.63 an hour). Everything
  together uses about 63 GB of GPU memory. Many instance types keep billing until you delete them.
- **The Brev CLI.**
  - macOS: `brew install brevdev/homebrew-brev/brev`.
  - Linux or WSL: `bash asr_eval/scripts/brev/install_cli_wsl.sh`.
- **Commit before syncing.** `sync_app.sh` sends what is committed, not your working copy.
- **Browser model files**, once, before the first `npm run dev`:

  ```bash
  cd web && mkdir -p public/mediapipe public/vendor public/models
  cp -r node_modules/@mediapipe/tasks-vision/wasm public/mediapipe/
  cp node_modules/webgazer/dist/webgazer.js public/vendor/
  cp -r node_modules/webgazer/dist/mediapipe/face_mesh public/mediapipe/
  curl -sSL -o public/mediapipe/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
  curl -sSL -o public/models/mobileone_s0_gaze.onnx https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobileone_s0_gaze.onnx
  ```

- **The person's lip recordings** are not in git; they are published on Hugging Face. Send a local copy with:

  ```bash
  tar -czf /tmp/enroll.tgz -C data enroll && brev copy /tmp/enroll.tgz <instance>:/home/shadeform/enroll.tgz
  brev exec <instance> 'cd ~/unspoken && rm -rf data/enroll && tar -xzf ~/enroll.tgz -C data'
  ```

- **Tell the person first.** Their mouth video and voice clips go to the GPU server over SSH. They are processed
  in memory and not stored. The page mentions this only on the calibration screen, so say it yourself and get
  their OK.
- **Flags.** Pass them as environment variables to `start_app.sh`; the full list is in
  [TECHNICAL_VERBOSE §2](docs/TECHNICAL_VERBOSE.md).
- **Logs.** `~/unspoken/.run/logs/`.
- **Benchmarks.** `scripts/brev/bench_latency.py` and `scripts/brev/serving/`.
- **Known problems and fixes.** [BREV.md §5](docs/BREV.md).

## Data

We do not re-host anyone's data. The fetch scripts download each public dataset from its source.

| Dataset | Licence | Used for |
|---|---|---|
| [bigP3BCI](https://physionet.org/content/bigp3bci/1.0.0/), PhysioNet, Study F (10 ALS participants) and Study H | CC BY 4.0 | EEG training on sessions 1 and 2, testing on session 3; EEG with gaze |
| [BNCI2014-008](http://bnci-horizon-2020.eu/database/data-sets/008-2014/), Riccio et al. 2013, 8 participants with ALS | CC BY-NC-ND 4.0 | the EEG replay in the app |
| [GRID](https://zenodo.org/records/3625687), Cooke et al. | CC BY 4.0 | lip reading and lips with voice; s1 and s2 never trained on |
| MIRACL-VC1, Rekik, Ben-Hamadou & Mahdi 2016 | research use only | testing lip reading with personal recordings |
| [TORGO](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html), Rudzicz, Namasivayam & Wolff 2012 | academic research use | testing on dysarthric speech |
| [LibriSpeech](https://www.openslr.org/12/), [MUSAN](https://www.openslr.org/17/), [Speech Commands](https://www.tensorflow.org/datasets/catalog/speech_commands) | CC BY 4.0 | speech and noise the voice model must reject |
| [ESC-50](https://github.com/karolpiczak/ESC-50) | CC BY-NC 3.0 | everyday sounds the voice model must reject (testing only) |
| [MIMIC-IV Demo v2.2](https://physionet.org/content/mimic-iv-demo/2.2/), Johnson et al. | ODbL 1.0 | one recorded ICU event in the vision preview, display only |

**Data we made** is published on Hugging Face under CC BY 4.0, at
[AumPandya01/unspoken-data](https://huggingface.co/datasets/AumPandya01/unspoken-data). It holds:
- the NeMo Gym tasks and training labels;
- the team's own voice recordings and lip recordings, shared with everyone's agreement.

How it was made: [DATA_RELEASE](docs/DATA_RELEASE.md). Full citations and consent:
[DATASETS_AND_CONSENT](docs/DATASETS_AND_CONSENT.md).

## Docs

| Doc | What is in it |
|---|---|
| [TECHNICAL_VERBOSE](docs/TECHNICAL_VERBOSE.md) | everything in one place, including every known gap (§10) |
| [RESULTS](docs/RESULTS.md) | every model against its baseline, with caveats |
| [BOUNTIES](docs/BOUNTIES.md) | the Nemotron Lightning, NeMo Gym and Red Hat vLLM write-ups |
| [BREV](docs/BREV.md), [SERVING](docs/SERVING.md) | running on Brev, and how we made it fast |
| [ARCHITECTURE](docs/ARCHITECTURE.md), [EVALUATION](docs/EVALUATION.md) | the design, and how we tested it |
| [MODEL_CARDS_AND_LICENSES](docs/MODEL_CARDS_AND_LICENSES.md), [DATA_RELEASE](docs/DATA_RELEASE.md) | models, licences and the data we published |
