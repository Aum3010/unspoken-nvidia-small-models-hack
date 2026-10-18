# Unspoken

**Because everyone deserves to be heard.**

Picture someone in a hospital bed who is fully awake and understands everything around them, but cannot speak.
They are thirsty, or in pain, or they want their family, and nobody can tell.

Unspoken gives that person a way to talk again. It listens to a weak voice, reads silent lip movements, follows
the eyes, and notices a deliberate blink or a nod. Small models each read one of these signals, NVIDIA Nemotron
3.5 Lightning decides what to do when they are unsure, and plain code decides what gets said out loud.

This is a research prototype, not a medical device.

## Results

- **76%** of phrases understood in loud background noise when lips and voice work together. Voice alone got 8%,
  lips alone 53.5%.
- **0.985** reward and **0** unsafe actions for the coordinator on held-out NeMo Gym tests.
- **61%** of spoken commands from people with dysarthria recognised, up from 40%, with no wrong ones.
- **96%** lip-reading accuracy once a person records their own phrases, up from 72%.
- **1** wrong letter per 120 for EEG spelling with ALS patients, down from 24.
- **407 ms** to read the lips after the person stops, down from 898 ms.
- **0** errors with 64 requests at once on vLLM, with every model sharing one GPU.
- **31B** parameters in total, under the 33B limit.

How each number was measured, and its caveats, are in [docs/RESULTS.md](docs/RESULTS.md).

## How it works

```
lips  ─┐                                        ┌─ you pick ─┐
voice ─┼─► models propose ─► coordinator asks ─►            ├─► you confirm ─► policy gate ─► voice
eyes  ─┤   readings + quality  (Nemotron 3.5     └─ or fix ───┘                 (exact text only)
EEG*  ─┘                        Lightning)                              *recorded replays only
```

| Part | Model |
|---|---|
| Voice | NVIDIA Nemotron Speech Streaming 0.6B, scoring the person's phrases |
| Lips | Auto-AVSR, matched against the person's own recordings |
| Eyes, blinks, nods | MediaPipe and MobileGaze, in the browser |
| EEG | a P300 decoder on recorded data |
| Speech | Kokoro |
| Coordinator | NVIDIA Nemotron 3.5 Lightning 30B-A3B on Red Hat vLLM |

When the models are unsure, the coordinator picks the next step. It can ask about one reading or offer a few, fix
a misread word, ask for another try, switch to gaze or spelling, or ignore background noise. It answers with one
typed tool call that code checks, and anything invalid falls back to fixed rules. The person confirms with a look,
a blink or a nod, and only then does the policy gate sign the exact text for the voice to speak.

In demo mode (`UNSPOKEN_AUTO_SPEAK=1`, the default) a reading that the models already agree on is spoken without
asking. Set it to `0` to require a confirmation every time.

## Quickstart

On an NVIDIA Brev GPU, with the UI on your laptop (first-time setup: [docs/BREV.md](docs/BREV.md) §4b):

```bash
bash scripts/brev/sync_app.sh <instance>
brev exec <instance> 'bash ~/unspoken/scripts/brev/start_app.sh'
bash scripts/brev/connect.sh <instance>
cd web && npm install && npm run dev        # http://localhost:8800
```

On an Apple Silicon Mac: `./scripts/setup_models.sh`, then `./scripts/run_all.sh`.

Tests: `python -m pytest -q`

## Docs

| Doc | What is in it |
|---|---|
| [TECHNICAL_VERBOSE](docs/TECHNICAL_VERBOSE.md) | the whole system: every part, flag, number and known gap |
| [RESULTS](docs/RESULTS.md) | every result, with method and caveats |
| [BOUNTIES](docs/BOUNTIES.md) | the Nemotron Lightning, NeMo Gym and Red Hat vLLM write-ups |
| [BREV](docs/BREV.md) | setting up and running on Brev |
| [DATA_RELEASE](docs/DATA_RELEASE.md) | the datasets we used, the data we published, and consent |
| [MODEL_CARDS_AND_LICENSES](docs/MODEL_CARDS_AND_LICENSES.md) | every model and its licence |
| [experiments/](docs/experiments/) | the detailed lip, EEG and NeMo Gym experiment write-ups |
