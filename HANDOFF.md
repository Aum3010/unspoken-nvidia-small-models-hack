# Handoff: branch `ui` (2026-10-18)

The full reference is [docs/TECHNICAL_VERBOSE.md](docs/TECHNICAL_VERBOSE.md). This page is the short version:
what is on the branch, how to run it, and what is open.

## TL;DR

- **App:** one liquid-glass page (`web/src/glass/`) at http://localhost:8800.
  - Voice, lips, gaze, blink and head run at once.
  - Phrase cards; big Yes / No targets.
  - Agent activity and history on the right.
  - Eye calibration on its own black screen, saved and reused.
- **Backend:** all six services plus vLLM run on one Brev RTX PRO 6000 (`scripts/brev/`). NVIDIA MPS is on by
  default. The laptop reaches the gateway through an SSH tunnel.
- **Agents:**

  | Role | Model |
  |---|---|
  | Voice | Nemotron Speech en 0.6B, 20 fixed phrases, the evaluated decision code |
  | Lips | Auto-AVSR matched against the person's own enrolled phrases (22 phrases, `data/enroll/`) |
  | Coordinator | Nemotron 3.5 Lightning on vLLM, now with `propose_correction` and `ignore_input` |
  | Speech out | Kokoro |
  | EEG replay | P300 decoder |
  | Gaze | MediaPipe iris + MobileGaze |

- **Speech:** **demo auto-speak is the default** (`UNSPOKEN_AUTO_SPEAK=1`). An accepted reading (the voice agent
  accepted it, or lips and voice agree) is spoken without the person's Yes, and audited as such. Ambiguous readings,
  lip-only matches and corrections wait for Yes/No. Set `UNSPOKEN_AUTO_SPEAK=0` to restore "nothing is spoken
  until confirmed".
- **Latency** (Brev, MPS on):
  - one message takes 0.44 s on the box;
  - the voice result is ready 64 ms after the end-of-utterance pause, seen from the laptop (early decode);
  - a lip message takes about 2.5 s for the person, mostly the 0.9 s stillness wait and the 0.68 s video upload.
- **Coordinator (NeMo Gym, 72 tasks):** deployed 0.931 with **0 unsafe actions**. Every wrong move is
  `ignore_input` on real input.
- **Budget:** 31.01B of 33B (32.59B with the coordinator's weight-file count).

## How to run

```bash
# laptop (Windows: brev runs inside WSL Ubuntu); commit first: sync ships the committed tree
bash scripts/brev/sync_app.sh marginal-sapphire-whale
git archive --format=tar.gz -o /tmp/enroll.tgz HEAD:data enroll && brev copy /tmp/enroll.tgz marginal-sapphire-whale:/home/shadeform/enroll.tgz
brev exec marginal-sapphire-whale 'cd ~/unspoken && rm -rf data/enroll && tar -xzf ~/enroll.tgz -C data'
brev exec marginal-sapphire-whale 'bash ~/unspoken/scripts/brev/start_app.sh'      # start | status | stop; UNSPOKEN_MPS=0 for no MPS
bash scripts/brev/connect.sh marginal-sapphire-whale                               # tunnel :8000
cd web && npm run dev                                                              # http://localhost:8800
```

A fresh instance needs `scripts/brev/setup_app.sh` once. The voice agent also needs the NeMo venv from
`asr_eval/scripts/setup_brev.sh`.

## What changed on `ui` recently

**From the team** (Aum3010, 2026-10-17/18):
- the patient UI: Yes/No targets, zone gaze targeting, layout calibration;
- lip enrollment (from `yash-lipreading-v2`);
- the coordinator's two new tools;
- auto-speak;
- evidence noise screening in the gateway, and utterance replacement;
- one clip in flight per agent;
- OpenBLAS limited to 1 thread.

**Added on top** (local commits, not pushed yet):
- **MPS:** on by default, plus a fix so `UNSPOKEN_MPS=0` really stops it.
- **Early voice decode:** fed through the team's noise screening and one-clip rule.
- **NeMo Gym:** knows the two new tools; misread and negation cases added (72 tasks; data regenerated).
- **Result files:** latency, lip speed vs accuracy, and the early-decode check.
- **Docs:** this page, TECHNICAL_VERBOSE, SERVING, EVALUATION, ARCHITECTURE, BREV, README, web/README, model
  cards, consent.

## Open items (for the team)

1. **Auto-speak by default.** It contradicts the README's product rule.
   - It now covers accepted readings only (`evaluations/safety_tests/test_auto_speak.py`).
   - It is scheduled at once, so a No rarely lands in time.
2. **`ignore_input` overuse.** It drops a person who failed twice, a genuine request next to an injection, and
   clear negated readings. **The validator is intentionally unchanged.**
3. **The correction validator does not check negations.** It would accept "I don't need water" → "I need water".
4. **`data/enroll/` is personal data in git.** These are encoder features of a real person's recordings.
5. **The UI no longer discloses** that clips go to Brev.
6. **The enrollment threshold is unmeasured.** It was set from 10 recordings, and strangers matched about 12% of
   the time.
7. **Speed:**
   - lip video bitrate (needs an accuracy check);
   - coordinator output length;
   - vLLM prefix cache at 0%;
   - GPU isolation for vLLM (do last);
   - native FP4 for the coordinator: not now, see SERVING.md.
8. **Smaller items:**
   - `zero_infinity=True` in `services/vsr/phrases.py`;
   - the camera-open sharing and single tracking loop were reverted;
   - the Mac path needs NeMo for the voice agent;
   - unused `webgazer` dependency;
   - 8 lint warnings.

## Housekeeping

- **Billing.** `marginal-sapphire-whale` bills $2.63/hr until deleted.
  - `start_app.sh stop` frees the GPU but does not stop billing.
  - Back up the recordings first:
    `brev copy marginal-sapphire-whale:/workspace/asr_eval_data/own/ asr_eval/data/own/`.
- **Rotate the Brev API key** that was pasted into an earlier chat session.
- **Voice recordings are personal data.** Keep them out of git, and delete both copies after the hackathon.
