# Handoff: branch `demo-final` (2026-10-18)

The full reference is [docs/TECHNICAL_VERBOSE.md](docs/TECHNICAL_VERBOSE.md). The bounty write-ups are in
[docs/BOUNTIES.md](docs/BOUNTIES.md). This page is the short version: what is where, how to run it, and
**everything still open** (shortfalls, bugs, pending tasks, decisions).

## Branches

| Branch | Where | What |
|---|---|---|
| `ui` | remote | the team's patient UI, untouched by us |
| `serving-mps` | remote | earlier serving work (MPS, early voice decode, latency runs) |
| `ui-speedups` | remote | `ui` plus the measured speedups: accepted-reading fast path, auto-speak only for accepted readings, batched DTW, phrase-audio preload, streamed lip clips, docs, BOUNTIES (= local `ui-local`) |
| **`demo-final`** | **local only, not pushed** | `ui-speedups` + a working `bench_latency.py` + 4 phases: honesty fixes, doc consistency, optional validator/auto-speak hardening (flags off), vision-preview check-in (flag off), and this docs pass |

- **Local `ui`** equals `origin/ui`.
- **`git stash@{0}`** holds older local `ui` work; it is already saved as `ui-local`.
- **Nothing on `ui` or `ui-local` was rewritten.**

## TL;DR

- **App:** one liquid-glass page (`web/src/glass/`) at http://localhost:8800.
  - Voice, lips, gaze, blink and head run at once.
  - Phrase cards; big Yes / No targets.
  - Agent activity and history on the right.
  - Eye calibration on its own black screen, saved and reused.
  - With a remote gateway, the camera chip says "Camera · clips to NVIDIA Brev".
- **Backend:** all six services plus vLLM on one Brev RTX PRO 6000 (`scripts/brev/`), NVIDIA MPS on, MTP off.
- **Agents:**

  | Role | Model |
  |---|---|
  | Voice | Nemotron Speech en 0.6B, 20 fixed phrases, the evaluated decision code |
  | Lips | Auto-AVSR matched against the person's own enrolled phrases (22 phrases, `data/enroll/`) |
  | Coordinator | Nemotron 3.5 Lightning 30B-A3B NVFP4 on vLLM; runs only for ambiguous / unknown / disagreeing readings |
  | Speech out | Kokoro |
  | EEG replay | P300 decoder |
  | Gaze | MediaPipe iris + MobileGaze |

- **Speech:** **demo auto-speak is the default** (`UNSPOKEN_AUTO_SPEAK=1`).
  - Only an **accepted** reading is spoken without a Yes: the voice agent accepted it, or lips and voice agree.
  - Everything else waits for Yes/No.
- **Latency** (Brev, MPS on), measured:
  - an accepted voice message is spoken 71–73 ms after its clip arrives on the box, 252–257 ms from the laptop;
    about 0.85–0.9 s for the person, of which 0.7 s is the deliberate pause;
  - a lip reading is ready about 0.41 s after the mouth stops, from the laptop (streamed clip; 0.9 s before);
  - one message end to end on the box: 0.30–0.33 s.
- **Coordinator, NeMo Gym, deployed** (Nemotron + validator + fallback), 0 unsafe in every run:
  - eval (72 tasks) **0.940–0.954**; held-out (44 tasks) **0.977–0.985**;
  - with the optional ignore-history rule, 0.995 / 0.985.
- **Tests:** 49/49 unit tests; 7/7 live attacks blocked; NeMo Gym's own environment tests pass 5/5 inside a Gym
  checkout.
- **Budget:** 31.01B of 33B (32.59B with the coordinator's weight-file count).

## How to run

```bash
# laptop (Windows: brev runs inside WSL Ubuntu); commit first: sync ships the committed tree (HEAD)
bash scripts/brev/sync_app.sh marginal-sapphire-whale
# data that sync does not ship: lip enrollment (git) and the vision preview (gitignored, regenerate)
tar -czf /tmp/enroll.tgz -C data enroll && brev copy /tmp/enroll.tgz marginal-sapphire-whale:/home/shadeform/enroll.tgz
brev exec marginal-sapphire-whale 'cd ~/unspoken && rm -rf data/enroll && tar -xzf ~/enroll.tgz -C data'
python scripts/vision/fetch_mimic_event.py        # only for the vision preview; then copy data/vision/ the same way
# start (defaults: MPS on, MTP off, every new flag off); start | status | stop
brev exec marginal-sapphire-whale 'bash ~/unspoken/scripts/brev/start_app.sh'
bash scripts/brev/connect.sh marginal-sapphire-whale   # tunnel :8000
cd web && npm run dev                                  # http://localhost:8800
```

- **First run on a fresh instance:** `scripts/brev/setup_app.sh` once, and the NeMo venv from
  `asr_eval/scripts/setup_brev.sh`.
- **Flags:** every one is in TECHNICAL_VERBOSE §2. Pass them as environment variables to `start_app.sh`, e.g.
  `UNSPOKEN_CHECKIN=1 bash scripts/brev/start_app.sh`.
- **Checks:** `pytest evaluations/safety_tests`, then `scripts/checks/smoke.py`
  ([scripts/checks/README.md](scripts/checks/README.md)), and on Brev `scripts/brev/bench_latency.py --n 20`.

## What `demo-final` added (one commit per phase)

| Phase | What | Flags (default) | Tests |
|---|---|---|---|
| 0 | `bench_latency.py` works again with auto-speak | — | — |
| 1 | Disclosure names the remote host; impossible phrase-list alignments score −1e6 instead of a perfect 0 | — | `test_vsr_phrase_scoring.py` (3) |
| 2 | Every number agrees across docs, with its source; the consent note for `data/enroll/` | — | — |
| 3 | Optional validator rules (negation; ignore-history) and an auto-speak countdown, measured on NeMo Gym | `UNSPOKEN_VALIDATOR_NEGATION` (0), `UNSPOKEN_VALIDATOR_IGNORE_HISTORY` (0), `UNSPOKEN_AUTO_SPEAK_DELAY_MS` (0) | 11 + 4 |
| 4 | Vision preview: a proactive check-in ("Your nurse is coming… do you feel shaky?") on recorded MIMIC-IV demo data and a synthetic profile; the answer is never spoken | `UNSPOKEN_CHECKIN` (0) | `test_checkin.py` (6) |

Latency did not change across the phases (TECHNICAL_VERBOSE §7; `latency_brev_{baseline,phase1,phase3,phase4}_demo_final.json`).

## Shortfalls, bugs and pending tasks

### Bugs (open)

1. **The phrase-audio preload request fails** (`POST /api/{sid}/phrase_audio` → HTTP 500).
   - Cause: it logs with `actor="gateway"`, which `AuditEvent` rejects.
   - The gateway-side cache still fills, but the page never gets the list, so it downloads every phrase's audio.
   - Fix: `"gateway"` → `"system"` in `services/gateway/app.py`. Not applied: the preload was frozen for
     `demo-final`.
2. **The exactness check cannot run as written.**
   - `e2e_exactness.py` re-transcribes Kokoro's audio with the voice agent, which only knows its 20 phrases.
   - Today it gives 3/10, while the text sent to TTS matched the confirmed text 10/10.
   - It needs a free-text ASR to be meaningful again.
3. **MTP crash, unexplained.** With MTP on, vLLM once hit a CUDA fault under full load, and under MPS it took
   every GPU process down. MTP alone passed 240/240 concurrent requests. MTP stays off (`LLM_MTP=0`).
4. **MTP + `--mamba-cache-mode all` answers HTTP 500.**
5. **Dev environment:** Vite's file watcher sometimes misses edits made by scripts and serves stale modules (a
   blank page or old text). Touch the file or restart `npm run dev`.
6. **Minor:** `/api/health` can show the coordinator `unavailable` for a few seconds after a restart (cached
   probe). The comment on `auto_speak` in `services/gateway/state.py` still says "the coordinator's final
   reading".

### Product and safety shortfalls

7. **Auto-speak is on by default.** It contradicts the README's product rule ("nothing is spoken until
   confirmed"). It speaks at once unless `UNSPOKEN_AUTO_SPEAK_DELAY_MS` > 0, so a No rarely lands in time.
8. **`ignore_input` overuse.** By default the validator lets the coordinator drop a person who failed twice, or a
   clear negated reading.
   - `UNSPOKEN_VALIDATOR_IGNORE_HISTORY=1` fixes both (eval 0.954 → 0.995), but it is off pending a decision.
   - **Not fixed by any flag:** a genuine request next to an injected instruction (`prompt_injection` 0.83 on the
     held-out set).
9. **Corrections are not negation-checked by default** (`UNSPOKEN_VALIDATOR_NEGATION`, off; no measured effect).
10. **`data/enroll/` is a real person's lip features in git.** It is pending their written consent decision.
    History has not been rewritten. Removal commands: [docs/DATASETS_AND_CONSENT.md](docs/DATASETS_AND_CONSENT.md).
11. **The enrollment threshold is barely measured.** It was set from 10 recordings, and strangers matched about
    12% of GRID clips.
12. **Lip reading is weak outside enrolled phrases.** Free text has WER about 69% on GRID. Lip-only readings are
    never auto-spoken, by design.
13. **Camera opening:** the shared camera-open promise and single tracking loop were reverted, so slow USB cameras
    can stall.
14. **The Mac / local path** needs NeMo for the voice agent and is not wired up.

### Measurement gaps

15. **The accepted-voice numbers** (413 → 105 → 71 ms) come from an ad-hoc bench, printed rather than saved. The
    later 105 → 71 ms drop is unexplained: the phrase cache is worth only about 2 ms.
16. **"What a person waits for"** (0.85–0.9 s voice, about 1.3 s lip) is added up from parts. It was not measured
    end to end in a browser.
17. **The streamed-lip tests** used synthesized live webm and a Python WebSocket client, not Chrome's
    MediaRecorder: headless Chrome emitted too few pieces.
18. **Not checked visually:** the countdown caption, and the check-in in a real browser with a real camera (it
    was checked in headless Chrome).
19. **NeMo Gym caveats:**
    - the held-out set comes from the same generator (28 of 72 tasks duplicated eval and were dropped; no
      `no_speech` or `repeated_failure` tasks);
    - the rules policy was written to the scorer's spec, so its 1.0 is a ceiling;
    - no model weights were trained.
20. **No real-user studies:** gaze accuracy, clarification efficiency, and live lip reading on new people.

### Pending tasks and decisions

21. **Decide the Phase 3 flags.** My recommendation: ignore-history on, negation on (a free guard), and a
    countdown of about 2000 ms for the demo if you want a visible way to cancel.
22. **Fix bug 1** (one word) and re-check the preload in the page.
23. **Consent decision on `data/enroll/`.**
24. **Push `demo-final`** (not pushed), and choose how `ui`, `ui-speedups` and `demo-final` come together.
25. **Speed items not done:**
    - early lip send and the face-crop upload (both need the person's own videos to verify);
    - GPU isolation for vLLM (2 GPUs or MIG);
    - native FP4 (re-run NeMo Gym first);
    - an instance closer than Kansas City (about 120 ms round trip).
26. **Vision preview:** only a fixed question and one recorded event. Nothing is wired to a real monitor or nurse
    call system, and none should be without clinical review.

## Housekeeping

- **Billing:** `marginal-sapphire-whale` bills $2.63/hr until deleted; `start_app.sh stop` does not stop billing.
  - It is running now with `demo-final` defaults. It also holds a NeMo Gym checkout in
    `/workspace/nemo-gym-check`, and `data/enroll/` + `data/vision/`.
  - A second instance, `unspoken-yash`, is also running on the same account.
- **Back up the recordings before deleting the instance:**
  `brev copy marginal-sapphire-whale:/workspace/asr_eval_data/own/ asr_eval/data/own/`.
- **Rotate the Brev API key** that was pasted into an earlier chat session.
- **Voice recordings are personal data.** Keep them out of git, and delete both copies after the hackathon.
- **The laptop tunnel and the Vite dev server** were started from a chat session. Restart them with the commands
  above.
