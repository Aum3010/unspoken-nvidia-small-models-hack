# Live checks used on `demo-final`

Small scripts that produced the numbers in `docs/TECHNICAL_VERBOSE.md` §7, `docs/RESULTS.md` and the phase
checks of `demo-final`. They talk to a running stack; none of them changes the system.

**Inputs not in git:**
- `<clips>` is a folder holding:
  - `need_water.wav` and `pain.wav`, which the voice agent accepts;
  - `water.wav` and `dont_need_water.wav`, which it marks ambiguous;
  - a `grid/` folder of GRID `.mpg` clips.
- Render the phrases with Kokoro, or record them with consent. Never commit voice recordings.

| Script | Where | What it checks |
|---|---|---|
| `smoke.py <repo> <clips> [out.json]` | laptop, through the tunnel | health; accepted voice auto-spoken; ambiguous waits; No rejects; Yes speaks by the person; streamed lip clip read |
| `laptop_accept.py <clips> [n]` | laptop | accepted voice: clip sent → spoken (p50/p95) |
| `laptop_lip.py <repo> <grid dir>` | laptop | lip reading after the clip ends: whole upload vs streamed over the WebSocket, same reading? |
| `countdown_live.py <clips>` | laptop, gateway with `UNSPOKEN_AUTO_SPEAK_DELAY_MS=2500` | countdown shown; No inside it speaks nothing; no answer speaks once |
| `checkin_live.py off\|on <clips>` | laptop, gateway with `UNSPOKEN_CHECKIN=0/1` | check-in routes absent / nurse alert first, answer logged, nothing spoken, queued while a message is pending |
| `accepted_bench.py` | Brev (`~/unspoken`, venv-app) | accepted vs ambiguous voice messages, clip → spoken/asked, on the box |
| `stream_equiv.py [n]` | Brev | streamed vs whole-clip lip readings identical? (live webm VP9/VP8) and the time after the end |
| `lip_profile.py` | Brev | lip path stage timings: decode, face detection, crop, encoder, DTW |

**On Windows:** set `PYTHONIOENCODING=utf-8` for `checkin_live.py` (it prints 🔔).
