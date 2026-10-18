# Unspoken web UI

One liquid-glass page over the live camera (React 19 + Vite). Every input runs at once: voice, lips, gaze, blink
and head. The design follows `unspoken_liquid_glass_3.html`, without its bottom dock. Full details:
[docs/TECHNICAL_VERBOSE.md](../docs/TECHNICAL_VERBOSE.md) §5.

```bash
npm install
npm run dev        # http://localhost:8800 · proxies /api and /ws to the gateway on 127.0.0.1:8000
npm run build      # type-check + production build
npx oxlint         # lint
```

**Where the gateway runs:** locally (`../scripts/run_all.sh`), or on a Brev GPU through the tunnel
(`../scripts/brev/connect.sh`).

**Browser model files** are not in git; `../scripts/setup_models.sh` puts them in `public/`:
- `public/mediapipe/face_landmarker.task` and `public/mediapipe/wasm/` (Face Landmarker);
- `public/models/mobileone_s0_gaze.onnx` (MobileGaze).

## Code map

| Path | What it does |
|---|---|
| `src/glass/GlassApp.tsx` | The page and its controller: gateway session (WebSocket) → ink, captions, phrase cards; rail and Yes/No actions; sensors; agent activity feed; history; auto-speak playback |
| `src/glass/engine.ts` | Imperative canvas engine: handwriting ink (faint sketch, then drawn by the pen), the agent-activity nucleus, phrase cards, gaze dwell with zone targeting, blink selection |
| `src/glass/sensors.ts` | Always-on capture: voice detector (early send at 350 ms of quiet, utterance end at 700 ms, 16 kHz WAV); mouth-motion clips (webm, 2.5 Mbps, streamed in 250 ms pieces over the session WebSocket while recording); head nod/shake |
| `src/glass/Calibration.tsx` | Eye calibration on its own black screen: 11 dots on the real targets, accuracy on 3 unseen points, saved calibration reused on the same window size |
| `src/glass/glass.css`, `glyphs.ts` | Styles, and the single-stroke Hershey cursive glyphs the ink is drawn with |
| `src/lib/face.ts` | MediaPipe Face Landmarker: deliberate blink, head pose, mouth landmarks |
| `src/lib/gaze.ts`, `ridge.ts`, `appearance.ts` | Gaze: iris/eyelid geometry + MobileGaze eye-image angles (ONNX Runtime Web, WebGPU → wasm) → per-user ridge regression; head fine-tune; learns from each gaze selection |
| `src/lib/capture.ts` | Camera, microphone and clip recording |
| `src/lib/api.ts`, `types.ts` | Gateway REST + WebSocket client (including early voice decode: `captureAudioEarly`, `commitAudio`, `discardAudio`); types mirror `shared/schemas.py` |

## What the person can do

- **Phrases:** open the phrase cards. Choosing one speaks it at once, through the policy gate.
- **Yes:** confirm the pending reading. A nod, or a blink with nothing under the gaze, does the same.
- **No:** reject or undo. A shake does the same.
- **Clear:** start a new message.
- **Help:** a red flash, then speaks "I need help now."
- **Keys:** Space = blink, P = phrases, C = clear, E = calibrate, Esc = close.
- **Demo auto-speak** (`UNSPOKEN_AUTO_SPEAK=1` on the gateway, the current default): accepted readings are spoken
  without a Yes ("Understanding you…"), and No or a shake stops them. Everything else asks "Did you mean this?".
  With `UNSPOKEN_AUTO_SPEAK=0`, nothing is spoken until the person confirms.
- **Phrase audio** is preloaded when the session opens: a fixed phrase plays from the page as soon as the gateway
  reports it spoken.
- **One clip in flight per agent.** Capture pauses while:
  - a message waits for confirmation;
  - speech plays;
  - calibration is on screen.

Confirmation methods are audited:

| Input | Method recorded |
|---|---|
| Eye dwell | `gaze_dwell_confirmed` |
| Pointer dwell, nod | `other_explicit_user_confirmed` |
| Blink | `deliberate_blink_confirmed` |
| Space | `keyboard_confirmed` |
| Click | `accessible_button_confirmed` |
| Auto-speak | `demo_auto_spoken`, by the system |

## Known issues

- **Privacy text:** the camera chip says "on device" even when clips go to a remote gateway.
- **Unused code:** the `webgazer` dependency, the `speak` action and the `pick:` action.
- **Lint:** 8 warnings.
