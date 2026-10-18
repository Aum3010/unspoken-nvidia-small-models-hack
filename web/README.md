# Unspoken web UI

One liquid-glass page over the live camera (React 19 + Vite). Every input runs at once: voice, lips, gaze,
blink and head. Readings appear as faint ink and glass bubbles, and nothing is spoken until the person
confirms it. The design follows `unspoken_liquid_glass_3.html` (without its bottom dock).

```bash
npm install
npm run dev        # http://localhost:8800 · proxies /api and /ws to the gateway on 127.0.0.1:8000
npm run build      # type-check + production build
npx oxlint         # lint
```

The gateway is either local (`../scripts/run_all.sh`) or on a Brev GPU through the tunnel (`../scripts/brev/connect.sh`).

Browser model files are not in git; `../scripts/setup_models.sh` puts them in `public/`:
- `public/mediapipe/` (Face Landmarker wasm + `face_landmarker.task`)
- `public/models/mobileone_s0_gaze.onnx` (MobileGaze)

## Code map

| Path | What it does |
|---|---|
| `src/glass/GlassApp.tsx` | The page and its controller: gateway session (WebSocket) → ink, bubbles, captions; rail actions; sensors; agent activity feed; history |
| `src/glass/engine.ts` | Imperative canvas engine ported from the design: handwriting ink (tentative = faint sketch, final = drawn by the pen), the agent-activity nucleus, spawning/melting bubbles, gaze dwell with error-scaled slack, blink selection |
| `src/glass/sensors.ts` | Always-on capture: voice VAD (AudioWorklet, adaptive noise floor, 350 ms pre-roll → 16 kHz WAV), mouth-motion clips (webm), head nod/shake from the face transform |
| `src/glass/Calibration.tsx` | Eye calibration on its own black screen: 18 dots in two passes, accuracy on 5 unseen dots, saved calibration, live camera/face/eye-model status, measured failure reason |
| `src/glass/glass.css`, `glyphs.ts` | Styles (from the design) and the single-stroke Hershey cursive glyphs the ink is drawn with |
| `src/lib/face.ts` | MediaPipe Face Landmarker: one tracking loop per page, deliberate blink, head pose |
| `src/lib/gaze.ts`, `ridge.ts`, `appearance.ts` | Gaze: iris/eyelid geometry + MobileGaze eye-image angles (ONNX Runtime Web, WebGPU → wasm) → per-user ridge regression, head fine-tune, implicit learning from gaze selections |
| `src/lib/capture.ts` | Camera/mic (one shared request each) and clip recording |
| `src/lib/api.ts`, `types.ts` | Gateway REST + WebSocket client; types mirror `shared/schemas.py` |

## How a message flows

1. A voice or mouth clip goes to the gateway, which asks the agent. The reading comes back as candidates.
2. The coordinator decides how to ask. One candidate gets a confirm question; several become bubbles.
3. The person chooses by dwell, blink or click. The choice is written as faint ink.
4. The person confirms with Speak, Yes, a blink or a nod. The policy gate signs the exact text, Kokoro speaks it,
   the pen draws it in, and it goes to History.

Confirmation methods are audited:

| Input | Method recorded |
|---|---|
| Eye dwell | `gaze_dwell_confirmed` |
| Pointer dwell, nod | `other_explicit_user_confirmed` |
| Blink | `deliberate_blink_confirmed` |
| Space | `keyboard_confirmed` |
| Click | `accessible_button_confirmed` |

Keys: Space = blink, P = phrases, C = clear, Esc = close bubbles.

## Debugging gaze

In dev builds, `window.__gaze` exposes the frame counters (`gazeDebug`: frames, face found, drops for missing
iris/pose/eye-model, blinks, samples) with `gazeState` and `appearanceState`. The calibration screen shows the same
reasons when a calibration fails. Only one app or tab can use the webcam at a time; a busy camera shows as
"in use by another app or tab".
