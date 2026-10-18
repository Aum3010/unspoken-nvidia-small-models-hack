// Appearance-based gaze direction from the face image: MobileGaze MobileOne-S0
// (MIT, github.com/yakhyo/gaze-estimation, L2CS-style, trained on Gaze360; ~1M parameters, 4.8 MB ONNX).
// Runs in the browser with ONNX Runtime Web (WebGPU, falling back to WebAssembly). The face box comes
// from the MediaPipe landmarks the app already computes, so there is no extra face detector.
// Its yaw/pitch are fed to the personal gaze calibration (lib/gaze.ts) as extra features alongside the
// iris geometry: the eye image carries cues the landmarks miss, and it is steadier when the head moves.

import * as ort from 'onnxruntime-web/webgpu'
// the runtime's own .mjs/.wasm, served by Vite as plain URLs (it loads them at run time)
import ortMjs from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url'
import ortWasm from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url'

const MODEL_URL = '/models/mobileone_s0_gaze.onnx'
const SIZE = 448
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225]
const BINS = 90, BIN_WIDTH = 4, OFFSET = 180   // Gaze360 binning used by the released weights
const MIN_INTERVAL_MS = 80                     // at most ~12 runs per second

export interface Appearance { yaw: number; pitch: number; t: number }   // radians

let session: ort.InferenceSession | null = null
let loading: Promise<boolean> | null = null
let busy = false
let lastRun = 0
let latest: Appearance | null = null
let canvas: HTMLCanvasElement | null = null
const input = new Float32Array(3 * SIZE * SIZE)

export const appearanceState = { ready: false, provider: '' as string, error: '' as string, ms: 0 }

/** Loads the model once; resolves false (and the gaze falls back to geometry only) if it cannot run. */
export function loadAppearance(): Promise<boolean> {
  loading ??= (async () => {
    ort.env.wasm.wasmPaths = { mjs: ortMjs, wasm: ortWasm }
    for (const ep of ['webgpu', 'wasm'] as const) {
      try {
        session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: [ep] })
        appearanceState.ready = true
        appearanceState.provider = ep
        return true
      } catch (e: any) {
        appearanceState.error = `${ep}: ${e?.message ?? e}`
      }
    }
    console.warn('MobileGaze unavailable, gaze uses iris geometry only:', appearanceState.error)
    return false
  })()
  return loading
}

/** Most recent estimate, if it is fresh enough to describe the current frame. */
export function currentAppearance(maxAgeMs = 250): Appearance | null {
  return latest && performance.now() - latest.t <= maxAgeMs ? latest : null
}

function decode(logits: Float32Array): number {
  let max = -Infinity
  for (let i = 0; i < BINS; i++) max = Math.max(max, logits[i])
  let sum = 0, acc = 0
  for (let i = 0; i < BINS; i++) { const e = Math.exp(logits[i] - max); sum += e; acc += e * i }
  return ((acc / sum) * BIN_WIDTH - OFFSET) * Math.PI / 180
}

/** Square face crop around the landmarks, a little larger than the face (like a detector box). */
function faceBox(lm: { x: number; y: number }[], w: number, h: number) {
  let x0 = 1, y0 = 1, x1 = 0, y1 = 0
  for (const p of lm) { x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y); x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y) }
  const cx = (x0 + x1) / 2 * w, cy = (y0 + y1) / 2 * h
  const side = Math.max((x1 - x0) * w, (y1 - y0) * h) * 1.15
  return { sx: cx - side / 2, sy: cy - side / 2, side }
}

/** Runs the model on the current video frame if it is idle; the result arrives via currentAppearance(). */
export async function updateAppearance(video: HTMLVideoElement, landmarks: { x: number; y: number }[]): Promise<void> {
  const now = performance.now()
  if (!session || busy || now - lastRun < MIN_INTERVAL_MS || !video.videoWidth) return
  busy = true
  lastRun = now
  try {
    const { sx, sy, side } = faceBox(landmarks, video.videoWidth, video.videoHeight)
    canvas ??= Object.assign(document.createElement('canvas'), { width: SIZE, height: SIZE })
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!
    ctx.drawImage(video, sx, sy, side, side, 0, 0, SIZE, SIZE)
    const px = ctx.getImageData(0, 0, SIZE, SIZE).data
    const plane = SIZE * SIZE
    for (let i = 0; i < plane; i++) {
      input[i] = (px[i * 4] / 255 - MEAN[0]) / STD[0]
      input[plane + i] = (px[i * 4 + 1] / 255 - MEAN[1]) / STD[1]
      input[2 * plane + i] = (px[i * 4 + 2] / 255 - MEAN[2]) / STD[2]
    }
    const out = await session.run({ input: new ort.Tensor('float32', input, [1, 3, SIZE, SIZE]) })
    latest = { yaw: decode(out.yaw.data as Float32Array), pitch: decode(out.pitch.data as Float32Array), t: now }
    appearanceState.ms = Math.round(performance.now() - now)
  } catch (e) {
    console.warn('MobileGaze step failed', e)
  } finally {
    busy = false
  }
}
