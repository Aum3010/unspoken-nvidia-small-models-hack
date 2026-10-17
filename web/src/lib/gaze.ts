// Webcam gaze from the MediaPipe Face Landmarker we already run (lib/face.ts):
// iris position inside each eye + head pose -> screen point, via a ridge regression
// fitted on the calibration dots. No second face model, no extra camera stream.
// It is an *estimate*, not clinical eye tracking, so callers use dwell, hysteresis
// and a separate confirmation.

import { onFace, trackCameraInBackground, type FaceFrame } from './face'
import { fitRidge, predictRidge, type RidgeModel } from './ridge'

export interface GazePoint { x: number; y: number; t: number }
type Listener = (p: GazePoint | null) => void

const listeners = new Set<Listener>()
const FEAT_FRAMES = 4      // feature averaging window (frames)
const LAMBDA = 5           // ridge penalty on z-scored features
const LOST_MS = 600        // emit null after this long without a face
const BLINK = 0.35         // eyelid closure above which iris landmarks are unreliable
const KEEP = 0.8           // per calibration dot, keep the 80% of samples closest to its median
const MAX_IMPLICIT = 300   // samples learned from confirmed gaze selections (oldest dropped first)

export const gazeState = {
  calibrated: false,
  accuracyPx: null as number | null,
  running: false,
  savedAt: null as number | null,   // when the restored calibration was made
}

let stopTracking: (() => void) | null = null
let offFace: (() => void) | null = null
let recent: number[][] = []
let current: { f: number[]; t: number } | null = null
let lastFace = 0
let blinking = false
let history: { f: number[]; t: number }[] = []   // last few seconds of features, for implicit learning
const samples: { f: number[]; x: number; y: number }[] = []
const implicit: { f: number[]; x: number; y: number }[] = []
let model: RidgeModel | null = null

// One-euro filter: steady when the gaze rests, responsive when it moves (Casiez et al., 2012)
class OneEuro {
  private x: number | null = null
  private dx = 0
  private t = 0
  constructor(private minCutoff = 0.8, private beta = 0.004, private dCutoff = 1) {}
  private static a(cutoff: number, dt: number) { const r = 2 * Math.PI * cutoff * dt; return r / (r + 1) }
  filter(v: number, t: number) {
    if (this.x === null) { this.x = v; this.t = t; return v }
    const dt = Math.max(1e-3, (t - this.t) / 1000)
    this.t = t
    this.dx += OneEuro.a(this.dCutoff, dt) * ((v - this.x) / dt - this.dx)
    this.x += OneEuro.a(this.minCutoff + this.beta * Math.abs(this.dx), dt) * (v - this.x)
    return this.x
  }
  reset() { this.x = null; this.dx = 0 }
}
const fx = new OneEuro(), fy = new OneEuro()

// The fitted calibration is kept on this device (a few dozen numbers, no images) so it can be
// reused after a reload. It is only valid for the same window size and camera position.
const STORE = 'unspoken.gaze.v4'
interface Saved { model: RidgeModel; accuracyPx: number; w: number; h: number; savedAt: number }

export function saveCalibration() {
  if (!model || gazeState.accuracyPx == null || !gazeState.calibrated) return
  const s: Saved = { model, accuracyPx: gazeState.accuracyPx, w: window.innerWidth, h: window.innerHeight, savedAt: Date.now() }
  try { localStorage.setItem(STORE, JSON.stringify(s)) } catch { /* storage unavailable */ }
  gazeState.savedAt = s.savedAt
}

function restoreCalibration() {
  try {
    const s: Saved | null = JSON.parse(localStorage.getItem(STORE) ?? 'null')
    if (!s || Math.abs(s.w - window.innerWidth) > 40 || Math.abs(s.h - window.innerHeight) > 40) return
    model = s.model
    gazeState.accuracyPx = s.accuracyPx
    gazeState.calibrated = s.accuracyPx < 220
    gazeState.savedAt = s.savedAt
  } catch { /* corrupt or unavailable: calibrate again */ }
}
restoreCalibration()

// MediaPipe face mesh indices (478-point model with irises)
const R = { outer: 33, inner: 133, top: 159, bottom: 145, iris: 468 }
const L = { outer: 263, inner: 362, top: 386, bottom: 374, iris: 473 }

type P = { x: number; y: number }
function eye(lm: P[], e: typeof R): [number, number, number] {
  const o = lm[e.outer], i = lm[e.inner], c = lm[e.iris]
  const ux = i.x - o.x, uy = i.y - o.y
  const w = Math.hypot(ux, uy) || 1e-6
  const nx = ux / w, ny = uy / w
  const rx = c.x - (o.x + i.x) / 2, ry = c.y - (o.y + i.y) / 2
  const h = (rx * nx + ry * ny) / w            // along the eye: left/right
  const v = (-rx * ny + ry * nx) / w           // across the eye: up/down
  const open = Math.hypot(lm[e.top].x - lm[e.bottom].x, lm[e.top].y - lm[e.bottom].y) / w  // lids follow vertical gaze
  return [h, v, open]
}

function features(f: FaceFrame): number[] | null {
  const lm = f.landmarks
  if (!lm || lm.length < 478 || !f.pose) return null
  const [hR, vR, oR] = eye(lm, R)
  const [hL, vL, oL] = eye(lm, L)
  const p = f.pose
  const hm = (hR - hL) / 2, vm = (vR + vL) / 2  // the eye axes point opposite ways horizontally
  // head rotation (forward axis) + translation, taken for either matrix layout (row- or column-major;
  // the unused entries are constant and drop out), plus quadratic eye terms for screen edges
  const pose = [p[2], p[6], p[8], p[9], p[3], p[7], p[11], p[12], p[13], p[14]]
  return [hR, vR, oR, hL, vL, oL, ...pose, hm * hm, vm * vm, hm * vm]
}

function onFrame(f: FaceFrame) {
  const feat = f.present ? features(f) : null
  if (!feat) {
    recent = []
    if (f.t - lastFace > LOST_MS) { current = null; fx.reset(); fy.reset(); emit(null) }
    return
  }
  lastFace = f.t
  // during a blink the iris is hidden: hold the last estimate instead of jumping
  blinking = f.blinkL > BLINK || f.blinkR > BLINK
  if (blinking) return
  recent = [...recent.slice(-(FEAT_FRAMES - 1)), feat]
  const avg = feat.map((_, j) => recent.reduce((a, r) => a + r[j], 0) / recent.length)
  current = { f: avg, t: f.t }
  history = [...history.filter((h) => f.t - h.t < 4000), current]
  if (!gazeState.running || !model) return
  const [px, py] = predictRidge(model, avg)
  const x = Math.min(window.innerWidth, Math.max(0, px)), y = Math.min(window.innerHeight, Math.max(0, py))
  emit({ x: fx.filter(x, f.t), y: fy.filter(y, f.t), t: f.t })
}

/** Drop each calibration dot's outlying samples (glances away, landmark glitches), then fit. */
function refit() {
  const groups = new Map<string, typeof samples>()
  for (const s of samples) {
    const k = `${Math.round(s.x)},${Math.round(s.y)}`
    groups.set(k, [...(groups.get(k) ?? []), s])
  }
  const d = samples[0]?.f.length ?? 0
  const sd = Array.from({ length: d }, (_, j) => {
    const m = samples.reduce((a, s) => a + s.f[j], 0) / samples.length
    return Math.sqrt(samples.reduce((a, s) => a + (s.f[j] - m) ** 2, 0) / samples.length) || 1
  })
  const kept: typeof samples = []
  for (const g of groups.values()) {
    const med = Array.from({ length: d }, (_, j) => g.map((s) => s.f[j]).sort((a, b) => a - b)[Math.floor(g.length / 2)])
    const dist = (s: (typeof samples)[number]) => s.f.reduce((a, v, j) => a + ((v - med[j]) / sd[j]) ** 2, 0)
    kept.push(...[...g].sort((a, b) => dist(a) - dist(b)).slice(0, Math.max(1, Math.ceil(g.length * KEEP))))
  }
  const all = [...kept, ...implicit]
  model = all.length >= 20 ? fitRidge(all.map((s) => s.f), all.map((s) => [s.x, s.y]), LAMBDA) : null
}

export async function startGaze(): Promise<void> {
  if (!stopTracking) {
    stopTracking = await trackCameraInBackground()  // keeps face tracking alive on every page
    offFace = onFace(onFrame)
  }
  gazeState.running = true
}

export function pauseGaze() {
  gazeState.running = false
  stopTracking?.(); offFace?.()
  stopTracking = offFace = null
  fx.reset(); fy.reset()
  emit(null)
}

export function forgetCalibration() {
  try { localStorage.removeItem(STORE) } catch { /* ignore */ }
  clearCalibration()
}

export function clearCalibration() {
  samples.length = 0
  implicit.length = 0
  model = null
  fx.reset(); fy.reset()
  gazeState.calibrated = false
  gazeState.accuracyPx = null
  gazeState.savedAt = null
}

/** Feed one explicit calibration sample: the person is looking at (x, y) right now. */
export function recordCalibrationPoint(x: number, y: number) {
  if (blinking || !current || performance.now() - current.t > 200) return  // blink or no fresh face frame
  samples.push({ f: current.f, x, y })
  refit()
}

/**
 * Learn from a completed gaze selection: for the dwell that just ended, the person was looking
 * at the target, so its centre labels those frames. Skips the first third (eyes still arriving).
 */
export function learnFromSelection(x: number, y: number, dwellMs: number) {
  if (!model) return
  const now = performance.now()
  const frames = history.filter((h) => now - h.t < dwellMs * 0.67)
  if (frames.length < 5) return
  implicit.push(...frames.map((h) => ({ f: h.f, x, y })))
  implicit.splice(0, Math.max(0, implicit.length - MAX_IMPLICIT))
  refit()
}

export function calibrationSampleCount() { return samples.length }

function emit(p: GazePoint | null) {
  listeners.forEach((l) => l(p))
}

export function onGaze(l: Listener) {
  listeners.add(l)
  return () => { listeners.delete(l) }
}

/** Collect gaze for `ms` while the person fixates a known point; returns mean error in px. */
export function measureAccuracy(target: { x: number; y: number }, ms = 2500): Promise<number | null> {
  return new Promise((resolve) => {
    const errs: number[] = []
    const off = onGaze((p) => { if (p) errs.push(Math.hypot(p.x - target.x, p.y - target.y)) })
    setTimeout(() => {
      off()
      if (errs.length < 5) return resolve(null)
      errs.sort((a, b) => a - b)
      const trimmed = errs.slice(0, Math.ceil(errs.length * 0.8))
      resolve(trimmed.reduce((a, b) => a + b, 0) / trimmed.length)
    }, ms)
  })
}
