// Webcam gaze from the MediaPipe Face Landmarker we already run (lib/face.ts):
// iris position and eyelids inside each eye, plus the gaze direction MobileGaze reads from the
// face image (lib/appearance.ts) -> screen point, via a ridge regression fitted on the calibration
// dots. No second face model, no extra camera stream.
//
// Positioning: the gaze puts the cursor near the target (fast, ~2 cm accurate); once the gaze settles,
// small head movements fine-tune the cursor from there (slow, precise), as in gaze + head pointing
// (Kyto et al., CHI 2018). A real eye jump re-anchors the cursor at the new gaze point.
// It is an *estimate*, not clinical eye tracking, so callers use dwell, hysteresis
// and a separate confirmation.

import { appearanceState, currentAppearance, loadAppearance, updateAppearance } from './appearance'
import { onFace, trackCameraInBackground, type FaceFrame } from './face'
import { fitRidge, predictRidge, type RidgeModel } from './ridge'

export interface GazePoint { x: number; y: number; t: number }
type Listener = (p: GazePoint | null) => void

const listeners = new Set<Listener>()
const FEAT_FRAMES = 4      // feature averaging window (frames)
const LAMBDA = 3           // ridge penalty on z-scored features (best on held-out dots in recorded sessions)
const LOST_MS = 600        // emit null after this long without a face
const BLINK = 0.35         // eyelid closure above which iris landmarks are unreliable
const KEEP = 0.8           // per calibration dot, keep the 80% of samples closest to its median
const MAX_IMPLICIT = 300   // samples learned from confirmed gaze selections (oldest dropped first)

export const gazeState = {
  calibrated: false,
  accuracyPx: null as number | null,
  running: false,
  savedAt: null as number | null,   // when the restored calibration was made
  appearance: false,                // calibration includes the MobileGaze features
  calibrating: false,               // pure gaze while calibrating/measuring (no head fine-tune)
}

// Head fine-tune settings (saved per device)
export const headAssist = (() => {
  const d = { enabled: true, sensitivity: 3 }
  try { return { ...d, ...JSON.parse(localStorage.getItem('unspoken.gaze.head') ?? '{}') } } catch { return d }
})() as { enabled: boolean; sensitivity: number }
export function setHeadAssist(p: Partial<typeof headAssist>) {
  Object.assign(headAssist, p)
  try { localStorage.setItem('unspoken.gaze.head', JSON.stringify(headAssist)) } catch { /* ignore */ }
  anchor = null
}
export function setCalibrating(on: boolean) { gazeState.calibrating = on; anchor = null }

// Whether new calibrations use the MobileGaze eye-image features (saved per device)
export const appearanceSetting = (() => {
  try { return { enabled: true, ...JSON.parse(localStorage.getItem('unspoken.gaze.appearance') ?? '{}') } } catch { return { enabled: true } }
})() as { enabled: boolean }
export function setAppearanceSetting(enabled: boolean) {
  appearanceSetting.enabled = enabled
  try { localStorage.setItem('unspoken.gaze.appearance', JSON.stringify(appearanceSetting)) } catch { /* ignore */ }
}

// Diagnostics: per-frame features labelled with the dot being looked at, for offline tuning.
// Numbers only (landmark-derived features, model angles, dot positions) -- no images.
type DiagFrame = { t: number; phase: 'calib' | 'test'; tx: number; ty: number; geo: number[]; app: [number, number] | null; pred: [number, number] | null }
let diag: DiagFrame[] = []
let label: { phase: 'calib' | 'test'; x: number; y: number; until: number } | null = null

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
  private minCutoff: number
  private beta: number
  private dCutoff: number
  constructor(minCutoff = 0.8, beta = 0.004, dCutoff = 1) { this.minCutoff = minCutoff; this.beta = beta; this.dCutoff = dCutoff }
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
const STORE = 'unspoken.gaze.v6'
interface Saved { model: RidgeModel; appearance: boolean; accuracyPx: number; w: number; h: number; savedAt: number }

export function saveCalibration() {
  if (!model || gazeState.accuracyPx == null || !gazeState.calibrated) return
  const s: Saved = { model, appearance: gazeState.appearance, accuracyPx: gazeState.accuracyPx, w: window.innerWidth, h: window.innerHeight, savedAt: Date.now() }
  try { localStorage.setItem(STORE, JSON.stringify(s)) } catch { /* storage unavailable */ }
  gazeState.savedAt = s.savedAt
}

function restoreCalibration() {
  try {
    const s: Saved | null = JSON.parse(localStorage.getItem(STORE) ?? 'null')
    if (!s || Math.abs(s.w - window.innerWidth) > 40 || Math.abs(s.h - window.innerHeight) > 40) return
    model = s.model
    gazeState.appearance = s.appearance
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

// Why frames were used or dropped since the last reset: the calibration screen shows it, so a failing
// calibration says what is missing (face, head pose, the eye-image model) instead of "eyes not seen".
export const gazeDebug = { frames: 0, ok: 0, noFace: 0, noIris: 0, noPose: 0, noAppearance: 0, blink: 0, samples: 0, stale: 0, lastFeatureAt: 0 }
export function resetGazeDebug() { for (const k of Object.keys(gazeDebug) as (keyof typeof gazeDebug)[]) gazeDebug[k] = 0 }

function features(f: FaceFrame): number[] | null {
  const lm = f.landmarks
  if (!lm || lm.length < 478) { gazeDebug.noIris++; return null }
  if (!f.pose) { gazeDebug.noPose++; return null }
  // appearance features: required when the calibration was made with them, so samples and
  // predictions always have the same layout
  const app = gazeState.appearance ? currentAppearance() : null
  const [hR, vR, oR] = eye(lm, R)
  const [hL, vL, oL] = eye(lm, L)
  const p = f.pose
  const hm = (hR - hL) / 2, vm = (vR + vL) / 2  // the eye axes point opposite ways horizontally
  // head rotation (forward axis) + translation, taken for either matrix layout (row- or column-major;
  // the unused entries are constant and drop out), plus quadratic eye terms for screen edges
  const pose = [p[2], p[6], p[8], p[9], p[3], p[7], p[11], p[12], p[13], p[14]]
  const geo = [hR, vR, oR, hL, vL, oL, ...pose, hm * hm, vm * vm, hm * vm]   // full set, logged for diagnostics
  if (label && performance.now() < label.until) {
    const a = currentAppearance()
    diag.push({ t: Math.round(f.t), phase: label.phase, tx: Math.round(label.x), ty: Math.round(label.y), geo: geo.map((v) => +v.toFixed(6)),
      app: a ? [+a.yaw.toFixed(5), +a.pitch.toFixed(5)] : null, pred: null })
  }
  if (gazeState.appearance && !app) { gazeDebug.noAppearance++; return null }
  // The model leaves out the head-pose terms: on recorded sessions they fit one sitting well (~70 px)
  // but a calibration reused minutes later, after a small posture change, jumped to 180-220 px.
  // Eye-in-head features (+ the eye-image angles) stayed at 63-86 px. Head movement is handled by
  // the head fine-tune instead (position()).
  const eyes = [hR, vR, oR, hL, vL, oL, hm * hm, vm * vm, hm * vm]
  return app ? [...eyes, app.yaw, app.pitch] : eyes
}

// Head direction from 2D landmarks, in image units with known signs (no matrix-layout ambiguity):
// yaw grows when the nose moves toward image-right, pitch grows when it moves down.
function headDir(lm: P[]): [number, number] {
  const nose = lm[1], l = lm[234], r = lm[454], top = lm[10], chin = lm[152]
  const w = Math.hypot(r.x - l.x, r.y - l.y) || 1e-6, h = Math.hypot(chin.x - top.x, chin.y - top.y) || 1e-6
  return [(nose.x - (l.x + r.x) / 2) / w, (nose.y - (top.y + chin.y) / 2) / h]
}

// cursor = gaze anchor + a bounded head offset since the anchor was set
const JUMP_MS = 150          // a gaze shift must persist this long to count as moving to a new target
const RECENTRE_S = 4         // the head reference slowly follows posture drift (time constant, seconds)
let anchor: { gx: number; gy: number; hx: number; hy: number; away: number | null; t: number } | null = null
const hfx = new OneEuro(1.0, 0.02), hfy = new OneEuro(1.0, 0.02)

function position(gx: number, gy: number, lm: P[], t: number): [number, number] {
  if (!headAssist.enabled || gazeState.calibrating || !gazeState.calibrated) { anchor = null; return [gx, gy] }
  const [rx, ry] = headDir(lm)
  const hx = hfx.filter(rx, t), hy = hfy.filter(ry, t)
  const err = gazeState.accuracyPx ?? 80
  if (!anchor) anchor = { gx, gy, hx, hy, away: null, t }
  // the eyes moved further than the normal gaze error: follow them (new target)
  if (Math.hypot(gx - anchor.gx, gy - anchor.gy) > Math.max(70, err)) {
    anchor.away ??= t
    if (t - anchor.away > JUMP_MS) anchor = { gx, gy, hx, hy, away: null, t }
  } else anchor.away = null
  // slow posture drift must not accumulate into an offset: let the reference creep toward the head
  const dt = Math.max(0, (t - anchor.t) / 1000)
  anchor.t = t
  const a = 1 - Math.exp(-dt / RECENTRE_S)
  anchor.hx += a * (hx - anchor.hx); anchor.hy += a * (hy - anchor.hy)
  // turning to the user's right moves the nose toward image-left (camera is not mirrored) -> cursor right.
  // The head only corrects the gaze error, so its reach is bounded to about that error.
  const k = headAssist.sensitivity * 0.25 * window.innerWidth
  const reach = Math.max(60, err * 1.2)
  const ox = Math.max(-reach, Math.min(reach, -k * (hx - anchor.hx)))
  const oy = Math.max(-reach, Math.min(reach, k * 0.8 * (hy - anchor.hy)))
  return [anchor.gx + ox, anchor.gy + oy]
}

function onFrame(f: FaceFrame) {
  if (f.present && f.landmarks && appearanceState.ready) void updateAppearance(f.source, f.landmarks)
  gazeDebug.frames++
  if (!f.present) gazeDebug.noFace++
  const feat = f.present ? features(f) : null
  if (!feat) {
    recent = []
    if (f.t - lastFace > LOST_MS) { current = null; fx.reset(); fy.reset(); emit(null) }
    return
  }
  lastFace = f.t
  // during a blink the iris is hidden: hold the last estimate instead of jumping
  blinking = f.blinkL > BLINK || f.blinkR > BLINK
  if (blinking) { gazeDebug.blink++; return }
  gazeDebug.ok++
  gazeDebug.lastFeatureAt = f.t
  recent = [...recent.slice(-(FEAT_FRAMES - 1)), feat]
  const avg = feat.map((_, j) => recent.reduce((a, r) => a + r[j], 0) / recent.length)
  current = { f: avg, t: f.t }
  history = [...history.filter((h) => f.t - h.t < 4000), current]
  if (!gazeState.running || !model) return
  const [px, py] = predictRidge(model, avg)
  const last = diag[diag.length - 1]
  if (last && last.t === Math.round(f.t)) last.pred = [Math.round(px), Math.round(py)]
  const [cx, cy] = position(fx.filter(px, f.t), fy.filter(py, f.t), f.landmarks!, f.t)
  emit({ x: Math.min(window.innerWidth, Math.max(0, cx)), y: Math.min(window.innerHeight, Math.max(0, cy)), t: f.t })
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
  await loadAppearance()   // MobileGaze; if it cannot load, gaze uses iris geometry only
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
  anchor = null
  gazeState.appearance = appearanceState.ready && appearanceSetting.enabled   // eye-image features when available and wanted
  diag = []
  fx.reset(); fy.reset()
  gazeState.calibrated = false
  gazeState.accuracyPx = null
  gazeState.savedAt = null
}

/** Feed one explicit calibration sample: the person is looking at (x, y) right now. */
export function recordCalibrationPoint(x: number, y: number) {
  label = { phase: 'calib', x, y, until: performance.now() + 120 }
  if (blinking || !current || performance.now() - current.t > 200) { gazeDebug.stale++; return }  // blink or no fresh face frame
  samples.push({ f: current.f, x, y })
  gazeDebug.samples++
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

/** Downloads the last calibration's labelled features as JSON (for offline tuning). */
export function downloadDiagnostics() {
  const data = {
    version: 1, createdAt: new Date().toISOString(), screen: { w: window.innerWidth, h: window.innerHeight },
    accuracyPx: gazeState.accuracyPx, usedAppearance: gazeState.appearance, appearance: { ...appearanceState },
    geoFeatures: 'hR,vR,openR,hL,vL,openL,pose[2,6,8,9,3,7,11,12,13,14],hm2,vm2,hm*vm', frames: diag,
  }
  const a = document.createElement('a')
  a.href = URL.createObjectURL(new Blob([JSON.stringify(data)], { type: 'application/json' }))
  a.download = `gaze-calibration-${Date.now()}.json`
  a.click()
  URL.revokeObjectURL(a.href)
}

function emit(p: GazePoint | null) {
  listeners.forEach((l) => l(p))
}

export function onGaze(l: Listener) {
  listeners.add(l)
  return () => { listeners.delete(l) }
}

/** Collect gaze for `ms` while the person fixates a known point; returns mean error in px. */
export function measureAccuracy(target: { x: number; y: number }, ms = 2500): Promise<number | null> {
  label = { phase: 'test', x: target.x, y: target.y, until: performance.now() + ms }
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

// dev builds: inspect a live calibration from the console or an automated browser
if (import.meta.env.DEV) Object.assign(window, { __gaze: { gazeDebug, gazeState, appearanceState, calibrationSampleCount } })
