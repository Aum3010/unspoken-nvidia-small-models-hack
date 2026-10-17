// WebGazer.js wrapper: lazy load, calibration bookkeeping, smoothed gaze stream.
// WebGazer returns *estimated* screen coordinates from a webcam — not clinical
// eye tracking — so callers must use dwell, hysteresis and separate confirmation.

declare global {
  interface Window { webgazer?: any }
}

export interface GazePoint { x: number; y: number; t: number }
type Listener = (p: GazePoint | null) => void

const listeners = new Set<Listener>()
let started = false
let loading: Promise<void> | null = null
let smooth: GazePoint | null = null
let lastRaw = 0
const ALPHA = 0.28

export const gazeState = {
  calibrated: false,
  accuracyPx: null as number | null,
  running: false,
}

function loadScript(): Promise<void> {
  if (window.webgazer) return Promise.resolve()
  if (loading) return loading
  loading = new Promise((resolve, reject) => {
    const s = document.createElement('script')
    s.src = '/vendor/webgazer.js'
    s.async = true
    s.onload = () => resolve()
    s.onerror = () => reject(new Error('WebGazer failed to load'))
    document.head.appendChild(s)
  })
  return loading
}

export async function startGaze(): Promise<void> {
  await loadScript()
  const wg = window.webgazer
  if (started) {
    wg.resume()
    gazeState.running = true
    return
  }
  wg.params.showVideoPreview = false
  wg.params.showFaceOverlay = false
  wg.params.showFaceFeedbackBox = false
  wg.saveDataAcrossSessions(false)
  wg.setRegression('ridge')
  wg.applyKalmanFilter(true)
  wg.setGazeListener((data: { x: number; y: number } | null) => {
    if (!data) {
      if (performance.now() - lastRaw > 600) emit(null)
      return
    }
    lastRaw = performance.now()
    smooth = smooth
      ? { x: smooth.x + ALPHA * (data.x - smooth.x), y: smooth.y + ALPHA * (data.y - smooth.y), t: lastRaw }
      : { x: data.x, y: data.y, t: lastRaw }
    emit(smooth)
  })
  await wg.begin()
  wg.showVideoPreview(false).showPredictionPoints(false)
  started = true
  gazeState.running = true
}

export function pauseGaze() {
  if (started) window.webgazer?.pause()
  gazeState.running = false
  emit(null)
}

export function clearCalibration() {
  window.webgazer?.clearData?.()
  gazeState.calibrated = false
  gazeState.accuracyPx = null
}

/** Feed one explicit calibration sample: the person looked at (x, y). */
export function recordCalibrationPoint(x: number, y: number) {
  window.webgazer?.recordScreenPosition?.(x, y, 'click')
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
