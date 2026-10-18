// MediaPipe Face Landmarker (runs locally in the browser, wasm + GPU delegate).
// Used for: mouth-motion clips for the lip agent, deliberate-blink detection, head nod/shake,
// and the iris + head-pose features used for gaze (lib/gaze.ts). Landmarks do not decode words.

import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'
import { getCamera } from './capture'

export interface FaceFrame {
  present: boolean
  landmarks: { x: number; y: number; z: number }[] | null
  mouthBox: { x: number; y: number; w: number; h: number } | null
  mouthOpen: number      // 0..1 jawOpen blendshape
  blinkL: number
  blinkR: number
  yaw: number            // rough left/right head turn, -1..1
  headPitch: number      // head pose from the face transform, degrees (nod axis)
  headYaw: number        // degrees (shake axis)
  pose: number[] | null  // 4x4 facial transformation matrix (flattened), head rotation + translation
  source: HTMLVideoElement  // the frame these landmarks belong to (used for the face crop in lib/appearance.ts)
  t: number
}

type FrameListener = (f: FaceFrame) => void
type BlinkListener = (durationMs: number) => void

let landmarker: FaceLandmarker | null = null
let initPromise: Promise<FaceLandmarker> | null = null
const frameListeners = new Set<FrameListener>()
const blinkListeners = new Set<BlinkListener>()

export const blinkConfig = { threshold: 0.5, minMs: 380, maxMs: 1800 }
let closedSince: number | null = null

export const LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
export const LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
export const FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
export const LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
export const RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

export function initFace(): Promise<FaceLandmarker> {
  if (landmarker) return Promise.resolve(landmarker)
  if (initPromise) return initPromise
  initPromise = (async () => {
    const fileset = await FilesetResolver.forVisionTasks('/mediapipe/wasm')
    landmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: '/mediapipe/face_landmarker.task', delegate: 'GPU' },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    })
    return landmarker
  })()
  return initPromise
}

export function onFace(l: FrameListener) { frameListeners.add(l); return () => { frameListeners.delete(l) } }
export function onBlink(l: BlinkListener) { blinkListeners.add(l); return () => { blinkListeners.delete(l) } }

/** Run the landmarker on every new video frame until the returned stop() is called. */
// One tracking loop for the whole page, whoever registers a video (the room camera, gaze's background
// video): the landmarker runs once per camera frame and a blink is never counted twice.
const tracked: HTMLVideoElement[] = []
let looping = false
let lastVideo: HTMLVideoElement | null = null
let lastTs = -1

export function trackVideo(video: HTMLVideoElement): () => void {
  tracked.push(video)
  if (!looping) {
    looping = true
    initFace().then(loop).catch((e) => { looping = false; console.warn('face landmarker unavailable', e) })
  }
  let stopped = false
  return () => {
    if (stopped) return
    stopped = true
    const i = tracked.indexOf(video)
    if (i >= 0) tracked.splice(i, 1)
  }
}

function loop() {
  if (!tracked.length) { looping = false; lastVideo = null; return }
  // the first registered video that has frames (they all show the same camera stream)
  const video = tracked.find((v) => v.readyState >= 2)
  if (video && landmarker) {
    if (video !== lastVideo) { lastVideo = video; lastTs = -1 }
    if (video.currentTime !== lastTs) {
      lastTs = video.currentTime
      process(video)
    }
  }
  requestAnimationFrame(loop)
}

function process(video: HTMLVideoElement) {
  const lm_ = landmarker!
  const t = performance.now()
  const res = lm_.detectForVideo(video, t)
  const lm = res.faceLandmarks?.[0] ?? null
  const shapes = Object.fromEntries((res.faceBlendshapes?.[0]?.categories ?? []).map((c) => [c.categoryName, c.score]))
  let mouthBox = null
  let yaw = 0
  if (lm) {
    const pts = LIPS_OUTER.map((i) => lm[i])
    const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y)
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2
    const w = (Math.max(...xs) - Math.min(...xs)) * 1.7
    mouthBox = { x: cx - w / 2, y: cy - w / 2 * (video.videoWidth / Math.max(1, video.videoHeight)), w, h: w * (video.videoWidth / Math.max(1, video.videoHeight)) }
    const nose = lm[1], l = lm[234], r = lm[454]
    yaw = ((nose.x - l.x) / Math.max(1e-3, r.x - l.x) - 0.5) * 2
  }
  // column-major 4x4; ZYX Euler angles from its rotation part
  const M = res.facialTransformationMatrixes?.[0]?.data
  const headPitch = M ? Math.atan2(M[6], M[10]) * 180 / Math.PI : 0
  const headYaw = M ? Math.asin(Math.max(-1, Math.min(1, -M[2]))) * 180 / Math.PI : 0
  const frame: FaceFrame = {
    present: !!lm, landmarks: lm, mouthBox, yaw, headPitch, headYaw, pose: M ? [...M] : null, source: video,
    mouthOpen: shapes.jawOpen ?? 0, blinkL: shapes.eyeBlinkLeft ?? 0, blinkR: shapes.eyeBlinkRight ?? 0, t,
  }
  detectBlink(frame)
  frameListeners.forEach((f) => f(frame))
}

// Background tracking for when no page video is tracked yet (gaze needs face frames).
// Reference-counted: one off-screen video and one tracking loop, however many users. If a visible
// video is already tracked (the room camera), its frames are used instead of a second tracker.
let bg: { video: HTMLVideoElement; stop: () => void } | null = null
let bgUsers = 0
export async function trackCameraInBackground(): Promise<() => void> {
  if (tracked.length > 0 && !bg) return () => {}
  bgUsers++
  if (!bg) {
    const video = document.createElement('video')
    // off-screen but rendered: display:none would stop frame decoding
    Object.assign(video.style, { position: 'fixed', width: '2px', height: '2px', opacity: '0', pointerEvents: 'none', left: '0', top: '0' })
    video.muted = true
    video.playsInline = true
    document.body.appendChild(video)
    video.srcObject = await getCamera()
    await video.play()
    bg = { video, stop: trackVideo(video) }
  }
  let released = false
  return () => {
    if (released) return
    released = true
    if (--bgUsers === 0 && bg) { bg.stop(); bg.video.remove(); bg = null }
  }
}

function detectBlink(f: FaceFrame) {
  const closed = f.present && f.blinkL > blinkConfig.threshold && f.blinkR > blinkConfig.threshold
  if (closed && closedSince === null) closedSince = f.t
  if (!closed && closedSince !== null) {
    const d = f.t - closedSince
    closedSince = null
    // deliberate = longer than a natural blink (~100-300 ms), shorter than eyes-resting
    if (d >= blinkConfig.minMs && d <= blinkConfig.maxMs) blinkListeners.forEach((l) => l(d))
  }
}
