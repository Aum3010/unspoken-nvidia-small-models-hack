// Always-on inputs. Nothing waits for a button: the microphone, the mouth and the
// head are watched continuously, and a clip is cut only around activity.
//   voice -> energy VAD with an adaptive noise floor, 350 ms pre-roll, 16 kHz WAV
//   lips  -> mouth-opening variability from MediaPipe landmarks, webm clip
//   head  -> nod / shake from the face transform (pitch / yaw oscillation)
// Clips go to the local gateway only, and only while `canCapture()` allows it
// (never while a message waits for confirmation or while speech is playing).

import { getMic, record, type Recording } from '../lib/capture'
import { onFace, type FaceFrame } from '../lib/face'

/* ============ voice ============ */
interface VoiceOpts {
  canCapture: () => boolean
  onSpeech: (active: boolean) => void
  /** The utterance is over (700 ms of quiet). `early` says the same clip was already sent by onEarly. */
  onClip: (wav: Blob, ms: number, early: boolean) => void
  /** 350 ms of quiet: the utterance has probably ended, so its decode can start while we wait to be sure. */
  onEarly?: (wav: Blob) => void
  /** Speech resumed (or capture was cancelled) after onEarly: that clip will not be used. */
  onEarlyCancel?: () => void
  /** Every ~45 ms block, capture or not: is there voice (or a whisper) above the room's noise? */
  onLevel?: (voice: boolean) => void
}

const TAP = `class Tap extends AudioWorkletProcessor {
  constructor(){ super(); this.buf = new Float32Array(2048); this.n = 0 }
  process(inputs){
    const ch = inputs[0] && inputs[0][0]
    if (ch) for (let i = 0; i < ch.length; i++){ this.buf[this.n++] = ch[i]; if (this.n === 2048){ this.port.postMessage(this.buf.slice(0)); this.n = 0 } }
    return true
  }
}
registerProcessor('unspoken-tap', Tap)`

const PRE_BLOCKS = 8          // ≈350 ms of audio before the onset
const ONSET_BLOCKS = 3        // ≈130 ms above threshold to start
const HANG_BLOCKS = 16        // ≈700 ms of quiet to end (unchanged: pauses inside a phrase must not split it)
const EARLY_BLOCKS = 8        // ≈350 ms of quiet: start decoding early; the clip then keeps exactly this audio
const MIN_VOICED_BLOCKS = 8   // ≈350 ms of voice, or the clip is dropped
// after ≈1.5 s of voice the utterance is a long sentence: its end needs ≈1.1 s of quiet, so a pause inside it does
// not cut it in two (short phrases keep the 700 ms end)
const LONG_VOICED_BLOCKS = 35, LONG_HANG_BLOCKS = 24
const MAX_MS = 10000

export async function startVoice(o: VoiceOpts): Promise<() => void> {
  const stream = await getMic()
  const ac = new AudioContext()
  const url = URL.createObjectURL(new Blob([TAP], { type: 'application/javascript' }))
  await ac.audioWorklet.addModule(url)
  URL.revokeObjectURL(url)
  const src = ac.createMediaStreamSource(stream)
  const tap = new AudioWorkletNode(ac, 'unspoken-tap')
  const sink = ac.createGain(); sink.gain.value = 0
  src.connect(tap); tap.connect(sink); sink.connect(ac.destination)
  const sr = ac.sampleRate, blockMs = 2048 / sr * 1000

  // the first 1.5 s only learn the room's noise floor: no clip, no "voice" (the floor starts as a guess)
  const VOICE_WARMUP_MS = 1500, t0 = performance.now()
  let floor = -60, above = 0, quiet = 0, voiced = 0
  let clip: Float32Array[] | null = null
  let early: { wav: Blob; blocks: number } | null = null   // the clip as sent at EARLY_BLOCKS of quiet
  const pre: Float32Array[] = []
  const cancelEarly = () => { if (early) { early = null; o.onEarlyCancel?.() } }
  const end = (keep: boolean) => {
    if (clip && keep && voiced >= MIN_VOICED_BLOCKS) {
      // committed clip = the early one, so the decode already running is for exactly what is submitted
      if (early) o.onClip(early.wav, early.blocks * blockMs, true)
      else o.onClip(toWav(clip, sr), clip.length * blockMs, false)
      early = null
    } else cancelEarly()
    if (clip) o.onSpeech(false)
    clip = null; voiced = 0; quiet = 0
  }
  tap.port.onmessage = (ev: MessageEvent<Float32Array>) => {
    const b = ev.data
    let sum = 0; for (let i = 0; i < b.length; i++) sum += b[i] * b[i]
    const db = 10 * Math.log10(sum / b.length + 1e-12)
    if (performance.now() - t0 < VOICE_WARMUP_MS) { floor += (db - floor) * 0.3; return }
    const loud = db > Math.max(floor + 12, -52)
    o.onLevel?.(loud)
    if (!clip) {
      // the floor follows the room while nobody speaks
      floor += (db - floor) * (db < floor ? 0.2 : loud ? 0.002 : 0.04)
      pre.push(b); if (pre.length > PRE_BLOCKS) pre.shift()
      above = loud ? above + 1 : 0
      if (above >= ONSET_BLOCKS && o.canCapture()) { clip = pre.splice(0); voiced = above; o.onSpeech(true) }
      return
    }
    if (!o.canCapture()) { end(false); return }
    clip.push(b)
    if (db > floor + 7) { quiet = 0; voiced++; cancelEarly() } else quiet++
    if (quiet === EARLY_BLOCKS && !early && voiced >= MIN_VOICED_BLOCKS && o.onEarly) {
      early = { wav: toWav(clip, sr), blocks: clip.length }
      o.onEarly(early.wav)
    }
    if (clip.length * blockMs >= MAX_MS) { cancelEarly(); end(true) }
    else if (quiet >= (voiced >= LONG_VOICED_BLOCKS ? LONG_HANG_BLOCKS : HANG_BLOCKS)) end(true)
  }
  return () => { end(false); tap.port.onmessage = null; src.disconnect(); ac.close() }
}

function toWav(blocks: Float32Array[], sr: number): Blob {
  const n = blocks.reduce((a, b) => a + b.length, 0)
  const all = new Float32Array(n); let o = 0
  for (const b of blocks) { all.set(b, o); o += b.length }
  // 16 kHz is what every ASR here expects; linear resampling is plenty for speech
  const out = 16000, m = Math.floor(n * out / sr)
  const pcm = new Int16Array(m)
  for (let i = 0; i < m; i++) {
    const x = i * sr / out, j = Math.floor(x), f = x - j
    const v = all[j] * (1 - f) + (all[Math.min(n - 1, j + 1)] ?? 0) * f
    pcm[i] = Math.max(-32768, Math.min(32767, Math.round(v * 32767)))
  }
  const buf = new ArrayBuffer(44 + pcm.byteLength), dv = new DataView(buf)
  const str = (off: number, s: string) => { for (let i = 0; i < s.length; i++) dv.setUint8(off + i, s.charCodeAt(i)) }
  str(0, 'RIFF'); dv.setUint32(4, 36 + pcm.byteLength, true); str(8, 'WAVE'); str(12, 'fmt ')
  dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true); dv.setUint32(24, out, true)
  dv.setUint32(28, out * 2, true); dv.setUint16(32, 2, true); dv.setUint16(34, 16, true); str(36, 'data'); dv.setUint32(40, pcm.byteLength, true)
  new Int16Array(buf, 44).set(pcm)
  return new Blob([buf], { type: 'audio/wav' })
}

/* ============ lips ============ */
interface LipOpts {
  stream: MediaStream
  canCapture: () => boolean
  onMouthing: (active: boolean) => void
  /** movingMs: how long the lips moved in the clip, on the lip landmarks (the lip agent re-checks it) */
  onClip: (video: Blob, ms: number, stream: { id: string; chunks: number }, movingMs: number) => void
  /** a piece of the clip being recorded (sent while the person is still mouthing) */
  onPiece?: (id: string, piece: Blob, seq: number, mime: string) => void
  /** the recording was dropped: its pieces will not be used */
  onDrop?: (id: string) => void
}

const LIP_WINDOW_MS = 600
const LIP_MOVING = 0.012      // std of lip gap / face height over the window
const LIP_STILL_MS = 900
// a long sentence has pauses inside it: once the lips have moved this long, the clip ends only after a longer
// stillness, so it is not cut in two (short phrases keep the quick 0.9 s end)
const LIP_LONG_MOVING_MS = 1500, LIP_LONG_STILL_MS = 1400
const LIP_MIN_MS = 900
const LIP_MAX_MS = 8000

// A clip is only cut around real lip movement, measured on the lip landmarks (lighting cannot fake it, unlike
// pixel change): the face must be tracked steadily first (the camera's first second adjusts exposure and the
// landmarks settle), the lips must keep moving to start a clip, and a clip whose lips moved for less than
// LIP_MIN_MOVING_MS in total is dropped. The lip agent re-checks the moving time it is sent.
const FACE_WARMUP_MS = 1500
const LIP_CONFIRM_MS = 350
export const LIP_MIN_MOVING_MS = 450

export function startLips(o: LipOpts): () => void {
  const hist: { t: number; g: number }[] = []
  let rec: Recording | null = null, recT0 = 0, movingSince = 0, stillSince = 0, confirmed = false, recId = ''
  let faceSince = 0, movingMs = 0, lastT = 0
  const stop = (send: boolean) => {
    const r = rec; if (!r) return
    rec = null
    if (confirmed) o.onMouthing(false)
    const ms = performance.now() - recT0, id = recId
    const moved = movingMs
    if (send && confirmed && ms >= LIP_MIN_MS && moved >= LIP_MIN_MOVING_MS) r.stop().then((b) => o.onClip(b, ms, { id, chunks: r.chunks() }, moved))
    else { r.cancel(); o.onDrop?.(id) }
    confirmed = false; movingMs = 0
  }
  const off = onFace((f: FaceFrame) => {
    const now = f.t
    const dt = lastT ? Math.min(100, now - lastT) : 0
    lastT = now
    if (!f.present || !f.landmarks) { stop(true); hist.length = 0; faceSince = 0; return }
    faceSince ||= now
    if (now - faceSince < FACE_WARMUP_MS) { hist.length = 0; return }
    const lm = f.landmarks
    const faceH = Math.hypot(lm[10].x - lm[152].x, lm[10].y - lm[152].y) || 1
    hist.push({ t: now, g: Math.hypot(lm[13].x - lm[14].x, lm[13].y - lm[14].y) / faceH })
    while (hist.length && now - hist[0].t > LIP_WINDOW_MS) hist.shift()
    const mean = hist.reduce((a, h) => a + h.g, 0) / hist.length
    const std = Math.sqrt(hist.reduce((a, h) => a + (h.g - mean) ** 2, 0) / hist.length)
    const moving = std > LIP_MOVING
    if (!o.canCapture()) { stop(false); return }
    if (moving) {
      stillSince = 0
      if (!movingSince) movingSince = now
      if (rec) movingMs += dt
      // start recording at the first movement so the onset is kept; commit after LIP_CONFIRM_MS of it
      if (!rec) {
        const id = recId = `lip${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`
        rec = record(o.stream, 'video', LIP_MAX_MS + 500, o.onPiece && ((p, seq, mime) => o.onPiece!(id, p, seq, mime)))
        recT0 = performance.now(); movingMs = 0
      }
      if (!confirmed && now - movingSince > LIP_CONFIRM_MS) { confirmed = true; o.onMouthing(true) }
    } else {
      movingSince = 0
      if (!stillSince) stillSince = now
      if (rec && !confirmed && now - stillSince > 300) stop(false)
      if (rec && confirmed && now - stillSince > (movingMs > LIP_LONG_MOVING_MS ? LIP_LONG_STILL_MS : LIP_STILL_MS)) stop(true)
    }
    if (rec && performance.now() - recT0 > LIP_MAX_MS) stop(true)
  })
  return () => { off(); stop(false) }
}

/* ============ mouth activity (no capture) ============ */
/** Whether the mouth is moving, from the same lip-gap measure as startLips, but always on: it keeps reporting
 *  while capture is paused (a clip being read, a message waiting), so the room always knows. */
export function watchMouth(onChange: (moving: boolean) => void): () => void {
  const hist: { t: number; g: number }[] = []
  let moving = false, faceSince = 0
  return onFace((f: FaceFrame) => {
    let now = false
    if (f.present && f.landmarks && (faceSince ||= f.t) && f.t - faceSince >= FACE_WARMUP_MS) {
      const lm = f.landmarks
      const faceH = Math.hypot(lm[10].x - lm[152].x, lm[10].y - lm[152].y) || 1
      hist.push({ t: f.t, g: Math.hypot(lm[13].x - lm[14].x, lm[13].y - lm[14].y) / faceH })
      while (hist.length && f.t - hist[0].t > LIP_WINDOW_MS) hist.shift()
      const mean = hist.reduce((a, h) => a + h.g, 0) / hist.length
      now = Math.sqrt(hist.reduce((a, h) => a + (h.g - mean) ** 2, 0) / hist.length) > LIP_MOVING
    } else { hist.length = 0; if (!f.present) faceSince = 0 }
    if (now !== moving) { moving = now; onChange(now) }
  })
}

/* ============ head ============ */
export function startHead(onGesture: (g: 'nod' | 'shake') => void): () => void {
  let baseP: number | null = null, baseY: number | null = null
  let nodOut = 0, shakeSide = 0, shakeT = 0, cooldown = 0
  return onFace((f: FaceFrame) => {
    if (!f.present) { baseP = baseY = null; return }
    const now = f.t, p = f.headPitch, y = f.headYaw
    if (baseP === null || baseY === null) { baseP = p; baseY = y; return }
    const dp = p - baseP, dy = y - baseY
    // the resting pose drifts slowly with posture; gestures are fast excursions from it
    if (Math.abs(dp) < 4) baseP += dp * 0.03
    if (Math.abs(dy) < 5) baseY += dy * 0.03
    if (now < cooldown) return
    // nod: out past 9° on the pitch axis and back within a second, head not turning
    if (Math.abs(dp) > 9 && Math.abs(dy) < 8 && !nodOut) nodOut = now
    if (nodOut && now - nodOut > 1000) nodOut = 0
    if (nodOut && Math.abs(dp) < 4) { nodOut = 0; cooldown = now + 1500; onGesture('nod'); return }
    // shake: past 12° to one side, then the other, within 1.2 s
    const side = dy > 12 ? 1 : dy < -12 ? -1 : 0
    if (side && Math.abs(dp) < 8) {
      if (shakeSide && side === -shakeSide && now - shakeT < 1200) { shakeSide = 0; cooldown = now + 1500; onGesture('shake'); return }
      if (side !== shakeSide) { shakeSide = side; shakeT = now }
    }
    if (shakeSide && now - shakeT > 1200) shakeSide = 0
  })
}
