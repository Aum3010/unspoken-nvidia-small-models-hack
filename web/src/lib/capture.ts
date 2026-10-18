// Media permission + recording helpers. Nothing is uploaded until the person
// finishes a capture; clips are sent to the local gateway only.

let cameraStream: MediaStream | null = null
let micStream: MediaStream | null = null
// One open at a time, shared by every caller (room camera, gaze, lips). Parallel getUserMedia calls on
// the same device stall on slow USB cameras: one measured here took 12 s to open, and a second request
// made meanwhile never delivered a frame.
let cameraOpening: Promise<MediaStream> | null = null
let micOpening: Promise<MediaStream> | null = null

export function getCamera(): Promise<MediaStream> {
  if (cameraStream && cameraStream.getVideoTracks().some((t) => t.readyState === 'live')) return Promise.resolve(cameraStream)
  cameraOpening ??= navigator.mediaDevices.getUserMedia({
    // 1280x720: more pixels across the eyes for gaze; lip clips are cropped and resampled anyway
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 }, facingMode: 'user' },
    audio: false,
  }).then((s) => (cameraStream = s)).finally(() => { cameraOpening = null })
  return cameraOpening
}

export function getMic(): Promise<MediaStream> {
  if (micStream && micStream.getAudioTracks().some((t) => t.readyState === 'live')) return Promise.resolve(micStream)
  // Keep the raw signal: browser noise suppression would change what the voice model hears.
  micOpening ??= navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
  }).then((s) => (micStream = s)).finally(() => { micOpening = null })
  return micOpening
}

export function stopCamera() {
  cameraStream?.getTracks().forEach((t) => t.stop())
  cameraStream = null
}

function pickMime(kind: 'video' | 'audio'): string | undefined {
  const opts = kind === 'video'
    ? ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4']
    : ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4']
  return opts.find((m) => MediaRecorder.isTypeSupported(m))
}

export interface Recording {
  stop: () => Promise<Blob>
  cancel: () => void
  startedAt: number
}

export function record(stream: MediaStream, kind: 'video' | 'audio', maxMs = 10000): Recording {
  const mimeType = pickMime(kind)
  const tracks = kind === 'video' ? stream.getVideoTracks() : stream.getAudioTracks()
  const rec = new MediaRecorder(new MediaStream(tracks), mimeType ? { mimeType, videoBitsPerSecond: 2_500_000 } : undefined)
  const chunks: Blob[] = []
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data)
  let resolveStop: (b: Blob) => void = () => {}
  const done = new Promise<Blob>((r) => (resolveStop = r))
  rec.onstop = () => resolveStop(new Blob(chunks, { type: rec.mimeType || mimeType }))
  rec.start(250)
  const timer = window.setTimeout(() => rec.state === 'recording' && rec.stop(), maxMs)
  return {
    startedAt: performance.now(),
    stop: () => {
      window.clearTimeout(timer)
      if (rec.state === 'recording') rec.stop()
      return done
    },
    cancel: () => {
      window.clearTimeout(timer)
      rec.onstop = null
      if (rec.state === 'recording') rec.stop()
    },
  }
}
