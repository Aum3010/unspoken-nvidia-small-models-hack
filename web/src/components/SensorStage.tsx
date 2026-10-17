import { AnimatePresence, motion } from 'framer-motion'
import { Camera, Loader2, Mic, ShieldCheck, Square, Video } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { getCamera, getMic, record, type Recording } from '../lib/capture'
import { FACE_OVAL, LEFT_EYE, LIPS_INNER, LIPS_OUTER, RIGHT_EYE, onBlink, onFace, trackVideo, type FaceFrame } from '../lib/face'
import { Badge } from './meta'
import './sensor.css'

type Mode = 'lip' | 'audio'
const MAX_LIP_MS = 8000
const MAX_AUDIO_MS = 10000

interface Props {
  mode: Mode
  busy: boolean
  onLipClip: (b: Blob) => void
  onAudioClip: (b: Blob, enhance: boolean) => void
}

export function SensorStage({ mode, busy, onLipClip, onAudioClip }: Props) {
  const video = useRef<HTMLVideoElement>(null)
  const overlay = useRef<HTMLCanvasElement>(null)
  const inset = useRef<HTMLCanvasElement>(null)
  const [camState, setCamState] = useState<'idle' | 'starting' | 'live' | 'denied'>('idle')
  const [face, setFace] = useState<FaceFrame | null>(null)
  const [rec, setRec] = useState<Recording | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const [enhance, setEnhance] = useState(true)
  const [blinkFlash, setBlinkFlash] = useState(false)
  const [micStream, setMicStream] = useState<MediaStream | null>(null)

  const startCamera = useCallback(async () => {
    setCamState('starting')
    try {
      const s = await getCamera()
      if (video.current) {
        video.current.srcObject = s
        await video.current.play()
      }
      setCamState('live')
    } catch {
      setCamState('denied')
    }
  }, [])

  useEffect(() => { startCamera() }, [startCamera])

  useEffect(() => {
    if (camState !== 'live' || !video.current) return
    const stop = trackVideo(video.current)
    let last = 0
    const off = onFace((f) => {
      draw(f)
      const now = performance.now()
      if (now - last > 120) { setFace(f); last = now }
    })
    const offBlink = onBlink(() => { setBlinkFlash(true); setTimeout(() => setBlinkFlash(false), 500) })
    return () => { stop(); off(); offBlink() }
  }, [camState])

  const draw = (f: FaceFrame) => {
    const c = overlay.current, v = video.current
    if (!c || !v) return
    const w = (c.width = v.clientWidth * devicePixelRatio)
    const h = (c.height = v.clientHeight * devicePixelRatio)
    const ctx = c.getContext('2d')!
    ctx.clearRect(0, 0, w, h)
    if (!f.landmarks) return
    // object-fit: cover mapping
    const vw = v.videoWidth, vh = v.videoHeight
    const scale = Math.max(w / vw, h / vh)
    const ox = (w - vw * scale) / 2, oy = (h - vh * scale) / 2
    const P = (i: number) => [f.landmarks![i].x * vw * scale + ox, f.landmarks![i].y * vh * scale + oy] as const
    const css = getComputedStyle(document.documentElement)
    const lip = css.getPropertyValue('--lip').trim() || '#ff9ecb'
    const gaze = css.getPropertyValue('--gaze').trim() || '#7cc8ff'

    ctx.fillStyle = 'rgba(255,255,255,0.28)'
    for (let i = 0; i < f.landmarks.length; i += 7) {
      const [x, y] = P(i)
      ctx.beginPath(); ctx.arc(x, y, 1.1 * devicePixelRatio, 0, Math.PI * 2); ctx.fill()
    }
    const path = (idx: number[], color: string, width: number, glow = 0) => {
      ctx.beginPath()
      idx.forEach((i, k) => { const [x, y] = P(i); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y) })
      ctx.closePath()
      ctx.strokeStyle = color; ctx.lineWidth = width * devicePixelRatio
      ctx.shadowColor = color; ctx.shadowBlur = glow * devicePixelRatio
      ctx.stroke(); ctx.shadowBlur = 0
    }
    path(FACE_OVAL, 'rgba(255,255,255,0.18)', 1)
    path(LEFT_EYE, gaze, 1.4, 6); path(RIGHT_EYE, gaze, 1.4, 6)
    path(LIPS_OUTER, lip, 2.2, 14); path(LIPS_INNER, lip, 1.2, 4)

    // mouth ROI brackets
    if (f.mouthBox) {
      const x = f.mouthBox.x * vw * scale + ox, y = f.mouthBox.y * vh * scale + oy
      const bw = f.mouthBox.w * vw * scale, bh = f.mouthBox.h * vh * scale
      const k = Math.min(bw, bh) * 0.22
      ctx.strokeStyle = lip; ctx.lineWidth = 2 * devicePixelRatio; ctx.shadowColor = lip; ctx.shadowBlur = 10
      ctx.beginPath()
      ;[[x, y, 1, 1], [x + bw, y, -1, 1], [x, y + bh, 1, -1], [x + bw, y + bh, -1, -1]].forEach(([cx, cy, sx, sy]) => {
        ctx.moveTo(cx, cy + sy * k); ctx.lineTo(cx, cy); ctx.lineTo(cx + sx * k, cy)
      })
      ctx.stroke(); ctx.shadowBlur = 0
      // inset crop: grayscale mouth patch, like the VSR model input
      const ic = inset.current
      if (ic) {
        const ictx = ic.getContext('2d')!
        ictx.filter = 'grayscale(1) contrast(1.15)'
        ictx.drawImage(v, f.mouthBox.x * vw, f.mouthBox.y * vh, f.mouthBox.w * vw, f.mouthBox.h * vh, 0, 0, ic.width, ic.height)
      }
    }
  }

  // recording timers
  useEffect(() => {
    if (!rec) return
    const id = window.setInterval(() => setElapsed(performance.now() - rec.startedAt), 50)
    return () => window.clearInterval(id)
  }, [rec])

  const toggleLip = useCallback(async () => {
    if (rec) {
      const blob = await rec.stop()
      setRec(null); setElapsed(0)
      onLipClip(blob)
      return
    }
    const s = await getCamera()
    const r = record(s, 'video', MAX_LIP_MS)
    setRec(r)
    setTimeout(() => {
      // auto-stop path: if still the same recording, submit it
      setRec((cur) => {
        if (cur === r) { r.stop().then(onLipClip); setElapsed(0); return null }
        return cur
      })
    }, MAX_LIP_MS + 50)
  }, [rec, onLipClip])

  const toggleAudio = useCallback(async () => {
    if (rec) {
      const blob = await rec.stop()
      setRec(null); setElapsed(0)
      onAudioClip(blob, enhance)
      return
    }
    const s = await getMic()
    setMicStream(s)
    const r = record(s, 'audio', MAX_AUDIO_MS)
    setRec(r)
    setTimeout(() => {
      setRec((cur) => {
        if (cur === r) { r.stop().then((b) => onAudioClip(b, enhance)); setElapsed(0); return null }
        return cur
      })
    }, MAX_AUDIO_MS + 50)
  }, [rec, onAudioClip, enhance])

  // keyboard: R toggles recording
  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 'r' || e.key === 'R') { e.preventDefault(); if (!busy) (mode === 'lip' ? toggleLip : toggleAudio)() }
    }
    window.addEventListener('keydown', k)
    return () => window.removeEventListener('keydown', k)
  }, [mode, toggleLip, toggleAudio, busy])

  const max = mode === 'lip' ? MAX_LIP_MS : MAX_AUDIO_MS
  const hint = !face?.present ? 'Bring your face into view'
    : Math.abs(face.yaw) > 0.45 ? 'Turn a little toward the camera'
      : mode === 'lip' ? 'Mouth tracked — ready to read lips' : 'Ready to listen'
  const hintOk = !!face?.present && Math.abs(face.yaw) <= 0.45

  return (
    <div className="glass sensor">
      <div className="sensor-head">
        <div className="panel-title" style={{ margin: 0 }}>
          {mode === 'lip' ? <Video size={15} /> : <Mic size={15} />}
          {mode === 'lip' ? 'Silent-mouth capture' : 'Residual-voice capture'}
        </div>
        <Badge modality={mode === 'lip' ? 'lip' : 'audio'} live={camState === 'live'} />
      </div>

      <div className={`stage ${rec ? 'recording' : ''}`}>
        <video ref={video} playsInline muted className="stage-video" />
        <canvas ref={overlay} className="stage-overlay" />
        {camState !== 'live' && (
          <div className="stage-empty">
            {camState === 'starting' ? <Loader2 className="spin" /> : <Camera size={30} />}
            <div className="display" style={{ fontSize: '1.6rem' }}>
              {camState === 'denied' ? 'Camera unavailable' : 'Starting camera…'}
            </div>
            {camState === 'denied' && <button className="btn sm" onClick={startCamera}>Allow camera</button>}
            <div className="faint" style={{ fontSize: '.85rem' }}>The board below works without a camera.</div>
          </div>
        )}

        {camState === 'live' && (
          <>
            <div className={`stage-hint ${hintOk ? 'ok' : ''}`}>
              <span className="dot" />{hint}
            </div>
            <div className="stage-hud">
              <Meter label="Mouth" value={face?.mouthOpen ?? 0} color="var(--lip)" />
              <Meter label="Blink" value={Math.min(face?.blinkL ?? 0, face?.blinkR ?? 0)} color="var(--gaze)" />
            </div>
            {mode === 'lip' && (
              <div className="inset">
                <canvas ref={inset} width={96} height={96} />
                <span>model view · 96px</span>
              </div>
            )}
            <AnimatePresence>
              {blinkFlash && (
                <motion.div className="blink-flash" initial={{ opacity: 0, scale: .9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}>
                  Deliberate blink
                </motion.div>
              )}
            </AnimatePresence>
          </>
        )}

        {rec && (
          <div className="rec-pill">
            <span className="rec-dot" /> {mode === 'lip' ? 'Reading lips' : 'Listening'} · {(elapsed / 1000).toFixed(1)}s
            <span className="rec-bar"><span style={{ width: `${Math.min(100, (elapsed / max) * 100)}%` }} /></span>
          </div>
        )}
        {busy && !rec && (
          <div className="processing">
            <Loader2 className="spin" size={18} />
            <span className="shimmer">{mode === 'lip' ? 'Lip-reading agent decoding n-best…' : 'Voice agent comparing raw and enhanced audio…'}</span>
          </div>
        )}
      </div>

      {mode === 'audio' && <Waveform stream={micStream} active={!!rec} />}

      <div className="sensor-actions">
        <button
          className={`rec-btn ${rec ? 'on' : ''}`}
          onClick={mode === 'lip' ? toggleLip : toggleAudio}
          disabled={busy || (mode === 'lip' && camState !== 'live')}
          aria-label={rec ? 'Stop and send' : mode === 'lip' ? 'Start mouthing' : 'Start speaking'}
        >
          <span className="rec-core">{rec ? <Square size={22} fill="currentColor" /> : mode === 'lip' ? <Video size={24} /> : <Mic size={24} />}</span>
        </button>
        <div className="grow">
          <div style={{ fontWeight: 600, fontSize: '1.05rem' }}>
            {rec ? 'Tap again when you finish' : mode === 'lip' ? 'Mouth a short phrase silently' : 'Say a phrase, however quietly'}
          </div>
          <div className="faint" style={{ fontSize: '.86rem' }}>
            Press <kbd>R</kbd> to start/stop · max {max / 1000}s · clips go to this Mac only
          </div>
        </div>
        {mode === 'audio' && (
          <label className="toggle">
            <input type="checkbox" checked={enhance} onChange={(e) => setEnhance(e.target.checked)} />
            <span className="toggle-track"><span /></span>
            <span style={{ fontSize: '.88rem' }}>Compare enhanced</span>
          </label>
        )}
      </div>
      <div className="privacy"><ShieldCheck size={14} /> On-device: MediaPipe in browser · VSR on Metal (MPS) · nothing stored</div>
    </div>
  )
}

function Meter({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="meter">
      <span>{label}</span>
      <span className="meter-track"><span style={{ width: `${Math.round(value * 100)}%`, background: color }} /></span>
    </div>
  )
}

function Waveform({ stream, active }: { stream: MediaStream | null; active: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const [level, setLevel] = useState(-90)
  useEffect(() => {
    if (!stream || !ref.current) return
    const ac = new AudioContext()
    const src = ac.createMediaStreamSource(stream)
    const an = ac.createAnalyser()
    an.fftSize = 2048
    src.connect(an)
    const buf = new Float32Array(an.fftSize)
    let raf = 0, last = 0
    const c = ref.current
    const color = getComputedStyle(document.documentElement).getPropertyValue('--audio').trim() || '#ffc46b'
    const loop = (t: number) => {
      raf = requestAnimationFrame(loop)
      an.getFloatTimeDomainData(buf)
      const w = (c.width = c.clientWidth * devicePixelRatio), h = (c.height = c.clientHeight * devicePixelRatio)
      const ctx = c.getContext('2d')!
      ctx.clearRect(0, 0, w, h)
      const bars = 72, step = Math.floor(buf.length / bars)
      let sum = 0
      for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i]
      const rms = Math.sqrt(sum / buf.length)
      for (let b = 0; b < bars; b++) {
        let m = 0
        for (let j = 0; j < step; j++) m = Math.max(m, Math.abs(buf[b * step + j]))
        // log-ish scaling so whispers are visible
        const v = Math.min(1, Math.log10(1 + m * 400) / 2.4)
        const bh = Math.max(2, v * h * 0.9)
        ctx.fillStyle = color
        ctx.globalAlpha = 0.35 + v * 0.65
        const bw = w / bars
        ctx.beginPath()
        ctx.roundRect(b * bw + bw * 0.2, (h - bh) / 2, bw * 0.6, bh, bw * 0.3)
        ctx.fill()
      }
      if (t - last > 150) { setLevel(20 * Math.log10(rms + 1e-9)); last = t }
    }
    raf = requestAnimationFrame(loop)
    return () => { cancelAnimationFrame(raf); ac.close() }
  }, [stream])
  return (
    <div className={`wave ${active ? 'active' : ''}`}>
      <canvas ref={ref} />
      <span className="mono wave-level">{stream ? `${level.toFixed(0)} dBFS` : 'mic idle'}</span>
    </div>
  )
}
