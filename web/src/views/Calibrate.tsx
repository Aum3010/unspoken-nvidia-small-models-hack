import { AnimatePresence, motion } from 'framer-motion'
import { Eye, Loader2, RotateCcw, ScanEye } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { getCamera } from '../lib/capture'
import { blinkConfig, onBlink, trackVideo } from '../lib/face'
import { clearCalibration, gazeState, measureAccuracy, onGaze, recordCalibrationPoint, startGaze } from '../lib/gaze'
import { useSelection } from '../lib/selection'
import './calibrate.css'

const POINTS = [[0.1, 0.12], [0.5, 0.12], [0.9, 0.12], [0.1, 0.5], [0.5, 0.5], [0.9, 0.5], [0.1, 0.88], [0.5, 0.88], [0.9, 0.88]]
const HOLD_MS = 1700

type Phase = 'intro' | 'starting' | 'points' | 'measure' | 'done' | 'error'

export function Calibrate({ onDone }: { onDone: () => void }) {
  const sel = useSelection()
  const [phase, setPhase] = useState<Phase>('intro')
  const [idx, setIdx] = useState(0)
  const [acc, setAcc] = useState<number | null>(gazeState.accuracyPx)
  const [blinks, setBlinks] = useState(0)
  const [gaze, setGaze] = useState<{ x: number; y: number } | null>(null)
  const video = useRef<HTMLVideoElement>(null)

  useEffect(() => onGaze((p) => setGaze(p)), [])
  useEffect(() => onBlink(() => setBlinks((b) => b + 1)), [])
  useEffect(() => {
    let stop = () => {}
    getCamera().then((s) => {
      if (!video.current) return
      video.current.srcObject = s
      video.current.play().then(() => { stop = trackVideo(video.current!) })
    }).catch(() => undefined)
    return () => stop()
  }, [])

  const begin = async () => {
    setPhase('starting')
    try {
      clearCalibration()
      await startGaze()
      setIdx(0)
      setPhase('points')
    } catch {
      setPhase('error')
    }
  }

  // hands-free calibration: hold each point while samples are recorded
  useEffect(() => {
    if (phase !== 'points') return
    const [px, py] = POINTS[idx]
    const x = px * window.innerWidth, y = py * window.innerHeight
    const iv = window.setInterval(() => recordCalibrationPoint(x, y), 90)
    const t = window.setTimeout(() => {
      window.clearInterval(iv)
      if (idx + 1 < POINTS.length) setIdx(idx + 1)
      else setPhase('measure')
    }, HOLD_MS + 500)
    return () => { window.clearInterval(iv); window.clearTimeout(t) }
  }, [phase, idx])

  useEffect(() => {
    if (phase !== 'measure') return
    measureAccuracy({ x: window.innerWidth / 2, y: window.innerHeight / 2 }, 2600).then((a) => {
      gazeState.accuracyPx = a
      gazeState.calibrated = a != null && a < 220
      setAcc(a)
      setPhase('done')
    })
  }, [phase])

  const quality = acc == null ? 'none' : acc < 120 ? 'good' : acc < 220 ? 'fair' : 'poor'

  if (phase === 'points' || phase === 'measure') {
    const [px, py] = phase === 'measure' ? [0.5, 0.5] : POINTS[idx]
    return (
      <div className="calib-full">
        <div className="calib-instr">{phase === 'measure' ? 'Keep looking at the centre dot — measuring accuracy' : `Look at the dot · ${idx + 1} of ${POINTS.length}`}</div>
        <AnimatePresence mode="wait">
          <motion.div key={phase + idx} className={`calib-dot ${phase}`} style={{ left: `${px * 100}%`, top: `${py * 100}%` }}
            initial={{ scale: 0, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0, opacity: 0 }} transition={{ duration: .35 }}>
            <svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="44" className="calib-ring" style={{ animationDuration: `${phase === 'measure' ? 2600 : HOLD_MS + 500}ms` }} /></svg>
            <span />
          </motion.div>
        </AnimatePresence>
        {gaze && <div className="calib-gaze" style={{ transform: `translate(${gaze.x}px, ${gaze.y}px)` }} />}
      </div>
    )
  }

  return (
    <div className="container calib">
      <div className="calib-grid">
        <div className="glass calib-card">
          <div className="calib-icon"><ScanEye size={26} /></div>
          <h1 className="display">Set up your eyes</h1>
          <p className="muted">
            Webcam gaze is an <b>estimate</b>, not clinical eye tracking. We calibrate it to you, measure its error, and
            only enable gaze selection when it is good enough. A separate, longer look is always needed to confirm.
          </p>
          <ol className="calib-steps">
            <li>Sit about an arm's length from the screen, face lit from the front.</li>
            <li>Look at each of the 9 dots until its ring closes — no clicking needed.</li>
            <li>We then test accuracy at the centre.</li>
          </ol>
          {phase === 'done' && (
            <div className={`calib-result ${quality}`}>
              <Eye size={20} />
              <div className="grow">
                <b>{quality === 'good' ? 'Gaze ready' : quality === 'fair' ? 'Low confidence — usable with large targets' : 'Too imprecise — gaze selection stays off'}</b>
                <div style={{ fontSize: '.85rem', opacity: .85 }}>Mean error ≈ {acc ? Math.round(acc) : '—'} px</div>
              </div>
            </div>
          )}
          {phase === 'error' && <div className="calib-result poor"><b>Could not start gaze tracking.</b> Check camera permission.</div>}
          <div className="row wrap" style={{ marginTop: 8 }}>
            <button className="btn lg primary" onClick={begin} disabled={phase === 'starting'}>
              {phase === 'starting' ? <Loader2 className="spin" size={20} /> : phase === 'done' ? <RotateCcw size={20} /> : <Eye size={20} />}
              {phase === 'done' ? 'Calibrate again' : 'Start calibration'}
            </button>
            {phase === 'done' && quality !== 'poor' && (
              <button className="btn lg" onClick={() => { sel.setSettings({ mode: 'gaze' }); onDone() }}>Use gaze to communicate</button>
            )}
          </div>
        </div>

        <div className="glass calib-card">
          <div className="panel-title">Deliberate blink</div>
          <div className="calib-video"><video ref={video} muted playsInline /></div>
          <p className="muted" style={{ margin: 0 }}>
            Close both eyes for about half a second. Natural blinks (under {blinkConfig.minMs} ms) are ignored.
          </p>
          <div className="blink-count">
            <motion.span key={blinks} initial={{ scale: 1.4, color: 'var(--gaze)' }} animate={{ scale: 1, color: 'var(--text)' }} className="display">{blinks}</motion.span>
            <span className="faint">deliberate blinks detected</span>
          </div>
          <label className="slider">
            <span>Minimum hold <b>{blinkConfig.minMs} ms</b></span>
            <input type="range" min={250} max={900} step={10} defaultValue={blinkConfig.minMs} onChange={(e) => { blinkConfig.minMs = +e.target.value; setBlinks((b) => b) }} />
          </label>
          <button className="btn" onClick={() => { sel.setSettings({ mode: 'scan' }); onDone() }}>Use blink scanning</button>
        </div>
      </div>
    </div>
  )
}
