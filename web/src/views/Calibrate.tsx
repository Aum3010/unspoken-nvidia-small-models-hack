import { AnimatePresence, motion } from 'framer-motion'
import { Eye, Loader2, RotateCcw, ScanEye } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { getCamera } from '../lib/capture'
import { blinkConfig, onBlink, trackVideo } from '../lib/face'
import { calibrationSampleCount, clearCalibration, forgetCalibration, gazeState, measureAccuracy, onGaze, recordCalibrationPoint, saveCalibration, startGaze } from '../lib/gaze'
import { useSelection } from '../lib/selection'
import './calibrate.css'

const POINTS = [[0.1, 0.12], [0.5, 0.12], [0.9, 0.12], [0.1, 0.5], [0.5, 0.5], [0.9, 0.5], [0.1, 0.88], [0.5, 0.88], [0.9, 0.88]]
// two passes, the second in a different order, so drift and order effects average out
const SEQUENCE = [...POINTS, ...[4, 8, 0, 6, 2, 7, 1, 5, 3].map((i) => POINTS[i])]
// accuracy is tested on points that were not trained on
const TEST_POINTS = [[0.5, 0.5], [0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7]]
const SETTLE_MS = 600   // eyes travel to a new dot (~200-400 ms) before samples count
const RECORD_MS = 1000
const SAMPLE_MS = 50
const TEST_MS = 1200

type Phase = 'intro' | 'starting' | 'points' | 'measure' | 'done' | 'noface' | 'error'

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

  // hands-free calibration: let the eyes settle on each dot, then record samples
  useEffect(() => {
    if (phase !== 'points') return
    const [px, py] = SEQUENCE[idx]
    const x = px * window.innerWidth, y = py * window.innerHeight
    let iv = 0
    const settle = window.setTimeout(() => { iv = window.setInterval(() => recordCalibrationPoint(x, y), SAMPLE_MS) }, SETTLE_MS)
    const next = window.setTimeout(() => {
      window.clearInterval(iv)
      if (idx + 1 < SEQUENCE.length) setIdx(idx + 1)
      else if (calibrationSampleCount() < SEQUENCE.length * 5) { setAcc(null); setPhase('noface') }  // eyes were not seen
      else { setIdx(0); setPhase('measure') }
    }, SETTLE_MS + RECORD_MS)
    return () => { window.clearTimeout(settle); window.clearInterval(iv); window.clearTimeout(next) }
  }, [phase, idx])

  // accuracy: mean error over unseen test points, each measured after the eyes settle
  const errors = useRef<number[]>([])
  useEffect(() => {
    if (phase !== 'measure') return
    if (idx === 0) errors.current = []
    const [px, py] = TEST_POINTS[idx]
    let cancelled = false
    const t = window.setTimeout(() => {
      measureAccuracy({ x: px * window.innerWidth, y: py * window.innerHeight }, TEST_MS).then((e) => {
        if (cancelled) return
        if (e != null) errors.current.push(e)
        if (idx + 1 < TEST_POINTS.length) { setIdx(idx + 1); return }
        const a = errors.current.length >= 3 ? errors.current.reduce((x, y) => x + y, 0) / errors.current.length : null
        gazeState.accuracyPx = a
        gazeState.calibrated = a != null && a < 220
        saveCalibration()  // reused next time (same window size and camera position)
        setAcc(a)
        setPhase('done')
      })
    }, SETTLE_MS)
    return () => { cancelled = true; window.clearTimeout(t) }
  }, [phase, idx])

  const quality = acc == null ? 'none' : acc < 120 ? 'good' : acc < 220 ? 'fair' : 'poor'

  if (phase === 'points' || phase === 'measure') {
    const [px, py] = phase === 'measure' ? TEST_POINTS[idx] : SEQUENCE[idx]
    return (
      <div className="calib-full">
        <div className="calib-instr">{phase === 'measure' ? `Checking accuracy · look at the dot · ${idx + 1} of ${TEST_POINTS.length}` : `Look at the dot · ${idx + 1} of ${SEQUENCE.length}`}</div>
        <AnimatePresence mode="wait">
          <motion.div key={phase + idx} className={`calib-dot ${phase}`} style={{ left: `${px * 100}%`, top: `${py * 100}%` }}
            initial={{ scale: 0, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0, opacity: 0 }} transition={{ duration: .35 }}>
            <svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="44" className="calib-ring" style={{ animationDuration: `${SETTLE_MS + (phase === 'measure' ? TEST_MS : RECORD_MS)}ms` }} /></svg>
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
            Webcam gaze is an <b>estimate</b> from your iris position and head pose, not clinical eye tracking. We calibrate it to you, measure its error, and
            only enable gaze selection when it is good enough. A separate, longer look is always needed to confirm.
          </p>
          <ol className="calib-steps">
            <li>Sit about an arm's length from the screen, face lit from the front.</li>
            <li>Look at each dot until its ring closes — two rounds of 9, no clicking needed (about 30 seconds).</li>
            <li>We then test accuracy on 5 new dots.</li>
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
          {phase === 'intro' && gazeState.calibrated && gazeState.savedAt && (
            <div className={`calib-result ${quality}`}>
              <Eye size={20} />
              <div className="grow">
                <b>Saved calibration in use</b>
                <div style={{ fontSize: '.85rem', opacity: .85 }}>
                  From {new Date(gazeState.savedAt).toLocaleString()} · error ≈ {acc ? Math.round(acc) : '—'} px. Recalibrate if the screen or bed moved.
                </div>
              </div>
              <button className="btn sm ghost" onClick={() => { forgetCalibration(); setAcc(null); setPhase('intro') }}>Forget</button>
            </div>
          )}
          {phase === 'error' && <div className="calib-result poor"><b>Could not start gaze tracking.</b> Check camera permission.</div>}
          {phase === 'noface' && <div className="calib-result poor"><b>Your eyes were not visible enough.</b> Face the camera with light on your face, then try again.</div>}
          <div className="panel-title" style={{ marginBottom: 0 }}>How to choose with the eyes</div>
          <div className="seg" role="radiogroup" aria-label="Gaze selection">
            <button aria-pressed={sel.gazeSelect === 'dwell'} onClick={() => sel.setSettings({ gazeSelect: 'dwell' })}>Look and hold (dwell)</button>
            <button aria-pressed={sel.gazeSelect === 'blink'} onClick={() => sel.setSettings({ gazeSelect: 'blink' })}>Look, then blink slowly</button>
          </div>
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
