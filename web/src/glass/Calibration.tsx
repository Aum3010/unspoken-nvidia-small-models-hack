// Eye calibration in the glass style. Shown when the app opens and from the Eyes chip.
// Webcam gaze is an estimate, so we measure its error and only switch selection to the
// eyes when it is usable; otherwise the pointer stays the stand-in for gaze.
// Its buttons are ordinary [data-a] targets, so they work by click, dwell or blink.

import { useEffect, useRef, useState } from 'react'
import { calibrationSampleCount, clearCalibration, gazeState, measureAccuracy, onGaze, recordCalibrationPoint, saveCalibration, setCalibrating, startGaze } from '../lib/gaze'

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

interface Props {
  /** Routes this screen's [data-a="calib:*"] actions here. */
  bind: (fn: ((a: string) => void) | null) => void
  /** Hide the room's dwell targets while dots are on screen. */
  onFocusMode: (on: boolean) => void
  onDone: (useEyes: boolean) => void
}

export function Calibration({ bind, onFocusMode, onDone }: Props) {
  const [phase, setPhase] = useState<Phase>('intro')
  const [idx, setIdx] = useState(0)
  const [acc, setAcc] = useState<number | null>(gazeState.accuracyPx)
  const [gaze, setGaze] = useState<{ x: number; y: number } | null>(null)

  useEffect(() => onGaze((p) => setGaze(p)), [])
  useEffect(() => { onFocusMode(phase === 'points' || phase === 'measure') }, [phase, onFocusMode])

  const quality = acc == null ? 'none' : acc < 120 ? 'good' : acc < 220 ? 'fair' : 'poor'
  const usable = quality === 'good' || quality === 'fair'

  useEffect(() => {
    bind(async (a) => {
      if (a === 'start' && phase !== 'starting') {
        setPhase('starting')
        try {
          setCalibrating(true)   // pure gaze while calibrating and measuring: no head fine-tune
          await startGaze()      // loads MobileGaze first, so the new calibration can use it
          clearCalibration()
          setIdx(0)
          setPhase('points')
        } catch {
          setCalibrating(false)
          setPhase('error')
        }
      } else if (a === 'skip') onDone(false)
      else if (a === 'continue') onDone(usable)
    })
    return () => bind(null)
  }, [bind, phase, usable, onDone])

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
      else if (calibrationSampleCount() < SEQUENCE.length * 5) { setCalibrating(false); setAcc(null); setPhase('noface') }  // eyes were not seen
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
        setCalibrating(false)
        setPhase('done')
      })
    }, SETTLE_MS)
    return () => { cancelled = true; window.clearTimeout(t) }
  }, [phase, idx])

  if (phase === 'points' || phase === 'measure') {
    const [px, py] = phase === 'measure' ? TEST_POINTS[idx] : SEQUENCE[idx]
    return (
      <>
        <div className="calib" style={{ background: 'rgba(3,5,9,.55)' }} />
        <div key={phase + idx} className="calib-dot" style={{ left: `${px * 100}%`, top: `${py * 100}%` }}>
          <svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="44" style={{ animationDuration: `${SETTLE_MS + (phase === 'measure' ? TEST_MS : RECORD_MS)}ms` }} /></svg>
        </div>
        <div className="calib-instr glass mono">{phase === 'measure' ? `Checking accuracy · look at the dot · ${idx + 1} of ${TEST_POINTS.length}` : `Look at the dot · ${idx + 1} of ${SEQUENCE.length}`}</div>
        {gaze && <div className="calib-gaze" style={{ transform: `translate(${gaze.x}px, ${gaze.y}px)` }} />}
      </>
    )
  }

  return (
    <div className="calib">
      <section className="calib-card glass" aria-label="Eye calibration" style={{ ['--c' as string]: '94,234,212' }}>
        <h1><span className="pebble">👁</span>Set up your eyes</h1>
        <p>
          Unspoken follows where you look, so you can choose without touching anything. Webcam gaze is an
          estimate, not clinical eye tracking: we tune it to you, measure its error, and only use it when it is good enough.
        </p>
        {phase !== 'done' && (
          <ol>
            <li>Sit about an arm's length from the screen, face lit from the front.</li>
            <li>Look at each dot until its ring closes: two rounds of 9, no clicking needed (about 30 seconds).</li>
            <li>We then test accuracy on 5 new dots.</li>
          </ol>
        )}
        {phase === 'done' && (
          <div className="calib-result" style={{ ['--c' as string]: usable ? '94,234,212' : '255,184,108' }}>
            <span className="pebble" />
            <div>
              <b>{quality === 'good' ? 'Your eyes are ready' : quality === 'fair' ? 'Usable, with some drift' : 'Too imprecise for gaze yet'}</b>
              <small>Mean error ≈ {acc ? `${Math.round(acc)} px` : 'not measured'} · {usable ? 'rest your gaze to choose, blink to select' : 'the pointer stays the stand-in for your eyes'}</small>
            </div>
          </div>
        )}
        {phase === 'noface' && (
          <div className="calib-result" style={{ ['--c' as string]: '255,128,112' }}>
            <span className="pebble" />
            <div><b>Your eyes were not visible enough</b><small>Face the camera with light on your face, then try again.</small></div>
          </div>
        )}
        {phase === 'error' && (
          <div className="calib-result" style={{ ['--c' as string]: '255,128,112' }}>
            <span className="pebble" />
            <div><b>Gaze tracking could not start</b><small>Allow the camera for this page, then try again.</small></div>
          </div>
        )}
        <div className="row">
          <button className="cbtn primary" data-a="calib:start" disabled={phase === 'starting'}>
            {phase === 'starting' ? 'Starting camera…' : phase === 'done' || phase === 'error' || phase === 'noface' ? 'Calibrate again' : 'Start calibration'}
          </button>
          {phase === 'done'
            ? <button className="cbtn" data-a="calib:continue">{usable ? 'Use my eyes' : 'Continue with pointer'}</button>
            : <button className="cbtn" data-a="calib:skip">Skip · use pointer</button>}
        </div>
        <div className="calib-perm mono">Camera and microphone stay on this device · lips, voice, blink and head run together</div>
      </section>
    </div>
  )
}
