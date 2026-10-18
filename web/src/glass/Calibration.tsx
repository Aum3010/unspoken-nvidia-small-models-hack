// Eye calibration in the glass style. Shown when the app opens and from the Eyes chip.
// Webcam gaze is an estimate, so we measure its error and only switch selection to the
// eyes when it is usable; otherwise the pointer stays the stand-in for gaze.
// Its buttons are ordinary [data-a] targets, so they work by click, dwell or blink.

import { useEffect, useState } from 'react'
import { clearCalibration, gazeState, measureAccuracy, onGaze, recordCalibrationPoint, startGaze } from '../lib/gaze'

const POINTS = [[0.1, 0.12], [0.5, 0.12], [0.9, 0.12], [0.1, 0.5], [0.5, 0.5], [0.9, 0.5], [0.1, 0.88], [0.5, 0.88], [0.9, 0.88]]
const HOLD_MS = 2200
const MEASURE_MS = 2600

type Phase = 'intro' | 'starting' | 'points' | 'measure' | 'done' | 'error'

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
          clearCalibration()
          await startGaze()
          setIdx(0)
          setPhase('points')
        } catch {
          setPhase('error')
        }
      } else if (a === 'skip') onDone(false)
      else if (a === 'continue') onDone(usable)
    })
    return () => bind(null)
  }, [bind, phase, usable, onDone])

  // hands-free: hold each dot while samples are recorded
  useEffect(() => {
    if (phase !== 'points') return
    const [px, py] = POINTS[idx]
    const x = px * window.innerWidth, y = py * window.innerHeight
    const iv = window.setInterval(() => recordCalibrationPoint(x, y), 90)
    const t = window.setTimeout(() => {
      window.clearInterval(iv)
      if (idx + 1 < POINTS.length) setIdx(idx + 1)
      else setPhase('measure')
    }, HOLD_MS)
    return () => { window.clearInterval(iv); window.clearTimeout(t) }
  }, [phase, idx])

  useEffect(() => {
    if (phase !== 'measure') return
    measureAccuracy({ x: window.innerWidth / 2, y: window.innerHeight / 2 }, MEASURE_MS).then((a) => {
      gazeState.accuracyPx = a
      gazeState.calibrated = a != null && a < 220
      setAcc(a)
      setPhase('done')
    })
  }, [phase])

  if (phase === 'points' || phase === 'measure') {
    const [px, py] = phase === 'measure' ? [0.5, 0.5] : POINTS[idx]
    return (
      <>
        <div className="calib" style={{ background: 'rgba(3,5,9,.55)' }} />
        <div key={phase + idx} className="calib-dot" style={{ left: `${px * 100}%`, top: `${py * 100}%` }}>
          <svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="44" style={{ animationDuration: `${phase === 'measure' ? MEASURE_MS : HOLD_MS}ms` }} /></svg>
        </div>
        <div className="calib-instr glass mono">{phase === 'measure' ? 'Keep looking at the centre · measuring accuracy' : `Look at the dot · ${idx + 1} of ${POINTS.length}`}</div>
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
            <li>Look at each of the 9 dots until its ring closes. No clicking needed.</li>
            <li>We then test accuracy at the centre.</li>
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
        {phase === 'error' && (
          <div className="calib-result" style={{ ['--c' as string]: '255,128,112' }}>
            <span className="pebble" />
            <div><b>Gaze tracking could not start</b><small>Allow the camera for this page, then try again.</small></div>
          </div>
        )}
        <div className="row">
          <button className="cbtn primary" data-a="calib:start" disabled={phase === 'starting'}>
            {phase === 'starting' ? 'Starting camera…' : phase === 'done' || phase === 'error' ? 'Calibrate again' : 'Start calibration'}
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
