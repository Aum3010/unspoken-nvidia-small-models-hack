// Eye calibration in the glass style. Shown when the app opens and from the Eyes chip.
// Webcam gaze is an estimate, so we measure its error and only switch selection to the
// eyes when it is usable; otherwise the pointer stays the stand-in for gaze.
// Its buttons are ordinary [data-a] targets, so they work by click, dwell or blink.

import { useEffect, useRef, useState } from 'react'
import { bubbleCentres } from './engine'
import { calibrationSampleCount, clearCalibration, eyesVisible, gazeState, hasGazeModel, measureAccuracy, onGaze, recordCalibrationPoint, saveCalibration, setCalibrating, startGaze } from '../lib/gaze'

const PHRASE_CARDS = 6   // the Phrases menu: its cards open in a 3 x 2 grid in the room
// accuracy is tested on points that were not trained on, including the gaps between dock tiles
const DOT_SAMPLES = 12   // each dot stays until it has this many samples (slow eye model = fewer per second)...
const MAX_DOT_MS = 4500  // ...or this long, so a dot the eyes never reach cannot hold calibration up
const READY_MS = 600    // eyes tracked this long before the first dot (the camera and eye model warm up first)
const READY_TIMEOUT_MS = 6000

/**
 * Calibrate on the layout itself (as the card calibration in yash-gaze-v3 did): Yes/No, every dock tile, which
 * sits at the screen edge where a grid-only fit has to extrapolate, and the spots where the phrase cards open. One short
 * pass (about 16 s) to spare a patient; selections made with the eyes keep refining it afterwards.
 */
function layoutPoints() {
  const W = window.innerWidth, H = window.innerHeight
  const centre = (el: HTMLElement) => { const r = el.getBoundingClientRect(); return [(r.left + r.width / 2) / W, (r.top + r.height / 2) / H] }
  const tiles = [...document.querySelectorAll<HTMLElement>('.rail button .ic')].map(centre)
  const answers = [...document.querySelectorAll<HTMLElement>('.answers button .ic')].map(centre)
  const cards = bubbleCentres(PHRASE_CARDS).map(([x, y]) => [x / W, y / H])
  const gaps = tiles.slice(1).map((t, i) => [t[0], (t[1] + tiles[i][1]) / 2])
  const between = [(cards[0][0] + cards[1][0]) / 2, (cards[0][1] + cards[3][1]) / 2]   // where four cards meet
  return { sequence: [...answers, ...tiles, ...cards], test: [...gaps, between] }
}
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
  /** where clips are processed: the gateway's host label, null when it is this device */
  remoteHost?: string | null
}

export function Calibration({ bind, onFocusMode, onDone, remoteHost }: Props) {
  const [phase, setPhase] = useState<Phase>('intro')
  const plan = useRef(layoutPoints())
  const [idx, setIdx] = useState(0)
  const [acc, setAcc] = useState<number | null>(gazeState.accuracyPx)
  const [gaze, setGaze] = useState<{ x: number; y: number } | null>(null)

  useEffect(() => onGaze((p) => setGaze(p)), [])
  // live status while dots are shown, so it is visible whether the eyes are actually being sampled
  const [live, setLive] = useState({ eyes: false, samples: 0, model: false })
  useEffect(() => {
    if (phase !== 'points' && phase !== 'measure') return
    const iv = window.setInterval(() => setLive({ eyes: eyesVisible(), samples: calibrationSampleCount(), model: hasGazeModel() }), 200)
    return () => window.clearInterval(iv)
  }, [phase])
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
          plan.current = layoutPoints()
          // wait until the eyes are really being tracked: the first frames after start-up stall the page
          const t0 = performance.now()
          let since = 0
          while (performance.now() - t0 < READY_TIMEOUT_MS) {
            await new Promise((r) => setTimeout(r, 100))
            since = eyesVisible() ? (since || performance.now()) : 0
            if (since && performance.now() - since >= READY_MS) break
          }
          setIdx(0)
          setPhase('points')
        } catch {
          setCalibrating(false)
          setPhase('error')
        }
      } else if (a === 'saved') onDone(true)
      else if (a === 'skip') onDone(false)
      else if (a === 'continue') onDone(usable)
    })
    return () => bind(null)
  }, [bind, phase, usable, onDone])

  // hands-free calibration: let the eyes settle on each dot, then record until it has enough samples;
  // the ring shows those samples coming in, not just time passing
  const [dotFill, setDotFill] = useState(0)
  useEffect(() => {
    if (phase !== 'points') return
    const [px, py] = plan.current.sequence[idx]
    const x = px * window.innerWidth, y = py * window.innerHeight
    const t0 = performance.now(), n0 = calibrationSampleCount()
    setDotFill(0)
    const iv = window.setInterval(() => {
      const t = performance.now() - t0
      if (t < SETTLE_MS) return
      recordCalibrationPoint(x, y)
      const got = calibrationSampleCount() - n0
      setDotFill(Math.min(1, got / DOT_SAMPLES))
      if ((got >= DOT_SAMPLES && t >= SETTLE_MS + RECORD_MS) || t >= MAX_DOT_MS) {
        window.clearInterval(iv)
        if (idx + 1 < plan.current.sequence.length) setIdx(idx + 1)
        else if (calibrationSampleCount() < plan.current.sequence.length * 5) { setCalibrating(false); setAcc(null); setPhase('noface') }  // eyes were not seen
        else { setIdx(0); setPhase('measure') }
      }
    }, SAMPLE_MS)
    return () => window.clearInterval(iv)
  }, [phase, idx])

  // accuracy: mean error over unseen test points, each measured after the eyes settle
  const errors = useRef<number[]>([])
  useEffect(() => {
    if (phase !== 'measure') return
    if (idx === 0) errors.current = []
    const [px, py] = plan.current.test[idx]
    let cancelled = false
    const t = window.setTimeout(() => {
      measureAccuracy({ x: px * window.innerWidth, y: py * window.innerHeight }, TEST_MS).then((e) => {
        if (cancelled) return
        if (e != null) errors.current.push(e)
        if (idx + 1 < plan.current.test.length) { setIdx(idx + 1); return }
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
    const [px, py] = phase === 'measure' ? plan.current.test[idx] : plan.current.sequence[idx]
    return (
      <>
        <div className="calib" style={{ background: 'rgba(3,5,9,.55)' }} />
        <div key={phase + idx} className="calib-dot" style={{ left: `${px * 100}%`, top: `${py * 100}%` }}>
          <svg viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="44" style={phase === 'points'
              ? { animation: 'none', strokeDashoffset: 276.5 * (1 - dotFill), transition: 'stroke-dashoffset .15s linear' }
              : { animationDuration: `${SETTLE_MS + TEST_MS}ms` }} />
          </svg>
        </div>
        <div className="calib-instr glass mono">
          {phase === 'measure' ? `Checking accuracy · look at the dot · ${idx + 1} of ${plan.current.test.length}` : `Look at the dot · ${idx + 1} of ${plan.current.sequence.length}`}
          <span className={`calib-live ${live.eyes ? 'ok' : 'bad'}`}>
            {live.eyes ? 'eyes seen' : 'eyes not found'} · {live.samples} samples{live.model ? '' : ' · pointer after first dot'}
          </span>
        </div>
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
            <li>Look at each dot until its ring closes: on Yes and No, every button of the dock, then where each phrase card appears. No clicking needed (about 20 seconds).</li>
            <li>We then test accuracy on 3 new dots.</li>
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
          {phase === 'intro' && gazeState.calibrated && <button className="cbtn" data-a="calib:saved">Keep saved calibration · ≈{Math.round(gazeState.accuracyPx ?? 0)} px</button>}
          {phase === 'done'
            ? <button className="cbtn" data-a="calib:continue">{usable ? 'Use my eyes' : 'Continue with pointer'}</button>
            : <button className="cbtn" data-a="calib:skip">Skip · use pointer</button>}
        </div>
        <div className="calib-perm mono">{remoteHost
          ? `Mouth and voice clips are processed on the ${remoteHost} GPU server, not stored · eye calibration stays on this device`
          : 'Camera and microphone stay on this device'} · lips, voice, blink and head run together</div>
      </section>
    </div>
  )
}
