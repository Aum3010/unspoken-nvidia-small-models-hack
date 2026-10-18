// Eye calibration on its own plain, solid black screen. Shown when the app opens and from the Eyes chip.
// Gaze comes from MediaPipe iris + eyelid geometry and MobileGaze's eye-image angles, fitted to this
// person with a ridge regression (lib/gaze.ts). It is an estimate, so we measure its error on dots it
// was not fitted on, and only switch selection to the eyes when it is usable; otherwise the pointer
// stays the stand-in. A good calibration is saved on this device and offered next time.
// Live status (camera, face, eye model) and, on failure, the measured reason are shown, so a failing
// calibration says what is missing. Buttons are ordinary [data-a] targets: click, dwell or blink.

import { useEffect, useRef, useState } from 'react'
import { appearanceState } from '../lib/appearance'
import { bubbleCentres } from './engine'
import { onFace } from '../lib/face'
import { calibrationSampleCount, clearCalibration, eyesVisible, gazeDebug, gazeState, measureAccuracy, onGaze, recordCalibrationPoint, resetGazeDebug, saveCalibration, setCalibrating, startGaze } from '../lib/gaze'

const PHRASE_CARDS = 6   // the Phrases menu: its cards open in a 3 x 2 grid in the room
const DOT_SAMPLES = 12   // each dot stays until it has this many samples (slow eye model = fewer per second)...
const MAX_DOT_MS = 4500  // ...or this long, so a dot the eyes never reach cannot hold calibration up
const READY_MS = 600     // eyes tracked this long before the first dot (the camera and eye model warm up first)
const READY_TIMEOUT_MS = 6000

/**
 * Calibrate on the layout itself (as the card calibration in yash-gaze-v3 did): Yes/No, every dock tile, which
 * sits at the screen edge where a grid-only fit has to extrapolate, and the spots where the phrase cards open. One short
 * pass to spare a patient; selections made with the eyes keep refining it afterwards. Accuracy is tested on points that
 * were not trained on: the gaps between dock tiles and where four cards meet.
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
const START_TIMEOUT_MS = 20000   // slow USB cameras can take >10 s to open

type Phase = 'intro' | 'starting' | 'points' | 'measure' | 'done' | 'noface' | 'error'
type Light = 'ok' | 'bad' | 'wait'

interface Props {
  /** Routes this screen's [data-a="calib:*"] actions here. */
  bind: (fn: ((a: string) => void) | null) => void
  /** Hide the room's dwell targets while dots are on screen. */
  onFocusMode: (on: boolean) => void
  onDone: (useEyes: boolean) => void
}

function L({ s, label }: { s: Light; label: string }) {
  return <span><i className={s} />{label}</span>
}

/** Plain-language cause of a camera that would not start (getUserMedia error names). */
function cameraProblem(e: Error): string {
  if (e.message === 'camera timeout') return 'The camera did not start within 20 seconds. Close other apps that use it, then try again.'
  if (e.name === 'NotReadableError') return 'The camera is in use by another app or browser tab (only one can use it at a time). Close that, then try again.'
  if (e.name === 'NotAllowedError') return 'Camera access is blocked for this page. Allow it in the address bar, then try again.'
  if (e.name === 'NotFoundError' || e.name === 'OverconstrainedError') return 'No usable camera was found. Connect one, then try again.'
  return `The camera could not start (${e.name || 'error'}: ${e.message}).`
}

/** What a failed calibration was missing, from the frame counters in lib/gaze.ts. */
function failureReason(): string {
  const d = gazeDebug, n = Math.max(1, d.frames), pct = (k: number) => `${Math.round((k / n) * 100)}%`
  if (d.frames === 0) return 'No camera frames reached the face tracker. Check that no other app is using the camera, then try again.'
  if (d.noFace / n > 0.4) return `Your face was not found in ${pct(d.noFace)} of frames. Face the camera, with light in front of you.`
  if (d.noAppearance / n > 0.4) return `The eye-image model (MobileGaze, ${appearanceState.provider || 'not loaded'}, ${appearanceState.ms} ms per run) did not keep up in ${pct(d.noAppearance)} of frames.`
  if (d.blink / n > 0.4) return `Your eyes looked closed in ${pct(d.blink)} of frames. Open them a little wider, or add light.`
  if (d.noIris + d.noPose > 0.4 * n) return 'The tracker found a face but not the eye landmarks. Move a little closer to the camera.'
  return `Only ${d.samples} samples were recorded (${d.stale} missed between frames). Try again in steadier light.`
}

export function Calibration({ bind, onFocusMode, onDone }: Props) {
  const [phase, setPhase] = useState<Phase>('intro')
  const plan = useRef(layoutPoints())
  const [idx, setIdx] = useState(0)
  const [acc, setAcc] = useState<number | null>(gazeState.accuracyPx)
  const [gaze, setGaze] = useState<{ x: number; y: number } | null>(null)
  const [why, setWhy] = useState('')
  const [lights, setLights] = useState<{ camera: Light; face: Light }>({ camera: 'wait', face: 'wait' })
  const errors = useRef<number[]>([])
  const seen = useRef({ frame: 0, face: 0 })
  // a calibration saved on this device for this window size (lib/gaze.ts restores it on load)
  const [saved] = useState(() => gazeState.calibrated && gazeState.savedAt != null ? { acc: gazeState.accuracyPx ?? 0, at: gazeState.savedAt } : null)

  useEffect(() => onGaze((p) => setGaze(p)), [])
  useEffect(() => { onFocusMode(phase === 'points' || phase === 'measure') }, [phase, onFocusMode])
  useEffect(() => { document.body.classList.add('calibrating'); return () => document.body.classList.remove('calibrating') }, [])

  // live status: is the camera delivering frames, and is a face in them
  useEffect(() => {
    const off = onFace((f) => { seen.current.frame = f.t; if (f.present) seen.current.face = f.t })
    const id = window.setInterval(() => {
      const now = performance.now(), { frame, face } = seen.current
      setLights({ camera: frame && now - frame < 1000 ? 'ok' : frame ? 'bad' : 'wait', face: face && now - face < 1000 ? 'ok' : frame ? 'bad' : 'wait' })
    }, 400)
    return () => { off(); window.clearInterval(id) }
  }, [])

  const quality = acc == null ? 'none' : acc < 120 ? 'good' : acc < 220 ? 'fair' : 'poor'
  const usable = quality === 'good' || quality === 'fair'

  useEffect(() => {
    bind(async (a) => {
      if (a === 'start' && phase !== 'starting') {
        setPhase('starting')
        try {
          setCalibrating(true)   // pure gaze while calibrating and measuring: no head fine-tune
          // loads MobileGaze and waits for the camera (shared with the room, opened once)
          await Promise.race([startGaze(), new Promise((_, rej) => setTimeout(() => rej(new Error('camera timeout')), START_TIMEOUT_MS))])
          clearCalibration()
          resetGazeDebug()
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
        } catch (e) {
          setCalibrating(false)
          setWhy(cameraProblem(e as Error))
          setPhase('error')
        }
      } else if (a === 'saved' && saved) {
        try { await startGaze(); onDone(true) } catch { setPhase('error') }
      } else if (a === 'skip') onDone(false)
      else if (a === 'continue') onDone(usable)
    })
    return () => bind(null)
  }, [bind, phase, usable, onDone, saved])

  // hands-free: let the eyes settle on each dot, then record until it has enough samples;
  // the ring shows those samples coming in, not just time passing
  const [dotFill, setDotFill] = useState(0)
  useEffect(() => {
    if (phase !== 'points') return
    const seq = plan.current.sequence
    const [px, py] = seq[idx]
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
        if (idx + 1 < seq.length) setIdx(idx + 1)
        else if (calibrationSampleCount() < seq.length * 5) { setCalibrating(false); setAcc(null); setWhy(failureReason()); setPhase('noface') }
        else { setIdx(0); setPhase('measure') }
      }
    }, SAMPLE_MS)
    return () => window.clearInterval(iv)
  }, [phase, idx])

  // accuracy: mean error over unseen test points, each measured after the eyes settle
  useEffect(() => {
    if (phase !== 'measure') return
    if (idx === 0) errors.current = []
    const test = plan.current.test
    const [px, py] = test[idx]
    let cancelled = false
    const t = window.setTimeout(() => {
      measureAccuracy({ x: px * window.innerWidth, y: py * window.innerHeight }, TEST_MS).then((e) => {
        if (cancelled) return
        if (e != null) errors.current.push(e)
        if (idx + 1 < test.length) { setIdx(idx + 1); return }
        const a = errors.current.length >= 3 ? errors.current.reduce((x, y) => x + y, 0) / errors.current.length : null
        gazeState.accuracyPx = a
        gazeState.calibrated = a != null && a < 220
        saveCalibration()  // reused next time (same window size and camera position)
        setAcc(a)
        if (a == null) setWhy(failureReason())
        setCalibrating(false)
        setPhase('done')
      })
    }, SETTLE_MS)
    return () => { cancelled = true; window.clearTimeout(t) }
  }, [phase, idx])

  if (phase === 'points' || phase === 'measure') {
    const { sequence, test } = plan.current
    const [px, py] = phase === 'measure' ? test[idx] : sequence[idx]
    const lost = lights.face === 'bad'
    return (
      <div className="calib">
        <div key={phase + idx} className="calib-dot" style={{ left: `${px * 100}%`, top: `${py * 100}%` }}>
          <svg viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="44" style={phase === 'points'
              ? { animation: 'none', strokeDashoffset: 276.5 * (1 - dotFill), transition: 'stroke-dashoffset .15s linear' }
              : { animationDuration: `${SETTLE_MS + TEST_MS}ms` }} />
          </svg>
        </div>
        <div className={`calib-instr ${lost ? 'bad' : ''}`}>
          {lost ? 'Face not found · look toward the camera'
            : phase === 'measure' ? `Checking accuracy · look at the dot · ${idx + 1} of ${test.length}` : `Look at the dot · ${idx + 1} of ${sequence.length}`}
          {' · '}{gazeDebug.samples} samples
        </div>
        {gaze && <div className="calib-gaze" style={{ transform: `translate(${gaze.x}px, ${gaze.y}px)` }} />}
      </div>
    )
  }

  return (
    <div className="calib">
      <section className="calib-card" aria-label="Eye calibration">
        <h1>Set up your eyes</h1>
        <p>
          Unspoken follows where you look, so you can choose without touching anything. Gaze is estimated on this
          device from your iris and the eye image. It is an estimate, not clinical eye tracking: we tune it to you,
          measure its error, and only use it when it is good enough.
        </p>
        <div className="calib-status" aria-live="polite">
          {/* loading the eye model briefly blocks the page, so frames pause: not a camera fault */}
          {phase === 'starting' && lights.camera !== 'ok'
            ? <L s="wait" label="Loading eye model…" />
            : <L s={lights.camera} label={lights.camera === 'ok' ? 'Camera on' : lights.camera === 'wait' ? 'Camera starting' : 'Camera stopped'} />}
          <L s={lights.face} label={lights.face === 'ok' ? 'Face in view' : lights.face === 'wait' ? 'Looking for your face' : 'Face not found'} />
          <L s={appearanceState.ready ? 'ok' : appearanceState.error ? 'bad' : 'wait'}
            label={appearanceState.ready ? `Eye model on ${appearanceState.provider}` : appearanceState.error ? 'Eye model unavailable (iris only)' : 'Eye model loads at start'} />
        </div>
        {(phase === 'intro' || phase === 'starting') && (
          <ol>
            <li>Sit about an arm's length from the screen, face lit from the front.</li>
            <li>Look at each dot until its ring closes: on Yes and No, every button of the dock, then where each phrase card appears. No clicking needed (about 20 seconds).</li>
            <li>We then test accuracy on a few new dots.</li>
          </ol>
        )}
        {phase === 'intro' && saved && (
          <div className="calib-result good">
            <b>Saved calibration on this device</b>
            <small>≈{Math.round(saved.acc)} px · {new Date(saved.at).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })} · same seat and camera position?</small>
          </div>
        )}
        {phase === 'done' && (
          <div className={`calib-result ${usable ? 'good' : 'warn'}`}>
            <b>{quality === 'good' ? 'Your eyes are ready' : quality === 'fair' ? 'Usable, with some drift' : acc == null ? 'Accuracy could not be measured' : 'Too imprecise for gaze yet'}</b>
            <small>
              {acc != null ? `Mean error ≈ ${Math.round(acc)} px on new dots · ` : `${why} · `}
              {gazeState.appearance ? `MobileGaze on ${appearanceState.provider}` : 'iris geometry only'} · {gazeDebug.samples} samples
              {' · '}{usable ? 'rest your gaze to choose, blink to select' : 'the pointer stays the stand-in for your eyes'}
            </small>
          </div>
        )}
        {phase === 'noface' && (
          <div className="calib-result bad">
            <b>Calibration did not get enough eye samples</b>
            <small>{why}</small>
          </div>
        )}
        {phase === 'error' && (
          <div className="calib-result bad">
            <b>Gaze tracking could not start</b>
            <small>{why}</small>
          </div>
        )}
        <div className="row">
          {phase === 'intro' && saved && <button className="cbtn primary" data-a="calib:saved">Use saved calibration</button>}
          <button className={`cbtn ${phase === 'intro' && saved ? '' : 'primary'}`} data-a="calib:start" disabled={phase === 'starting'}>
            {phase === 'starting' ? 'Starting camera…' : phase === 'intro' ? (saved ? 'Calibrate again' : 'Start calibration') : 'Calibrate again'}
          </button>
          {phase === 'done'
            ? <button className="cbtn" data-a="calib:continue">{usable ? 'Use my eyes' : 'Continue with pointer'}</button>
            : <button className="cbtn" data-a="calib:skip">Skip · use pointer</button>}
        </div>
      </section>
    </div>
  )
}
