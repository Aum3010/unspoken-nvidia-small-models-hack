import { useEffect, useRef, useState } from 'react'
import {
  calibrationSampleCount, clearCalibration, gazeState, measureAccuracy, onGaze, recordCalibrationPoint, saveCalibration,
  setCalibrating, startGaze,
} from '../lib/gaze'
import { useSelection } from '../lib/selection'
import './card-calibration.css'

// Calibrate on the card positions themselves. The caller passes the exact card rectangles (from the
// same fixed geometry that places the cards), so calibration targets and cards cannot drift apart.
// Each card is spotlit in turn; the person only looks at its centre. Dwell is paused meanwhile.
// Afterwards accuracy is measured on every card: mean error (px) and how often the gaze was nearest
// the right card ("hit rate"), which is what matters with snapping.

const SETTLE_MS = 600, RECORD_MS = 1000, SAMPLE_MS = 50, TEST_MS = 1200

export type Card = { x: number; y: number; left: number; top: number; width: number; height: number }

export function CardCalibration({ cards, onDone }: { cards: Card[]; onDone: (r: { errorPx: number | null; hitRate: number | null }) => void }) {
  const sel = useSelection()
  const [spot, setSpot] = useState<Card | null>(null)
  const [label, setLabel] = useState('Getting ready…')
  const prevMode = useRef(sel.mode)

  useEffect(() => {
    let cancelled = false
    const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))
    const nearest = (x: number, y: number) =>
      cards.reduce((b, c, i) => (Math.hypot(x - c.x, y - c.y) < Math.hypot(x - cards[b].x, y - cards[b].y) ? i : b), 0)
    const run = async () => {
      sel.setSettings({ mode: 'pointer' })     // pause dwell while calibrating
      setCalibrating(true)
      await startGaze()
      clearCalibration()
      // two passes, the second in reverse order, so order and drift effects average out
      const seq = [...cards, ...[...cards].reverse()]
      for (let i = 0; i < seq.length && !cancelled; i++) {
        const c = seq[i]
        setSpot(c); setLabel(`Look at the dot · ${i + 1} of ${seq.length}`)
        await sleep(SETTLE_MS)
        const t0 = performance.now()
        while (performance.now() - t0 < RECORD_MS && !cancelled) { recordCalibrationPoint(c.x, c.y); await sleep(SAMPLE_MS) }
      }
      if (cancelled) return
      if (calibrationSampleCount() < seq.length * 5) { finish(null, null); return }
      // accuracy on every card, with the fitted model
      const errs: number[] = []
      let hits = 0, total = 0
      for (let i = 0; i < cards.length && !cancelled; i++) {
        const c = cards[i]
        setSpot(c); setLabel(`Checking · ${i + 1} of ${cards.length}`)
        await sleep(SETTLE_MS)
        const off = onGaze((p) => { if (p) { total++; if (nearest(p.x, p.y) === i) hits++ } })
        const e = await measureAccuracy({ x: c.x, y: c.y }, TEST_MS)
        off()
        if (e != null) errs.push(e)
      }
      finish(errs.length >= 3 ? errs.reduce((a, b) => a + b, 0) / errs.length : null, total ? hits / total : null)
    }
    const finish = (errorPx: number | null, hitRate: number | null) => {
      gazeState.accuracyPx = errorPx
      gazeState.calibrated = errorPx != null && errorPx < 220
      saveCalibration()
      setCalibrating(false)
      sel.setSettings({ mode: gazeState.calibrated ? 'gaze' : prevMode.current })
      onDone({ errorPx, hitRate })
    }
    run().catch(() => finish(null, null))
    return () => { cancelled = true; setCalibrating(false) }
  }, [])  // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="cardcal" aria-live="polite">
      {spot && (
        <div className="cardcal-spot" style={{ left: spot.left - 8, top: spot.top - 8, width: spot.width + 16, height: spot.height + 16 }}>
          {/* children of the spotlight are placed inside its 3px border, hence the -3 */}
          <span className="cardcal-dot" style={{ left: spot.x - spot.left + 8 - 3, top: spot.y - spot.top + 8 - 3 }} />
        </div>
      )}
      <div className={`cardcal-label${spot && spot.y > window.innerHeight / 2 ? ' top' : ''}`}>{label}</div>
    </div>
  )
}
