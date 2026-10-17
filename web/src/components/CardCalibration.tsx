import { useEffect, useRef, useState } from 'react'
import {
  calibrationSampleCount, clearCalibration, gazeState, measureAccuracy, recordCalibrationPoint, saveCalibration,
  setCalibrating, startGaze,
} from '../lib/gaze'
import { useSelection } from '../lib/selection'
import './card-calibration.css'

// Calibrate on the cards themselves: the layout is fixed, so the gaze model is fitted exactly at the
// card centres the person will look at (the regression is most accurate at its calibration points),
// and afterwards the pointer snaps to the nearest card centre. Each card is spotlit in turn; the
// person only has to look at it. Dwell selection is paused meanwhile so nothing gets activated.

const SETTLE_MS = 600, RECORD_MS = 1000, SAMPLE_MS = 50, TEST_MS = 1200, TESTS = 4

type Spot = { x: number; y: number; r: DOMRect }

export function CardCalibration({ onDone }: { onDone: (accuracyPx: number | null) => void }) {
  const sel = useSelection()
  const [spot, setSpot] = useState<Spot | null>(null)
  const [label, setLabel] = useState('Getting ready…')
  const prevMode = useRef(sel.mode)

  useEffect(() => {
    let cancelled = false
    const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))
    const run = async () => {
      sel.setSettings({ mode: 'pointer' })     // pause dwell while calibrating
      setCalibrating(true)
      await startGaze()
      clearCalibration()
      const spots: Spot[] = sel.targetRects().map((r) => ({ x: r.left + r.width / 2, y: r.top + r.height / 2, r }))
      if (spots.length < 4) { finish(null); return }
      // two passes over small layouts (second in reverse), one over large ones
      const seq = spots.length <= 12 ? [...spots, ...[...spots].reverse()] : spots
      for (let i = 0; i < seq.length && !cancelled; i++) {
        const s = seq[i]
        setSpot(s); setLabel(`Look at the highlighted card · ${i + 1} of ${seq.length}`)
        await sleep(SETTLE_MS)
        const t0 = performance.now()
        while (performance.now() - t0 < RECORD_MS && !cancelled) { recordCalibrationPoint(s.x, s.y); await sleep(SAMPLE_MS) }
      }
      if (cancelled) return
      if (calibrationSampleCount() < seq.length * 5) { finish(null); return }
      // accuracy on a spread of cards, measured with the fitted model
      const step = Math.max(1, Math.floor(spots.length / TESTS))
      const tests = spots.filter((_, i) => i % step === 0).slice(0, TESTS)
      const errs: number[] = []
      for (let i = 0; i < tests.length && !cancelled; i++) {
        setSpot(tests[i]); setLabel(`Checking accuracy · ${i + 1} of ${tests.length}`)
        await sleep(SETTLE_MS)
        const e = await measureAccuracy({ x: tests[i].x, y: tests[i].y }, TEST_MS)
        if (e != null) errs.push(e)
      }
      finish(errs.length >= 2 ? errs.reduce((a, b) => a + b, 0) / errs.length : null)
    }
    const finish = (acc: number | null) => {
      gazeState.accuracyPx = acc
      gazeState.calibrated = acc != null && acc < 220
      saveCalibration()
      setCalibrating(false)
      sel.setSettings({ mode: prevMode.current === 'pointer' ? 'gaze' : prevMode.current, cardSnap: gazeState.calibrated })
      onDone(acc)
    }
    run().catch(() => finish(null))
    return () => { cancelled = true }
  }, [])  // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="cardcal" aria-live="polite">
      {spot && (
        <div className="cardcal-spot" style={{ left: spot.r.left - 8, top: spot.r.top - 8, width: spot.r.width + 16, height: spot.r.height + 16 }}>
          <span className="cardcal-dot" style={{ left: spot.x - spot.r.left + 8, top: spot.y - spot.r.top + 8 }} />
        </div>
      )}
      <div className={`cardcal-label${spot && spot.y > window.innerHeight * 0.6 ? ' top' : ''}`}>{label}</div>
    </div>
  )
}
