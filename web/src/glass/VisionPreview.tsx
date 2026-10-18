// Vision preview (UNSPOKEN_CHECKIN=1 on the gateway only): recorded ICU data from the MIMIC-IV demo next to a
// SYNTHETIC profile, and a button (or V) that starts a proactive check-in. Rendered only when the gateway reports
// the flag; the answer goes to the nurse, never to speech.
import { useEffect, useState } from 'react'
import { api, type VisionData } from '../lib/api'

function Spark({ points, color, lo, hi }: { points: { t_min: number; value: number }[]; color: string; lo: number; hi: number }) {
  if (points.length < 2) return <svg className="spark" viewBox="0 0 120 28" />
  const t0 = points[0].t_min, t1 = points[points.length - 1].t_min || 1
  const xy = points.map((p) => `${((p.t_min - t0) / (t1 - t0 || 1)) * 118 + 1},${27 - ((p.value - lo) / (hi - lo)) * 26}`)
  return <svg className="spark" viewBox="0 0 120 28"><polyline points={xy.join(' ')} fill="none" stroke={`rgb(${color})`} strokeWidth="1.6" /></svg>
}

export function VisionPreview() {
  const [data, setData] = useState<VisionData | null>(null)
  // a failure here must never take the page down: the panel just shows no data
  useEffect(() => { Promise.resolve().then(() => api.visionData()).then(setData).catch(() => setData(null)) }, [])
  const ev = data?.event
  const hr = ev?.heart_rate.points ?? [], spo2 = ev?.spo2.points ?? []
  const last = (ps: { value: number }[]) => (ps.length ? ps[ps.length - 1].value : null)
  return (
    <section className="vision glass" aria-label="Vision preview">
      {/* the dataset stays named (MIMIC-IV demo, ODbL); the footer says the profile is synthetic */}
      <header><div>Vision preview</div><span className="mono">Recorded ICU data (MIMIC-IV)</span></header>
      <div className="vgreet">Good morning, {data?.profile?.preferred_name ?? 'Raj'}</div>
      {ev ? (
        <div className="vrows">
          <div><span>Glucose (charted)</span><b className="vlow">{ev.glucose.value} {ev.glucose.unit}</b></div>
          <div><span>Heart rate · {ev.window_hours} h</span><Spark points={hr} color="255,128,112" lo={40} hi={120} /><b>{last(hr) ?? '–'} bpm</b></div>
          <div><span>SpO2 · {ev.window_hours} h</span><Spark points={spo2} color="94,200,255" lo={85} hi={100} /><b>{last(spo2) ?? '–'} %</b></div>
        </div>
      ) : <div className="vrows">no recorded event on the server (scripts/vision/fetch_mimic_event.py)</div>}
      <div className="vfoot">
        <button className="cbtn" data-a="vision:checkin">Start check-in · V</button>
        <span className="mono">{data?.profile?.synthetic ? 'profile: SYNTHETIC' : ''}</span>
      </div>
    </section>
  )
}
