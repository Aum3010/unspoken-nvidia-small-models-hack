import { Circle, Plus, Square, Trash2, UserRound } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'
import { getCamera, record, type Recording } from '../lib/capture'
import './calibrate.css'
import './phrases.css'

const TARGET = 3          // recordings per phrase
const MAX_MS = 4000       // one phrase per clip

interface Enrolled { phrase: string; count: number }

// Personal enrollment: record your own phrases; lip capture then matches against them.
// Recordings are kept only on this Mac as model features and are used only for that matching.
export function MyPhrases({ notify }: { notify: (t: string, k?: 'error' | 'info') => void }) {
  const video = useRef<HTMLVideoElement>(null)
  const [enrolled, setEnrolled] = useState<Enrolled[]>([])
  const [suggestions, setSuggestions] = useState<string[]>([])
  const [target, setTarget] = useState('')
  const [custom, setCustom] = useState('')
  const [rec, setRec] = useState<Recording | null>(null)
  const recRef = useRef<Recording | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    api.enrollList().then((r) => { setEnrolled(r.phrases); setSuggestions(r.suggestions) })
      .catch((e) => notify(`Lip reading unavailable: ${e.message}`, 'error'))
    getCamera().then((s) => { if (video.current) { video.current.srcObject = s; video.current.play() } })
      .catch(() => notify('Camera permission needed', 'error'))
  }, [notify])

  const count = (p: string) => enrolled.find((e) => e.phrase.toLowerCase() === p.toLowerCase())?.count ?? 0

  const start = async () => {
    if (!target) return
    const r = record(await getCamera(), 'video', MAX_MS)
    recRef.current = r
    setRec(r)
    setTimeout(() => { if (recRef.current === r) void finish(r) }, MAX_MS + 50)  // auto-stop
  }

  const finish = async (r: Recording) => {
    recRef.current = null
    setRec(null)
    setBusy(true)
    try {
      const res = await api.enrollAdd(await r.stop(), target)
      setEnrolled(res.phrases)
      notify(`Saved “${res.phrase}” (${res.count}/${TARGET})`)
    } catch (e: any) {
      notify(e.message, 'error')
    } finally {
      setBusy(false)
    }
  }

  const remove = async (p: string) => {
    try { setEnrolled((await api.enrollRemove(p)).phrases); if (target === p) setTarget('') } catch (e: any) { notify(e.message, 'error') }
  }

  const add = () => { const p = custom.trim(); if (p) { setTarget(p); setCustom('') } }
  const rows = [...enrolled.map((e) => e.phrase), ...(target && !count(target) ? [target] : [])]

  return (
    <div className="container calib">
      <div className="calib-grid">
        <div className="glass calib-card">
          <div className="calib-icon"><UserRound size={26} /></div>
          <h1 className="display">My phrases</h1>
          <p className="muted">
            Record each phrase {TARGET} times. Lip reading then matches what you mouth against <b>your own</b> recordings.
            They stay on this Mac and are used only for that.
          </p>
          <div className="calib-video"><video ref={video} muted playsInline /></div>
          <div className="enroll-target">{target ? <>Mouth: <b>“{target}”</b> <span className="faint">({count(target)}/{TARGET})</span></> : <span className="faint">Pick a phrase on the right</span>}</div>
          {rec
            ? <button className="btn lg danger" onClick={() => finish(rec)}><Square size={18} /> Stop</button>
            : <button className="btn lg primary" disabled={!target || busy} onClick={start}><Circle size={18} /> {busy ? 'Saving…' : 'Record'}</button>}
        </div>

        <div className="glass calib-card">
          <div className="panel-title">Recorded</div>
          {rows.length === 0 && <p className="faint">Nothing yet.</p>}
          <ul className="enroll-list">
            {rows.map((p) => (
              <li key={p} className={p === target ? 'on' : ''}>
                <button className="link grow" onClick={() => setTarget(p)}>{p}</button>
                <span className="mono faint">{'●'.repeat(Math.min(count(p), TARGET))}{'○'.repeat(Math.max(0, TARGET - count(p)))}</span>
                {count(p) > 0 && <button className="icon-btn" aria-label={`Delete ${p}`} onClick={() => remove(p)}><Trash2 size={15} /></button>}
              </li>
            ))}
          </ul>
          <div className="row" style={{ gap: 8 }}>
            <input className="input grow" placeholder="Add your own phrase" value={custom} onChange={(e) => setCustom(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && add()} />
            <button className="btn sm" onClick={add} aria-label="Add phrase"><Plus size={16} /></button>
          </div>
          <div className="panel-title">Suggestions</div>
          <div className="enroll-sugg">
            {suggestions.filter((s) => !count(s)).map((s) => <button key={s} className="chip" onClick={() => setTarget(s)}>{s}</button>)}
          </div>
        </div>
      </div>
    </div>
  )
}
