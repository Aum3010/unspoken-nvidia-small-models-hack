// Recording page (/?record): the person records each phrase a few times; the lip agent then matches new
// mouth clips against these recordings (services/vsr/enroll.py). Clips are cut by the same mouth-motion
// detector the room uses (sensors.ts startLips), so recordings and live clips are segmented alike.
// Only Auto-AVSR encoder features are stored, on the machine running the lip agent; no video is kept.

import { useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'
import { getCamera } from '../lib/capture'
import { startLips } from './sensors'

type Item = { phrase: string; count: number }
const GOAL = 8   // takes per phrase to aim for; more takes is what raises accuracy

export function MyPhrases({ onClose }: { onClose: () => void }) {
  const [items, setItems] = useState<Item[]>([])
  const [suggestions, setSuggestions] = useState<string[]>([])
  const [negatives, setNegatives] = useState<string[]>([])
  const [neg, setNeg] = useState('[unknown] ')
  const [custom, setCustom] = useState('')
  const [target, setTarget] = useState<string | null>(null)
  const [status, setStatus] = useState<{ text: string; tone?: 'good' | 'warn' } | null>(null)
  const [mouthing, setMouthing] = useState(false)
  const [error, setError] = useState('')
  const video = useRef<HTMLVideoElement>(null)
  const busy = useRef(false)

  const isNeg = (p: string) => p.startsWith(neg)
  const show = (p: string) => (isNeg(p) ? p.slice(neg.length) : p)
  const count = (p: string) => items.find((i) => i.phrase.toLowerCase() === p.toLowerCase())?.count ?? 0

  useEffect(() => {
    api.enrollList()
      .then((r) => { setItems(r.phrases); setSuggestions(r.suggestions); setNegatives(r.negatives); setNeg(r.negative_prefix); setError('') })
      .catch((e: Error) => setError(e.message || 'Lip agent unavailable'))
    getCamera().then((s) => { if (video.current) { video.current.srcObject = s; video.current.play().catch(() => undefined) } }).catch(() => undefined)
  }, [])

  // while a phrase is chosen: every mouthed clip is one recording of it
  useEffect(() => {
    if (!target) return
    let stop = () => {}
    let alive = true
    getCamera().then((stream) => {
      if (!alive) return
      stop = startLips({
        stream,
        canCapture: () => !busy.current,
        onMouthing: setMouthing,
        onClip: (blob) => {
          busy.current = true
          setStatus({ text: 'Saving…' })
          api.enrollAdd(blob, target)
            .then((r) => {
              setItems(r.phrases)
              setStatus({ text: `Saved take ${r.count}${r.count < GOAL ? ` of ${GOAL}. Mouth it again.` : '. Pick another phrase, or keep going.'}`, tone: 'good' })
            })
            .catch((e: Error) => setStatus({ text: `${e.message || 'Not saved'}. Try again.`, tone: 'warn' }))
            .finally(() => { setTimeout(() => { busy.current = false }, 600) })
        },
      })
    })
    return () => { alive = false; stop(); setMouthing(false) }
  }, [target])

  const choose = (p: string) => {
    const phrase = p.trim()
    if (!phrase) return
    setTarget(phrase)
    setStatus({ text: `Mouth “${show(phrase)}” silently, then hold still for a moment.` })
  }
  const remove = (p: string) => {
    if (!confirm(`Delete all ${count(p)} recordings of “${show(p)}”?`)) return
    api.enrollRemove(p).then((r) => setItems(r.phrases)).catch(() => undefined)
  }

  // the 20 phrases and the negatives always show, with their take counts; fewest takes first is easiest to follow
  const row = (p: string) => {
    const n = count(p)
    return (
      <li key={p} className={target === p ? 'on' : ''}>
        <button onClick={() => choose(p)}>{show(p)}</button>
        <span className={`mono ${n >= GOAL ? 'ok' : n ? '' : 'none'}`}>{n}/{GOAL}</span>
        {n > 0 && <button className="x" aria-label={`Delete ${show(p)}`} onClick={() => remove(p)}>×</button>}
      </li>
    )
  }
  const known = new Set([...suggestions, ...negatives.map((s) => neg + s)].map((s) => s.toLowerCase()))
  const extra = items.filter((i) => !known.has(i.phrase.toLowerCase())).map((i) => i.phrase)
  const total = items.reduce((a, i) => a + i.count, 0)

  return (
    <div className="calib myp" role="dialog" aria-label="Record phrases">
      <section className="calib-card myp-card">
        <div className="myp-head">
          <h1>Record phrases</h1>
          <button className="cbtn" onClick={onClose}>Done</button>
        </div>
        <p>Pick a phrase and mouth it silently, the way you usually would. Each mouthed clip is saved as one take. Aim for {GOAL} takes per phrase, starting with the ones that look alike. {total} takes recorded.</p>
        {error && <div className="calib-result bad"><b>Lip agent unavailable</b><small>{error}</small></div>}

        <div className="myp-rec">
          <video ref={video} muted playsInline className={mouthing ? 'on' : ''} />
          <div>
            <b>{target ? `${isNeg(target) ? 'Must become unknown: ' : ''}“${show(target)}”` : 'Choose a phrase below'}</b>
            <small className={status?.tone ?? ''}>{mouthing ? 'Recording…' : status?.text ?? 'Face the camera with light on your face.'}</small>
            {target && <button className="cbtn" onClick={() => { setTarget(null); setStatus(null) }}>Stop</button>}
          </div>
        </div>

        <h2>The 20 phrases</h2>
        <ul className="myp-list">{[...suggestions].sort((a, b) => count(a) - count(b)).map(row)}</ul>

        <h2>Must become unknown</h2>
        <p className="myp-note">Near-misses look almost the same on the lips. Recording them teaches the matcher to answer “unknown” instead of the phrase they resemble.</p>
        <ul className="myp-list">{negatives.map((s) => row(neg + s))}</ul>

        {extra.length > 0 && <>
          <h2>Other recordings</h2>
          <ul className="myp-list">{extra.map(row)}</ul>
        </>}

        <h2>Add your own</h2>
        <form className="myp-add" onSubmit={(e) => { e.preventDefault(); choose(custom); setCustom('') }}>
          <input value={custom} onChange={(e) => setCustom(e.target.value)} placeholder="Type a phrase" />
          <button className="cbtn primary" disabled={!custom.trim()}>Record</button>
        </form>
      </section>
    </div>
  )
}
