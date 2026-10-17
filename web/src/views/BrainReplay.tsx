import { AnimatePresence, motion } from 'framer-motion'
import { Brain, Check, FastForward, Info, Loader2, Pause, Play, Send, Trash2, X } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { LineChart } from '../components/Charts'
import { api } from '../lib/api'
import './brain.css'

const MATRIX = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789_'

interface Subject { subject: string; sex: string; age: string; alsfrs_r: string; onset: string }
interface Summary {
  subject: string; meta: Record<string, string>; channels: string[]
  models: Record<string, { test_auc: number; accuracy_by_reps: number[] }>
  erp: { times: number[]; target: number[][]; nontarget: number[][] }
  chars: { index: number; split: string }[]
}
interface Flash { i: number; t: number; code: number; score: number }
interface Trial {
  subject: string; char_index: number; split: string; model: string
  flashes: Flash[]
  decoded_by_reps: { char: string; row: number; col: number; margin: number }[]
  prediction: { char: string; row: number; col: number; margin: number }
  trace: { fs: number; channels: string[]; data: number[][] }
  ground_truth: { char: string }
}

export function BrainReplay({ notify, goCommunicate }: { notify: (m: string, k?: 'error' | 'info') => void; goCommunicate: () => void }) {
  const [subjects, setSubjects] = useState<Subject[]>([])
  const [subject, setSubject] = useState('A01')
  const [summary, setSummary] = useState<Summary | null>(null)
  const [char, setChar] = useState(15)
  const [trial, setTrial] = useState<Trial | null>(null)
  const [loading, setLoading] = useState(false)
  const [decoded, setDecoded] = useState<{ char: number; pred: string; truth: string }[]>([])

  useEffect(() => { api.eeg<{ subjects: Subject[] }>('subjects').then((r) => setSubjects(r.subjects)).catch((e) => notify(`EEG service: ${e.message}`, 'error')) }, [notify])
  useEffect(() => {
    setSummary(null); setTrial(null); setDecoded([])
    api.eeg<Summary>(`subjects/${subject}`).then(setSummary).catch((e) => notify(e.message, 'error'))
  }, [subject, notify])
  useEffect(() => {
    setLoading(true)
    api.eeg<Trial>(`subjects/${subject}/trial/${char}`).then(setTrial).catch((e) => notify(e.message, 'error')).finally(() => setLoading(false))
  }, [subject, char, notify])

  const send = async () => {
    try {
      await api.captureEeg(subject, decoded.map((d) => d.char))
      notify('Replay result sent to the communication panel (labelled RECORDED EEG)', 'info')
      goCommunicate()
    } catch (e: any) { notify(e.message, 'error') }
  }

  const prim = summary?.models.xdawn_riemann_lr
  const base = summary?.models.decimated_lda

  return (
    <div className="container brain">
      <div className="replay-banner">
        <Brain size={20} />
        <div className="grow">
          <b>EEG RESEARCH REPLAY</b> — previously recorded signal, newly run inference. No live brain input: there is no EEG headset attached.
        </div>
        <span className="badge" style={{ ['--c' as any]: 'var(--eeg)' }}>RECORDED EEG</span>
      </div>

      <div className="brain-grid">
        <aside className="glass panel">
          <div className="panel-title">BNCI2014-008 · ALS P300 speller</div>
          <div className="subjects">
            {subjects.map((s) => (
              <button key={s.subject} className="subj" aria-pressed={s.subject === subject} onClick={() => setSubject(s.subject)}>
                <span className="mono subj-id">{s.subject}</span>
                <span className="grow">
                  <span className="subj-line">{s.sex.toLowerCase()} · {s.age} y</span>
                  <span className="subj-sub">ALSFRS-R {s.alsfrs_r} · {s.onset.toLowerCase()} onset</span>
                </span>
              </button>
            ))}
            {!subjects.length && <div className="faint row"><Loader2 className="spin" size={16} /> loading participants…</div>}
          </div>
          <div className="source-note">
            <Info size={14} />
            <span>Riccio et al. 2013 · 8 ch, 256 Hz · 6×6 matrix, 4 Hz flashes, 10 repetitions · CC BY-NC-ND 4.0 — raw data is never redistributed.</span>
          </div>
        </aside>

        <section className="stack" style={{ gap: 18 }}>
          <div className="glass panel">
            <div className="panel-title">Choose a recorded character selection <span className="spacer" /><span className="faint" style={{ letterSpacing: 0, textTransform: 'none' }}>Labels stay hidden until the model has decoded it</span></div>
            <div className="chars">
              {Array.from({ length: 35 }, (_, i) => {
                const d = decoded.find((x) => x.char === i)
                return (
                  <button key={i} className={`charbtn ${i < 15 ? 'train' : ''} ${d ? (d.pred === d.truth ? 'ok' : 'bad') : ''}`}
                    aria-pressed={i === char} onClick={() => setChar(i)} title={i < 15 ? 'Calibration (training) character' : 'Held-out test character'}>
                    {d ? d.pred : i + 1}
                  </button>
                )
              })}
            </div>
            <div className="row wrap faint" style={{ fontSize: '.78rem', marginTop: 10, gap: 16 }}>
              <span><i className="legend-sw train" /> calibration (training) 1–15</span>
              <span><i className="legend-sw" /> held-out test 16–35</span>
            </div>
          </div>

          {trial && !loading ? (
            <Replay key={`${subject}-${char}`} trial={trial} onDecoded={(pred, truth) => setDecoded((d) => [...d.filter((x) => x.char !== char), { char, pred, truth }].sort((a, b) => a.char - b.char))}
              onNext={() => setChar((c) => Math.min(34, c + 1))} />
          ) : (
            <div className="glass panel replay-loading"><Loader2 className="spin" /> Loading recording and running inference…</div>
          )}

          <div className="glass panel">
            <div className="panel-title">Decoded selections <span className="spacer" />
              {decoded.length > 0 && <button className="link" onClick={() => setDecoded([])}><Trash2 size={13} /> clear</button>}
            </div>
            <div className="decoded-row">
              <div className="decoded-str display">{decoded.length ? decoded.map((d) => d.pred).join('') : <span className="faint">—</span>}</div>
              <button className="btn primary" disabled={!decoded.length} onClick={send}><Send size={18} /> Route to communication</button>
            </div>
            <div className="faint" style={{ fontSize: '.8rem' }}>Arrives as a candidate labelled “Recorded EEG replay result”. A recorded label is not live consent; an operator confirms read-out.</div>
          </div>
        </section>

        <aside className="stack" style={{ gap: 18 }}>
          <div className="glass panel">
            <div className="panel-title">Held-out performance · {subject}</div>
            {prim && base ? (
              <>
                <div className="kpis">
                  <div><span className="kpi-v">{prim.test_auc.toFixed(3)}</span><span className="kpi-l">flash AUC · xDAWN+Riemann</span></div>
                  <div><span className="kpi-v">{Math.round(prim.accuracy_by_reps[9] * 100)}%</span><span className="kpi-l">letters @ 10 reps (20 test)</span></div>
                </div>
                <LineChart x={[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]} height={190} yDomain={[0, 1]} xLabel="reps" yFmt={(v) => `${Math.round(v * 100)}%`}
                  series={[
                    { name: 'xDAWN + tangent space + LR', color: 'var(--eeg)', values: prim.accuracy_by_reps },
                    { name: 'Decimated shrinkage LDA', color: 'var(--text-3)', values: base.accuracy_by_reps, dashed: true },
                    { name: 'Chance (1/36)', color: 'var(--hairline-strong)', values: Array(10).fill(1 / 36), dashed: true },
                  ]} />
                <div className="faint" style={{ fontSize: '.78rem' }}>Trained only on calibration runs 1–3; scored on runs 4–7. Healthy-volunteer or ALS benchmark results are not a promise for a new person.</div>
              </>
            ) : <div className="faint row"><Loader2 className="spin" size={16} /> training on calibration runs…</div>}
          </div>
          {summary && (
            <div className="glass panel">
              <div className="panel-title">Grand-average ERP (calibration data)</div>
              {(['Cz', 'Pz'] as const).map((ch) => {
                const k = summary.channels.indexOf(ch)
                return (
                  <div key={ch}>
                    <div className="mono faint" style={{ fontSize: '.75rem' }}>{ch} · µV</div>
                    <LineChart x={summary.erp.times.map((t) => Math.round(t * 1000))} height={150} xLabel="ms" marker={300} xFmt={(v) => `${v}`}
                      series={[
                        { name: 'Target', color: 'var(--eeg)', values: summary.erp.target[k] },
                        { name: 'Non-target', color: 'var(--text-3)', values: summary.erp.nontarget[k] },
                      ]} />
                  </div>
                )
              })}
            </div>
          )}
        </aside>
      </div>
    </div>
  )
}

function Replay({ trial, onDecoded, onNext }: { trial: Trial; onDecoded: (pred: string, truth: string) => void; onNext: () => void }) {
  const [pos, setPos] = useState(0)              // number of flashes shown
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(2)
  const [revealed, setRevealed] = useState(false)
  const raf = useRef(0)
  const start = useRef<{ wall: number; t: number } | null>(null)
  const total = trial.flashes.length
  const done = pos >= total

  useEffect(() => {
    if (!playing) return
    const t0 = trial.flashes[Math.min(pos, total - 1)].t
    start.current = { wall: performance.now(), t: t0 }
    const loop = () => {
      const elapsed = ((performance.now() - start.current!.wall) / 1000) * speed + start.current!.t
      let n = pos
      while (n < total && trial.flashes[n].t <= elapsed) n++
      setPos(n)
      if (n >= total) { setPlaying(false); return }
      raf.current = requestAnimationFrame(loop)
    }
    raf.current = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, speed])

  useEffect(() => {
    if (done && !revealed) {
      const t = setTimeout(() => { setRevealed(true); onDecoded(trial.prediction.char, trial.ground_truth.char) }, 450)
      return () => clearTimeout(t)
    }
  }, [done, revealed, trial, onDecoded])

  const shown = trial.flashes.slice(0, pos)
  const current = pos > 0 && playing ? trial.flashes[pos - 1] : null
  const rowEv = useMemo(() => [1, 2, 3, 4, 5, 6].map((r) => shown.filter((f) => f.code === r).reduce((a, f) => a + f.score, 0)), [pos])
  const colEv = useMemo(() => [7, 8, 9, 10, 11, 12].map((c) => shown.filter((f) => f.code === c).reduce((a, f) => a + f.score, 0)), [pos])
  const bestR = rowEv.indexOf(Math.max(...rowEv)), bestC = colEv.indexOf(Math.max(...colEv))
  const rep = Math.min(10, Math.floor(pos / 12) + (pos % 12 ? 1 : 0))
  const lit = (r: number, c: number) => current && (current.code === r + 1 || current.code === c + 7)
  const norm = (v: number[]) => { const mn = Math.min(...v, 0), mx = Math.max(...v, 1e-6); return v.map((x) => (x - mn) / (mx - mn || 1)) }
  const rN = norm(rowEv), cN = norm(colEv)
  const correct = trial.prediction.char === trial.ground_truth.char

  return (
    <div className="glass panel replay">
      <div className="panel-title">
        Replay · character {trial.char_index + 1} · <span style={{ color: trial.split === 'held_out_test' ? 'var(--eeg)' : 'var(--warn)' }}>{trial.split === 'held_out_test' ? 'held-out test' : 'calibration (seen in training)'}</span>
        <span className="spacer" /><span className="mono" style={{ letterSpacing: 0 }}>rep {rep}/10 · flash {pos}/{total}</span>
      </div>
      <div className="replay-grid">
        <div className="matrix-wrap">
          <div className="matrix">
            {MATRIX.split('').map((ch, i) => {
              const r = Math.floor(i / 6), c = i % 6
              const isPred = revealed && r === trial.prediction.row && c === trial.prediction.col
              const isTruth = revealed && ch === trial.ground_truth.char
              const lead = pos > 24 && r === bestR && c === bestC && !revealed
              return (
                <div key={i} className={`cell ${lit(r, c) ? 'lit' : ''} ${lead ? 'lead' : ''} ${isPred ? 'pred' : ''} ${isTruth && !isPred ? 'truth' : ''}`}>
                  {ch === '_' ? '␣' : ch}
                </div>
              )
            })}
          </div>
          <div className="col-bars">{cN.map((v, i) => <span key={i} style={{ height: `${8 + v * 40}px`, opacity: .35 + v * .65 }} />)}</div>
          <div className="row-bars">{rN.map((v, i) => <span key={i} style={{ width: `${8 + v * 40}px`, opacity: .35 + v * .65 }} />)}</div>
        </div>
        <div className="replay-side">
          <Trace trial={trial} pos={pos} />
          <div className="row wrap">
            <button className="btn primary" onClick={() => (done ? null : setPlaying((p) => !p))} disabled={done}>
              {playing ? <Pause size={18} /> : done ? <Check size={18} /> : <Play size={18} />} {playing ? 'Pause' : done ? 'Decoded' : pos ? 'Resume' : 'Replay recording'}
            </button>
            <button className="btn" onClick={() => { setPlaying(false); setPos(total) }} disabled={done}><FastForward size={18} /> Decode now</button>
            <div className="seg">
              {[1, 2, 4].map((s) => <button key={s} aria-pressed={speed === s} onClick={() => setSpeed(s)}>{s}×</button>)}
            </div>
          </div>
          <div className="faint" style={{ fontSize: '.8rem' }}>
            Each flash is a recorded row/column intensification (125 ms on, 250 ms apart). Bars accumulate the classifier's per-flash P300 score — computed now, from the EEG alone.
          </div>
          <AnimatePresence>
            {revealed && (
              <motion.div className={`verdict ${correct ? 'ok' : 'bad'}`} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                <div className="verdict-top">
                  <div className="verdict-cell display">{trial.prediction.char}</div>
                  <div className="grow">
                    <div className="verdict-title">Decoded “{trial.prediction.char}”</div>
                    <div className={`verdict-truth ${correct ? 'ok' : 'bad'}`}>{correct ? <Check size={15} /> : <X size={15} />} truth “{trial.ground_truth.char}”</div>
                  </div>
                  <button className="btn sm" onClick={onNext}>Next</button>
                </div>
                <div className="mono faint verdict-meta">margin {trial.prediction.margin.toFixed(2)} · ground truth revealed after inference<br />decoded after 1…10 reps: {trial.decoded_by_reps.map((d) => d.char).join(' ')}</div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  )
}

function Trace({ trial, pos }: { trial: Trial; pos: number }) {
  const W = 520, H = 150
  const fs = trial.trace.fs
  const chans = ['Cz', 'Pz', 'PO7', 'PO8'].map((c) => trial.trace.channels.indexOf(c)).filter((i) => i >= 0)
  const tNow = pos ? trial.flashes[pos - 1].t + 0.6 : 0.6
  const win = 3.0
  const t0 = Math.max(0, tNow - win)
  const i0 = Math.floor(t0 * fs), i1 = Math.floor(tNow * fs)
  return (
    <svg className="trace-svg" viewBox={`0 0 ${W} ${H}`} width="100%" aria-label="Recorded EEG trace">
      {trial.flashes.filter((f) => f.t >= t0 && f.t <= tNow).map((f) => (
        <line key={f.i} x1={((f.t - t0) / win) * W} x2={((f.t - t0) / win) * W} y1={0} y2={H}
          stroke={f.code <= 6 ? 'var(--eeg)' : 'var(--ai)'} strokeOpacity={0.15 + Math.max(0, Math.min(1, f.score / 3)) * 0.6} />
      ))}
      {chans.map((ch, k) => {
        const d = trial.trace.data[ch].slice(i0, i1)
        const all = trial.trace.data[ch]
        const mean = all.reduce((a, b) => a + b, 0) / all.length
        const sd = Math.sqrt(all.reduce((a, b) => a + (b - mean) ** 2, 0) / all.length) || 1
        const band = H / chans.length
        const y0 = band * (k + 0.5)
        return (
          <g key={ch}>
            <path fill="none" stroke="var(--text-2)" strokeWidth="1.1" strokeLinejoin="round"
              d={d.map((v, i) => `${i ? 'L' : 'M'}${(((i / fs) / win) * W).toFixed(1)} ${(y0 - ((v - mean) / sd) * band * 0.16).toFixed(1)}`).join(' ')} />
            <text x={4} y={y0 - 8} fontSize="10" fill="var(--text-3)" fontFamily="var(--font-mono)">{trial.trace.channels[ch]}</text>
          </g>
        )
      })}
    </svg>
  )
}
