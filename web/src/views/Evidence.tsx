import { AlertTriangle, BookOpen, Cpu, ExternalLink, FlaskConical, ShieldCheck } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import './evidence.css'

const pct = (v?: number | null, d = 0) => (v == null ? '—' : `${(v * 100).toFixed(d)}%`)

const SOURCES: [string, string, string][] = [
  ['Chaplin (VSR pipeline, Auto-AVSR wrapper)', 'https://github.com/amanvirparhar/chaplin', 'MIT'],
  ['Auto-AVSR', 'https://github.com/mpc001/auto_avsr', 'Apache-2.0 (code); model per dataset terms'],
  ['MobileGaze (MobileOne-S0 gaze model)', 'https://github.com/yakhyo/gaze-estimation', 'MIT'],
  ['ONNX Runtime Web', 'https://github.com/microsoft/onnxruntime', 'MIT'],
  ['MediaPipe Face Landmarker', 'https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker', 'Apache-2.0'],
  ['Whisper (MLX port)', 'https://github.com/openai/whisper', 'MIT'],
  ['RNNoise', 'https://github.com/xiph/rnnoise', 'BSD-3'],
  ['Kokoro-82M', 'https://huggingface.co/hexgrad/Kokoro-82M', 'Apache-2.0'],
  ['NVIDIA Nemotron 3.5 Lightning', 'https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4', 'OpenMDW-1.1'],
  ['NeMo Gym', 'https://github.com/NVIDIA-NeMo/gym', 'Apache-2.0'],
  ['P300 Speller BCI (pipeline reference)', 'https://github.com/weichkai/p300-speller-bci', 'MIT'],
  ['BNCI2014-008 ALS P300 (Riccio et al. 2013)', 'https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_008.html', 'CC BY-NC-ND 4.0'],
  ['GRID audiovisual corpus', 'https://zenodo.org/records/3625687', 'CC BY 4.0'],
  ['pyRiemann / MNE / MOABB', 'https://moabb.neurotechx.com/', 'BSD-3'],
  ['Personalized one-shot lipreading for an ALS patient (motivation)', 'https://arxiv.org/abs/2111.01740', 'link only'],
  ['VSR for ICU patients (motivation)', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC9844948/', 'link only'],
]

export function Evidence() {
  const [ev, setEv] = useState<any>(null)
  const [ledger, setLedger] = useState<any>(null)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => {
    api.evals().then(setEv).catch((e) => setErr(e.message))
    api.ledger().then(setLedger).catch(() => undefined)
  }, [])

  const total = ledger?.total_b ?? ledger?.components.reduce((a: number, c: any) => a + (c.params_b ?? 0), 0) ?? 0
  const color = (name: string) => /Coordinator/.test(name) ? 'var(--ai)' : /Visual|VSR|Mouth/.test(name) ? 'var(--lip)'
    : /ASR|enhancement/.test(name) ? 'var(--audio)' : /TTS/.test(name) ? 'var(--voice)' : /Gaze|Face/.test(name) ? 'var(--gaze)' : 'var(--eeg)'

  return (
    <div className="container evidence">
      <header className="ev-head">
        <div className="eyebrow">Evaluation</div>
        <h1 className="display">Measured, not promised.</h1>
        <p className="muted">Separate baselines per modality and for the coordinated system, on held-out data the models never trained on. Failures are kept and shown. Benchmark numbers are not real-patient accuracy.</p>
        {ev?.generated_at && <div className="mono faint">results generated {new Date(ev.generated_at).toLocaleString()}</div>}
        {err && <div className="ev-warn"><AlertTriangle size={16} /> No evaluation results yet — run <code className="mono">./scripts/run_evals.sh</code> ({err})</div>}
      </header>

      {ledger && (
        <section className="glass panel">
          <div className="panel-title"><Cpu size={14} /> Distinct deployed weights · ≤ {ledger.budget_b}B budget</div>
          <div className="budget">
            <div className="budget-bar">
              {ledger.components.filter((c: any) => c.params_b).map((c: any) => (
                <span key={c.component} style={{ width: `${(c.params_b / ledger.budget_b) * 100}%`, background: color(c.component) }} title={`${c.component}: ${c.params_b}B`} />
              ))}
            </div>
            <div className="budget-num"><span className="display">{total.toFixed(2)}B</span><span className="faint"> of {ledger.budget_b}B · {ledger.verified ? 'counted from weight files on disk · ' : ''}MoE total parameters counted, quantisation does not reduce count, shared coordinator weights counted once</span></div>
          </div>
          <table className="tbl">
            <thead><tr><th>Component</th><th>Model</th><th className="num">Params</th><th>Source / note</th></tr></thead>
            <tbody>
              {ledger.components.map((c: any) => (
                <tr key={c.component}><td>{c.component}</td><td>{c.model}</td><td className="num mono">{c.params_b == null ? 'verify' : c.params_b >= 0.01 ? `${c.params_b.toFixed(3)}B` : `${(c.params_b * 1000).toFixed(2)}M`}</td><td className="mono faint">{c.source}{c.note ? ` · ${c.note}` : ''}</td></tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {ev?.coordinator && <CoordinatorSection c={ev.coordinator} />}

      <div className="ev-grid">
        {ev?.eeg && (
          <section className="glass panel">
            <div className="panel-title"><FlaskConical size={14} /> EEG replay · BNCI2014-008 · 8 ALS participants</div>
            <table className="tbl">
              <thead><tr><th>Participant</th><th className="num">AUC</th><th className="num">1 rep</th><th className="num">5 reps</th><th className="num">10 reps</th><th className="num">LDA 10</th></tr></thead>
              <tbody>
                {ev.eeg.subjects.map((s: any) => {
                  const m = s.models.xdawn_riemann_lr, b = s.models.decimated_lda
                  return (
                    <tr key={s.subject}><td className="mono">{s.subject} <span className="faint">ALSFRS-R {s.meta.alsfrs_r}</span></td>
                      <td className="num mono">{m.test_auc.toFixed(3)}</td><td className="num mono">{pct(m.accuracy_by_reps[0])}</td>
                      <td className="num mono">{pct(m.accuracy_by_reps[4])}</td><td className="num mono strong">{pct(m.accuracy_by_reps[9])}</td><td className="num mono faint">{pct(b.accuracy_by_reps[9])}</td></tr>
                  )
                })}
              </tbody>
              <tfoot><tr><td>Mean</td><td className="num mono">{ev.eeg.mean.auc.toFixed(3)}</td><td className="num mono">{pct(ev.eeg.mean.acc1)}</td><td className="num mono">{pct(ev.eeg.mean.acc5)}</td><td className="num mono strong">{pct(ev.eeg.mean.acc10)}</td><td className="num mono faint">{pct(ev.eeg.mean.lda10)}</td></tr></tfoot>
            </table>
            <p className="note">Train: 15 calibration characters (runs 1–3). Test: 20 online characters (runs 4–7) — no repeated flashes of a test character leak into training. Chance = 2.8%.</p>
          </section>
        )}

        {ev?.vsr && (
          <section className="glass panel">
            <div className="panel-title"><FlaskConical size={14} /> Lip reading · {ev.vsr.dataset}</div>
            <div className="kpi-row">
              <Kpi v={pct(ev.vsr.wer)} l="top-1 WER" />
              <Kpi v={pct(ev.vsr.oracle_wer)} l="best-of-n-best WER" />
              <Kpi v={pct(ev.vsr.cer)} l="top-1 CER" />
              <Kpi v={`${ev.vsr.latency_ms_median} ms`} l="median latency" />
            </div>
            <p className="note">{ev.vsr.n_clips} clips from {ev.vsr.speakers.join(', ')} — speakers and grammar never seen by the LRS3-trained model. Paper LRS3 WER (19–20%) is not our webcam accuracy; this is why candidates are confirmed, never asserted.</p>
            {ev.vsr.still && <p className="note"><b>Still-face test:</b> {ev.vsr.still.n} clips with no articulation → flagged “little mouth movement” {pct(ev.vsr.still.flag_rate)} · unflagged fluent phrase {pct(ev.vsr.still.false_phrase_rate)}.</p>}
            <Examples rows={ev.vsr.examples} />
          </section>
        )}

        {ev?.audio && (
          <section className="glass panel">
            <div className="panel-title"><FlaskConical size={14} /> Weak speech · raw vs enhanced</div>
            <table className="tbl">
              <thead><tr><th>Condition</th><th className="num">WER raw</th><th className="num">WER enhanced</th><th className="num">Fabricated (raw/enh)</th></tr></thead>
              <tbody>
                {ev.audio.conditions.map((c: any) => (
                  <tr key={c.name}><td>{c.name}</td><td className="num mono">{pct(c.wer_raw)}</td><td className="num mono">{pct(c.wer_enhanced)}</td><td className="num mono">{c.fabricated_raw} / {c.fabricated_enh}</td></tr>
                ))}
              </tbody>
            </table>
            <p className="note">{ev.audio.note}</p>
            {ev.audio.examples?.length > 0 && (
              <div className="examples">
                {ev.audio.examples.slice(0, 4).map((x: any, i: number) => (
                  <div key={i} className="example">
                    <div><span className="faint">{x.condition} · truth</span> {x.truth}</div>
                    <div><span className="faint">raw</span> {x.raw || '∅'} <span className="faint">· enhanced</span> <span className="bad">{x.enhanced || '∅'}</span></div>
                  </div>
                ))}
              </div>
            )}
          </section>
        )}

        {ev?.safety && (
          <section className="glass panel">
            <div className="panel-title"><ShieldCheck size={14} /> Confirmation safety</div>
            <div className="kpi-row">
              <Kpi v={`${ev.safety.passed}/${ev.safety.total}`} l="policy tests passed" />
              <Kpi v={String(ev.safety.unsafe_assertions)} l="unconfirmed text reaching TTS" good />
              <Kpi v={pct(ev.safety.exactness)} l="spoken = confirmed string" good />
            </div>
            <ul className="tests">{ev.safety.cases.map((c: any) => <li key={c.name} className={c.passed ? 'ok' : 'bad'}>{c.passed ? '✓' : '✗'} {c.name}</li>)}</ul>
          </section>
        )}
      </div>

      <section className="glass panel">
        <div className="panel-title"><BookOpen size={14} /> Sources, licences & attribution</div>
        <div className="sources">
          {SOURCES.map(([n, u, l]) => (
            <a key={u} href={u} target="_blank" rel="noreferrer" className="source"><span className="grow">{n}</span><span className="mono faint">{l}</span><ExternalLink size={14} /></a>
          ))}
        </div>
        <p className="note">Patient videos found online are linked for motivation only — never downloaded, cropped, or trained on. Product performance is measured on public datasets and opted-in local recordings.</p>
      </section>
    </div>
  )
}

function Kpi({ v, l, good }: { v: string; l: string; good?: boolean }) {
  return <div className={`kpi ${good ? 'good' : ''}`}><span className="display">{v}</span><span>{l}</span></div>
}

function Examples({ rows }: { rows?: any[] }) {
  if (!rows?.length) return null
  return (
    <div className="examples">
      {rows.slice(0, 6).map((r, i) => (
        <div key={i} className="example">
          <div><span className="faint">truth</span> {r.truth}</div>
          <div><span className="faint">top-1</span> <span className={r.wer > 0.5 ? 'bad' : ''}>{r.top1 || '∅'}</span> <span className="mono faint">WER {pct(r.wer)}</span></div>
        </div>
      ))}
    </div>
  )
}

function CoordinatorSection({ c }: { c: any }) {
  const rows = Object.entries(c.policies ?? {}) as [string, any][]
  const cats = [...new Set(rows.flatMap(([, v]) => Object.keys(v.by_category ?? {})))]
  return (
    <section className="glass panel">
      <div className="panel-title"><FlaskConical size={14} /> Coordinator · NeMo Gym environment “{c.env}” · {c.n_tasks} tasks</div>
      <div className="coord-eval">
        <table className="tbl">
          <thead><tr><th>Policy</th><th className="num">Reward</th><th className="num">Valid tool call</th><th className="num">Correct action</th><th className="num">Unsafe</th><th className="num">By model</th><th className="num">Median latency</th></tr></thead>
          <tbody>
            {rows.map(([k, v]) => (
              <tr key={k}><td>{v.label ?? k}</td><td className="num mono strong">{v.reward_mean.toFixed(3)}</td><td className="num mono">{pct(v.valid_rate)}</td><td className="num mono">{pct(v.action_rate)}</td><td className="num mono">{pct(v.unsafe_rate)}</td><td className="num mono">{v.llm_share != null ? pct(v.llm_share) : v.label?.startsWith('Raw') ? '100%' : '0%'}</td><td className="num mono">{v.latency_ms_median ?? '—'} ms</td></tr>
            ))}
          </tbody>
        </table>
        {cats.length > 0 && (
          <table className="tbl">
            <thead><tr><th>Category</th>{rows.map(([k, v]) => <th key={k} className="num">{v.short ?? k}</th>)}</tr></thead>
            <tbody>
              {cats.map((cat) => <tr key={cat}><td>{cat.replace(/_/g, ' ')}</td>{rows.map(([k, v]) => <td key={k} className="num mono">{v.by_category?.[cat] != null ? v.by_category[cat].toFixed(2) : '—'}</td>)}</tr>)}
            </tbody>
          </table>
        )}
      </div>
      <p className="note">{c.note}</p>
    </section>
  )
}
