import { AnimatePresence, motion } from 'framer-motion'
import { Bell, Check, ChevronDown, CircleSlash, Code2, Pencil, RotateCcw, Sparkles, Volume2, X } from 'lucide-react'
import { useMemo, useState } from 'react'
import { api } from '../lib/api'
import type { Candidate, ConfirmationMethod, Evidence, SessionState } from '../lib/types'
import { DwellButton } from './DwellButton'
import { FLAG_TEXT, MODALITY, METHOD_TEXT } from './meta'
import './composer.css'

interface Props {
  state: SessionState
  onSpoken: (audioUrl: string | null, text: string) => void
  onSwitch: (method: string) => void
  onRetake: (modality: string) => void
  notify: (msg: string, kind?: 'error' | 'info') => void
}

const STATUS: Record<string, { label: string; color: string }> = {
  collecting: { label: 'Waiting for input', color: 'var(--text-3)' },
  proposed: { label: 'Reading proposed', color: 'var(--ai)' },
  ambiguous: { label: 'Uncertain — your choice', color: 'var(--warn)' },
  clarifying: { label: 'Choose what you mean', color: 'var(--ai)' },
  selected: { label: 'Selected — confirm to speak', color: 'var(--gaze)' },
  confirmed: { label: 'Confirmed', color: 'var(--voice)' },
  spoken: { label: 'Spoken', color: 'var(--voice)' },
  rejected: { label: 'None of these', color: 'var(--text-3)' },
}

export function Composer({ state, onSpoken, onSwitch, onRetake, notify }: Props) {
  const m = state.message
  const plan = state.plan
  const [editing, setEditing] = useState<Candidate | null>(null)
  const [showMore, setShowMore] = useState(false)
  const [showCall, setShowCall] = useState(false)
  const [help, setHelp] = useState(false)

  const shown = useMemo(() => {
    const ids = state.shown_candidate_ids.length ? state.shown_candidate_ids : m.candidates.slice(0, 3).map((c) => c.id)
    const sel = m.selected_candidate_id
    const list = ids.map((id) => m.candidates.find((c) => c.id === id)).filter(Boolean) as Candidate[]
    if (sel && !list.find((c) => c.id === sel)) list.unshift(m.candidates.find((c) => c.id === sel)!)
    return list
  }, [state.shown_candidate_ids, m.candidates, m.selected_candidate_id])
  const others = m.candidates.filter((c) => !shown.includes(c))
  const selected = m.candidates.find((c) => c.id === m.selected_candidate_id) ?? null
  const evidenceById = Object.fromEntries(state.evidence.map((e) => [e.event_id, e]))

  const act = async (fn: () => Promise<unknown>) => {
    try { await fn() } catch (e: any) { notify(e.message ?? String(e), 'error') }
  }
  const select = (c: Candidate, method: ConfirmationMethod) => act(() => api.select(c.id, method))
  const confirm = (method: ConfirmationMethod) => act(async () => {
    const r = await api.confirm(method)
    onSpoken(r.audio_url, r.text)
  })

  const st = STATUS[m.status] ?? STATUS.collecting
  const spokenLast = m.status === 'spoken' ? state.spoken[state.spoken.length - 1] : null

  return (
    <div className="glass composer">
      <div className="composer-head">
        <span className="chip" style={{ ['--c' as any]: st.color }}><span className="dot" />{st.label}</span>
        <span className="grow" />
        {m.source_mode === 'prerecorded' && <span className="badge" style={{ ['--c' as any]: 'var(--eeg)' }}>RECORDED EEG · NOT LIVE CONSENT</span>}
        <button className="btn sm ghost" onClick={() => act(api.newMessage)} title="Start a new message"><RotateCcw size={15} /> New</button>
      </div>

      {/* Coordinator note */}
      <AnimatePresence mode="wait">
        {plan && m.status !== 'spoken' && (
          <motion.div key={plan.rationale + plan.name} className="coord" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
            <div className="coord-avatar"><Sparkles size={16} /></div>
            <div className="grow">
              <div className="coord-who">
                {plan.by === 'llm' ? 'Coordinator · Nemotron 3.5 Lightning' : 'Coordinator · deterministic policy'}
                <span className="mono faint">{plan.name}{plan.latency_ms != null ? ` · ${plan.latency_ms} ms` : ''}</span>
                <button className="link" onClick={() => setShowCall((v) => !v)}><Code2 size={13} /> tool call</button>
              </div>
              <div className="coord-text">{plan.rationale || 'Waiting for your choice.'}</div>
              <div className="coord-foot">It chooses how to ask. Only you decide what you mean.</div>
              {showCall && <pre className="code">{JSON.stringify({ name: plan.name, arguments: plan.arguments, by: plan.by, valid: plan.valid, validation_errors: plan.validation_errors, raw: plan.raw }, null, 2)}</pre>}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {state.agreement && state.agreement.word_overlap < 1 && m.status !== 'spoken' && (
        <div className="disagree">
          <div><span className="badge" style={{ ['--c' as any]: 'var(--lip)' }}>Lips</span> “{state.agreement.lip}”</div>
          <div><span className="badge" style={{ ['--c' as any]: 'var(--audio)' }}>Voice</span> “{state.agreement.audio}”</div>
          <div className="faint" style={{ fontSize: '.85rem' }}>Agents disagree ({Math.round(state.agreement.word_overlap * 100)}% word overlap). Nothing is merged — you choose.</div>
        </div>
      )}

      {/* Main body */}
      <div className="composer-body">
        {m.status === 'spoken' && spokenLast ? (
          <SpokenCard text={spokenLast.text} method={spokenLast.method} onReplay={() => onSpoken(`/api/${state.session_id}/audio/${spokenLast.message_id}`, spokenLast.text)} />
        ) : plan?.name === 'request_retake' && !m.candidates.length ? (
          <ActionCard
            title="Let's try that once more"
            text={plan.rationale}
            flags={state.evidence.flatMap((e) => e.quality_flags)}
            primary={{ label: 'Try again', onClick: () => onRetake(plan.arguments.modality) }}
            secondary={{ label: 'Use the phrase board', onClick: () => onSwitch('board') }}
          />
        ) : plan?.name === 'switch_input_method' && !selected ? (
          <ActionCard
            title={plan.arguments.method === 'spelling' ? "Let's spell it instead" : 'Try another way'}
            text={plan.rationale}
            flags={[]}
            primary={{ label: plan.arguments.method === 'spelling' ? 'Open spelling board' : `Switch to ${plan.arguments.method}`, onClick: () => onSwitch(plan.arguments.method) }}
          />
        ) : m.candidates.length === 0 ? (
          <Empty />
        ) : (
          <div className="cands">
            <AnimatePresence initial={false}>
              {shown.map((c, i) => (
                <motion.div key={c.id} layout initial={{ opacity: 0, y: 14, scale: .98 }} animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, scale: .98 }} transition={{ delay: i * 0.05, duration: .35, ease: [0.22, 1, 0.36, 1] }}>
                  <CandidateCard c={c} index={i} ev={c.evidence_event_id ? evidenceById[c.evidence_event_id] : undefined}
                    selected={c.id === m.selected_candidate_id} onSelect={(meth) => select(c, meth)} onEdit={() => setEditing(c)} />
                </motion.div>
              ))}
            </AnimatePresence>
            {others.length > 0 && (
              <button className="more" onClick={() => setShowMore((v) => !v)}>
                <ChevronDown size={16} style={{ transform: showMore ? 'rotate(180deg)' : 'none', transition: 'transform .2s' }} />
                {showMore ? 'Hide' : 'Show'} {others.length} more raw reading{others.length > 1 ? 's' : ''} (not chosen by coordinator)
              </button>
            )}
            {showMore && others.map((c, i) => (
              <CandidateCard key={c.id} c={c} index={shown.length + i} ev={c.evidence_event_id ? evidenceById[c.evidence_event_id] : undefined}
                selected={c.id === m.selected_candidate_id} onSelect={(meth) => select(c, meth)} onEdit={() => setEditing(c)} compact />
            ))}
          </div>
        )}
      </div>

      {state.evidence.length > 0 && m.status !== 'spoken' && <EvidenceStrip evidence={state.evidence} />}

      {/* Confirmation dock — spatially separate from candidates */}
      <div className="dock">
        <div className="dock-preview">
          {selected && m.status !== 'spoken' ? (
            <>
              <span className="faint" style={{ fontSize: '.8rem', letterSpacing: '.1em', textTransform: 'uppercase' }}>Will say exactly</span>
              <div className="dock-text">“{selected.text}”</div>
            </>
          ) : (
            <span className="faint">{m.status === 'spoken' ? 'Ready for your next message.' : 'Select a reading or a phrase. Nothing is spoken until you confirm.'}</span>
          )}
        </div>
        <div className="dock-actions">
          <DwellButton kind="confirm" className="btn lg primary speak" disabled={!selected || m.status !== 'selected'} onActivate={confirm} ringColor="var(--voice)">
            <Volume2 size={22} /> Speak it
          </DwellButton>
          <DwellButton className="btn lg" disabled={!m.candidates.length || m.status === 'spoken'} onActivate={() => act(api.reject)}>
            <CircleSlash size={20} /> None of these
          </DwellButton>
          <DwellButton className="btn lg" onActivate={() => act(api.undo)}>
            <RotateCcw size={20} /> Undo
          </DwellButton>
          <DwellButton className="btn lg" ringColor="var(--warn)" onActivate={() => setHelp(true)}>
            <Bell size={20} /> Attention
          </DwellButton>
        </div>
      </div>

      <AnimatePresence>
        {help && (
          <motion.div className="help-banner" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
            <Bell size={20} />
            <div className="grow"><b>I'd like someone's attention.</b><div style={{ fontSize: '.85rem', opacity: .8 }}>Shown on screen only — not a nurse call or emergency alarm.</div></div>
            <button className="icon-btn" onClick={() => setHelp(false)} aria-label="Dismiss"><X size={18} /></button>
          </motion.div>
        )}
      </AnimatePresence>

      {editing && <EditDialog c={editing} onClose={() => setEditing(null)} onSave={(t) => act(async () => { await api.edit(editing.id, t); setEditing(null) })} />}
    </div>
  )
}

function CandidateCard({ c, ev, selected, onSelect, onEdit, index, compact }: {
  c: Candidate; ev?: Evidence; selected: boolean; index: number; compact?: boolean
  onSelect: (m: ConfirmationMethod) => void; onEdit: () => void
}) {
  const meta = MODALITY[c.origin === 'spelling' ? 'spelling' : c.modality]
  const hyp = ev?.hypotheses.find((h) => h.text === c.text)
  const words = hyp?.words?.some((w) => w.p != null) ? hyp.words : null
  return (
    <div className={`cand ${selected ? 'selected' : ''} ${compact ? 'compact' : ''}`} style={{ ['--c' as any]: meta.color }}>
      <DwellButton className="cand-main" onActivate={onSelect} aria-pressed={selected}>
        <span className="cand-index mono">{index + 1}</span>
        <span className="cand-text">
          {words ? words.map((w, i) => (
            <span key={i} className={w.p != null && w.p < 0.5 ? 'unsure' : ''} title={w.p != null ? `word prob ${w.p}` : ''}>{w.word} </span>
          )) : c.text}
        </span>
        <span className="cand-meta">
          {c.supporting_modalities.map((mm) => (
            <span key={mm} className="badge" style={{ ['--c' as any]: MODALITY[mm].color }}>{MODALITY[mm].label}</span>
          ))}
          {c.origin === 'user_edit' && <span className="badge">your edit</span>}
          {c.score != null && <span className="mono faint" title={hyp?.score_kind ?? 'raw model score'}>raw {c.score.toFixed(2)}</span>}
        </span>
        <span className="cand-check">{selected ? <Check size={22} /> : null}</span>
      </DwellButton>
      <button className="cand-edit" onClick={onEdit} aria-label={`Correct “${c.text}”`} title="Correct this text"><Pencil size={16} /></button>
    </div>
  )
}

function EvidenceStrip({ evidence }: { evidence: Evidence[] }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="evs">
      <button className="evs-toggle" onClick={() => setOpen((v) => !v)}>
        <ChevronDown size={15} style={{ transform: open ? 'rotate(180deg)' : 'none', transition: 'transform .2s' }} />
        Raw evidence · {evidence.length} specialist output{evidence.length > 1 ? 's' : ''}
        <span className="grow" />
        {evidence.flatMap((e) => e.quality_flags).slice(0, 4).map((f, i) => (
          <span key={i} className="chip" style={{ ['--c' as any]: 'var(--warn)', height: 24 }}>{FLAG_TEXT[f] ?? f}</span>
        ))}
      </button>
      {open && evidence.map((e) => (
        <div key={e.event_id} className="ev">
          <div className="row wrap" style={{ gap: 8 }}>
            <span className="badge" style={{ ['--c' as any]: MODALITY[e.modality].color }}>{MODALITY[e.modality].badge}</span>
            <span className="mono faint">{e.provenance.checkpoint} · {e.provenance.device} · {e.provenance.latency_ms} ms</span>
          </div>
          {e.modality === 'lip' && e.quality.mouth_thumbs && (
            <div className="thumbs">{(e.quality.mouth_thumbs as string[]).map((t, i) => <img key={i} src={t} alt="" />)}</div>
          )}
          {e.modality === 'lip' && (
            <div className="mono faint" style={{ fontSize: '.75rem' }}>
              face {Math.round((e.quality.face_detected_ratio ?? 0) * 100)}% · mouth motion {e.quality.mouth_motion} (min {e.quality.mouth_motion_threshold}) · top-2 margin {e.quality.top2_margin ?? '—'}
            </div>
          )}
          {e.modality === 'audio' && e.quality.raw && (
            <div className="ab">
              <div><span className="faint">Raw audio →</span> “{e.quality.raw.text || '∅'}”</div>
              {e.quality.enhanced && <div><span className="faint">RNNoise + gain →</span> “{e.quality.enhanced.text || '∅'}”</div>}
              {e.quality.diff && (
                <div className="diff">{(e.quality.diff as any[]).map((d, i) => d.op === 'equal'
                  ? <span key={i}>{d.raw.join(' ')} </span>
                  : <span key={i} className="diff-x" title="raw vs enhanced differ"><s>{d.raw.join(' ') || '∅'}</s> → {d.enhanced.join(' ') || '∅'} </span>)}
                </div>
              )}
              <div className="mono faint" style={{ fontSize: '.75rem' }}>
                input {e.quality.input_dbfs} dBFS · speech {Math.round((e.quality.speech_ratio ?? 0) * 100)}% · SNR≈{e.quality.snr_db_est ?? '—'} dB · gain +{e.quality.gain_db} dB
              </div>
              {e.quality.enhanced_audio && <audio controls src={e.quality.enhanced_audio} style={{ width: '100%', height: 36 }} />}
            </div>
          )}
          <ol className="nbest">
            {e.hypotheses.map((h) => <li key={h.id}><span>{h.text}</span><span className="mono faint">{h.score.toFixed(2)}</span></li>)}
            {e.hypotheses.length === 0 && <li className="faint">No hypotheses — the agent declined to guess.</li>}
          </ol>
        </div>
      ))}
    </div>
  )
}

function SpokenCard({ text, method, onReplay }: { text: string; method: string; onReplay: () => void }) {
  return (
    <motion.div className="spoken" initial={{ opacity: 0, scale: .97 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: .5, ease: [0.22, 1, 0.36, 1] }}>
      <div className="spoken-label"><Check size={16} /> Spoken · confirmed by {METHOD_TEXT[method] ?? method}</div>
      <div className="display spoken-text">“{text}”</div>
      <button className="btn sm" onClick={onReplay}><Volume2 size={16} /> Say it again</button>
    </motion.div>
  )
}

function ActionCard({ title, text, flags, primary, secondary }: {
  title: string; text: string; flags: string[]
  primary: { label: string; onClick: () => void }; secondary?: { label: string; onClick: () => void }
}) {
  return (
    <div className="action-card">
      <div className="display" style={{ fontSize: '2rem' }}>{title}</div>
      <div className="muted">{text}</div>
      {flags.length > 0 && <div className="row wrap">{[...new Set(flags)].map((f) => <span key={f} className="chip" style={{ ['--c' as any]: 'var(--warn)' }}>{FLAG_TEXT[f] ?? f}</span>)}</div>}
      <div className="row wrap" style={{ marginTop: 6 }}>
        <DwellButton className="btn lg primary" onActivate={primary.onClick}>{primary.label}</DwellButton>
        {secondary && <DwellButton className="btn lg" onActivate={secondary.onClick}>{secondary.label}</DwellButton>}
      </div>
    </div>
  )
}

function Empty() {
  return (
    <div className="empty">
      <div className="empty-orbs" aria-hidden>
        {['var(--lip)', 'var(--audio)', 'var(--gaze)', 'var(--eeg)'].map((c, i) => (
          <motion.span key={i} style={{ background: c }} animate={{ y: [0, -8, 0] }} transition={{ repeat: Infinity, duration: 3, delay: i * 0.3, ease: 'easeInOut' }} />
        ))}
      </div>
      <div className="display" style={{ fontSize: 'clamp(1.9rem, 3vw, 2.6rem)' }}>Say it your way.</div>
      <div className="muted" style={{ maxWidth: 440 }}>Mouth a phrase, whisper it, pick from the board, or spell with your eyes. Every reading waits here for you to choose and confirm.</div>
    </div>
  )
}

function EditDialog({ c, onClose, onSave }: { c: Candidate; onClose: () => void; onSave: (t: string) => void }) {
  const [t, setT] = useState(c.text)
  return (
    <div className="modal" role="dialog" aria-modal aria-label="Correct text" onClick={onClose}>
      <div className="glass modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="panel-title">Correct the reading</div>
        <div className="faint" style={{ fontSize: '.85rem' }}>Original kept in the audit log: “{c.text}”</div>
        <input autoFocus className="input" value={t} onChange={(e) => setT(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && onSave(t)} />
        <div className="row" style={{ justifyContent: 'flex-end' }}>
          <button className="btn" onClick={onClose}>Cancel</button>
          <button className="btn primary" onClick={() => onSave(t)} disabled={!t.trim()}>Use my text</button>
        </div>
      </div>
    </div>
  )
}
