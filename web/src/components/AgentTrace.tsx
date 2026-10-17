import { AnimatePresence, motion } from 'framer-motion'
import { ChevronDown, History as HistoryIcon, Volume2 } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { SpokenItem, TimelineItem } from '../lib/types'
import { AGENT, METHOD_TEXT, MODALITY } from './meta'
import './trace.css'

const NODES = [
  { id: 'vsr', label: 'Lips', x: 14, y: 18 },
  { id: 'audio', label: 'Voice', x: 14, y: 42 },
  { id: 'gaze', label: 'Eyes', x: 14, y: 66 },
  { id: 'eeg', label: 'EEG', x: 14, y: 90 },
  { id: 'coordinator', label: 'Coordinator', x: 52, y: 54 },
  { id: 'user', label: 'Person', x: 52, y: 14 },
  { id: 'policy', label: 'Policy', x: 84, y: 34 },
  { id: 'tts', label: 'Speech', x: 84, y: 76 },
]
const EDGES = [['vsr', 'coordinator'], ['audio', 'coordinator'], ['gaze', 'coordinator'], ['eeg', 'coordinator'],
  ['coordinator', 'user'], ['user', 'policy'], ['policy', 'tts']]

export function AgentGraph({ timeline }: { timeline: TimelineItem[] }) {
  const [, tick] = useState(0)
  useEffect(() => { const id = setInterval(() => tick((x) => x + 1), 500); return () => clearInterval(id) }, [])
  const now = Date.now()
  const active = new Set(timeline.filter((t) => now - t.ts < 3500 || t.level === 'pending').map((t) => t.agent))
  const pos = Object.fromEntries(NODES.map((n) => [n.id, n]))
  return (
    <svg className="agent-graph" viewBox="0 0 100 104" role="img" aria-label="Agents and their recent activity">
      {EDGES.map(([a, b]) => {
        const A = pos[a], B = pos[b]
        const on = active.has(a) && (active.has(b) || b === 'coordinator')
        return (
          <g key={a + b}>
            <path d={`M${A.x} ${A.y} C ${(A.x + B.x) / 2} ${A.y}, ${(A.x + B.x) / 2} ${B.y}, ${B.x} ${B.y}`}
              fill="none" stroke={on ? AGENT[a]?.color ?? 'var(--text-3)' : 'var(--hairline-strong)'} strokeWidth={on ? 0.9 : 0.5}
              strokeDasharray={on ? '2 2' : undefined} className={on ? 'flow' : ''} />
          </g>
        )
      })}
      {NODES.map((n) => {
        const on = active.has(n.id)
        const c = AGENT[n.id]?.color ?? 'var(--text)'
        return (
          <g key={n.id} transform={`translate(${n.x} ${n.y})`}>
            {on && <circle r="7" fill={c} opacity=".18"><animate attributeName="r" values="4;8;4" dur="1.6s" repeatCount="indefinite" /></circle>}
            <circle r="3.4" fill={on ? c : 'var(--bg-elev)'} stroke={c} strokeWidth="0.8" />
            <text y={n.id === 'user' ? -6 : 8.5} textAnchor="middle" fontSize="4.2" fill={on ? 'var(--text)' : 'var(--text-3)'} fontWeight={on ? 600 : 400}>{n.label}</text>
          </g>
        )
      })}
    </svg>
  )
}

function ago(ts: number) {
  const s = Math.max(0, Math.round((Date.now() - ts) / 1000))
  return s < 60 ? `${s}s` : `${Math.round(s / 60)}m`
}

export function AgentTrace({ timeline }: { timeline: TimelineItem[] }) {
  const items = [...timeline].reverse().slice(0, 40)
  return (
    <div className="glass panel trace">
      <div className="panel-title">Agent activity <span className="spacer" /><span className="mono" style={{ letterSpacing: 0 }}>{timeline.length}</span></div>
      <AgentGraph timeline={timeline} />
      <div className="trace-list scroll">
        <AnimatePresence initial={false}>
          {items.map((t) => {
            const a = AGENT[t.agent] ?? { label: t.agent, color: 'var(--text-3)' }
            return (
              <motion.div key={t.id} className={`trace-item ${t.level}`} layout initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }}
                style={{ ['--c' as any]: t.level === 'error' ? 'var(--danger)' : t.level === 'warn' ? 'var(--warn)' : a.color }}>
                <span className="trace-dot" />
                <div className="grow">
                  <div className="trace-top">
                    <span className="trace-agent">{a.label}</span>
                    <span className="faint mono">{t.latency_ms != null ? `${t.latency_ms} ms · ` : ''}{ago(t.ts)}</span>
                  </div>
                  <div className={`trace-title ${t.level === 'pending' ? 'shimmer' : ''}`}>{t.title}</div>
                  {t.detail && <div className="trace-detail">{t.detail}</div>}
                </div>
              </motion.div>
            )
          })}
        </AnimatePresence>
        {!items.length && <div className="faint" style={{ fontSize: '.88rem', padding: '8px 2px' }}>Specialist agents report here as they work.</div>}
      </div>
    </div>
  )
}

export function History({ spoken, onReplay }: { spoken: SpokenItem[]; onReplay: (s: SpokenItem) => void }) {
  const [open, setOpen] = useState<string | null>(null)
  return (
    <div className="glass panel history">
      <div className="panel-title"><HistoryIcon size={14} /> Conversation</div>
      <div className="history-list scroll">
        {[...spoken].reverse().map((s) => {
          const m = MODALITY[s.origin === 'spelling' ? 'spelling' : s.modality]
          return (
            <div key={s.message_id} className="hist" style={{ ['--c' as any]: m.color }}>
              <div className="hist-row">
                <button className="icon-btn" onClick={() => onReplay(s)} aria-label={`Replay “${s.text}”`}><Volume2 size={16} /></button>
                <div className="grow">
                  <div className="hist-text">{s.text}</div>
                  <div className="hist-meta">
                    <span className="badge" style={{ ['--c' as any]: m.color }}>{m.badge}</span>
                    <span className="faint">confirmed by {METHOD_TEXT[s.method] ?? s.method}</span>
                    <span className="faint">· {new Date(s.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                  </div>
                </div>
                <button className="icon-btn" onClick={() => setOpen(open === s.message_id ? null : s.message_id)} aria-label="Show audit">
                  <ChevronDown size={16} style={{ transform: open === s.message_id ? 'rotate(180deg)' : 'none' }} />
                </button>
              </div>
              {open === s.message_id && (
                <div className="audit">
                  <div><span className="faint">Origin</span> {s.origin} · <span className="faint">source</span> {s.source_mode}</div>
                  <div><span className="faint">Supported by</span> {s.supporting_modalities.join(', ')}</div>
                  {s.raw.map((r, i) => <div key={i}><span className="faint">Raw {r.modality}</span> {r.hypotheses.slice(0, 3).map((h) => `“${h}”`).join(' · ') || '∅'}</div>)}
                  <div><span className="faint">TTS</span> Kokoro-82M · {s.tts_latency_ms ?? '—'} ms · exact confirmed text</div>
                </div>
              )}
            </div>
          )
        })}
        {!spoken.length && <div className="faint" style={{ fontSize: '.88rem' }}>Confirmed messages will appear here.</div>}
      </div>
    </div>
  )
}
