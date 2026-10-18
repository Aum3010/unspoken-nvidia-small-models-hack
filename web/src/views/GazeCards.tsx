import { ArrowLeft, Bell, Eraser, ThumbsDown, ThumbsUp, Volume2, X } from 'lucide-react'
import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { PHRASES } from '../components/Board'
import { CardCalibration, type Card } from '../components/CardCalibration'
import { DwellButton } from '../components/DwellButton'
import { api } from '../lib/api'
import { gazeState, startGaze } from '../lib/gaze'
import { useSelection } from '../lib/selection'
import type { ConfirmationMethod, SessionState } from '../lib/types'
import './gaze-cards.css'

// Full-screen gaze page with 9 cards at FIXED positions: a 3x3 grid whose card centres are at
// 1/6, 1/2 and 5/6 of the window width and height, with wide gaps between cards. The same numbers
// place the cards and define the calibration targets, so they always coincide; changing labels
// never moves a card and the page never scrolls. These 9 cards are the only gaze targets here, and
// the gaze pointer snaps to the nearest card centre.

const CENTRES = [1 / 6, 1 / 2, 5 / 6]
const CARD_W = 0.26, CARD_H = 0.22          // fractions of the window: leaves ~7% / ~11% gaps between cards

function geometry(w: number, h: number): Card[] {
  return CENTRES.flatMap((fy) => CENTRES.map((fx) => {
    const width = CARD_W * w, height = CARD_H * h, x = fx * w, y = fy * h
    return { x, y, left: x - width / 2, top: y - height / 2, width, height }
  }))
}

interface Slot { label: ReactNode; sub?: ReactNode; icon?: ReactNode; tone?: string; kind?: 'select' | 'confirm'; disabled?: boolean; onActivate: (m: ConfirmationMethod) => void }

export function GazeCards({ state, speak, notify, onExit }: {
  state: SessionState | null
  speak: (url: string | null, text: string) => void
  notify: (t: string, k?: 'error' | 'info') => void
  onExit: () => void
}) {
  const sel = useSelection()
  const [size, setSize] = useState({ w: window.innerWidth, h: window.innerHeight })
  const cards = useMemo(() => geometry(size.w, size.h), [size])
  const [group, setGroup] = useState<number | null>(null)
  const [calibrating, setCalibrating] = useState(!gazeState.calibrated)
  const [result, setResult] = useState<{ errorPx: number | null; hitRate: number | null } | null>(null)
  const [attention, setAttention] = useState(false)

  useEffect(() => {
    const onResize = () => setSize({ w: window.innerWidth, h: window.innerHeight })
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

  // on this page: gaze dwell with snapping to the nearest card; restore the previous setting on exit
  useEffect(() => {
    const before = { mode: sel.mode, cardSnap: sel.cardSnap }
    if (gazeState.calibrated) startGaze().then(() => sel.setSettings({ mode: 'gaze', cardSnap: true })).catch(() => notify('Gaze tracking could not start', 'error'))
    else sel.setSettings({ cardSnap: true })
    return () => sel.setSettings(before)
  }, [])  // eslint-disable-line react-hooks/exhaustive-deps

  const m = state?.message
  const selected = m?.candidates.find((c) => c.id === m.selected_candidate_id) ?? null
  const act = async (fn: () => Promise<unknown>) => { try { await fn() } catch (e: any) { notify(e.message ?? String(e), 'error') } }
  const choose = (text: string) => act(() => api.board(text, 'board'))
  const confirm = (method: ConfirmationMethod) => act(async () => {
    const r = await api.confirm(method)
    speak(r.audio_url, r.text)
  })

  const top: Slot[] = group === null
    ? [
        ...PHRASES.map((g, i): Slot => ({ label: g.group, sub: g.items.slice(0, 2).join(' · '), tone: 'group', onActivate: () => setGroup(i) })),
        { label: 'Yes', icon: <ThumbsUp size={34} />, tone: 'yes', onActivate: () => choose('Yes') },
        { label: 'No', icon: <ThumbsDown size={34} />, tone: 'no', onActivate: () => choose('No') },
      ]
    : PHRASES[group].items.slice(0, 6).map((t): Slot => ({ label: t, tone: 'phrase', onActivate: () => choose(t) }))
  const bottom: Slot[] = [
    group === null
      ? { label: attention ? 'Stop asking' : 'Attention', icon: <Bell size={30} />, tone: 'warn', onActivate: () => setAttention((a) => !a) }
      : { label: 'Back', icon: <ArrowLeft size={30} />, tone: 'nav', onActivate: () => setGroup(null) },
    { label: selected ? `“${selected.text}”` : 'Speak it', sub: selected ? 'Speak it' : 'Choose a phrase first', icon: <Volume2 size={32} />,
      tone: 'speak', kind: 'confirm', disabled: !selected || m?.status !== 'selected', onActivate: confirm },
    { label: 'Clear', icon: <Eraser size={30} />, tone: 'nav', disabled: !selected, onActivate: () => act(api.undo) },
  ]
  const slots = [...top, ...bottom]

  // Rendered straight into <body>: the app animates pages with a CSS transform, and a transformed
  // ancestor would make these position:fixed cards relative to it instead of the screen, so they
  // would sit away from the calibration targets.
  return createPortal(
    <div className="gazecards">
      {cards.map((c, i) => {
        const s = slots[i]
        return (
          <DwellButton key={i} kind={s.kind ?? 'select'} disabled={s.disabled} onActivate={s.onActivate}
            className={`gc-card ${s.tone ?? ''}`} ringColor={s.tone === 'speak' ? 'var(--voice)' : s.tone === 'warn' ? 'var(--warn)' : undefined}
            style={{ left: c.left, top: c.top, width: c.width, height: c.height }}>
            {s.icon}
            <span className="gc-label">{s.label}</span>
            {s.sub && <span className="gc-sub">{s.sub}</span>}
          </DwellButton>
        )
      })}

      {attention && <div className="gc-attention"><Bell size={18} /> I'd like someone's attention <span className="faint">(on screen only)</span></div>}

      {/* caregiver controls: ordinary buttons in the corner gaps, never gaze targets */}
      <div className="gc-controls">
        {result && (
          <span className="gc-result">
            {result.errorPx == null ? 'Eyes not visible enough — try again' : `Error ≈ ${Math.round(result.errorPx)} px · right card ${Math.round((result.hitRate ?? 0) * 100)}%`}
          </span>
        )}
        <button className="btn sm" onClick={() => { setResult(null); setCalibrating(true) }}>Calibrate</button>
        <button className="icon-btn" onClick={onExit} aria-label="Leave gaze cards"><X size={18} /></button>
      </div>

      {calibrating && (
        <CardCalibration cards={cards} onDone={(r) => {
          setCalibrating(false)
          setResult(r)
          if (gazeState.calibrated) sel.setSettings({ mode: 'gaze', cardSnap: true })
        }} />
      )}
    </div>,
    document.body,
  )
}
