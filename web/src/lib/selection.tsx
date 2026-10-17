// One selection engine for every accessible input:
//   pointer/keyboard  -> native button activation
//   hover dwell       -> head-mouse / pointer rests on a target
//   gaze              -> gaze estimate rests on a target (with hysteresis), then either
//                        dwell (ring fills) or look + deliberate blink, per user setting
//   blink scan        -> highlight cycles through targets, deliberate blink selects
// Every activation reports *how* the person acted, which becomes the audited
// confirmation method. Confirm targets need a longer dwell (or a held look before the blink).

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { onBlink } from './face'
import { gazeState, learnFromSelection, onGaze, type GazePoint } from './gaze'
import type { ConfirmationMethod } from './types'

export type InputMode = 'pointer' | 'hover' | 'gaze' | 'scan'
export type GazeSelect = 'dwell' | 'blink'

interface Target {
  id: string
  el: HTMLElement
  kind: 'select' | 'confirm'
  activate: (method: ConfirmationMethod) => void
  setProgress: (p: number) => void
  setFocus: (f: boolean) => void
}

interface Settings {
  mode: InputMode
  gazeSelect: GazeSelect
  cardSnap: boolean   // gaze goes to the nearest card centre (after calibrating on the cards themselves)
  dwellMs: number
  confirmDwellMs: number
  scanMs: number
}

interface Ctx extends Settings {
  setSettings: (s: Partial<Settings>) => void
  targetRects: () => DOMRect[]   // on-screen targets (the fixed card layout), in reading order
  register: (t: Target) => () => void
  gaze: GazePoint | null
}

const SelectionContext = createContext<Ctx | null>(null)
const HYSTERESIS_PX = 36
// gaze is an estimate: tolerate misses in proportion to the measured calibration error
const gazeSlack = () => Math.max(HYSTERESIS_PX, Math.min(140, (gazeState.accuracyPx ?? 80) * 0.6))
const COOLDOWN_MS = 900
const BLINK_AIM_MS = 350          // look at a target this long before a blink can select it
const BLINK_CONFIRM_AIM_MS = 700  // longer for confirm targets (Speak it)

function load(): Settings {
  const d: Settings = { mode: 'pointer', gazeSelect: 'dwell', cardSnap: false, dwellMs: 1100, confirmDwellMs: 1700, scanMs: 1500 }
  try { return { ...d, ...JSON.parse(localStorage.getItem('unspoken.selection') ?? '{}') } } catch { return d }
}

export function SelectionProvider({ children }: { children: ReactNode }) {
  const [settings, setSettingsState] = useState<Settings>(load)
  const [gaze, setGaze] = useState<GazePoint | null>(null)
  const targets = useRef(new Map<string, Target>())
  const current = useRef<{ id: string; since: number } | null>(null)
  const cooldown = useRef<{ id: string; until: number } | null>(null)
  const scanIdx = useRef(0)
  const scanFocus = useRef<string | null>(null)
  const pointer = useRef<GazePoint | null>(null)
  const gazeRef = useRef<GazePoint | null>(null)
  const s = useRef(settings)
  s.current = settings

  // lets CSS give gaze users larger, more widely spaced targets
  useEffect(() => { document.documentElement.dataset.input = settings.mode }, [settings.mode])

  const setSettings = useCallback((p: Partial<Settings>) => {
    setSettingsState((prev) => {
      const next = { ...prev, ...p }
      try { localStorage.setItem('unspoken.selection', JSON.stringify(next)) } catch { /* private mode */ }
      return next
    })
  }, [])

  const register = useCallback((t: Target) => {
    targets.current.set(t.id, t)
    return () => {
      targets.current.delete(t.id)
      if (current.current?.id === t.id) current.current = null
    }
  }, [])

  const live = () => [...targets.current.values()]
  const onScreen = (r: DOMRect) => r.width > 0 && r.bottom > 0 && r.right > 0 && r.top < window.innerHeight && r.left < window.innerWidth
  const targetRects = useCallback(() => [...targets.current.values()]
    .sort((a, b) => (a.el.compareDocumentPosition(b.el) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1))
    .map((t) => t.el.getBoundingClientRect()).filter(onScreen), [])
  useEffect(() => {
    targets.current.forEach((t) => { t.setProgress(0); t.setFocus(false) })
    current.current = null
  }, [settings.mode, settings.gazeSelect])

  // gaze + pointer streams
  useEffect(() => onGaze((p) => { gazeRef.current = p; setGaze(p) }), [])
  useEffect(() => {
    const mv = (e: PointerEvent) => { pointer.current = { x: e.clientX, y: e.clientY, t: performance.now() } }
    const leave = () => { pointer.current = null }
    window.addEventListener('pointermove', mv)
    document.addEventListener('pointerleave', leave)
    return () => { window.removeEventListener('pointermove', mv); document.removeEventListener('pointerleave', leave) }
  }, [])

  const centre = (t: Target) => { const r = t.el.getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2] as const }

  // pointing loop: which target is the gaze/pointer on, and dwell progress
  useEffect(() => {
    let raf = 0
    const hit = (p: GazePoint, pad: number) => {
      for (const t of live()) {
        const r = t.el.getBoundingClientRect()
        if (r.width === 0) continue
        if (p.x >= r.left - pad && p.x <= r.right + pad && p.y >= r.top - pad && p.y <= r.bottom + pad) return t
      }
      return null
    }
    // magnetic: a gaze point in the gap between targets goes to the nearest target edge within `pad`
    const nearest = (p: GazePoint, pad: number) => {
      let best: Target | null = null, bestD = pad
      for (const t of live()) {
        const r = t.el.getBoundingClientRect()
        if (r.width === 0 || t.el.hasAttribute('disabled')) continue
        const d = Math.hypot(Math.max(r.left - p.x, 0, p.x - r.right), Math.max(r.top - p.y, 0, p.y - r.bottom))
        if (d <= bestD) { best = t; bestD = d }
      }
      return best
    }
    // card snap: the target whose centre is nearest the gaze (the calibration was made on these centres)
    const nearestCentre = (p: GazePoint, cur: Target | null) => {
      let best: Target | null = null, bestD = Infinity, curD = Infinity
      for (const t of live()) {
        const r = t.el.getBoundingClientRect()
        if (!onScreen(r) || t.el.hasAttribute('disabled')) continue
        const d = Math.hypot(p.x - (r.left + r.width / 2), p.y - (r.top + r.height / 2))
        if (t === cur) curD = d
        if (d < bestD) { best = t; bestD = d }
      }
      const reach = Math.max(150, (gazeState.accuracyPx ?? 80) * 2.5)
      if (bestD > reach) return null
      return cur && curD <= bestD * 1.25 + 20 ? cur : best   // hysteresis: stay unless clearly closer to another card
    }
    const loop = () => {
      raf = requestAnimationFrame(loop)
      const { mode, gazeSelect, cardSnap, dwellMs, confirmDwellMs } = s.current
      if (mode !== 'gaze' && mode !== 'hover') return
      const p = mode === 'gaze' ? gazeRef.current : pointer.current
      const now = performance.now()
      let t: Target | null = null
      if (p) {
        // hysteresis: stay on the current target while the point is near it
        const slack = mode === 'gaze' ? gazeSlack() : HYSTERESIS_PX
        const cur = current.current ? targets.current.get(current.current.id) ?? null : null
        t = mode === 'gaze' && cardSnap ? nearestCentre(p, cur)
          : cur && hit(p, slack) === cur ? cur : hit(p, 0) ?? (mode === 'gaze' ? nearest(p, slack) : null)
      }
      if (cooldown.current && t?.id === cooldown.current.id && now < cooldown.current.until) t = null
      if (!t || t.el.hasAttribute('disabled')) {
        if (current.current) { const c = targets.current.get(current.current.id); c?.setProgress(0); c?.setFocus(false) }
        current.current = null
        return
      }
      if (current.current?.id !== t.id) {
        if (current.current) { const c = targets.current.get(current.current.id); c?.setProgress(0); c?.setFocus(false) }
        current.current = { id: t.id, since: now }
      }
      if (mode === 'gaze' && gazeSelect === 'blink') {
        t.setFocus(true)  // look + blink: highlight the aimed target; the blink handler selects it
        return
      }
      const need = t.kind === 'confirm' ? confirmDwellMs : dwellMs
      const prog = Math.min(1, (now - current.current.since) / need)
      t.setProgress(prog)
      if (prog >= 1) {
        t.setProgress(0)
        current.current = null
        cooldown.current = { id: t.id, until: now + COOLDOWN_MS * 3 }
        if (mode === 'gaze') learnFromSelection(...centre(t), need)
        t.activate(mode === 'gaze' ? 'gaze_dwell_confirmed' : 'other_explicit_user_confirmed')
      }
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [])

  // look + blink: a deliberate blink selects the target the gaze has been resting on
  useEffect(() => {
    if (settings.mode !== 'gaze' || settings.gazeSelect !== 'blink') return
    return onBlink((ms) => {
      const c = current.current
      const t = c && targets.current.get(c.id)
      if (!c || !t || t.el.hasAttribute('disabled')) return
      // aim time is measured up to the start of the blink
      const aimed = performance.now() - ms - c.since
      if (aimed < (t.kind === 'confirm' ? BLINK_CONFIRM_AIM_MS : BLINK_AIM_MS)) return
      learnFromSelection(...centre(t), Math.min(aimed, 1200) + ms)
      t.setFocus(false)
      current.current = null
      t.activate('deliberate_blink_confirmed')
    })
  }, [settings.mode, settings.gazeSelect])

  // blink / switch scanning
  useEffect(() => {
    if (settings.mode !== 'scan') {
      targets.current.forEach((t) => { if (scanFocus.current === t.id) t.setFocus(false) })
      scanFocus.current = null
      return
    }
    const ordered = () => live()
      .filter((t) => !t.el.hasAttribute('disabled') && t.el.getBoundingClientRect().width > 0)
      .sort((a, b) => (a.el.compareDocumentPosition(b.el) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1))
    const advance = () => {
      const list = ordered()
      if (!list.length) return
      scanIdx.current = (scanIdx.current + 1) % list.length
      targets.current.forEach((t) => t.setFocus(false))
      list[scanIdx.current].setFocus(true)
      scanFocus.current = list[scanIdx.current].id
      list[scanIdx.current].el.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
    }
    const id = window.setInterval(advance, settings.scanMs)
    const choose = (method: ConfirmationMethod) => {
      const t = scanFocus.current && targets.current.get(scanFocus.current)
      if (t) { t.activate(method); scanIdx.current = -1 }
    }
    const offBlink = onBlink(() => choose('deliberate_blink_confirmed'))
    const key = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !(e.target instanceof HTMLInputElement)) { e.preventDefault(); choose('keyboard_confirmed') }
    }
    window.addEventListener('keydown', key)
    return () => { window.clearInterval(id); offBlink(); window.removeEventListener('keydown', key) }
  }, [settings.mode, settings.scanMs])

  const value = useMemo(() => ({ ...settings, setSettings, register, gaze, targetRects }), [settings, setSettings, register, gaze, targetRects])
  return <SelectionContext.Provider value={value}>{children}</SelectionContext.Provider>
}

export function useSelection() {
  const c = useContext(SelectionContext)
  if (!c) throw new Error('SelectionProvider missing')
  return c
}

let _uid = 0
export function useDwellTarget(kind: 'select' | 'confirm', activate: (m: ConfirmationMethod) => void) {
  const { register } = useSelection()
  const ref = useRef<HTMLButtonElement | null>(null)
  const [progress, setProgress] = useState(0)
  const [focus, setFocus] = useState(false)
  const act = useRef(activate)
  act.current = activate
  const id = useMemo(() => `t${++_uid}`, [])
  useEffect(() => {
    if (!ref.current) return
    return register({ id, el: ref.current, kind, activate: (m) => act.current(m), setProgress, setFocus })
  }, [id, kind, register])
  return { ref, progress, focus }
}
