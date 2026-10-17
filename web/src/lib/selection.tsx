// One selection engine for every accessible input:
//   pointer/keyboard  -> native button activation
//   hover dwell       -> head-mouse / pointer rests on a target
//   gaze dwell        -> WebGazer estimate rests on a target (with hysteresis)
//   blink scan        -> highlight cycles through targets, deliberate blink selects
// Every activation reports *how* the person acted, which becomes the audited
// confirmation method. Confirm targets need a longer dwell than select targets.

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { onBlink } from './face'
import { onGaze, type GazePoint } from './gaze'
import type { ConfirmationMethod } from './types'

export type InputMode = 'pointer' | 'hover' | 'gaze' | 'scan'

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
  dwellMs: number
  confirmDwellMs: number
  scanMs: number
}

interface Ctx extends Settings {
  setSettings: (s: Partial<Settings>) => void
  register: (t: Target) => () => void
  gaze: GazePoint | null
}

const SelectionContext = createContext<Ctx | null>(null)
const HYSTERESIS_PX = 36
const COOLDOWN_MS = 900

function load(): Settings {
  const d: Settings = { mode: 'pointer', dwellMs: 1100, confirmDwellMs: 1700, scanMs: 1500 }
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

  // gaze + pointer streams
  useEffect(() => onGaze((p) => { gazeRef.current = p; setGaze(p) }), [])
  useEffect(() => {
    const mv = (e: PointerEvent) => { pointer.current = { x: e.clientX, y: e.clientY, t: performance.now() } }
    const leave = () => { pointer.current = null }
    window.addEventListener('pointermove', mv)
    document.addEventListener('pointerleave', leave)
    return () => { window.removeEventListener('pointermove', mv); document.removeEventListener('pointerleave', leave) }
  }, [])

  // dwell loop
  useEffect(() => {
    let raf = 0
    const hit = (p: GazePoint, pad: number) => {
      for (const t of targets.current.values()) {
        const r = t.el.getBoundingClientRect()
        if (r.width === 0) continue
        if (p.x >= r.left - pad && p.x <= r.right + pad && p.y >= r.top - pad && p.y <= r.bottom + pad) return t
      }
      return null
    }
    const loop = () => {
      raf = requestAnimationFrame(loop)
      const { mode, dwellMs, confirmDwellMs } = s.current
      if (mode !== 'gaze' && mode !== 'hover') return
      const p = mode === 'gaze' ? gazeRef.current : pointer.current
      const now = performance.now()
      let t: Target | null = null
      if (p) {
        // hysteresis: stay on the current target while the point is near it
        const cur = current.current && targets.current.get(current.current.id)
        t = cur && hit(p, HYSTERESIS_PX) === cur ? cur : hit(p, 0)
      }
      if (cooldown.current && t?.id === cooldown.current.id && now < cooldown.current.until) t = null
      if (!t || t.el.hasAttribute('disabled')) {
        if (current.current) targets.current.get(current.current.id)?.setProgress(0)
        current.current = null
        return
      }
      if (current.current?.id !== t.id) {
        if (current.current) targets.current.get(current.current.id)?.setProgress(0)
        current.current = { id: t.id, since: now }
      }
      const need = t.kind === 'confirm' ? confirmDwellMs : dwellMs
      const prog = Math.min(1, (now - current.current.since) / need)
      t.setProgress(prog)
      if (prog >= 1) {
        t.setProgress(0)
        current.current = null
        cooldown.current = { id: t.id, until: now + COOLDOWN_MS * 3 }
        t.activate(mode === 'gaze' ? 'gaze_dwell_confirmed' : 'other_explicit_user_confirmed')
      }
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [])

  // blink / switch scanning
  useEffect(() => {
    if (settings.mode !== 'scan') {
      targets.current.forEach((t) => t.setFocus(false))
      scanFocus.current = null
      return
    }
    const ordered = () => [...targets.current.values()]
      .filter((t) => !t.el.hasAttribute('disabled') && t.el.getBoundingClientRect().width > 0)
      .sort((a, b) => (a.el.compareDocumentPosition(b.el) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1))
    const advance = () => {
      const list = ordered()
      if (!list.length) return
      scanIdx.current = (scanIdx.current + 1) % list.length
      list.forEach((t, i) => t.setFocus(i === scanIdx.current))
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

  const value = useMemo(() => ({ ...settings, setSettings, register, gaze }), [settings, setSettings, register, gaze])
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
