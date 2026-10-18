import { useEffect, useRef, useState } from 'react'
import type { ConfirmationMethod, ServiceHealth, SessionState } from './types'

export const SESSION_ID = (() => {
  try {
    const k = 'unspoken.session'
    const v = sessionStorage.getItem(k) ?? `s-${Math.random().toString(36).slice(2, 9)}`
    sessionStorage.setItem(k, v)
    return v
  } catch {
    return 'demo-session'
  }
})()

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init)
  if (!r.ok) {
    let detail = r.statusText
    try { detail = (await r.json()).detail ?? detail } catch { /* not json */ }
    throw new Error(detail)
  }
  return r.json()
}

const post = <T,>(path: string, body?: unknown) =>
  req<T>(path, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body ?? {}) })

const base = `/api/${SESSION_ID}`

export const api = {
  health: () => req<ServiceHealth>('/api/health'),
  ledger: () => req<{ budget_b: number; components: any[] }>('/api/ledger'),
  captureLip: (blob: Blob, movingMs?: number) => {
    const f = new FormData()
    f.append('video', blob, blob.type.includes('mp4') ? 'clip.mp4' : 'clip.webm')
    if (movingMs != null) f.append('moving_ms', String(Math.round(movingMs)))   // lip-landmark movement time
    return req<{ event: any; state: SessionState }>(`${base}/capture/lip`, { method: 'POST', body: f })
  },
  captureAudio: (blob: Blob, enhance = true) => {
    const f = new FormData()
    f.append('audio', blob, blob.type.includes('wav') ? 'clip.wav' : blob.type.includes('mp4') ? 'clip.m4a' : 'clip.webm')
    f.append('enhance', String(enhance))
    return req<{ event: any; state: SessionState }>(`${base}/capture/audio`, { method: 'POST', body: f })
  },
  captureEeg: (subject: string, chars: number[]) => post<SessionState>(`${base}/capture/eeg`, { subject, chars }),
  board: (text: string, origin: 'board' | 'spelling') => post<SessionState>(`${base}/board`, { text, origin }),
  select: (candidate_id: string, method: string) => post<SessionState>(`${base}/select`, { candidate_id, method }),
  edit: (candidate_id: string, text: string) => post<SessionState>(`${base}/edit`, { candidate_id, text }),
  confirm: (method: ConfirmationMethod) =>
    post<{ message_id: string; audio_url: string | null; text: string; state: SessionState }>(`${base}/confirm`, { method }),
  reject: () => post<SessionState>(`${base}/reject`),
  undo: () => post<SessionState>(`${base}/undo`),
  newMessage: () => post<SessionState>(`${base}/new`),
  settings: (s: Partial<SessionState['settings']>) => post<SessionState>(`${base}/settings`, s),
  safetyProbe: () => post<{ outcome: string }>(`${base}/safety/attempt_unconfirmed_speech`),
  audit: () => req<any[]>(`${base}/audit`),
  eeg: <T = any,>(path: string) => req<T>(`/api/eeg/${path}`),
  evals: () => req<any>('/api/evals'),
}

/** Live session state over WebSocket, with automatic reconnect. */
export function useSession() {
  const [state, setState] = useState<SessionState | null>(null)
  const [connected, setConnected] = useState(false)
  const retry = useRef(0)

  useEffect(() => {
    let ws: WebSocket | null = null
    let closed = false
    let ping: number | undefined
    const connect = () => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      ws = new WebSocket(`${proto}://${location.host}/ws/${SESSION_ID}`)
      ws.onopen = () => {
        setConnected(true)
        retry.current = 0
        ping = window.setInterval(() => ws?.readyState === 1 && ws.send('ping'), 15000)
      }
      ws.onmessage = (e) => {
        const msg = JSON.parse(e.data)
        if (msg.type === 'state') setState(msg.state)
      }
      ws.onclose = () => {
        setConnected(false)
        window.clearInterval(ping)
        if (!closed) setTimeout(connect, Math.min(8000, 500 * 2 ** retry.current++))
      }
    }
    connect()
    return () => { closed = true; window.clearInterval(ping); ws?.close() }
  }, [])

  return { state, setState, connected }
}

export function useHealth(intervalMs = 4000) {
  const [health, setHealth] = useState<ServiceHealth | null>(null)
  useEffect(() => {
    let alive = true
    const tick = () => api.health().then((h) => alive && setHealth(h)).catch(() => alive && setHealth(null))
    tick()
    const id = window.setInterval(tick, intervalMs)
    return () => { alive = false; window.clearInterval(id) }
  }, [intervalMs])
  return health
}
