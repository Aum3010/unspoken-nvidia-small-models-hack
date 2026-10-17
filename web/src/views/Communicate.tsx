import { motion } from 'framer-motion'
import { AudioLines, Crosshair, Eye, Grid3x3, Keyboard, MousePointer2, ScanLine, ShieldAlert, Smile } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { AgentTrace, History } from '../components/AgentTrace'
import { PhraseBoard, SpellBoard } from '../components/Board'
import { Composer } from '../components/Composer'
import { SensorStage } from '../components/SensorStage'
import { api } from '../lib/api'
import { gazeState, startGaze } from '../lib/gaze'
import { useSelection, type InputMode } from '../lib/selection'
import type { SessionState, SpokenItem } from '../lib/types'
import './communicate.css'

export type Channel = 'lip' | 'audio' | 'board' | 'spell'

const CHANNELS: { id: Channel; label: string; icon: any; color: string; hint: string }[] = [
  { id: 'lip', label: 'Lips', icon: Smile, color: 'var(--lip)', hint: 'Silent mouthing' },
  { id: 'audio', label: 'Weak speech', icon: AudioLines, color: 'var(--audio)', hint: 'Quiet or unclear voice' },
  { id: 'board', label: 'Phrases', icon: Grid3x3, color: 'var(--board)', hint: 'Large targets' },
  { id: 'spell', label: 'Spell', icon: Keyboard, color: 'var(--board)', hint: 'Letter by letter' },
]

const MODES: { id: InputMode; label: string; icon: any; desc: string }[] = [
  { id: 'pointer', label: 'Touch / click', icon: MousePointer2, desc: 'Buttons, touch, keyboard (Tab + Enter)' },
  { id: 'hover', label: 'Hover dwell', icon: Crosshair, desc: 'Rest a head-mouse or pointer on a target' },
  { id: 'gaze', label: 'Gaze dwell', icon: Eye, desc: 'Look at a target until the ring fills' },
  { id: 'scan', label: 'Blink scan', icon: ScanLine, desc: 'Targets light up in turn; a long blink (or Space) selects' },
]

interface Props {
  state: SessionState | null
  connected: boolean
  channel: Channel
  setChannel: (c: Channel) => void
  notify: (m: string, k?: 'error' | 'info') => void
  speak: (url: string | null, text: string) => void
  goCalibrate: () => void
}

export function Communicate({ state, connected, channel, setChannel, notify, speak, goCalibrate }: Props) {
  const sel = useSelection()
  const [busy, setBusy] = useState(false)

  // keep the gateway informed about which inputs the person can use
  useEffect(() => {
    if (!state) return
    const inputs = sel.mode === 'gaze' ? ['gaze', 'button', 'keyboard'] : sel.mode === 'scan' ? ['blink', 'button', 'keyboard'] : ['button', 'keyboard']
    const cal = gazeState.calibrated && (gazeState.accuracyPx ?? 999) < 220
    if (JSON.stringify(inputs) !== JSON.stringify(state.settings.available_inputs) || cal !== state.settings.gaze_calibrated) {
      api.settings({ available_inputs: inputs, gaze_calibrated: cal }).catch(() => undefined)
    }
  }, [sel.mode, state?.settings.available_inputs, state?.settings.gaze_calibrated])

  const setMode = async (m: InputMode) => {
    if (m === 'gaze') {
      if (!gazeState.calibrated) { goCalibrate(); return }
      try { await startGaze() } catch { notify('Gaze tracking could not start', 'error'); return }
    }
    sel.setSettings({ mode: m })
  }

  const run = useCallback(async (fn: () => Promise<unknown>) => {
    setBusy(true)
    try { await fn() } catch (e: any) { notify(e.message ?? 'Something went wrong', 'error') } finally { setBusy(false) }
  }, [notify])

  const onSwitch = (method: string) => {
    if (method === 'spelling') setChannel('spell')
    else if (method === 'board') setChannel('board')
    else if (method === 'lip' || method === 'audio') setChannel(method)
    else if (method === 'gaze') setMode('gaze')
    else if (method === 'blink') setMode('scan')
    api.newMessage().catch(() => undefined)
  }

  const replay = (s: SpokenItem) => speak(`/api/${state?.session_id}/audio/${s.message_id}`, s.text)

  if (!state) {
    return (
      <div className="container comm-loading">
        <div className="display shimmer" style={{ fontSize: '2.4rem' }}>{connected ? 'Opening session…' : 'Connecting to the local gateway…'}</div>
        <div className="faint">Start the services with <code className="mono">./scripts/run_all.sh</code></div>
      </div>
    )
  }

  const sensorChannel: 'lip' | 'audio' = channel === 'audio' ? 'audio' : 'lip'

  return (
    <div className="container comm">
      <div className="channel-bar">
        {CHANNELS.map((c) => {
          const Icon = c.icon
          return (
            <button key={c.id} className="channel" aria-pressed={channel === c.id} onClick={() => setChannel(c.id)} style={{ ['--c' as any]: c.color }}>
              <span className="channel-icon"><Icon size={20} /></span>
              <span><span className="channel-label">{c.label}</span><span className="channel-hint">{c.hint}</span></span>
              {channel === c.id && <motion.span layoutId="chan" className="channel-glow" />}
            </button>
          )
        })}
      </div>

      <div className="comm-grid">
        <section className="col-left">
          <SensorStage
            mode={sensorChannel}
            busy={busy}
            onLipClip={(b) => run(() => api.captureLip(b))}
            onAudioClip={(b, enhance) => run(() => api.captureAudio(b, enhance))}
          />
          <div className="glass panel">
            <div className="panel-title">How you select</div>
            <div className="modes">
              {MODES.map((m) => {
                const Icon = m.icon
                return (
                  <button key={m.id} className="mode" aria-pressed={sel.mode === m.id} onClick={() => setMode(m.id)}>
                    <Icon size={18} />
                    <span className="grow"><b>{m.label}</b><span>{m.desc}</span></span>
                  </button>
                )
              })}
            </div>
            {(sel.mode === 'gaze' || sel.mode === 'hover') && (
              <label className="slider">
                <span>Dwell <b>{(sel.dwellMs / 1000).toFixed(1)}s</b> · confirm <b>{(sel.confirmDwellMs / 1000).toFixed(1)}s</b></span>
                <input type="range" min={600} max={2500} step={100} value={sel.dwellMs}
                  onChange={(e) => sel.setSettings({ dwellMs: +e.target.value, confirmDwellMs: Math.round(+e.target.value * 1.5) })} />
              </label>
            )}
            {sel.mode === 'scan' && (
              <label className="slider">
                <span>Scan speed <b>{(sel.scanMs / 1000).toFixed(1)}s</b> per target</span>
                <input type="range" min={800} max={3000} step={100} value={sel.scanMs} onChange={(e) => sel.setSettings({ scanMs: +e.target.value })} />
              </label>
            )}
            {sel.mode === 'gaze' && (
              <div className="gaze-q">
                <span className="chip" style={{ ['--c' as any]: (gazeState.accuracyPx ?? 999) < 120 ? 'var(--gaze)' : 'var(--warn)' }}>
                  <span className="dot" />Calibration error ≈ {gazeState.accuracyPx ? `${Math.round(gazeState.accuracyPx)} px` : '—'}
                </span>
                <button className="btn sm" onClick={goCalibrate}>Recalibrate</button>
              </div>
            )}
          </div>
          <SafetyProbe notify={notify} />
        </section>

        <section className="col-center">
          <Composer state={state} onSpoken={speak} onSwitch={onSwitch} onRetake={(m) => setChannel(m === 'audio' ? 'audio' : 'lip')} notify={notify} />
          {channel === 'spell' ? <SpellBoard notify={notify} /> : <PhraseBoard notify={notify} />}
        </section>

        <section className="col-right">
          <AgentTrace timeline={state.timeline} />
          <History spoken={state.spoken} onReplay={replay} />
        </section>
      </div>
    </div>
  )
}

function SafetyProbe({ notify }: { notify: (m: string, k?: 'error' | 'info') => void }) {
  return (
    <button className="probe" onClick={() => api.safetyProbe().then((r) => notify(`Policy gate: ${r.outcome}`, 'info'))}>
      <ShieldAlert size={16} />
      <span className="grow"><b>Test the policy gate</b><span>Simulate a model trying to speak an unconfirmed guess</span></span>
    </button>
  )
}
