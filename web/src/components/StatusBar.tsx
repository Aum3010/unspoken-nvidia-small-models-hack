import type { ServiceHealth } from '../lib/types'

const SERVICES: { key: string; label: string; color: string }[] = [
  { key: 'vsr', label: 'Lips', color: 'var(--lip)' },
  { key: 'audio', label: 'Voice', color: 'var(--audio)' },
  { key: 'coordinator', label: 'Coordinator', color: 'var(--ai)' },
  { key: 'eeg', label: 'EEG replay', color: 'var(--eeg)' },
  { key: 'tts', label: 'Speech', color: 'var(--voice)' },
]

const STATE_TEXT: Record<string, string> = {
  ready: 'Ready', loading: 'Loading', unavailable: 'Unavailable',
}

export function StatusBar({ health, gazeState }: { health: ServiceHealth | null; gazeState: 'Ready' | 'Calibrate' | 'Low confidence' | 'Off' }) {
  return (
    <div className="row wrap" style={{ gap: 6 }} role="status" aria-label="Subsystem status">
      {SERVICES.map((s) => {
        const st = health?.[s.key]?.state ?? (health ? 'unavailable' : 'loading')
        const c = st === 'ready' ? s.color : st === 'loading' ? 'var(--warn)' : 'var(--danger)'
        const label = s.key === 'eeg' && st === 'ready' ? 'Replay only' : STATE_TEXT[st] ?? st
        return (
          <span key={s.key} className={`chip ${st === 'loading' ? 'pulse' : ''}`} style={{ ['--c' as any]: c }}
            title={health?.[s.key]?.error ?? health?.[s.key]?.checkpoint ?? ''}>
            <span className="dot" />{s.label}<span className="chip-state" style={{ opacity: .7, fontWeight: 500 }}>· {label}</span>
          </span>
        )
      })}
      <span className="chip" style={{ ['--c' as any]: gazeState === 'Ready' ? 'var(--gaze)' : gazeState === 'Off' ? 'var(--text-3)' : 'var(--warn)' }}>
        <span className="dot" />Eyes<span className="chip-state" style={{ opacity: .7, fontWeight: 500 }}>· {gazeState}</span>
      </span>
    </div>
  )
}
