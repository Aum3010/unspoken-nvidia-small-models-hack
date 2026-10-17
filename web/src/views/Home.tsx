import { motion } from 'framer-motion'
import { ArrowRight, AudioLines, Brain, Check, Cpu, Eye, Fingerprint, GitMerge, Lock, ShieldCheck, Smile, Sparkles, Volume2 } from 'lucide-react'
import type { ServiceHealth } from '../lib/types'
import './home.css'

interface Props {
  health: ServiceHealth | null
  go: (view: 'communicate' | 'calibrate' | 'brain' | 'evidence', channel?: 'lip' | 'audio' | 'board') => void
}

const MODES = [
  { key: 'lip', svc: 'vsr', title: 'Lips', icon: Smile, color: 'var(--lip)', badge: 'LIVE CAMERA',
    text: 'Mouth words silently. A visual-speech model proposes n-best readings from your lip movements.', model: 'Auto-AVSR · 250M · Metal' },
  { key: 'audio', svc: 'audio', title: 'Weak speech', icon: AudioLines, color: 'var(--audio)', badge: 'LIVE MICROPHONE',
    text: 'Whisper, however faintly. Raw and denoised transcripts are compared word by word — never silently merged.', model: 'Whisper + RNNoise · MLX' },
  { key: 'eyes', svc: null, title: 'Eyes', icon: Eye, color: 'var(--gaze)', badge: 'LIVE CAMERA',
    text: 'Look to choose. Calibrated gaze dwell, or deliberate-blink scanning, over large phrase and spelling targets.', model: 'WebGazer + MediaPipe · in browser' },
  { key: 'brain', svc: 'eeg', title: 'Brain signals', icon: Brain, color: 'var(--eeg)', badge: 'RECORDED EEG DEMO',
    text: 'Research replay: a P300 decoder runs now on EEG recorded from people with ALS. No headset, no live brain input.', model: 'xDAWN + Riemann · BNCI2014-008' },
] as const

const STEPS = [
  { icon: Fingerprint, title: 'You express', text: 'Lips, voice, eyes, or the board — whatever you can use today.', color: 'var(--text)' },
  { icon: Cpu, title: 'Specialists propose', text: 'Small models each read one signal and return raw hypotheses with quality flags.', color: 'var(--lip)' },
  { icon: Sparkles, title: 'Coordinator asks', text: 'Nemotron chooses the smallest useful question — never your answer.', color: 'var(--ai)' },
  { icon: Check, title: 'You confirm', text: 'A separate, deliberate action. Nothing is spoken before it.', color: 'var(--gaze)' },
  { icon: Volume2, title: 'Your voice speaks', text: 'Exactly the confirmed text, rendered locally by Kokoro.', color: 'var(--voice)' },
]

const PRINCIPLES = [
  { icon: ShieldCheck, text: 'No unverified interpretation is ever spoken as yours' },
  { icon: GitMerge, text: 'When agents disagree, you choose — nothing is merged' },
  { icon: Brain, text: 'Recorded EEG is labelled as replay, everywhere' },
  { icon: Lock, text: 'Runs on this Mac. Clips are never stored' },
]

export function Home({ health, go }: Props) {
  return (
    <div className="home">
      <section className="container hero">
        <motion.div className="hero-copy" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .9, ease: [0.22, 1, 0.36, 1] }}>
          <div className="eyebrow"><span className="eyebrow-dot" /> Assistive communication · research prototype</div>
          <h1 className="display hero-title">
            When speech disappears,<br /><em>communication</em> shouldn't.
          </h1>
          <p className="hero-sub">
            Unspoken combines small specialist models across lips, voice, eyes and recorded brain signals —
            coordinated to ask, never to assume — so a person can say exactly what they mean.
          </p>
          <div className="row wrap" style={{ gap: 12 }}>
            <button className="btn lg primary" onClick={() => go('communicate', 'lip')}>Start communicating <ArrowRight size={20} /></button>
            <button className="btn lg" onClick={() => go('brain')}><Brain size={20} /> Brain-signal replay</button>
          </div>
        </motion.div>
        <HeroVisual />
      </section>

      <section className="container">
        <div className="modes-grid">
          {MODES.map((m, i) => {
            const Icon = m.icon
            const st = m.svc ? health?.[m.svc]?.state ?? 'loading' : 'browser'
            const stText = st === 'ready' ? (m.key === 'brain' ? 'Replay only' : 'Ready') : st === 'browser' ? 'Calibrate' : st === 'loading' ? 'Loading' : 'Unavailable'
            return (
              <motion.button key={m.key} className="mode-card glass" style={{ ['--c' as any]: m.color }}
                initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: .15 + i * .08, duration: .7, ease: [0.22, 1, 0.36, 1] }}
                onClick={() => m.key === 'eyes' ? go('calibrate') : m.key === 'brain' ? go('brain') : go('communicate', m.key)}>
                <div className="mode-top">
                  <span className="mode-icon"><Icon size={26} /></span>
                  <span className="badge" style={{ ['--c' as any]: m.color }}>{m.badge}</span>
                </div>
                <div className="display mode-title">{m.title}</div>
                <p>{m.text}</p>
                <div className="mode-foot">
                  <span className="mono faint">{m.model}</span>
                  <span className="chip" style={{ ['--c' as any]: st === 'ready' ? m.color : st === 'browser' ? 'var(--gaze)' : st === 'loading' ? 'var(--warn)' : 'var(--danger)' }}>
                    <span className="dot" />{stText}
                  </span>
                </div>
                <span className="mode-glow" />
              </motion.button>
            )
          })}
        </div>
      </section>

      <section className="container how">
        <div className="section-head">
          <div className="eyebrow">How it works</div>
          <h2 className="display">Many small models. <em>One</em> person in control.</h2>
          <p className="muted">The valuable multi-agent behaviour is active uncertainty resolution: when lip reading is unsure, another modality asks the person — and the person decides.</p>
        </div>
        <div className="steps">
          {STEPS.map((s, i) => {
            const Icon = s.icon
            return (
              <motion.div key={s.title} className="step" style={{ ['--c' as any]: s.color }}
                initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: .4 + i * .08, duration: .6 }}>
                <span className="step-n mono">0{i + 1}</span>
                <span className="step-icon"><Icon size={22} /></span>
                <div className="step-title">{s.title}</div>
                <div className="step-text">{s.text}</div>
              </motion.div>
            )
          })}
        </div>
      </section>

      <section className="container">
        <div className="principles glass">
          {PRINCIPLES.map((p) => {
            const Icon = p.icon
            return <div key={p.text} className="principle"><Icon size={20} /><span>{p.text}</span></div>
          })}
        </div>
        <p className="disclaimer">
          Unspoken is an assistive communication research prototype. It is not a medical device, not a nurse-call or emergency system,
          and does not diagnose. It is not a mind reader: EEG features here replay published recordings with honest labels.
          <button className="link" onClick={() => go('evidence')}>See evaluation & model ledger →</button>
        </p>
      </section>
    </div>
  )
}

function HeroVisual() {
  const sources = [
    { c: 'var(--lip)', y: 70, label: 'lips' },
    { c: 'var(--audio)', y: 150, label: 'voice' },
    { c: 'var(--gaze)', y: 230, label: 'eyes' },
    { c: 'var(--eeg)', y: 310, label: 'EEG' },
  ]
  return (
    <motion.div className="hero-visual" initial={{ opacity: 0, scale: .96 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 1.2, delay: .2 }}>
      <svg viewBox="0 0 560 380" width="100%" aria-hidden>
        <defs>
          <linearGradient id="voice-g" x1="0" x2="1">
            <stop offset="0" stopColor="var(--ai)" />
            <stop offset="1" stopColor="var(--voice)" />
          </linearGradient>
          <filter id="glow"><feGaussianBlur stdDeviation="4" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
        </defs>
        {sources.map((s, i) => (
          <g key={s.label}>
            <motion.path
              d={`M20 ${s.y} C 120 ${s.y + (i % 2 ? -26 : 26)}, 180 ${s.y}, 250 ${(s.y + 190) / 2} S 300 190, 330 190`}
              fill="none" stroke={s.c} strokeWidth="2.2" strokeLinecap="round" filter="url(#glow)"
              initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: .9 }} transition={{ duration: 1.6, delay: .3 + i * .15, ease: 'easeInOut' }} />
            <motion.circle r="4" fill={s.c} filter="url(#glow)"
              animate={{ offsetDistance: ['0%', '100%'] } as any}
              style={{ offsetPath: `path("M20 ${s.y} C 120 ${s.y + (i % 2 ? -26 : 26)}, 180 ${s.y}, 250 ${(s.y + 190) / 2} S 300 190, 330 190")` } as any}
              transition={{ repeat: Infinity, duration: 3.2, delay: i * .6, ease: 'easeInOut' }} />
            <text x="22" y={s.y - 12} fontSize="12" fill="var(--text-3)" fontFamily="var(--font-mono)" letterSpacing="1">{s.label.toUpperCase()}</text>
          </g>
        ))}
        {/* coordinator node */}
        <motion.circle cx="340" cy="190" r="26" fill="var(--bg-elev)" stroke="var(--ai)" strokeWidth="2" filter="url(#glow)"
          animate={{ r: [24, 28, 24] }} transition={{ repeat: Infinity, duration: 2.6 }} />
        <text x="340" y="194" textAnchor="middle" fontSize="11" fill="var(--ai)" fontFamily="var(--font-mono)">ASK</text>
        {/* confirm gate */}
        <rect x="392" y="172" width="36" height="36" rx="10" fill="var(--bg-elev)" stroke="var(--gaze)" strokeWidth="2" />
        <path d="M402 190 l6 6 l11 -12" stroke="var(--gaze)" strokeWidth="2.6" fill="none" strokeLinecap="round" strokeLinejoin="round" />
        <line x1="366" y1="190" x2="392" y2="190" stroke="var(--hairline-strong)" strokeWidth="2" />
        {/* voice */}
        {Array.from({ length: 12 }, (_, i) => {
          const h = [10, 22, 36, 52, 70, 84, 84, 70, 52, 36, 22, 10][i]
          return (
            <motion.rect key={i} x={446 + i * 9} width="5" rx="2.5" fill="url(#voice-g)"
              animate={{ height: [h * .35, h, h * .5, h * .85, h * .35], y: [190 - h * .175, 190 - h / 2, 190 - h * .25, 190 - h * .425, 190 - h * .175] }}
              transition={{ repeat: Infinity, duration: 1.6 + (i % 4) * .15, ease: 'easeInOut', delay: i * .05 }} />
          )
        })}
      </svg>
      <div className="hero-quote">
        <span className="badge" style={{ ['--c' as any]: 'var(--voice)' }}>CONFIRMED</span>
        <span className="display">“I need water, please.”</span>
      </div>
    </motion.div>
  )
}
