import { AnimatePresence, motion } from 'framer-motion'
import { AArrowUp, Brain, Contrast, Eye, FlaskConical, Home as HomeIcon, MessageSquareText, Moon, Sun, UserRound, X } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { BrandMark } from './components/Brand'
import { StatusBar } from './components/StatusBar'
import { useHealth, useSession } from './lib/api'
import { gazeState } from './lib/gaze'
import { SelectionProvider, useSelection } from './lib/selection'
import { BrainReplay } from './views/BrainReplay'
import { Calibrate } from './views/Calibrate'
import { Communicate, type Channel } from './views/Communicate'
import { Evidence } from './views/Evidence'
import { Home } from './views/Home'
import { MyPhrases } from './views/MyPhrases'
import './app.css'

type View = 'home' | 'communicate' | 'phrases' | 'calibrate' | 'brain' | 'evidence'
interface Toast { id: number; text: string; kind: 'error' | 'info' }

const NAV: { id: View; label: string; icon: any }[] = [
  { id: 'home', label: 'Home', icon: HomeIcon },
  { id: 'communicate', label: 'Communicate', icon: MessageSquareText },
  { id: 'phrases', label: 'My phrases', icon: UserRound },
  { id: 'calibrate', label: 'Eyes', icon: Eye },
  { id: 'brain', label: 'Brain replay', icon: Brain },
  { id: 'evidence', label: 'Evidence', icon: FlaskConical },
]

function pref<T>(k: string, d: T): T {
  try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d } catch { return d }
}
function save(k: string, v: unknown) { try { localStorage.setItem(k, JSON.stringify(v)) } catch { /* ignore */ } }

export default function App() {
  return (
    <SelectionProvider>
      <Shell />
    </SelectionProvider>
  )
}

function Shell() {
  const [view, setView] = useState<View>(() => (location.hash.slice(1) as View) || 'home')
  const [channel, setChannel] = useState<Channel>('lip')
  const [theme, setTheme] = useState<'dark' | 'light'>(() => pref('unspoken.theme', 'dark'))
  const [contrast, setContrast] = useState<boolean>(() => pref('unspoken.contrast', false))
  const [scale, setScale] = useState<number>(() => pref('unspoken.scale', 1))
  const [toasts, setToasts] = useState<Toast[]>([])
  const [speaking, setSpeaking] = useState(false)
  const { state, connected } = useSession()
  const health = useHealth()
  const sel = useSelection()

  useEffect(() => { document.documentElement.dataset.theme = theme; save('unspoken.theme', theme) }, [theme])
  useEffect(() => { document.documentElement.dataset.contrast = contrast ? 'high' : 'normal'; save('unspoken.contrast', contrast) }, [contrast])
  useEffect(() => { document.documentElement.style.setProperty('--text-scale', String(scale)); save('unspoken.scale', scale) }, [scale])
  useEffect(() => { location.hash = view; window.scrollTo({ top: 0 }) }, [view])

  const notify = useCallback((text: string, kind: 'error' | 'info' = 'info') => {
    const id = Date.now() + Math.random()
    setToasts((t) => [...t.slice(-3), { id, text, kind }])
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), kind === 'error' ? 6500 : 4200)
  }, [])

  // Speech: Kokoro audio from the gateway, or the browser voice as a last resort
  // (still the exact confirmed text; never anything else).
  const speak = useCallback((url: string | null, text: string) => {
    setSpeaking(true)
    const done = () => setSpeaking(false)
    if (url) {
      const a = new Audio(url)
      a.onended = done
      a.onerror = () => fallback()
      a.play().catch(() => fallback())
    } else fallback()
    function fallback() {
      const u = new SpeechSynthesisUtterance(text)
      u.onend = done
      speechSynthesis.speak(u)
    }
  }, [])

  const gazeLabel = !gazeState.calibrated ? (sel.mode === 'gaze' ? 'Calibrate' : 'Off') : (gazeState.accuracyPx ?? 999) < 120 ? 'Ready' : 'Low confidence'

  return (
    <div className="app">
      <div className="aurora" aria-hidden />
      <div className="grain" aria-hidden />
      <header className="topbar">
        <div className="container topbar-inner">
          <button className="brand" onClick={() => setView('home')} aria-label="Unspoken home">
            <BrandMark speaking={speaking} />
            <span className="brand-name">Unspoken</span>
          </button>
          <nav className="nav" aria-label="Main">
            {NAV.map((n) => {
              const Icon = n.icon
              return (
                <button key={n.id} aria-current={view === n.id ? 'page' : undefined} onClick={() => setView(n.id)}>
                  <Icon size={16} /><span>{n.label}</span>
                </button>
              )
            })}
          </nav>
          <div className="grow" />
          <div className="topbar-status"><StatusBar health={health} gazeState={gazeLabel} /></div>
          <div className="row" style={{ gap: 6 }}>
            <button className="icon-btn" onClick={() => setScale((s) => (s >= 1.3 ? 1 : +(s + 0.15).toFixed(2)))} aria-label="Larger text" title="Text size"><AArrowUp size={18} /></button>
            <button className="icon-btn" onClick={() => setContrast((c) => !c)} aria-pressed={contrast} aria-label="High contrast" title="High contrast"><Contrast size={18} /></button>
            <button className="icon-btn" onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))} aria-label="Toggle theme" title="Theme">
              {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
            </button>
          </div>
        </div>
        {view === 'communicate' && <div className="container topbar-status-mobile"><StatusBar health={health} gazeState={gazeLabel} /></div>}
      </header>

      <main className="main">
        <AnimatePresence mode="wait">
          <motion.div key={view} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: .35, ease: [0.22, 1, 0.36, 1] }}>
            {view === 'home' && <Home health={health} go={(v, ch) => { if (ch) setChannel(ch); setView(v) }} />}
            {view === 'communicate' && (
              <Communicate state={state} connected={connected} channel={channel} setChannel={setChannel} notify={notify} speak={speak} goCalibrate={() => setView('calibrate')} />
            )}
            {view === 'phrases' && <MyPhrases notify={notify} />}
            {view === 'calibrate' && <Calibrate onDone={() => setView('communicate')} />}
            {view === 'brain' && <BrainReplay notify={notify} goCommunicate={() => setView('communicate')} />}
            {view === 'evidence' && <Evidence />}
          </motion.div>
        </AnimatePresence>
      </main>

      {sel.mode === 'gaze' && sel.gaze && <div className="gaze-cursor" style={{ left: sel.gaze.x, top: sel.gaze.y }} aria-hidden />}

      <div className="toasts" role="status" aria-live="polite">
        <AnimatePresence>
          {toasts.map((t) => (
            <motion.div key={t.id} className={`toast ${t.kind}`} initial={{ opacity: 0, y: 16, scale: .96 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, scale: .96 }}>
              <span className="grow">{t.text}</span>
              <button className="link" onClick={() => setToasts((x) => x.filter((y) => y.id !== t.id))} aria-label="Dismiss"><X size={14} /></button>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
      {!connected && view === 'communicate' && <div className="offline">Gateway offline — reconnecting. The phrase board needs the local gateway.</div>}
    </div>
  )
}
