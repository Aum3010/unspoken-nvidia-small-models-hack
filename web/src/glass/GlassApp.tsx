// The Unspoken room: one liquid-glass page wired to the local gateway.
//   every input runs at once: lips + blink + head (camera), voice (mic), gaze (eyes or pointer)
//   readings arrive as faint ink and glass bubbles; nothing is spoken until the person confirms
//   (Yes, a deliberate blink or a nod), then the policy gate signs the exact text for TTS
//   top right: live agent activity from the gateway timeline, then the spoken history

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { api, useHealth, useSession } from '../lib/api'
import { getCamera } from '../lib/capture'
import { onBlink, trackVideo } from '../lib/face'
import { gazeState, learnFromSelection, onGaze, saveCalibration, startGaze } from '../lib/gaze'
import type { Candidate, ConfirmationMethod, Modality, SessionState, TimelineItem } from '../lib/types'
import { Calibration } from './Calibration'
import * as E from './engine'
import { startHead, startLips, startVoice } from './sensors'
import './glass.css'

const METHOD: Record<E.How, ConfirmationMethod> = {
  eyes: 'gaze_dwell_confirmed', pointer: 'other_explicit_user_confirmed', blink: 'deliberate_blink_confirmed',
  space: 'keyboard_confirmed', click: 'accessible_button_confirmed', keyboard: 'keyboard_confirmed',
  touch: 'accessible_button_confirmed', nod: 'other_explicit_user_confirmed',
}
const HOW_TEXT: Record<E.How, string> = {
  eyes: 'your eyes', pointer: 'dwell', blink: 'a blink', space: 'the space key', click: 'a click', keyboard: 'the keyboard', touch: 'touch', nod: 'a nod',
}
const METHOD_NODE: Record<string, E.NodeId | undefined> = {
  gaze_dwell_confirmed: 'gaze', deliberate_blink_confirmed: 'blink', other_explicit_user_confirmed: 'gaze',
}
const METHOD_TEXT: Record<string, string> = {
  gaze_dwell_confirmed: 'gaze', deliberate_blink_confirmed: 'blink', accessible_button_confirmed: 'button',
  keyboard_confirmed: 'keyboard', other_explicit_user_confirmed: 'dwell / nod',
}
// modality -> its input's colour in the room
const MOD_NODE: Partial<Record<Modality, E.NodeId>> = { audio: 'voice', lip: 'lips', gaze: 'gaze', eeg_replay: 'eeg', gesture: 'head' }
const MOD_LABEL: Record<string, string> = { audio: 'Voice', lip: 'Lips', gaze: 'Eyes', gesture: 'Head', board: 'Phrase', eeg_replay: 'Recorded EEG', spelling: 'Spelled' }
const AGENTS: Record<string, { label: string; c: E.Rgb; node?: E.NodeId }> = {
  audio: { label: 'Voice agent', c: E.COL.voice, node: 'voice' },
  vsr: { label: 'Lip agent', c: E.COL.lips, node: 'lips' },
  eeg: { label: 'EEG replay', c: E.COL.eeg, node: 'eeg' },
  gaze: { label: 'Gaze', c: E.COL.gaze, node: 'gaze' },
  coordinator: { label: 'Coordinator', c: [255, 150, 192] },
  policy: { label: 'Policy gate', c: [94, 234, 212] },
  tts: { label: 'Voice out', c: [255, 236, 206] },
  user: { label: 'You', c: [241, 245, 255] },
}

const PHRASES: [string, string, string, string][] = [
  ['I need water', 'need', '94,200,255', '💧'], ['Please help me', 'care', '255,184,108', '🤲'], ['I am in pain', 'body', '255,128,112', '🩹'],
  ['Thank you', 'warmth', '181,232,107', '🙏'], ['I love you', 'warmth', '232,121,249', '❤️'], ['Move me, please', 'body', '165,180,252', '↻'],
]
const HELP_TEXT = 'I need help now.'
const PENDING = ['proposed', 'ambiguous', 'clarifying']
const words = (t: string) => t.split(/\s+/).filter(Boolean)

function shownCandidates(s: SessionState): Candidate[] {
  const m = s.message
  const ids = s.shown_candidate_ids.length ? s.shown_candidate_ids : m.candidates.slice(0, 3).map((c) => c.id)
  return ids.map((id) => m.candidates.find((c) => c.id === id)).filter(Boolean) as Candidate[]
}
function quote(t: string, n = 16) { const u = t.toUpperCase(); return `“${u.length > n ? u.slice(0, n - 1) + '…' : u}”` }

export function GlassApp() {
  const { state, connected } = useSession()
  const health = useHealth()
  // a saved calibration (same window size) is used straight away: no recalibrating after a reload
  const [calibrating, setCalibrating] = useState(() => !gazeState.calibrated)
  const [focusMode, setFocusMode] = useState(false)
  const [eyes, setEyes] = useState(() => gazeState.calibrated)
  const [cam, setCam] = useState<'starting' | 'on' | 'off'>('starting')
  const [mic, setMic] = useState<'starting' | 'on' | 'off'>('starting')
  const [hearing, setHearing] = useState(false)
  const [alarm, setAlarm] = useState(0)
  const camRef = useRef<HTMLVideoElement>(null)
  const histRef = useRef<HTMLDivElement>(null)
  // the latest render's values, for sensor and engine callbacks that outlive a render
  const live = useRef({ state: null as SessionState | null, health, connected, calibrating })
  useLayoutEffect(() => { live.current = { state, health, connected, calibrating } })
  const speakingUntil = useRef(0)
  const calibAct = useRef<((a: string) => void) | null>(null)
  const actRef = useRef<(a: string, el: HTMLElement | null, how: E.How) => void>(() => {})

  useEffect(() => E.mount(), [])
  useEffect(() => { E.setPaused(focusMode) }, [focusMode])
  useEffect(() => { E.setGazeSource(eyes ? 'eyes' : 'pointer'); E.setGazeError(gazeState.accuracyPx) }, [eyes])
  useEffect(() => onGaze((p) => E.feedGaze(p)), [])
  useEffect(() => {
    if (!gazeState.calibrated) return
    void startGaze().then(() => saveCalibration())   // keep what this session has learned, too
    setTimeout(() => { E.greet(); E.caption('Using your saved eye calibration', 'rest your eyes 1.2 s on anything, or blink, to choose') }, 500)
  }, [])

  // capture is allowed only while nothing waits for confirmation and nothing is being spoken
  const canCapture = useCallback((svc: string) => {
    const { state: s, health: h, connected: c, calibrating: cal } = live.current
    return !!s && c && !cal && performance.now() > speakingUntil.current &&
      !['selected', 'confirmed'].includes(s.message.status) && h?.[svc]?.state === 'ready'
  }, [])

  const play = useCallback((url: string | null, text: string) => {
    speakingUntil.current = performance.now() + 60000
    E.fireOut()
    const done = () => { speakingUntil.current = performance.now() + 700 }
    const fallback = () => { const u = new SpeechSynthesisUtterance(text); u.onend = done; u.onerror = done; speechSynthesis.speak(u) }
    if (url) {
      const a = new Audio(url)
      a.onended = done
      a.onerror = fallback
      a.play().catch(fallback)
    } else fallback()
  }, [])

  const run = useCallback((p: Promise<unknown>) => p.catch((e: Error) => E.caption(e.message || 'Something went wrong', 'the gateway refused that')), [])

  /* ---------- camera: the room, lips, blink, head ---------- */
  useEffect(() => {
    let stops: (() => void)[] = []
    let alive = true
    const start = async () => {
      try {
        const s = await getCamera()
        const v = camRef.current
        if (!alive || !v) return
        v.srcObject = s
        await v.play().catch(() => undefined)
        setCam('on')
        stops.push(trackVideo(v))
        stops.push(onBlink(() => E.blink('blink')))
        stops.push(startHead((g) => {
          E.fire('head', 1, g === 'nod' ? 'NOD' : 'SHAKE')
          actRef.current(g, null, 'nod')
        }))
        stops.push(startLips({
          stream: s,
          canCapture: () => canCapture('vsr'),
          onMouthing: (on) => { if (on) E.fire('lips', .6, 'WATCHING') },
          onClip: (b) => { E.fire('lips', 1, 'READING'); run(api.captureLip(b)) },
        }))
      } catch {
        setCam('off')
      }
    }
    start()
    return () => { alive = false; stops.forEach((f) => f()); stops = [] }
  }, [run, canCapture])

  /* ---------- microphone: voice ---------- */
  useEffect(() => {
    let stop: (() => void) | null = null
    let alive = true
    startVoice({
      canCapture: () => canCapture('audio'),
      onSpeech: (on) => { setHearing(on); if (on) E.fire('voice', .7, 'HEARING') },
      onClip: (b) => { E.fire('voice', 1, 'DECODING'); run(api.captureAudio(b, true)) },
    }).then((f) => { if (alive) { stop = f; setMic('on') } else f() }).catch(() => setMic('off'))
    return () => { alive = false; stop?.() }
  }, [run, canCapture])

  /* ---------- tell the gateway what the person can use ---------- */
  useEffect(() => {
    if (!state || calibrating) return
    const inputs = [...(eyes ? ['gaze'] : []), ...(cam === 'on' ? ['blink'] : []), 'button', 'keyboard']
    const cal = eyes && gazeState.calibrated
    if (JSON.stringify(inputs) !== JSON.stringify(state.settings.available_inputs) || cal !== state.settings.gaze_calibrated) {
      api.settings({ available_inputs: inputs, gaze_calibrated: cal }).catch(() => undefined)
    }
  }, [state, eyes, cam, calibrating])

  /* ---------- the six inputs in the activity nucleus ---------- */
  useEffect(() => {
    const svc = (name: string, on: boolean, tag: string): [E.NodeState, string] => {
      if (!on) return ['off', tag]
      const st = health?.[name]?.state
      return st === 'ready' ? ['on', 'LIVE'] : ['off', st === 'loading' ? 'LOADING' : 'OFFLINE']
    }
    E.setNode('voice', ...svc('audio', mic === 'on', mic === 'starting' ? 'STARTING' : 'NO MIC'))
    E.setNode('lips', ...svc('vsr', cam === 'on', cam === 'starting' ? 'STARTING' : 'NO CAMERA'))
    E.setNode('blink', cam === 'on' ? 'on' : 'off', cam === 'on' ? 'LIVE' : 'NO CAMERA')
    E.setNode('head', cam === 'on' ? 'on' : 'off', cam === 'on' ? 'LIVE' : 'NO CAMERA')
    E.setNode('gaze', 'on', eyes ? 'LIVE' : 'POINTER')
    E.setNode('eeg', health?.eeg?.state === 'ready' ? 'on' : 'off', health?.eeg?.state === 'ready' ? 'REPLAY' : 'OFFLINE')
  }, [health, mic, cam, eyes])

  /* ---------- gateway timeline -> pulses in the nucleus ---------- */
  const seen = useRef(new Set<string>())
  const primed = useRef(false)
  useEffect(() => {
    if (!state) return
    const fresh = state.timeline.filter((t) => !seen.current.has(t.id))
    fresh.forEach((t) => seen.current.add(t.id))
    if (!primed.current) { primed.current = true; return } // history from before this page opened: listed, not replayed
    for (const t of fresh) {
      const a = AGENTS[t.agent]
      if (a?.node) {
        const top = /“(.+?)”/.exec(t.detail)?.[1]
        const note = t.level === 'error' ? 'OFFLINE' : t.level === 'pending' ? 'DECODING'
          : top ? `${a.node === 'lips' ? 'SAW' : a.node === 'voice' ? 'HEARD' : 'READ'} ${quote(top)}` : 'NO READING'
        E.fire(a.node, t.level === 'error' ? .4 : 1, note)
      } else if (t.agent === 'coordinator') {
        E.setOrch(t.level === 'pending' ? 'Nemotron is planning…'
          : /ask_confirmation/.test(t.title) ? 'Asking which you meant'
            : /request_retake/.test(t.title) ? 'Asking for one more try'
              : /ignore_input/.test(t.title) ? 'Ignored as noise'
              : /switch_input_method/.test(t.title) ? 'Suggesting another way'
                : /offline|fallback/i.test(t.title) ? 'Deterministic policy' : t.title, 5000)
      } else if (t.agent === 'policy') {
        E.setOrch(t.title, 4000)
      } else if (t.agent === 'tts') {
        E.setOrch(t.level === 'warn' ? 'Browser voice' : 'Speaking for you', 3000)
      }
    }
  }, [state])

  /* ---------- session state -> ink, bubbles, caption ---------- */
  const lastSpoken = useRef<string | null>(null)
  useEffect(() => {
    if (!state || calibrating) return
    const m = state.message
    const shown = shownCandidates(state)

    if (m.status === 'selected') {
      const c = m.candidates.find((x) => x.id === m.selected_candidate_id)
      if (c) {
        E.setTranscript([], words(c.text))
        E.caption(`Will say exactly “${c.text}”`, 'look at Yes, blink, or nod to say it · No or shake to change it')
      }
    } else if (PENDING.includes(m.status) && shown.length) {
      // only the best reading, written in ink; Yes/blink/nod chooses it, No/shake rejects it
      E.setTranscript(words(shown[0].text), [])
      E.caption('Did you mean this?', 'Yes, blink or nod to choose it · No or shake if it is wrong')
    } else if (m.status === 'spoken') {
      const sp = state.spoken[state.spoken.length - 1]
      if (sp && sp.message_id !== lastSpoken.current) {
        lastSpoken.current = sp.message_id
        E.setTranscript(words(sp.text), [])
        E.caption(`Spoken · confirmed by ${METHOD_TEXT[sp.method] ?? sp.method}`)
      }
    } else if (m.status === 'rejected') {
      E.setTranscript([], [])
      E.caption('None of these', 'nothing will be spoken · try again, or open Phrases')
    } else if (state.plan?.name === 'ignore_input' && !m.candidates.length) {
      E.setTranscript([], [])
      E.caption('Listening', 'mouth or say what you need, or open Phrases')
    } else if (state.plan?.name === 'request_retake' && !m.candidates.length) {
      E.caption(state.plan.rationale || 'Let’s try that once more', 'mouth or say it again, or open Phrases')
    } else if (state.plan?.name === 'switch_input_method') {
      E.caption(state.plan.rationale, `try ${state.plan.arguments.method ?? 'another way'}`)
    }
  }, [state, calibrating])

  /* ---------- actions: from the rail, bubbles, blinks, nods and keys ---------- */
  const confirm = useCallback((how: E.How) => run(api.confirm(METHOD[how]).then((r) => play(r.audio_url, r.text))), [run, play])
  const act = (a: string, el: HTMLElement | null, how: E.How) => {
    // a completed eye dwell means the person was looking at that target: refine the calibration with it
    if (how === 'eyes' && el) {
      const r = (el.querySelector<HTMLElement>('.ic') ?? el).getBoundingClientRect()
      learnFromSelection(r.left + r.width / 2, r.top + r.height / 2, E.getDwell())
      saveCalibration()   // so what it learned survives a reload
    }
    if (a.startsWith('calib:')) { calibAct.current?.(a.slice(6)); return }
    if (a === 'calibrate') { E.closeBubbles(); setCalibrating(true); return }
    if (a === 'camera') { if (cam === 'off') location.reload(); return }
    const s = live.current.state
    if (!s) { E.caption('The gateway is offline', 'start it with ./scripts/run_all.sh'); return }
    const m = s.message
    // a phrase card is the person's own explicit choice (held with the eyes, or clicked), not a model's
    // reading, so choosing it is the confirmation: it is spoken straight away, through the same policy gate
    if (a.startsWith('say:')) {
      E.closeBubbles()
      run(api.board(a.slice(4), 'board').then(() => api.confirm(METHOD[how])).then((r) => play(r.audio_url, r.text)))
      return
    }
    if (a.startsWith('pick:')) { E.closeBubbles(); run(api.select(a.slice(5), METHOD[how])); return }
    switch (a) {
      case 'phrases':
        if (E.bubblesKey() === 'phrases') E.closeBubbles()
        else E.showBubbles(PHRASES.map(([t, cat, c, ic]) => ({ a: 'say:' + t, text: t, sub: cat, color: c, icon: ic })), el ?? document.querySelector('[data-a=phrases]')!, 'phrases')
        break
      case 'yes':
        if (m.status === 'selected') confirm(how)
        else if (PENDING.includes(m.status) && shownCandidates(s)[0]) run(api.select(shownCandidates(s)[0].id, METHOD[how]))
        else { E.closeBubbles(); run(api.board('Yes', 'board')) }
        break
      case 'no': case 'shake':
        // a phrase you picked yourself is simply dropped; a model's reading goes back to the choices
        if (m.status === 'selected') {
          const c = m.candidates.find((x) => x.id === m.selected_candidate_id)
          run(c && c.origin !== 'evidence' && c.origin !== 'eeg_replay' && m.candidates.length === 1 ? api.undo().then(api.undo) : api.undo())
        }
        else if (PENDING.includes(m.status)) { E.closeBubbles(); run(api.reject()) }
        else if (a === 'no') { E.closeBubbles(); run(api.board('No', 'board')) }
        break
      case 'speak': {
        const sp = s.spoken[s.spoken.length - 1]
        if (m.status === 'selected') confirm(how)
        else if (m.status === 'spoken' && sp) play(`/api/${s.session_id}/audio/${sp.message_id}`, sp.text)
        else E.caption('Nothing chosen yet', 'choose a phrase, or mouth, say or spell it')
        break
      }
      case 'clear':
        E.closeBubbles(); E.clearInk(); run(api.newMessage())
        setTimeout(() => { E.greet(); E.caption('Rest your gaze, or blink, to choose') }, 700)
        break
      case 'help':
        E.closeBubbles(); setAlarm(Date.now())
        run(api.board(HELP_TEXT, 'board'))
        break
      // a blink with nothing under the eyes, or a nod: confirms what is waiting
      case 'blink': case 'nod':
        if (m.status === 'selected') confirm(how)
        else if (PENDING.includes(m.status) && shownCandidates(s)[0]) run(api.select(shownCandidates(s)[0].id, METHOD[how]))
        break
    }
    if (['yes', 'speak', 'blink', 'nod'].includes(a) && m.status === 'selected') E.caption(`Confirmed by ${HOW_TEXT[how]}`, 'policy gate checks the exact text')
  }
  useLayoutEffect(() => { actRef.current = act })
  useEffect(() => E.onAction((a, el, how) => actRef.current(a, el, how)), [])

  const bindCalib = useCallback((fn: ((a: string) => void) | null) => { calibAct.current = fn }, [])
  const doneCalib = useCallback((useEyes: boolean) => {
    setEyes(useEyes)
    E.setGazeError(gazeState.accuracyPx)
    setCalibrating(false)
    setTimeout(() => { E.greet(); E.caption(useEyes ? 'Rest your eyes 1.2 s on anything, or blink, to choose' : 'Rest the pointer 1.2 s on anything, or blink, to choose') }, 500)
  }, [])

  const selected = state?.message.status === 'selected'
  const active = useMemo(() => {
    const up = (svc: string) => health?.[svc]?.state === 'ready'
    return 1 + (cam === 'on' ? 2 : 0) + (up('eeg') ? 1 : 0) + (mic === 'on' && up('audio') ? 1 : 0) + (cam === 'on' && up('vsr') ? 1 : 0)
  }, [health, cam, mic])
  useEffect(() => { const h = histRef.current; if (h) h.scrollTop = h.scrollHeight }, [state?.spoken.length])

  return (
    <>
      <div className="room" aria-hidden="true"><div className="ambient" /><video id="cam" ref={camRef} className={cam === 'on' ? 'on' : ''} autoPlay playsInline muted /><div className="veil" /></div>
      <canvas className="layer" id="light" />
      <canvas className="layer" id="ghost" />
      <canvas className="layer" id="ink" />
      {alarm > 0 && <div key={alarm} className="alarm" />}

      <div className="brand">
        <div className="name"><span className="logo" /><span>unspoken<i>.</i></span></div>
        <button className={`chip glass mono ${cam === 'on' ? 'live' : ''}`} data-a="camera"><b /><span>{cam === 'on' ? 'Camera · on device' : cam === 'off' ? 'Camera off · allow & reload' : 'Camera · starting'}</span></button>
        <span className={`chip glass mono ${mic === 'on' ? (hearing ? 'live' : 'ok') : ''}`}><b /><span>{mic === 'on' ? (health?.audio?.state !== 'ready' ? 'Mic · voice agent offline' : hearing ? 'Mic · hearing you' : 'Mic · listening') : mic === 'off' ? 'Mic off' : 'Mic · starting'}</span></span>
        <button className={`chip glass mono ${eyes ? 'ok' : 'warn'}`} data-a="calibrate"><b /><span>{eyes ? `Eyes · ≈${Math.round(gazeState.accuracyPx ?? 0)} px` : 'Eyes · pointer · calibrate'}</span></button>
      </div>
      {!connected && <div className="offline mono">Gateway offline · reconnecting</div>}

      <div className="side">
        <section className="twin glass" aria-label="Agent activity">
          <header><div>Agent activity</div><span className="mono">{active} of 6 inputs on</span></header>
          <canvas id="twin" />
          <Feed timeline={state?.timeline ?? []} />
        </section>
        <section className="chat glass" aria-label="What you said">
          <header><div>History</div><span className="mono">{state?.spoken.length ? `${state.spoken.length} spoken` : 'today'}</span></header>
          <div className="history" ref={histRef}>
            {!state?.spoken.length && <div className="hist-empty">What you say will collect here.</div>}
            {state?.spoken.map((sp) => {
              const via = [...sp.supporting_modalities.map((x) => MOD_NODE[x]).filter(Boolean), METHOD_NODE[sp.method]].filter(Boolean) as E.NodeId[]
              const names = [...sp.supporting_modalities.map((x) => MOD_LABEL[x] ?? x), METHOD_TEXT[sp.method] ?? sp.method].join(' · ')
              return (
                <div key={sp.message_id} className="msg" title="Click to hear it again" style={{ cursor: 'pointer' }}
                  onClick={() => play(`/api/${state.session_id}/audio/${sp.message_id}`, sp.text)}>
                  <div className="b">{sp.text}</div>
                  <div className="meta">{[...new Set(via)].map((v) => <i key={v} style={{ ['--c' as string]: E.COL[v].join(',') }} />)}<span>{names} · {new Date(sp.ts).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}</span></div>
                </div>
              )
            })}
          </div>
        </section>
      </div>

      {/* Yes and No are needed most often: two big targets far apart, well beyond the webcam gaze error */}
      <nav className="answers" aria-label="Answer">
        <button data-a="yes" {...(selected ? { 'data-confirm': '' } : {})} className={`answer glass ${selected ? 'ready' : ''}`} style={{ ['--c' as string]: '72,190,110' }}><span className="ic"><svg viewBox="0 0 24 24"><path d="m5 12 4.5 4.5L19 7" /></svg></span><span className="lb">Yes</span></button>
        <button data-a="no" className="answer glass" style={{ ['--c' as string]: '240,92,104' }}><span className="ic"><svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18" /></svg></span><span className="lb">No</span></button>
      </nav>

      <nav className="rail glass" aria-label="Always available">
        <button data-a="phrases" style={{ ['--c' as string]: '250,160,70' }}><span className="ic"><svg viewBox="0 0 24 24"><path d="M4 6h16M4 12h10M4 18h7" /></svg></span><span className="lb">Phrases</span></button>
        <button data-a="clear" style={{ ['--c' as string]: '126,132,148' }}><span className="ic"><svg viewBox="0 0 24 24"><path d="M5 19 19 5M8 5h11v11" /></svg></span><span className="lb">Clear</span></button>
        <button data-a="help" className="alert" style={{ ['--c' as string]: '236,72,64' }}><span className="ic"><svg viewBox="0 0 24 24"><path d="M12 3 2 21h20zM12 10v4M12 18h.01" /></svg></span><span className="lb">Help</span></button>
      </nav>

      <div className="caption mono" id="caption" aria-live="polite" />
      <div className="hint mono">Look or point to aim · blink or Space selects<br />Nod = yes · shake = no · P = phrases</div>
      <div className="gaze" id="gaze" />

      {calibrating && <Calibration bind={bindCalib} onFocusMode={setFocusMode} onDone={doneCalib} />}
    </>
  )
}

function Feed({ timeline }: { timeline: TimelineItem[] }) {
  const items = timeline.slice(-3)
  return (
    <div className="feed" aria-live="polite">
      {!items.length && <div className="act-empty">Agents report here as they work.</div>}
      {items.map((t) => {
        const a = AGENTS[t.agent] ?? { label: t.agent, c: [200, 200, 210] as E.Rgb }
        return (
          <div key={t.id} className={`act ${t.level}`} title={`${t.title}${t.detail ? ' — ' + t.detail : ''}`}>
            <i style={{ ['--c' as string]: a.c.join(',') }} />
            <b>{a.label}</b>
            <span>{t.title}{t.detail ? ` · ${t.detail}` : ''}</span>
            <em className="mono">{t.latency_ms != null ? `${t.latency_ms} ms` : ''}</em>
          </div>
        )
      })}
    </div>
  )
}
