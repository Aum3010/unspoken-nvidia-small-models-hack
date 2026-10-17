import { motion } from 'framer-motion'
import { ArrowLeft, Check, Delete, Space, ThumbsDown, ThumbsUp } from 'lucide-react'
import { useMemo, useState } from 'react'
import { api } from '../lib/api'
import { DwellButton } from './DwellButton'
import './board.css'

const PHRASES: { group: string; items: string[] }[] = [
  { group: 'Needs', items: ['I need water', 'I need the bathroom', 'Please call the nurse', 'I am in pain', 'I need suction', 'I want to sit up'] },
  { group: 'Comfort', items: ['I am too hot', 'I am too cold', 'Please move my pillow', 'Turn off the light', 'Turn me, please', 'Itchy — please scratch'] },
  { group: 'People', items: ['Where is my family?', 'Please call my family', 'I love you', 'Thank you', 'Please stay with me', 'I want to rest'] },
  { group: 'Talk', items: ['Please wait, I am thinking', 'Please repeat that', 'I don\'t understand', 'Ask me yes or no', 'Not now', 'I want to say something else'] },
]

const GROUPS = ['ABCDE', 'FGHIJ', 'KLMNO', 'PQRST', 'UVWXY', 'Z.?\'']
// Deterministic word completion from a fixed list — no language model involved.
const WORDS = 'i you yes no please thank water nurse pain bathroom family help hot cold tired hurt need want call doctor medicine more less stop wait sleep light pillow mouth head chest leg arm back breathe breathing sick scared okay love home hungry thirsty move turn sit up down left right now later today tomorrow wife husband mother father son daughter friend music tv phone glasses blanket window door'.split(' ')

export function PhraseBoard({ notify }: { notify: (m: string, k?: 'error' | 'info') => void }) {
  const [group, setGroup] = useState(0)
  const choose = (t: string) => api.board(t, 'board').catch((e) => notify(e.message, 'error'))
  return (
    <div className="glass board">
      <div className="board-head">
        <div className="panel-title" style={{ margin: 0 }}>Phrase board</div>
        <div className="seg">
          {PHRASES.map((g, i) => (
            <DwellButton key={g.group} aria-pressed={i === group} onActivate={() => setGroup(i)}>{g.group}</DwellButton>
          ))}
        </div>
      </div>
      <div className="yesno">
        <DwellButton className="tile yes" onActivate={() => choose('Yes')}><ThumbsUp size={28} /> Yes</DwellButton>
        <DwellButton className="tile no" onActivate={() => choose('No')}><ThumbsDown size={28} /> No</DwellButton>
      </div>
      <motion.div key={group} className="tiles" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .3 }}>
        {PHRASES[group].items.map((t) => (
          <DwellButton key={t} className="tile" onActivate={() => choose(t)}>{t}</DwellButton>
        ))}
      </motion.div>
      <div className="faint" style={{ fontSize: '.8rem' }}>Choosing a phrase selects it. It is spoken only after you confirm with <b>Speak it</b>.</div>
    </div>
  )
}

export function SpellBoard({ notify }: { notify: (m: string, k?: 'error' | 'info') => void }) {
  const [text, setText] = useState('')
  const [group, setGroup] = useState<number | null>(null)
  const lastWord = text.split(' ').pop() ?? ''
  const suggestions = useMemo(() => {
    if (!lastWord) return ['I', 'need', 'please', 'want']
    const w = lastWord.toLowerCase()
    return WORDS.filter((x) => x.startsWith(w) && x !== w).slice(0, 4)
  }, [lastWord])

  const type = (ch: string) => { setText((t) => (t + ch).slice(0, 140)); setGroup(null) }
  const complete = (w: string) => setText((t) => {
    const parts = t.split(' ')
    parts[parts.length - 1] = w
    return parts.join(' ') + ' '
  })
  const submit = () => text.trim() && api.board(text.trim(), 'spelling').then(() => setText('')).catch((e) => notify(e.message, 'error'))

  return (
    <div className="glass board">
      <div className="board-head">
        <div className="panel-title" style={{ margin: 0 }}>Spelling board</div>
        <span className="faint" style={{ fontSize: '.8rem' }}>Two steps: choose a group, then a letter</span>
      </div>
      <div className="spell-out">
        <input className="input spell-input" value={text} onChange={(e) => setText(e.target.value)} placeholder="Your words appear here"
          onKeyDown={(e) => e.key === 'Enter' && submit()} aria-label="Spelled text" />
        <DwellButton className="btn lg primary" disabled={!text.trim()} onActivate={submit}><Check size={20} /> Use</DwellButton>
      </div>
      <div className="suggest">
        {suggestions.map((s) => <DwellButton key={s} className="tile sm" onActivate={() => complete(s)}>{s}</DwellButton>)}
      </div>
      {group === null ? (
        <div className="groups">
          {GROUPS.map((g, i) => (
            <DwellButton key={g} className="tile group" onActivate={() => setGroup(i)}>
              {g.split('').map((c) => <span key={c}>{c}</span>)}
            </DwellButton>
          ))}
        </div>
      ) : (
        <div className="groups">
          {GROUPS[group].split('').map((c) => <DwellButton key={c} className="tile letter" onActivate={() => type(c)}>{c}</DwellButton>)}
          <DwellButton className="tile letter back" onActivate={() => setGroup(null)}><ArrowLeft /></DwellButton>
        </div>
      )}
      <div className="spell-ctrl">
        <DwellButton className="tile sm" onActivate={() => type(' ')}><Space size={18} /> Space</DwellButton>
        <DwellButton className="tile sm" onActivate={() => setText((t) => t.slice(0, -1))}><Delete size={18} /> Delete</DwellButton>
        <DwellButton className="tile sm" onActivate={() => setText('')}>Clear</DwellButton>
      </div>
    </div>
  )
}
