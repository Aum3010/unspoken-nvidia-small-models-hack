// The liquid-glass stage, ported from unspoken_liquid_glass_3.html: words written in
// light by a pen (tentative words stay a faint, trembling sketch), the agent-activity
// nucleus fed by six coloured inputs, glass bubbles that bloom and melt, and a gaze
// point (eyes, or the pointer as a stand-in) that selects by dwelling or blinking.
// Imperative on purpose: it draws every frame. React drives it through the exports.

import { G } from './glyphs'

export type NodeId = 'voice' | 'lips' | 'gaze' | 'blink' | 'eeg' | 'head'
export type NodeState = 'off' | 'on'
/** How a target was chosen. The controller maps this to an audited confirmation method. */
export type How = 'eyes' | 'pointer' | 'blink' | 'space' | 'click' | 'keyboard' | 'touch' | 'nod'
export type Rgb = [number, number, number]

const $ = <T extends Element = HTMLElement>(s: string) => document.querySelector(s) as T
const $$ = (s: string) => [...document.querySelectorAll<HTMLElement>(s)]
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches

let lightC: HTMLCanvasElement, ghostC: HTMLCanvasElement, inkC: HTMLCanvasElement, TC: HTMLCanvasElement
let L: CanvasRenderingContext2D, Gx: CanvasRenderingContext2D, K: CanvasRenderingContext2D, T: CanvasRenderingContext2D
let W = 0, H = 0, DPR = 1

let S = 4, CAP = 80, LINE_H = 160, BLOCK_W = 1000, BLOCK_CX = 500, BASE_Y = 400
function metrics() {
  const wide = W > 860, left = wide ? 130 : 76, right = wide ? 440 : 0
  CAP = Math.max(30, Math.min(W * 0.06, H * 0.1, 96))
  S = CAP / 21; LINE_H = CAP * 2.05
  BLOCK_CX = (W + left - right) / 2
  BLOCK_W = Math.min(W - left - right - (wide ? 70 : 20), 1200)
  BASE_Y = H * (wide ? 0.52 : 0.5)
}

/* ============ glyph -> pen paths ============ */
interface Stroke { xy: Float32Array; len: Float32Array; w: Float32Array; spd: Float32Array; n: number; start: number; L: number }
interface Geo { strokes: Stroke[]; total: number; width: number }

function rand(seed: number) { let s = seed >>> 0 || 1; return () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296 }
function hash(str: string) { let h = 2166136261; for (const c of str) h = Math.imul(h ^ c.charCodeAt(0), 16777619); return h >>> 0 }

function buildWord(text: string, seedKey: string): Geo {
  const r = rand(hash(seedKey + text))
  const slant = 0.12 + (r() - .5) * 0.03, squeeze = 1.04, wob = 0.25 + r() * 0.2, ph = r() * 6
  // 1. raw strokes in font units, letters laid end to end
  let x = 0; const raw: number[][][] = []
  for (const ch of text) {
    const g = G[ch] || G['?']
    for (const s of g.s) raw.push(s.map(([px, py]) => [x + px, py - 9]))
    x += g.w
  }
  const width = x
  // 2. join strokes whose ends meet, so the pen flows through a word
  const merged: number[][][] = []
  for (const s of raw) {
    const last = merged[merged.length - 1]
    if (last) {
      const a = last[last.length - 1], b = s[0]
      if (Math.hypot(a[0] - b[0], a[1] - b[1]) < 0.8) { last.push(...s.slice(1)); continue }
    }
    merged.push(s.slice())
  }
  // 3. hand: squeeze, slant, gentle baseline wave, scale to px
  const tf = ([u, v]: number[]) => { const vv = v + Math.sin(u * 0.045 + ph) * wob; return [(u * squeeze - vv * slant) * S, vv * S] }
  const strokes: Stroke[] = []; let total = 0
  for (const s of merged) {
    const P = s.map(tf)
    // Catmull-Rom subdivision for a smooth nib path
    const pts: number[][] = []
    for (let i = 0; i < P.length - 1; i++) {
      const p0 = P[Math.max(0, i - 1)], p1 = P[i], p2 = P[i + 1], p3 = P[Math.min(P.length - 1, i + 2)]
      const n = Math.max(1, Math.ceil(Math.hypot(p2[0] - p1[0], p2[1] - p1[1]) / 2.2))
      for (let k = 0; k < n; k++) {
        const t = k / n, t2 = t * t, t3 = t2 * t
        pts.push([
          0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3),
          0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3),
        ])
      }
    }
    pts.push(P[P.length - 1])
    if (pts.length < 2) pts.push([pts[0][0] + 0.6, pts[0][1] + 0.6]) // dots
    const n = pts.length, xy = new Float32Array(n * 2), len = new Float32Array(n), w = new Float32Array(n), spd = new Float32Array(n)
    let acc = 0
    for (let i = 0; i < n; i++) {
      xy[i * 2] = pts[i][0]; xy[i * 2 + 1] = pts[i][1]
      if (i) acc += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
      len[i] = acc
    }
    for (let i = 0; i < n; i++) {
      w[i] = S * 2.5
      // pen slows into curves, speeds through straights
      const c = pts[Math.max(0, i - 2)], d = pts[Math.min(n - 1, i + 2)]
      const a1 = Math.atan2(pts[i][1] - c[1], pts[i][0] - c[0]), a2 = Math.atan2(d[1] - pts[i][1], d[0] - pts[i][0])
      let turn = Math.abs(a2 - a1); if (turn > Math.PI) turn = 2 * Math.PI - turn
      spd[i] = 1 / (1 + turn * 2.2)
    }
    strokes.push({ xy, len, w, spd, n, start: total, L: acc })
    total += acc
  }
  return { strokes, total, width: width * squeeze * S }
}

/* ============ the written sentence ============ */
interface Word {
  text: string; final: boolean; geo: Geo; seed: string; dx: number | null; dxT: number; line: number
  ghost: number; gp: number; ink: number; reach: Float32Array[]; settledAt: number; cache: Map<number, Path2D> | null; revised?: boolean
}
interface Line { off: number | null; offT: number; y: number | null; yT: number; alpha: number; alphaT: number }
let words: Word[] = []
let lines: Line[] = []
let seedN = 0
const norm = (t: string) => t.toLowerCase().replace(/[^a-z0-9']/g, '')

function makeWord(text: string, final: boolean): Word {
  const seed = 'w' + (seedN++)
  const geo = buildWord(text.toLowerCase(), seed)
  return { text, final, geo, seed, dx: null, dxT: 0, line: 0, ghost: 0, gp: 0, ink: 0,
    reach: geo.strokes.map((s) => new Float32Array(s.n).fill(-1)), settledAt: 0, cache: null }
}
function rebuild(wd: Word, text: string) {
  wd.text = text; wd.geo = buildWord(text.toLowerCase(), wd.seed)
  wd.reach = wd.geo.strokes.map((s) => new Float32Array(s.n).fill(-1)); wd.cache = null
  wd.gp = wd.geo.total; wd.ink = Math.min(wd.ink, 0)
}

/** Final words are drawn in by the pen; tentative words stay a faint sketch. */
export function setTranscript(finalWords: string[], tentWords: string[]) {
  greeting = false
  const want = finalWords.map((t) => ({ t, final: true })).concat(tentWords.map((t) => ({ t, final: false })))
  const next: Word[] = []
  const n = Math.max(want.length, words.length)
  for (let i = 0; i < n; i++) {
    const have = words[i], d = want[i]
    if (!d) { if (have) dissolve(have); continue }
    if (!have) { next.push(makeWord(d.t, d.final)); continue }
    if (have.text === d.t) { have.final = have.final || d.final; next.push(have); continue }
    if (norm(have.text) === norm(d.t) && !have.final) { rebuild(have, d.t); have.final = d.final; next.push(have); continue }
    // the reading changed: the old word breaks apart, a new one is drawn
    dissolve(have); const nw = makeWord(d.t, d.final); nw.revised = true; next.push(nw)
  }
  words = next
  layout()
}
export const writtenText = () => words.map((w) => w.text).join(' ')

function layout() {
  metrics()
  const gap = CAP * 0.5
  let li = 0, x = 0; const lw: number[] = []
  for (const wd of words) {
    if (x > 0 && x + wd.geo.width > BLOCK_W) { lw[li] = x - gap; li++; x = 0 }
    wd.line = li; wd.dxT = x; if (wd.dx === null) wd.dx = x
    x += wd.geo.width + gap
  }
  lw[li] = Math.max(0, x - gap)
  const nl = words.length ? li + 1 : 0
  while (lines.length < nl) lines.push({ off: null, offT: 0, y: null, yT: 0, alpha: 0, alphaT: 1 })
  lines.length = Math.max(nl, 0)
  const firstVis = Math.max(0, nl - 3), vis = nl - firstVis
  lines.forEach((ln, i) => {
    ln.offT = BLOCK_CX - lw[i] / 2
    ln.yT = BASE_Y + (i - firstVis - (vis - 1) / 2) * LINE_H
    ln.alphaT = i < firstVis ? 0 : 1
    if (ln.off === null) { ln.off = ln.offT; ln.y = ln.yT }
  })
}

/* ============ particles ============ */
interface Part { x: number; y: number; vx: number; vy: number; life: number; max: number; r: number }
const parts: Part[] = []
function worldOf(wd: Word): [number, number] { const ln = lines[wd.line]; return ln ? [(ln.off ?? 0) + (wd.dx ?? 0), ln.y ?? 0] : [0, 0] }
function dissolve(wd: Word) {
  const [ox, oy] = worldOf(wd)
  for (const s of wd.geo.strokes) for (let i = 0; i < s.n; i += 4) {
    parts.push({ x: ox + s.xy[i * 2], y: oy + s.xy[i * 2 + 1], vx: (Math.random() - .5) * 30, vy: -12 - Math.random() * 40,
      life: 0, max: 700 + Math.random() * 700, r: 0.8 + Math.random() * 1.4 })
  }
}

/* ============ the pen ============ */
const pen = { pause: 0, active: null as Word | null }
function penStep(dt: number, now: number) {
  // first final word not yet fully inked
  const idx = words.findIndex((w) => w.final && w.ink < w.geo.total)
  const wd = idx >= 0 ? words[idx] : null
  if (!wd) { pen.active = null; return }
  if (pen.active !== wd) { pen.active = wd; pen.pause = 90 }
  if (reduced) { wd.ink = wd.geo.total; markReach(wd, now); wd.settledAt = now; return }
  // backlog makes the hand hurry, like someone catching up on dictation
  let backlog = 0; for (let i = idx; i < words.length; i++) if (words[i].final) backlog += words[i].geo.total - words[i].ink
  const base = 1500 * (S / 4.2) * (1 + Math.min(3, backlog / (S * 500)))
  if (pen.pause > 0) { pen.pause -= dt; return }
  const st = strokeAt(wd, wd.ink)
  const k = localIndex(st, wd.ink - st.start)
  const v = base * (0.45 + 0.55 * st.spd[k])
  const before = wd.ink
  wd.ink = Math.min(wd.geo.total, wd.ink + v * dt / 1000)
  if (wd.ink >= st.start + st.L - 0.01 && before < st.start + st.L && wd.ink < wd.geo.total) { wd.ink = st.start + st.L + 0.02; pen.pause = 55 }
  markReach(wd, now)
  if (wd.ink >= wd.geo.total) { wd.settledAt = now; pen.pause = 110 }
}
function strokeAt(wd: Word, g: number) { const ss = wd.geo.strokes; for (const s of ss) if (g < s.start + s.L) return s; return ss[ss.length - 1] }
function localIndex(s: Stroke, l: number) { let lo = 0, hi = s.n - 1; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (s.len[m] <= l) lo = m; else hi = m - 1 } return lo }
function markReach(wd: Word, now: number) {
  wd.geo.strokes.forEach((s, si) => {
    const R = wd.reach[si]; const l = wd.ink - s.start; if (l < 0) return
    for (let i = 0; i < s.n && s.len[i] <= l; i++) if (R[i] < 0) R[i] = now
  })
}

/* ============ drawing ============ */
function bucketPaths(wd: Word, upto: number) {
  const outer = new Map<number, Path2D>()
  wd.geo.strokes.forEach((s) => {
    const l = upto - s.start; if (l <= 0) return
    for (let i = 1; i < s.n; i++) {
      if (s.len[i - 1] >= l) break
      let x2 = s.xy[i * 2], y2 = s.xy[i * 2 + 1]
      if (s.len[i] > l) { const t = (l - s.len[i - 1]) / (s.len[i] - s.len[i - 1]); x2 = s.xy[i * 2 - 2] + (x2 - s.xy[i * 2 - 2]) * t; y2 = s.xy[i * 2 - 1] + (y2 - s.xy[i * 2 - 1]) * t }
      const key = Math.round(s.w[i] * 2) / 2
      let p = outer.get(key); if (!p) { p = new Path2D(); outer.set(key, p) }
      p.moveTo(s.xy[i * 2 - 2], s.xy[i * 2 - 1]); p.lineTo(x2, y2)
    }
  })
  return outer
}
function strokeInk(ctx: CanvasRenderingContext2D, outer: Map<number, Path2D>) {
  // a matte white tube lit from the top left: shadowed rim, body, soft sheen, sharp highlight
  ctx.lineCap = 'round'; ctx.lineJoin = 'round'
  for (const [w, p] of outer) {
    ctx.strokeStyle = '#a9adb6'; ctx.lineWidth = w; ctx.stroke(p)
    ctx.save(); ctx.translate(-w * .035, -w * .05); ctx.strokeStyle = '#d6d9df'; ctx.lineWidth = w * .93; ctx.stroke(p); ctx.restore()
    ctx.save(); ctx.translate(-w * .1, -w * .12); ctx.strokeStyle = '#eceef2'; ctx.lineWidth = w * .58; ctx.stroke(p); ctx.restore()
    ctx.save(); ctx.translate(-w * .16, -w * .2); ctx.strokeStyle = 'rgba(255,255,255,.95)'; ctx.lineWidth = w * .2; ctx.stroke(p); ctx.restore()
  }
}
function drawGhost(wd: Word, ox: number, oy: number, now: number, alpha: number) {
  // tentative: a faint, trembling sketch that has not been committed yet
  const amp = wd.final ? 0.2 : 0.6 + (wd.revised ? 0.4 : 0)
  const covered = wd.ink / wd.geo.total
  const a = alpha * wd.ghost * (1 - covered) * (0.32 + 0.1 * Math.sin(now / 420 + (wd.dx ?? 0)))
  if (a <= 0.005) return
  Gx.save(); Gx.translate(ox, oy)
  Gx.strokeStyle = `rgba(255,255,255,${a * 1.5})`; Gx.lineWidth = Math.max(1.2, S * 0.42); Gx.lineCap = 'round'; Gx.lineJoin = 'round'
  const t = now / 1000
  Gx.beginPath()
  for (const s of wd.geo.strokes) {
    const l = wd.gp - s.start; if (l <= 0) break
    for (let i = 0; i < s.n; i++) {
      if (s.len[i] > l) break
      const x = s.xy[i * 2] + Math.sin(t * 3.1 + i * 0.21 + s.start) * amp
      const y = s.xy[i * 2 + 1] + Math.cos(t * 2.7 + i * 0.17 + s.start) * amp
      if (i) Gx.lineTo(x, y); else Gx.moveTo(x, y)
    }
  }
  Gx.stroke(); Gx.restore()
}

let last = performance.now(), raf = 0
function frame(now: number) {
  raf = requestAnimationFrame(frame)
  const dt = Math.min(100, now - last); last = now
  const e = (k: number) => 1 - Math.pow(1 - k, dt / 16.7)

  for (const ln of lines) { ln.off! += (ln.offT - ln.off!) * e(0.07); ln.y! += (ln.yT - ln.y!) * e(0.07); ln.alpha += (ln.alphaT - ln.alpha) * e(0.06) }
  for (const wd of words) {
    wd.dx! += (wd.dxT - wd.dx!) * e(0.18)
    wd.ghost = Math.min(1, wd.ghost + dt / 260)
    wd.gp = Math.min(wd.geo.total, wd.gp + wd.geo.total * dt / (reduced ? 1 : 420))
  }
  penStep(dt, now)

  Gx.setTransform(DPR, 0, 0, DPR, 0, 0); Gx.clearRect(0, 0, W, H)
  K.setTransform(DPR, 0, 0, DPR, 0, 0); K.clearRect(0, 0, W, H)
  for (const wd of words) {
    const ln = lines[wd.line]; if (!ln || ln.alpha < 0.01) continue
    const ox = ln.off! + wd.dx!, oy = ln.y!
    drawGhost(wd, ox, oy, now, ln.alpha)
    if (wd.ink <= 0) continue
    K.save(); K.globalAlpha = ln.alpha; K.translate(ox, oy)
    const settled = wd.ink >= wd.geo.total && now - wd.settledAt > 950
    if (settled) {
      if (!wd.cache) wd.cache = bucketPaths(wd, wd.geo.total)
      strokeInk(K, wd.cache)
    } else strokeInk(K, bucketPaths(wd, wd.ink))
    K.restore()
  }

  L.setTransform(DPR, 0, 0, DPR, 0, 0); L.clearRect(0, 0, W, H)
  L.globalCompositeOperation = 'lighter'
  for (let i = parts.length - 1; i >= 0; i--) {
    const p = parts[i]; p.life += dt; if (p.life > p.max) { parts.splice(i, 1); continue }
    p.x += p.vx * dt / 1000; p.y += p.vy * dt / 1000; p.vx *= 0.985; p.vy *= 0.99
    L.fillStyle = `rgba(235,238,244,${(1 - p.life / p.max) * 0.48})`
    L.beginPath(); L.arc(p.x, p.y, p.r, 0, 7); L.fill()
  }
  L.globalCompositeOperation = 'source-over'

  drawNeurons(now, dt)
  dwellTick(now, dt)
}

/* ============ agent activity: a living nucleus (the coordinator) fed by six coloured inputs ============ */
export const COL: Record<NodeId, Rgb> = { voice: [255, 184, 108], lips: [255, 128, 112], gaze: [94, 234, 212], blink: [165, 180, 252], eeg: [232, 121, 249], head: [181, 232, 107] }
export const MODS: Record<NodeId, string> = { voice: 'Voice', lips: 'Lips', gaze: 'Gaze', blink: 'Blink', eeg: 'Brain', head: 'Head' }
const ANG: Record<NodeId, number> = { voice: 200, lips: 158, blink: 124, gaze: 246, head: 312, eeg: 44 }
const IDS = Object.keys(MODS) as NodeId[]
interface TNode { label: string; state: NodeState; tag: string; e: number; mix: number; note: string; noteT: number; rings: number[]; ph: number; curve?: number[] }
const tnodes = {} as Record<NodeId, TNode>
for (const k of IDS) tnodes[k] = { label: MODS[k], state: 'off', tag: 'OFF', e: 0, mix: 0, note: '', noteT: 0, rings: [], ph: Math.random() * 6 }
const orch = { e: 0 }
const comets: { id: NodeId; t0: number; k: number }[] = []
const waves: { t0: number }[] = []
const auras: { x: number; y: number; c: Rgb; t0: number; k: number }[] = []
const nodeScreen = {} as Record<NodeId, { x: number; y: number }>
let wobble = 0, orchLine = 'Ready', orchLineT = 0, orchHold = 3000, orchStickyUntil = 0
const rgba = (c: number[], a: number) => `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${a})`
const lift = (c: number[], k: number) => c.map((v) => v + (255 - v) * k), sink = (c: number[], k: number) => c.map((v) => v * (1 - k))

/** An input just did something: its node pulses, a comet runs to the nucleus, the room catches its colour. */
export function fire(id: NodeId, k = 1, note?: string) {
  const n = tnodes[id]
  n.e = Math.min(1.5, n.e + k); if (note) { n.note = note; n.noteT = performance.now() }
  n.rings.push(performance.now())
  comets.push({ id, t0: performance.now(), k })
  const p = nodeScreen[id]
  if (p) auras.push({ x: p.x, y: p.y, c: COL[id], t0: performance.now(), k: Math.min(1, k) })
  // a coordinator or policy line stays readable for a moment before input chatter replaces it
  if (performance.now() > orchStickyUntil) {
    const live = IDS.filter((m) => tnodes[m].mix > .25 || m === id).map((m) => MODS[m].toLowerCase())
    orchLine = live.length > 1 ? 'Fusing ' + live.slice(0, 3).join(' + ') : 'Listening to ' + MODS[id].toLowerCase()
    orchLineT = performance.now(); orchHold = 3000
  }
}
/** Speech leaves the nucleus, like a voice. */
export function fireOut() { waves.push({ t0: performance.now() }) }
export function setOrch(line: string, holdMs = 4000) {
  orchLine = line; orchLineT = performance.now(); orchHold = holdMs; orchStickyUntil = orchLineT + Math.min(holdMs, 2500)
  orch.e = Math.min(1.5, orch.e + .5)
}
export function setNode(id: NodeId, state: NodeState, tag: string) { tnodes[id].state = state; tnodes[id].tag = tag }

const ICONP: Record<NodeId, Path2D> = {
  voice: new Path2D('M4 10v4M8 7v10M12 4v16M16 7v10M20 10v4'),
  lips: new Path2D('M3 12c3-4 6-5 9-3 3-2 6-1 9 3-3 4-6 6-9 6s-6-2-9-6zM3 12h18'),
  gaze: new Path2D('M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12zM15 12a3 3 0 1 1-6 0 3 3 0 1 1 6 0z'),
  blink: new Path2D('M2 11s4 5 10 5 10-5 10-5M5 14l-2 2M19 14l2 2M12 16v3M8.5 15.5 7.5 18M15.5 15.5l1 2.5'),
  eeg: new Path2D('M2 12h4l2-5 3 10 3-12 2.5 7H22'),
  head: new Path2D('M16 9a4 4 0 1 1-8 0 4 4 0 1 1 8 0zM5 21c1-4 4-6 7-6s6 2 7 6M3 6c-1 1-1 3 0 4M21 6c1 1 1 3 0 4'),
}
function gummy(x: number, y: number, r: number, c: number[], alpha = 1, sq = 1, glowing = true) {
  T.save(); T.globalAlpha = alpha; T.translate(x, y)
  // light passes through the jelly: a coloured shadow, not a black one
  const sh = T.createRadialGradient(r * .2, r * 1.05, 0, r * .2, r * 1.05, r * 1.3)
  sh.addColorStop(0, rgba(sink(c, .25), glowing ? .55 : .3)); sh.addColorStop(1, rgba(c, 0))
  T.fillStyle = sh; T.beginPath(); T.ellipse(r * .2, r * 1.0, r * 1.25, r * .45, 0, 0, 7); T.fill()
  T.scale(sq, 1 / sq)
  const g = T.createRadialGradient(-r * .25, -r * .3, r * .08, 0, 0, r * 1.05)
  g.addColorStop(0, rgba(lift(c, .5), .97)); g.addColorStop(.55, rgba(c, .95)); g.addColorStop(1, rgba(sink(c, .38), .97))
  T.fillStyle = g; T.beginPath(); T.arc(0, 0, r, 0, 7); T.fill()
  // subsurface: the bottom of a gummy glows where light exits
  const ss = T.createRadialGradient(r * .15, r * .45, 0, r * .15, r * .45, r * .75)
  ss.addColorStop(0, rgba(lift(c, .65), .65)); ss.addColorStop(1, rgba(lift(c, .65), 0))
  T.fillStyle = ss; T.beginPath(); T.arc(0, 0, r, 0, 7); T.fill()
  T.strokeStyle = 'rgba(255,255,255,.28)'; T.lineWidth = 1; T.beginPath(); T.arc(0, 0, r - .6, 0, 7); T.stroke()
  // wet gloss
  T.fillStyle = 'rgba(255,255,255,.82)'; T.beginPath(); T.ellipse(-r * .34, -r * .44, r * .34, r * .17, -.6, 0, 7); T.fill()
  T.fillStyle = 'rgba(255,255,255,.5)'; T.beginPath(); T.arc(r * .42, r * .3, r * .08, 0, 7); T.fill()
  T.restore()
}

function drawNeurons(now: number, dt: number) {
  const r = TC.getBoundingClientRect(), w = r.width, h = r.height
  if (!w || !h) return
  if (TC.width !== Math.round(w * DPR) || TC.height !== Math.round(h * DPR)) { TC.width = Math.round(w * DPR); TC.height = Math.round(h * DPR) }
  T.setTransform(DPR, 0, 0, DPR, 0, 0); T.clearRect(0, 0, w, h)
  const d = Math.pow(0.93, dt / 16.7)
  const small = w < 300
  // parallax from where you are looking: the nucleus leans toward your eyes
  const px = ((sgx > 0 ? sgx / W : .5) - .5) * 8, py = ((sgy > 0 ? sgy / H : .5) - .5) * 6
  const cx = w * .5 + px * .4, cy = h * .55 + py * .4, R = Math.min(w, h) * (small ? .17 : .16)
  const rx = w * (small ? .37 : .36), ry = h * .36

  // nucleus colour = blend of whichever inputs are feeding it right now
  let wsum = .8
  const mc = [236 * .8, 230 * .8, 246 * .8]
  const pos = {} as Record<NodeId, { x: number; y: number; z: number }>
  for (const k of IDS) {
    const n = tnodes[k]; n.e *= d; n.mix = Math.max(n.mix * Math.pow(.985, dt / 16.7), Math.min(1, n.e))
    if (n.mix > .02) { mc[0] += COL[k][0] * n.mix; mc[1] += COL[k][1] * n.mix; mc[2] += COL[k][2] * n.mix; wsum += n.mix }
    const a = (ANG[k] + Math.sin(now / 2600 + n.ph) * 4) * Math.PI / 180
    const z = Math.sin(a + Math.PI / 2) * .5 + .5 // pseudo depth: lower = nearer
    pos[k] = { x: cx + Math.cos(a) * rx + px * (.6 + z), y: cy + Math.sin(a) * ry + py * (.6 + z) + Math.sin(now / 900 + n.ph) * 1.5, z }
    nodeScreen[k] = { x: r.left + pos[k].x, y: r.top + pos[k].y }
  }
  const mixC = mc.map((v) => v / wsum)
  orch.e *= d; wobble *= Math.pow(.95, dt / 16.7)

  // soft colour bloom behind the nucleus
  const bloom = T.createRadialGradient(cx, cy, 0, cx, cy, R * 3.2)
  bloom.addColorStop(0, rgba(mixC, .22 + Math.min(.4, orch.e * .3))); bloom.addColorStop(1, rgba(mixC, 0))
  T.fillStyle = bloom; T.fillRect(0, 0, w, h)

  // tendrils
  for (const k of IDS) {
    const n = tnodes[k], p = pos[k], off = n.state === 'off', e = Math.min(1, n.e)
    const ang = Math.atan2(p.y - cy, p.x - cx), ex = cx + Math.cos(ang) * R * .95, ey = cy + Math.sin(ang) * R * .95
    const mx = (p.x + ex) / 2 + Math.sin(now / 1100 + n.ph) * 6, my = (p.y + ey) / 2 + Math.cos(now / 1300 + n.ph) * 6
    n.curve = [p.x, p.y, mx, my, ex, ey]
    const g = T.createLinearGradient(p.x, p.y, ex, ey)
    g.addColorStop(0, rgba(COL[k], off ? .12 : .35 + e * .5)); g.addColorStop(1, rgba(COL[k], off ? .03 : .08 + e * .3))
    T.strokeStyle = g; T.lineWidth = off ? 1 : 1.6 + e * 3; T.lineCap = 'round'
    if (off) T.setLineDash([2, 4])
    T.beginPath(); T.moveTo(p.x, p.y); T.quadraticCurveTo(mx, my, ex, ey); T.stroke(); T.setLineDash([])
  }
  // comets running along tendrils
  for (let i = comets.length - 1; i >= 0; i--) {
    const c = comets[i], t = (now - c.t0) / 560, n = tnodes[c.id]
    if (t >= 1 || !n.curve) { comets.splice(i, 1); orch.e = Math.min(1.5, orch.e + .45 * c.k); wobble = Math.min(1, wobble + .5 * c.k); continue }
    const [ax, ay, bx, by, ex, ey] = n.curve
    for (let j = 0; j < 6; j++) {
      const tt = Math.max(0, t - j * .035), u = 1 - tt
      const x = u * u * ax + 2 * u * tt * bx + tt * tt * ex, y = u * u * ay + 2 * u * tt * by + tt * tt * ey
      const rr = (6 - j) * 1.3
      const g = T.createRadialGradient(x, y, 0, x, y, rr * 2)
      g.addColorStop(0, rgba(lift(COL[c.id], .5), .95 - j * .14)); g.addColorStop(1, rgba(COL[c.id], 0))
      T.fillStyle = g; T.beginPath(); T.arc(x, y, rr * 2, 0, 7); T.fill()
    }
  }
  // output waves leave the nucleus to the right, like a voice
  for (let i = waves.length - 1; i >= 0; i--) {
    const v = waves[i], t = (now - v.t0) / 1300; if (t >= 1) { waves.splice(i, 1); continue }
    for (let j = 0; j < 3; j++) {
      const tt = t - j * .12; if (tt <= 0) continue
      T.strokeStyle = rgba(lift(mixC, .4), (1 - tt) * .8); T.lineWidth = 2
      T.beginPath(); T.arc(cx, cy, R * (1.1 + tt * 2.2), -.55, .55); T.stroke()
    }
  }

  // the coordinator: a little clay brain with a face. It squashes when an input arrives.
  const e = Math.min(1, orch.e), breathe = 1 + Math.sin(now / 1400) * .02 + e * .06
  const sq = 1 + wobble * .08 * Math.sin(now / 60)
  const bw = R * 1.25 * breathe * sq, bh = R * 1.02 * breathe / sq, by = cy - R * .05
  const tint = [255, 150, 192].map((v, i) => v * .65 + mixC[i] * .35)
  T.save()
  const sh = T.createRadialGradient(cx + R * .15, cy + R * 1.0, 0, cx + R * .15, cy + R * 1.0, R * 1.4)
  sh.addColorStop(0, rgba(sink(tint, .3), .5)); sh.addColorStop(1, 'rgba(0,0,0,0)')
  T.fillStyle = sh; T.beginPath(); T.ellipse(cx + R * .15, cy + R * .98, R * 1.35, R * .4, 0, 0, 7); T.fill()
  const LOBES = [[-.62, -.05, .46], [-.38, -.42, .46], [.02, -.56, .5], [.42, -.4, .48], [.66, -.02, .44], [.42, .34, .44], [0, .4, .5], [-.4, .34, .44], [0, -.05, .66]]
  const brainPath = new Path2D()
  for (const [lx, ly, lr] of LOBES) {
    const j = 1 + .03 * Math.sin(now / 600 + lx * 9)
    brainPath.moveTo(cx + lx * bw + lr * bw * j, by + ly * bh)
    brainPath.arc(cx + lx * bw, by + ly * bh, lr * bw * j, 0, Math.PI * 2)
  }
  const body = T.createRadialGradient(cx - bw * .45, by - bh * .55, bw * .05, cx + bw * .1, by + bh * .2, bw * 1.25)
  body.addColorStop(0, rgba(lift(tint, .55), .97)); body.addColorStop(.55, rgba(tint, .95)); body.addColorStop(1, rgba(sink(tint, .42), .97))
  T.fillStyle = body; T.fill(brainPath)
  T.clip(brainPath)
  // folds: soft grooves with a lit lip under each, like pressed clay
  const folds = [
    [[-.04, -1.0], [.03, -.62], [0, -.3]],
    [[-.75, -.35], [-.5, -.1], [-.62, .2]], [[-.45, -.8], [-.3, -.6], [-.5, -.42]],
    [[.42, -.8], [.28, -.6], [.5, -.42]], [[.78, -.3], [.52, -.05], [.66, .25]],
    [[-.55, .55], [-.3, .4], [-.15, .62]], [[.55, .55], [.32, .42], [.15, .64]],
  ]
  for (const [dx, dy, c] of [[.5, .7, 'rgba(255,255,255,.5)'], [0, 0, rgba(sink(tint, .5), .45)]] as [number, number, string][]) {
    T.strokeStyle = c; T.lineWidth = Math.max(1.4, R * .06); T.lineCap = 'round'
    for (const f of folds) {
      T.beginPath(); T.moveTo(cx + f[0][0] * bw + dx, by + f[0][1] * bh + dy)
      T.quadraticCurveTo(cx + f[1][0] * bw + dx, by + f[1][1] * bh + dy, cx + f[2][0] * bw + dx, by + f[2][1] * bh + dy); T.stroke()
    }
  }
  const ig = T.createRadialGradient(cx, by, 0, cx, by, bw)
  ig.addColorStop(0, `rgba(255,255,255,${e * .35})`); ig.addColorStop(1, 'rgba(255,255,255,0)')
  T.fillStyle = ig; T.fillRect(cx - bw * 2, by - bh * 2, bw * 4, bh * 4)
  {
    const ss = T.createRadialGradient(cx + bw * .15, by + bh * .5, 0, cx + bw * .15, by + bh * .5, bw * .8)
    ss.addColorStop(0, rgba(lift(tint, .6), .6)); ss.addColorStop(1, rgba(lift(tint, .6), 0)); T.fillStyle = ss; T.fillRect(cx - bw * 2, by - bh * 2, bw * 4, bh * 4)
  }
  T.fillStyle = 'rgba(255,255,255,.8)'; T.beginPath(); T.ellipse(cx - bw * .45, by - bh * .62, bw * .26, bh * .11, -.45, 0, 7); T.fill()
  T.fillStyle = 'rgba(255,255,255,.55)'; T.beginPath(); T.ellipse(cx + bw * .5, by - bh * .5, bw * .09, bh * .05, .4, 0, 7); T.fill()
  T.restore()
  // face: it looks where you look, blinks, and grins when it is busy
  const fx = cx + px * .9, fy = by + bh * .12 + py * .7, blinkK = (now % 4600) < 140 ? .12 : 1
  T.fillStyle = 'rgba(28,24,38,.85)'
  for (const s of [-1, 1]) {
    T.beginPath(); T.ellipse(fx + s * bw * .2, fy - bh * .02, R * .07, R * .1 * blinkK, 0, 0, 7); T.fill()
    T.fillStyle = 'rgba(255,255,255,.9)'; T.beginPath(); T.arc(fx + s * bw * .2 - R * .02, fy - bh * .06, R * .022, 0, 7); T.fill(); T.fillStyle = 'rgba(28,24,38,.85)'
  }
  T.fillStyle = 'rgba(255,140,140,.35)'
  for (const s of [-1, 1]) { T.beginPath(); T.ellipse(fx + s * bw * .36, fy + bh * .14, R * .1, R * .055, 0, 0, 7); T.fill() }
  T.strokeStyle = 'rgba(28,24,38,.7)'; T.lineWidth = Math.max(1.5, R * .055); T.lineCap = 'round'
  T.beginPath(); T.arc(fx, fy + bh * .1, R * (.1 + e * .07), .18 * Math.PI, .82 * Math.PI); T.stroke()

  // inputs, far ones first
  const order = [...IDS].sort((a, b) => pos[a].z - pos[b].z)
  T.textAlign = 'center'; T.textBaseline = 'middle'
  for (const k of order) {
    const n = tnodes[k], p = pos[k], off = n.state === 'off', ee = Math.min(1, n.e)
    const rr = (small ? 12 : 20) * (.92 + p.z * .16) * (1 + ee * .18)
    n.rings = n.rings.filter((t0) => now - t0 < 900)
    for (const t0 of n.rings) { const t = (now - t0) / 900; T.strokeStyle = rgba(COL[k], (1 - t) * .8); T.lineWidth = 1.5; T.beginPath(); T.arc(p.x, p.y, rr * (1.2 + t * 2.4), 0, 7); T.stroke() }
    if (!off) { const hg = T.createRadialGradient(p.x, p.y, 0, p.x, p.y, rr * 3.2); hg.addColorStop(0, rgba(COL[k], .2 + ee * .5)); hg.addColorStop(1, rgba(COL[k], 0)); T.fillStyle = hg; T.beginPath(); T.arc(p.x, p.y, rr * 3.2, 0, 7); T.fill() }
    // every input is a gummy: translucent, glossy, squishy candy in its own colour, wearing its symbol
    {
      const c = off ? COL[k].map((v) => v * .82 + 120 * .18) : COL[k]
      const sq2 = 1 + (n.e > .05 ? Math.sin(now / 55 + n.ph) * .07 * Math.min(1, n.e) : Math.sin(now / 900 + n.ph) * .015)
      gummy(p.x, p.y, rr, c, off ? .8 : 1, sq2, !off)
      const ip = ICONP[k]; const s = rr * 1.12 / 24
      T.save(); T.translate(p.x - 12 * s, p.y - 12 * s - rr * .02); T.scale(s, s)
      T.lineCap = 'round'; T.lineJoin = 'round'; T.lineWidth = 2.3
      T.strokeStyle = rgba(sink(c, .45), .35); T.translate(.5, .9); T.stroke(ip); T.translate(-.5, -.9)
      T.strokeStyle = off ? 'rgba(255,255,255,.75)' : '#fff'; T.stroke(ip)
      T.restore()
    }
    if (!small) {
      const below = p.y > cy - 4 ? 1 : -1
      const ly = p.y + below * (rr + 13)
      T.font = '600 10.5px Inter, system-ui, sans-serif'
      T.fillStyle = off ? 'rgba(232,240,255,.38)' : rgba(lift(COL[k], .55), .95)
      T.fillText(n.label, p.x, ly)
      const fresh = n.note && now - n.noteT < 2600
      T.font = '500 7.5px "JetBrains Mono", monospace'
      T.fillStyle = fresh ? rgba(lift(COL[k], .3), 1) : off ? 'rgba(232,240,255,.22)' : 'rgba(232,240,255,.5)'
      T.fillText(fresh ? n.note : n.tag, p.x, ly + below * 11)
    }
  }
  if (!small) {
    T.font = '600 11px Inter, system-ui, sans-serif'; T.fillStyle = 'rgba(244,247,255,.92)'
    T.fillText('Coordinator', cx, cy + R * 1.5)
    const fresh = now - orchLineT < orchHold
    T.font = '400 9.5px Inter, system-ui, sans-serif'; T.fillStyle = rgba(lift(mixC, .5), fresh ? .95 : .5)
    T.fillText(fresh ? orchLine : 'Ready', cx, cy + R * 1.5 + 13)
  }

  // the room catches each input's colour
  for (let i = auras.length - 1; i >= 0; i--) {
    const a = auras[i], t = (now - a.t0) / 1600; if (t >= 1) { auras.splice(i, 1); continue }
    const rad = 260 + t * 260, al = Math.sin(Math.min(1, t * 4) * Math.PI / 2) * (1 - t) * .22 * a.k
    const g = L.createRadialGradient(a.x, a.y, 0, a.x, a.y, rad); g.addColorStop(0, rgba(a.c, al)); g.addColorStop(1, rgba(a.c, 0))
    L.fillStyle = g; L.fillRect(a.x - rad, a.y - rad, rad * 2, rad * 2)
  }
}

/* ============ spawning glass out of nowhere, and letting it melt back ============ */
function spawn(el: HTMLElement, from: Element, delay = 0) {
  const a = el.getBoundingClientRect(), b = from.getBoundingClientRect()
  const dx = (b.left + b.width / 2) - (a.left + a.width / 2), dy = (b.top + b.height / 2) - (a.top + a.height / 2)
  if (reduced) return
  el.animate([
    { transform: `translate(${dx}px,${dy}px) scale(.08,.12)`, opacity: 0, filter: 'blur(14px)', borderRadius: '999px' },
    { opacity: 1, offset: .45 },
    { transform: 'none', opacity: 1, filter: 'blur(0)' },
  ], { duration: 720, delay, easing: 'cubic-bezier(.18,1.25,.36,1)', fill: 'backwards' })
}
function melt(el: HTMLElement, to: Element | null) {
  if (!el.isConnected) return
  el.style.pointerEvents = 'none'
  el.removeAttribute('data-a')
  if (reduced || !to || !to.isConnected) { el.remove(); return }
  const a = el.getBoundingClientRect(), b = to.getBoundingClientRect()
  const dx = (b.left + b.width / 2) - (a.left + a.width / 2), dy = (b.top + b.height / 2) - (a.top + a.height / 2)
  el.animate([{ transform: 'none', opacity: 1, filter: 'blur(0)' }, { transform: `translate(${dx}px,${dy}px) scale(.1,.14)`, opacity: 0, filter: 'blur(12px)' }],
    { duration: 460, easing: 'cubic-bezier(.5,0,.75,0)', fill: 'forwards' }).onfinish = () => el.remove()
}

/* ============ bubbles bloom around the middle of the room ============ */
export interface BubbleItem { a: string; text: string; sub: string; color: string; icon: string }
let bubbles: HTMLElement[] = [], bubbleFrom: Element | null = null, bubbleKey = ''
export const bubblesKey = () => bubbleKey
/** Where bubble i of n is placed (its centre, before keeping a bubble of width w on screen). */
function bubbleSlot(i: number, n: number, w: number) {
  const cols = Math.min(3, n), rows = Math.ceil(n / cols)
  const cx = BLOCK_CX, cy = H * 0.5, rx = Math.min(W * 0.3, 420), ry = Math.min(H * 0.26, 220)
  const wide = W > 860, minX = wide ? 130 : 76, maxX = W - (wide ? 440 : 0) - 12
  const row = Math.floor(i / cols), inRow = Math.min(cols, n - row * cols), col = i % cols
  const x = cx + (col - (inRow - 1) / 2) * Math.min(rx * 0.95, (W - 160) / 3.2), y = cy + (row - (rows - 1) / 2) * Math.min(ry, 120)
  return { left: Math.max(minX, Math.min(maxX - w, x - w / 2)), y }
}
/** Screen centres the bubbles of an n-item menu will have (calibration puts dots there). */
export function bubbleCentres(n: number, w = 230): [number, number][] {
  return Array.from({ length: n }, (_, i) => { const s = bubbleSlot(i, n, w); return [s.left + w / 2, s.y] })
}
export function showBubbles(items: BubbleItem[], from: Element, key: string) {
  closeBubbles()
  bubbleFrom = from; bubbleKey = key
  const n = items.length
  items.forEach((it, i) => {
    const el = document.createElement('button'); el.className = 'bubble glass'; el.dataset.a = it.a
    el.style.setProperty('--c', it.color)
    const peb = document.createElement('span'); peb.className = 'pebble'; peb.textContent = it.icon
    const body = document.createElement('div'); body.textContent = it.text
    const sub = document.createElement('small'); sub.className = 'mono'; sub.textContent = it.sub; body.append(sub)
    el.append(peb, body)
    document.body.append(el)
    const { left, y } = bubbleSlot(i, n, el.offsetWidth)
    el.style.left = left + 'px'
    el.style.top = (y - el.offsetHeight / 2) + 'px'
    spawn(el, from, i * 55); bubbles.push(el)
  })
  inkC.style.opacity = ghostC.style.opacity = '.25'
}
export function closeBubbles() {
  bubbles.forEach((b) => melt(b, bubbleFrom)); bubbles = []; bubbleKey = ''
  if (inkC) inkC.style.opacity = ghostC.style.opacity = '1'
}

/* ============ captions ============ */
export function caption(text: string, em?: string) {
  const c = $('#caption'); if (!c) return
  c.textContent = text
  if (em) { if (text) c.append(' · '); const e = document.createElement('em'); e.textContent = em; c.append(e) }
}

/* ============ gaze (eyes, or the pointer as a stand-in) & blink ============ */
type Handler = (action: string, el: HTMLElement | null, how: How) => void
let handler: Handler = () => {}
export function onAction(h: Handler) { handler = h }

let DWELL = 1200
export const getDwell = () => DWELL
export function setDwell(ms: number) { DWELL = ms }
let source: 'pointer' | 'eyes' = 'pointer', paused = false
export function setGazeSource(s: 'pointer' | 'eyes') {
  source = s
  if (s === 'eyes') { hasPointer = false; gazeEl?.classList.remove('on') }
}
// measured webcam gaze error (px): how far a target may be from the estimate and still be the one meant
let gazeErr = 80
export function setGazeError(px: number | null) { gazeErr = Math.max(50, Math.min(160, px ?? 80)) }
export function setPaused(p: boolean) { paused = p }

let gazeEl: HTMLElement
let gx = -100, gy = -100, sgx = -100, sgy = -100, hot: HTMLElement | null = null, hotT0 = 0, hotDone = false, hasPointer = false
let clicked: HTMLElement | null = null // chosen by click: the gaze settling onto it must not choose it again
let dwelledAt = 0                       // a reflex click right after a dwell must not choose it again
function arrive() { hasPointer = true; sgx = ax = gx; sgy = ay = gy; gazeEl.classList.add('on'); document.body.classList.add('gazing') }
const onPointer = (ev: PointerEvent) => {
  if (ev.pointerType !== 'mouse' || source !== 'pointer') return
  gx = ev.clientX; gy = ev.clientY
  if (!hasPointer) arrive()
}
/** A smoothed gaze estimate from the eye tracker (screen px), or null when the eyes are lost. */
export function feedGaze(p: { x: number; y: number } | null) {
  if (source !== 'eyes') return
  if (!p) { gazeEl.classList.remove('on'); hasPointer = false; return }
  gx = p.x; gy = p.y
  if (!hasPointer) arrive()
}
function targetAt(x: number, y: number) {
  const el = document.elementFromPoint(x, y)
  const t = el && el.closest<HTMLElement>('[data-a]')
  return t && !(t as HTMLButtonElement).disabled ? t : null
}
/** Distance from a point to an element's box (0 inside). */
function boxDist(el: HTMLElement, x: number, y: number) {
  const r = el.getBoundingClientRect()
  return Math.hypot(Math.max(r.left - x, 0, x - r.right), Math.max(r.top - y, 0, y - r.bottom))
}
const usable = (el: HTMLElement) => !(el as HTMLButtonElement).disabled && el.getClientRects().length > 0
const centre = (el: HTMLElement) => { const r = (el.querySelector<HTMLElement>('.ic') ?? el).getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2] }
const SWITCH_PX = 40   // a neighbouring zone must win by this much before the choice changes
/** Of a group of targets, the one whose zone holds the point, keeping the current one near a boundary. */
function zonePick(group: HTMLElement[], d: (el: HTMLElement) => number): HTMLElement | null {
  let best: HTMLElement | null = null, bd = Infinity
  for (const el of group) { const v = d(el); if (v < bd) { bd = v; best = el } }
  if (hot && group.includes(hot) && d(hot) < bd + SWITCH_PX) return hot
  return best
}
/**
 * Webcam gaze is off by tens of pixels and drifts, so the eyes choose a screen area, not a point:
 * the left strip is the dock (top, middle, bottom third -> its buttons), the top band is Yes | No,
 * and while phrase cards are open the room is split between them. Anywhere else (the writing,
 * the agents panel) chooses nothing, so reading never selects.
 */
function eyeTarget(x: number, y: number): HTMLElement | null {
  const rail = $<HTMLElement>('.rail'), answers = $<HTMLElement>('.answers'), side = $<HTMLElement>('.side')
  const sideLeft = side ? side.getBoundingClientRect().left : W
  if (rail && x < rail.getBoundingClientRect().right + 90)
    return zonePick($$('.rail [data-a]').filter(usable), (el) => Math.abs(y - centre(el)[1]))
  if (answers && y < answers.getBoundingClientRect().bottom + 60 && x < sideLeft)
    return zonePick($$('.answers [data-a]').filter(usable), (el) => Math.abs(x - centre(el)[0]))
  const open = bubbles.filter(usable)
  if (open.length && x < sideLeft)
    return zonePick(open, (el) => { const [cx, cy] = centre(el); return Math.hypot(x - cx, y - cy) })
  // anything else (e.g. the calibration screen's buttons): the nearest within the gaze error
  let best: HTMLElement | null = null, bd = Infinity
  for (const el of $$('[data-a]')) {
    if (!usable(el) || el.closest('.brand, .rail, .answers, .side')) continue
    const d = boxDist(el, x, y)
    if (d < bd) { bd = d; best = el }
  }
  return bd <= gazeErr ? best : null
}
let lx = -100, ly = -100   // where the lens is drawn: on the gaze, or settled on the chosen target
let ax = -100, ay = -100   // eyes: fixation anchor; the estimate's jitter inside the gaze error does not move it
function dwellTick(now: number, dt = 16.7) {
  // eyes are not a mouse: the gaze point drifts and settles, it never snaps; the webcam estimate
  // jitters, so the eyes get a slower follow than the pointer
  let fx = gx, fy = gy
  if (source === 'eyes') {
    // a resting eye wobbles within the webcam error; only a move past ~half of it is a new look
    const R = gazeErr * 0.5, d = Math.hypot(gx - ax, gy - ay)
    if (ax < -50) { ax = gx; ay = gy } else if (d > R) { ax += (gx - ax) * (1 - R / d); ay += (gy - ay) * (1 - R / d) }
    fx = ax; fy = ay
  }
  const kz = 1 - Math.pow(source === 'eyes' ? 0.94 : 0.72, dt / 16.7); sgx += (fx - sgx) * kz; sgy += (fy - sgy) * kz
  const t = hasPointer && !paused ? (source === 'eyes' ? eyeTarget(sgx, sgy) : targetAt(sgx, sgy)) : null
  if (t !== hot) {
    if (hot) hot.classList.remove('hot')
    hot = t; hotT0 = now; hotDone = !!hot && hot === clicked
    if (hot && hot !== clicked) clicked = null
    if (hot) hot.classList.add('hot')
    gazeEl.classList.toggle('lock', !!hot && source === 'eyes')
  }
  // with the eyes, the lens settles on the chosen target, so it holds still while the dwell fills
  let tx = sgx, ty = sgy
  if (hot && source === 'eyes') {
    const r = (hot.querySelector<HTMLElement>('.ic') ?? hot).getBoundingClientRect()
    tx = r.left + r.width / 2; ty = r.top + r.height / 2
  }
  const kl = 1 - Math.pow(source === 'eyes' ? 0.88 : 0.78, dt / 16.7); lx += (tx - lx) * kl; ly += (ty - ly) * kl
  if (lx < -50) { lx = tx; ly = ty }
  gazeEl.style.transform = `translate(${lx}px,${ly}px)`
  // rail magnification: tiles swell as the gaze approaches, like the Mac Dock
  $$('.rail button').forEach((b) => {
    const r = b.getBoundingClientRect()
    const dd = Math.abs(sgy - (r.top + r.height / 2))
    const k = hasPointer && sgx < r.right + 120 ? Math.max(0, 1 - dd / 150) : 0
    const ic = b.querySelector<HTMLElement>('.ic'); if (ic) ic.style.transform = `scale(${1 + k * k * .32})`
  })
  // sheen follows the eyes across all glass
  $$('.glass').forEach((g) => { const r = g.getBoundingClientRect(); g.style.setProperty('--sx', (sgx - r.left) + 'px'); g.style.setProperty('--sy', (sgy - r.top) + 'px') })
  let p = 0
  if (hot && !hotDone) {
    // confirm targets need a longer look than select targets
    const need = hot.dataset.confirm != null ? DWELL * 1.5 : DWELL
    p = Math.min(1, (now - hotT0) / need)
    if (p >= 1) {
      hotDone = true; dwelledAt = now   // the controller learns the calibration from eye selections
      select(hot, source === 'eyes' ? 'eyes' : 'pointer'); p = 0
    }
  }
  gazeEl.style.setProperty('--p', String(p))
}
function select(el: HTMLElement, how: How) {
  const node: NodeId | null = how === 'eyes' || how === 'pointer' ? 'gaze' : how === 'blink' || how === 'space' ? 'blink' : how === 'nod' ? 'head' : null
  if (node) fire(node, 1, how === 'blink' || how === 'space' ? 'SELECT' : 'DWELL')
  el.animate([{ transform: 'scale(1.06)' }, { transform: 'scale(.94)' }, { transform: 'scale(1.06)' }], { duration: 260, easing: 'ease-out' })
  handler(el.dataset.a || '', el, how)
}
/** A deliberate blink selects whatever you are looking at; with nothing under the eyes it goes to the controller. */
export function blink(how: 'blink' | 'space') {
  if (paused) return
  gazeEl.classList.remove('blink'); void gazeEl.offsetWidth; gazeEl.classList.add('blink')
  if (hot) { hotDone = true; select(hot, how) }
  else { fire('blink', 1, 'BLINK'); handler('blink', null, how) }
}
const onKey = (e: KeyboardEvent) => {
  if (e.repeat || paused || e.target instanceof HTMLInputElement) return
  const k = e.key.toLowerCase()
  // Space is the blink stand-in; drop focus so it cannot also press the focused button on keyup
  if (k === ' ') { e.preventDefault(); (document.activeElement as HTMLElement | null)?.blur?.(); blink('space') }
  else if (k === 'p') handler('phrases', null, 'keyboard')
  else if (k === 'c') handler('clear', null, 'keyboard')
  else if (k === 'escape') closeBubbles()
}
// clicks and touch still work for carers and testing
const onClick = (e: MouseEvent) => {
  const el = (e.target as Element).closest<HTMLElement>('[data-a]'); if (!el || (el as HTMLButtonElement).disabled) return
  if (el === hot && performance.now() - dwelledAt < 700) return // already chosen by gaze
  clicked = el; if (el === hot) hotDone = true
  select(el, e.detail === 0 ? 'keyboard' : (e as PointerEvent).pointerType === 'touch' ? 'touch' : 'click')
}

/* ============ sizing ============ */
function resize() {
  DPR = Math.min(2, devicePixelRatio || 1); W = innerWidth; H = innerHeight
  for (const c of [lightC, ghostC, inkC]) { c.width = Math.round(W * DPR); c.height = Math.round(H * DPR) }
  const keep = words.map((w) => ({ w, f: w.ink / w.geo.total }))
  metrics()
  keep.forEach(({ w, f }) => { const fin = w.final; rebuild(w, w.text); w.final = fin; w.ink = f * w.geo.total; markReach(w, performance.now() - 2000); w.dx = null })
  lines = []; layout()
}

let greeting = false
export const isGreeting = () => greeting
export function clearInk() { words.forEach(dissolve); words = []; lines = []; pen.active = null; greeting = false }
export function greet() {
  if (words.length) return
  setTranscript('Hi. Look at anything to choose it.'.split(' '), [])
  greeting = true
}

/** Attach to the DOM rendered by GlassApp. Returns a teardown. */
export function mount() {
  lightC = $('#light'); ghostC = $('#ghost'); inkC = $('#ink'); TC = $('#twin'); gazeEl = $('#gaze')
  L = lightC.getContext('2d')!; Gx = ghostC.getContext('2d')!; K = inkC.getContext('2d')!; T = TC.getContext('2d')!
  resize()
  addEventListener('resize', resize)
  addEventListener('pointermove', onPointer)
  addEventListener('keydown', onKey)
  document.addEventListener('click', onClick)
  const leave = () => { if (source === 'pointer') gx = gy = -100 }
  document.addEventListener('mouseleave', leave)
  last = performance.now(); raf = requestAnimationFrame(frame)
  return () => {
    cancelAnimationFrame(raf)
    removeEventListener('resize', resize)
    removeEventListener('pointermove', onPointer)
    removeEventListener('keydown', onKey)
    document.removeEventListener('click', onClick)
    document.removeEventListener('mouseleave', leave)
    closeBubbles()
  }
}
