// Minimal SVG charts (no chart library): line series with axes, legend and hover readout.

import { useState } from 'react'

export interface Series { name: string; color: string; values: number[]; dashed?: boolean }

export function LineChart({ x, series, height = 200, yLabel, xLabel, yDomain, xFmt = (v) => String(v), yFmt = (v) => v.toFixed(1), marker }: {
  x: number[]; series: Series[]; height?: number; yLabel?: string; xLabel?: string
  yDomain?: [number, number]; xFmt?: (v: number) => string; yFmt?: (v: number) => string; marker?: number
}) {
  const [hover, setHover] = useState<number | null>(null)
  const W = 520, H = height, L = 44, R = 12, T = 12, B = 30
  const all = series.flatMap((s) => s.values)
  const [y0, y1] = yDomain ?? [Math.min(...all), Math.max(...all)]
  const pad = yDomain ? 0 : (y1 - y0) * 0.08
  const ymin = y0 - pad, ymax = y1 + pad
  const sx = (v: number) => L + ((v - x[0]) / (x[x.length - 1] - x[0] || 1)) * (W - L - R)
  const sy = (v: number) => T + (1 - (v - ymin) / (ymax - ymin || 1)) * (H - T - B)
  const ticks = 4
  const yt = Array.from({ length: ticks + 1 }, (_, i) => ymin + ((ymax - ymin) * i) / ticks)
  const xt = Array.from({ length: 5 }, (_, i) => x[Math.round(((x.length - 1) * i) / 4)])

  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={`${yLabel ?? ''} vs ${xLabel ?? ''}`}
        onMouseLeave={() => setHover(null)}
        onMouseMove={(e) => {
          const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect()
          const px = ((e.clientX - r.left) / r.width) * W
          let best = 0
          x.forEach((v, i) => { if (Math.abs(sx(v) - px) < Math.abs(sx(x[best]) - px)) best = i })
          setHover(best)
        }}>
        {yt.map((v, i) => (
          <g key={i}>
            <line x1={L} x2={W - R} y1={sy(v)} y2={sy(v)} stroke="var(--hairline)" />
            <text x={L - 8} y={sy(v) + 4} textAnchor="end" fontSize="10.5" fill="var(--text-3)" fontFamily="var(--font-mono)">{yFmt(v)}</text>
          </g>
        ))}
        {xt.map((v, i) => (
          <text key={i} x={sx(v)} y={H - 10} textAnchor="middle" fontSize="10.5" fill="var(--text-3)" fontFamily="var(--font-mono)">{xFmt(v)}</text>
        ))}
        {marker != null && <line x1={sx(marker)} x2={sx(marker)} y1={T} y2={H - B} stroke="var(--hairline-strong)" strokeDasharray="3 3" />}
        {series.map((s) => (
          <path key={s.name} fill="none" stroke={s.color} strokeWidth="2.2" strokeLinejoin="round" strokeLinecap="round"
            strokeDasharray={s.dashed ? '5 4' : undefined}
            d={s.values.map((v, i) => `${i ? 'L' : 'M'}${sx(x[i]).toFixed(1)} ${sy(v).toFixed(1)}`).join(' ')} />
        ))}
        {hover != null && (
          <g>
            <line x1={sx(x[hover])} x2={sx(x[hover])} y1={T} y2={H - B} stroke="var(--text-3)" strokeWidth=".8" />
            {series.map((s) => <circle key={s.name} cx={sx(x[hover])} cy={sy(s.values[hover])} r="3.6" fill={s.color} stroke="var(--bg)" strokeWidth="1.5" />)}
          </g>
        )}
      </svg>
      <div className="chart-legend">
        {series.map((s) => (
          <span key={s.name}>
            <i style={{ background: s.color }} />{s.name}
            {hover != null && <b className="mono"> {yFmt(s.values[hover])}</b>}
          </span>
        ))}
        {hover != null && <span className="faint mono">@ {xFmt(x[hover])}{xLabel ? ` ${xLabel}` : ''}</span>}
      </div>
    </div>
  )
}
