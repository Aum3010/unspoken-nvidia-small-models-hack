import { motion } from 'framer-motion'

export function BrandMark({ size = 38, speaking = false }: { size?: number; speaking?: boolean }) {
  return (
    <svg className="brand-mark" width={size} height={size} viewBox="0 0 64 64" aria-hidden>
      <defs>
        <linearGradient id="bm-g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="var(--voice)" />
          <stop offset="1" stopColor="var(--ai)" />
        </linearGradient>
      </defs>
      <rect x="1" y="1" width="62" height="62" rx="18" fill="var(--surface-2)" stroke="var(--hairline-strong)" />
      {[0, 1, 2, 3, 4].map((i) => {
        const h = [10, 22, 30, 22, 10][i]
        return (
          <motion.rect
            key={i} x={13 + i * 8.5} width="4.5" rx="2.25" fill="url(#bm-g)"
            initial={false}
            animate={speaking
              ? { height: [h * 0.4, h, h * 0.55, h * 0.9, h * 0.4], y: [32 - h * 0.2, 32 - h / 2, 32 - h * 0.27, 32 - h * 0.45, 32 - h * 0.2] }
              : { height: h * 0.3 + 3, y: 32 - (h * 0.3 + 3) / 2 }}
            transition={speaking ? { repeat: Infinity, duration: 0.9 + i * 0.07, ease: 'easeInOut' } : { duration: 0.6 }}
          />
        )
      })}
    </svg>
  )
}
