import type { ButtonHTMLAttributes, CSSProperties, ReactNode } from 'react'
import { useDwellTarget } from '../lib/selection'
import type { ConfirmationMethod } from '../lib/types'

interface Props extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'onClick'> {
  onActivate: (method: ConfirmationMethod) => void
  kind?: 'select' | 'confirm'
  ringColor?: string
  children: ReactNode
}

/** A large target that activates by click, keyboard, gaze dwell, hover dwell, or blink scan. */
export function DwellButton({ onActivate, kind = 'select', ringColor, className = '', style, children, ...rest }: Props) {
  const { ref, progress, focus } = useDwellTarget(kind, onActivate)
  const r = 48
  const c = 2 * Math.PI * r
  return (
    <button
      ref={ref}
      {...rest}
      className={`dwell ${className} ${focus ? 'scan-focus' : ''} ${progress > 0 ? 'gaze-hover' : ''}`}
      style={{ ...style, '--ring': ringColor ?? 'var(--gaze)' } as CSSProperties}
      onClick={(e) => onActivate(e.detail === 0 ? 'keyboard_confirmed' : 'accessible_button_confirmed')}
    >
      {children}
      {progress > 0 && (
        <span className="dwell-ring" aria-hidden>
          <svg viewBox="0 0 100 100" preserveAspectRatio="none">
            <rect x="1.5" y="1.5" width="97" height="97" rx="14" ry="14" fill="none"
              stroke={ringColor ?? 'var(--gaze)'} strokeWidth="3" vectorEffect="non-scaling-stroke"
              pathLength={c} strokeDasharray={c} strokeDashoffset={c * (1 - progress)}
              style={{ filter: `drop-shadow(0 0 6px ${ringColor ?? 'var(--gaze)'})` }} />
          </svg>
        </span>
      )}
    </button>
  )
}
