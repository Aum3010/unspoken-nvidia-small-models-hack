import { AudioLines, Brain, Eye, Grid3x3, Hand, Keyboard, Smile, type LucideIcon } from 'lucide-react'
import type { Modality } from '../lib/types'

export const MODALITY: Record<Modality | 'spelling', { label: string; color: string; icon: LucideIcon; badge: string }> = {
  lip: { label: 'Lips', color: 'var(--lip)', icon: Smile, badge: 'LIVE CAMERA' },
  audio: { label: 'Weak speech', color: 'var(--audio)', icon: AudioLines, badge: 'LIVE MICROPHONE' },
  gaze: { label: 'Eyes', color: 'var(--gaze)', icon: Eye, badge: 'LIVE CAMERA' },
  gesture: { label: 'Gesture', color: 'var(--gaze)', icon: Hand, badge: 'LIVE CAMERA' },
  board: { label: 'Board', color: 'var(--board)', icon: Grid3x3, badge: 'USER SELECTED' },
  spelling: { label: 'Spelling', color: 'var(--board)', icon: Keyboard, badge: 'USER SPELLED' },
  eeg_replay: { label: 'Recorded EEG', color: 'var(--eeg)', icon: Brain, badge: 'RECORDED EEG' },
}

export const AGENT: Record<string, { label: string; color: string }> = {
  vsr: { label: 'Lip-reading agent', color: 'var(--lip)' },
  audio: { label: 'Voice agent', color: 'var(--audio)' },
  eeg: { label: 'EEG replay agent', color: 'var(--eeg)' },
  coordinator: { label: 'Coordinator', color: 'var(--ai)' },
  policy: { label: 'Policy gate', color: 'var(--voice)' },
  tts: { label: 'Voice output', color: 'var(--voice)' },
  user: { label: 'Person', color: 'var(--text)' },
  gaze: { label: 'Gaze agent', color: 'var(--gaze)' },
}

export const FLAG_TEXT: Record<string, string> = {
  no_face: 'No face in view',
  partial_face: 'Face partly out of view',
  occlusion: 'Mouth hidden',
  low_mouth_motion: 'Little mouth movement',
  too_short: 'Too short',
  too_long: 'Long clip',
  low_signal: 'Very quiet',
  no_speech_detected: 'No voice detected',
  low_word_confidence: 'Some words unsure',
  enhancer_disagrees: 'Raw ≠ enhanced audio',
  out_of_distribution: 'Unusual input',
  calibration_poor: 'Calibration poor',
  recorded_data: 'Recorded data',
}

export const METHOD_TEXT: Record<string, string> = {
  gaze_dwell_confirmed: 'gaze dwell',
  deliberate_blink_confirmed: 'deliberate blink',
  accessible_button_confirmed: 'button',
  keyboard_confirmed: 'keyboard',
  other_explicit_user_confirmed: 'dwell',
}

export function Badge({ modality, live }: { modality: Modality | 'spelling'; live?: boolean }) {
  const m = MODALITY[modality]
  return <span className={`badge ${live ? 'live' : ''}`} style={{ ['--c' as any]: m.color }}>{m.badge}</span>
}
