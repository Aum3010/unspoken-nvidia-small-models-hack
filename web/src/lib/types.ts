// Mirrors shared/schemas.py — keep in sync.

export type Modality = 'lip' | 'audio' | 'gaze' | 'gesture' | 'board' | 'eeg_replay'
export type SourceMode = 'live' | 'prerecorded'
export type ConfirmationMethod =
  | 'gaze_dwell_confirmed'
  | 'deliberate_blink_confirmed'
  | 'accessible_button_confirmed'
  | 'keyboard_confirmed'
  | 'other_explicit_user_confirmed'

export interface Hypothesis {
  id: string
  text: string
  score: number
  score_kind: string
  words?: { word: string; p?: number; start?: number; end?: number }[] | null
}

export interface Evidence {
  event_id: string
  modality: Modality
  source_mode: SourceMode
  capture_time_ms: number
  source_ref: string
  hypotheses: Hypothesis[]
  quality_flags: string[]
  quality: Record<string, any>
  provenance: { service: string; checkpoint: string; preprocess: string; device: string; latency_ms?: number }
  status: 'proposed' | 'ambiguous' | 'confirmed' | 'rejected' | 'unknown'
}

export interface Candidate {
  id: string
  text: string
  origin: 'evidence' | 'user_edit' | 'board' | 'spelling' | 'eeg_replay'
  evidence_event_id?: string | null
  modality: Modality
  source_mode: SourceMode
  score?: number | null
  supporting_modalities: Modality[]
}

export type MessageStatus =
  | 'collecting' | 'proposed' | 'ambiguous' | 'clarifying' | 'selected' | 'confirmed' | 'spoken' | 'rejected'

export interface Message {
  id: string
  status: MessageStatus
  evidence_ids: string[]
  candidates: Candidate[]
  selected_candidate_id?: string | null
  confirmed_text?: string | null
  confirmation_method?: ConfirmationMethod | null
  source_mode: SourceMode
}

export interface ToolCall {
  name: string
  arguments: Record<string, any>
  rationale: string
  by: 'llm' | 'deterministic_fallback'
  latency_ms?: number | null
  valid: boolean
  validation_errors: string[]
  raw?: Record<string, any> | null
}

export interface TimelineItem {
  id: string
  ts: number
  agent: string
  title: string
  detail: string
  latency_ms?: number | null
  level: 'info' | 'warn' | 'error' | 'pending'
}

export interface SpokenItem {
  message_id: string
  text: string
  method: string
  ts: number
  modality: Modality
  origin: string
  source_mode: SourceMode
  supporting_modalities: Modality[]
  tts_latency_ms?: number | null
  raw: { modality: Modality; hypotheses: string[] }[]
}

export interface SessionState {
  session_id: string
  message: Message
  evidence: Evidence[]
  agreement: { lip: string; audio: string; word_overlap: number } | null
  plan: ToolCall | null
  shown_candidate_ids: string[]
  spoken: SpokenItem[]
  timeline: TimelineItem[]
  settings: {
    available_inputs: string[]
    gaze_calibrated: boolean
    use_llm: boolean
    voice: string
    speech_rate: number
  }
  audit_count: number
}

export type ServiceHealth = Record<string, { state: string; error?: string | null; [k: string]: any }>
