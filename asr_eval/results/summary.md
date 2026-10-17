# ASR evaluation summary

Generated 2026-10-17 19:59 · config `eval.yaml` · **test split only** (thresholds tuned on calib).

> ⚠️ No real human recordings of the 20 phrases in the test split yet: the headline positive columns are empty. TTS rows below are sanity checks only.
> ⚠️ Thresholds for parakeet-ctc-0.6b/text|phrases20, parakeet-ctc-0.6b/ctc_sum|phrases20, parakeet-ctc-0.6b/ctc_norm|phrases20, parakeet-ctc-0.6b/ctc_llr|phrases20, parakeet-tdt-0.6b-v2/text|phrases20, nemotron-streaming-0.6b/text|phrases20, nemotron-streaming-0.6b-160ms/text|phrases20, nemotron-streaming-0.6b-560ms/text|phrases20, whisper-large-v3-turbo/text|phrases20, whisper-large-v2/text|phrases20, whisper-small/text|phrases20 were tuned with **TTS positives** (no human calib speakers for that task) against real negatives. Record at least two speakers so tuning uses human speech.
> ⚠️ No threshold met the calib false-accept target for: parakeet-ctc-0.6b/text|torgo_words, nemotron-streaming-0.6b/text|torgo_words, nemotron-streaming-0.6b-160ms/text|torgo_words, nemotron-streaming-0.6b-560ms/text|torgo_words (strictest point used; see Thresholds).

## Headline: 20 phrases, real recordings, not degraded

| Model / method | Top-1 rank | Top-3 rank | Accepted correct | Wrong accept | Abstain | FA hard-neg | FA unknown speech | FA non-speech | FA silence |
|---|---|---|---|---|---|---|---|---|---|
| parakeet-ctc-0.6b/text | — | — | — | — | — | — | 0/399 (0.0%) [0.0–1.0] ≤0.8% | 0/182 (0.0%) [0.0–2.1] ≤1.6% | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| parakeet-ctc-0.6b/ctc_sum | — | — | — | — | — | — | 1/399 (0.3%) [0.0–1.4] | 0/182 (0.0%) [0.0–2.1] ≤1.6% | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| parakeet-ctc-0.6b/ctc_norm | — | — | — | — | — | — | 0/399 (0.0%) [0.0–1.0] ≤0.8% | 1/182 (0.5%) [0.1–3.0] | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| parakeet-ctc-0.6b/ctc_llr | — | — | — | — | — | — | 0/399 (0.0%) [0.0–1.0] ≤0.8% | 1/182 (0.5%) [0.1–3.0] | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| parakeet-tdt-0.6b-v2/text | — | — | — | — | — | — | 8/399 (2.0%) [1.0–3.9] | 0/182 (0.0%) [0.0–2.1] ≤1.6% | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| nemotron-streaming-0.6b/text | — | — | — | — | — | — | 0/399 (0.0%) [0.0–1.0] ≤0.8% | 0/182 (0.0%) [0.0–2.1] ≤1.6% | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| nemotron-streaming-0.6b-160ms/text | — | — | — | — | — | — | 5/399 (1.3%) [0.5–2.9] | 0/182 (0.0%) [0.0–2.1] ≤1.6% | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| nemotron-streaming-0.6b-560ms/text | — | — | — | — | — | — | 0/399 (0.0%) [0.0–1.0] ≤0.8% | 0/182 (0.0%) [0.0–2.1] ≤1.6% | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| whisper-large-v3-turbo/text | — | — | — | — | — | — | 1/399 (0.3%) [0.0–1.4] | 0/182 (0.0%) [0.0–2.1] ≤1.6% | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| whisper-large-v2/text | — | — | — | — | — | — | 0/399 (0.0%) [0.0–1.0] ≤0.8% | 1/182 (0.5%) [0.1–3.0] | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| whisper-small/text | — | — | — | — | — | — | 1/399 (0.3%) [0.0–1.4] | 0/182 (0.0%) [0.0–2.1] ≤1.6% | 0/180 (0.0%) [0.0–2.1] ≤1.7% |

FA = false accept (a phrase proposed on a negative clip): k/n (rate) [95% Wilson CI]; ≤x% = rule-of-three upper bound when k = 0. Ranking columns are before abstention; 'Accepted correct' is correct candidates over all positives.

## Per condition (parakeet-ctc-0.6b/ctc_sum)

### TTS — sanity only

| Condition | n | Top-1 rank | Accepted correct | Wrong accept | Abstain |
|---|---:|---|---|---|---|
| tts_clean | 340 | 340/340 (100.0%) | 338/340 (99.4%) | 0/340 (0.0%) | 2/340 (0.6%) |

## TORGO surrogate closed-vocabulary task: dysarthric vs control

Vocabulary (config): yes, no, up, down, left, right, go, one, two, three

| Model / method | Group | Top-1 rank | Accepted correct | Wrong accept | FA other words | WER / CER (all TORGO clips) |
|---|---|---|---|---|---|---|
| parakeet-ctc-0.6b/text | control | 103/118 (87.3%) | 24/118 (20.3%) | 0/118 (0.0%) | 5/150 (3.3%) [1.4–7.6] | 10.5% / 5.6% (n=470) |
| parakeet-ctc-0.6b/text | dysarthric | 73/90 (81.1%) | 12/90 (13.3%) | 1/90 (1.1%) | 2/57 (3.5%) [1.0–11.9] | 31.8% / 20.3% (n=232) |
| parakeet-ctc-0.6b/ctc_sum | control | 103/118 (87.3%) | 62/118 (52.5%) | 0/118 (0.0%) | 1/150 (0.7%) [0.1–3.7] | 10.5% / 5.6% (n=470) |
| parakeet-ctc-0.6b/ctc_sum | dysarthric | 77/90 (85.6%) | 32/90 (35.6%) | 0/90 (0.0%) | 0/57 (0.0%) [0.0–6.3] ≤5.3% | 31.8% / 20.3% (n=232) |
| parakeet-ctc-0.6b/ctc_norm | control | 103/118 (87.3%) | 62/118 (52.5%) | 0/118 (0.0%) | 1/150 (0.7%) [0.1–3.7] | 10.5% / 5.6% (n=470) |
| parakeet-ctc-0.6b/ctc_norm | dysarthric | 77/90 (85.6%) | 32/90 (35.6%) | 0/90 (0.0%) | 0/57 (0.0%) [0.0–6.3] ≤5.3% | 31.8% / 20.3% (n=232) |
| parakeet-ctc-0.6b/ctc_llr | control | 103/118 (87.3%) | 62/118 (52.5%) | 0/118 (0.0%) | 0/150 (0.0%) [0.0–2.5] ≤2.0% | 10.5% / 5.6% (n=470) |
| parakeet-ctc-0.6b/ctc_llr | dysarthric | 77/90 (85.6%) | 32/90 (35.6%) | 0/90 (0.0%) | 0/57 (0.0%) [0.0–6.3] ≤5.3% | 31.8% / 20.3% (n=232) |
| parakeet-tdt-0.6b-v2/text | control | 114/118 (96.6%) | 82/118 (69.5%) | 0/118 (0.0%) | 3/150 (2.0%) [0.7–5.7] | 5.8% / 2.7% (n=470) |
| parakeet-tdt-0.6b-v2/text | dysarthric | 74/90 (82.2%) | 42/90 (46.7%) | 1/90 (1.1%) | 0/57 (0.0%) [0.0–6.3] ≤5.3% | 27.2% / 19.1% (n=232) |
| nemotron-streaming-0.6b/text | control | 118/118 (100.0%) | 63/118 (53.4%) | 0/118 (0.0%) | 0/150 (0.0%) [0.0–2.5] ≤2.0% | 5.2% / 2.6% (n=470) |
| nemotron-streaming-0.6b/text | dysarthric | 84/90 (93.3%) | 36/90 (40.0%) | 1/90 (1.1%) | 0/57 (0.0%) [0.0–6.3] ≤5.3% | 27.0% / 20.0% (n=232) |
| nemotron-streaming-0.6b-160ms/text | control | 112/118 (94.9%) | 59/118 (50.0%) | 0/118 (0.0%) | 0/150 (0.0%) [0.0–2.5] ≤2.0% | 6.9% / 4.4% (n=470) |
| nemotron-streaming-0.6b-160ms/text | dysarthric | 76/90 (84.4%) | 35/90 (38.9%) | 0/90 (0.0%) | 0/57 (0.0%) [0.0–6.3] ≤5.3% | 30.9% / 23.9% (n=232) |
| nemotron-streaming-0.6b-560ms/text | control | 118/118 (100.0%) | 62/118 (52.5%) | 0/118 (0.0%) | 0/150 (0.0%) [0.0–2.5] ≤2.0% | 5.6% / 3.0% (n=470) |
| nemotron-streaming-0.6b-560ms/text | dysarthric | 84/90 (93.3%) | 36/90 (40.0%) | 1/90 (1.1%) | 0/57 (0.0%) [0.0–6.3] ≤5.3% | 27.8% / 20.3% (n=232) |
| whisper-large-v3-turbo/text | control | 108/118 (91.5%) | 14/118 (11.9%) | 0/118 (0.0%) | 0/150 (0.0%) [0.0–2.5] ≤2.0% | 5.6% / 2.7% (n=470) |
| whisper-large-v3-turbo/text | dysarthric | 82/90 (91.1%) | 5/90 (5.6%) | 0/90 (0.0%) | 0/57 (0.0%) [0.0–6.3] ≤5.3% | 23.6% / 14.0% (n=232) |
| whisper-large-v2/text | control | 99/118 (83.9%) | 69/118 (58.5%) | 0/118 (0.0%) | 2/150 (1.3%) [0.4–4.7] | 7.2% / 3.7% (n=470) |
| whisper-large-v2/text | dysarthric | 73/90 (81.1%) | 42/90 (46.7%) | 0/90 (0.0%) | 0/57 (0.0%) [0.0–6.3] ≤5.3% | 32.7% / 26.4% (n=232) |
| whisper-small/text | control | 112/118 (94.9%) | 87/118 (73.7%) | 0/118 (0.0%) | 1/150 (0.7%) [0.1–3.7] | 6.9% / 3.6% (n=470) |
| whisper-small/text | dysarthric | 79/90 (87.8%) | 46/90 (51.1%) | 0/90 (0.0%) | 0/57 (0.0%) [0.0–6.3] ≤5.3% | 37.5% / 47.3% (n=232) |

### TORGO words by condition (whisper-small/text, best on calib for this task; degraded rows are synthetic)

| Condition | Group | n | Top-1 rank | Accepted correct | Wrong accept |
|---|---|---:|---|---|---|
| bandpass_300-3400 | control | 118 | 108/118 (91.5%) | 82/118 (69.5%) | 0/118 (0.0%) |
| bandpass_300-3400 | dysarthric | 90 | 78/90 (86.7%) | 45/90 (50.0%) | 1/90 (1.1%) |
| clean | control | 118 | 112/118 (94.9%) | 87/118 (73.7%) | 0/118 (0.0%) |
| clean | dysarthric | 90 | 79/90 (87.8%) | 46/90 (51.1%) | 0/90 (0.0%) |
| gain_-12dB | control | 118 | 111/118 (94.1%) | 86/118 (72.9%) | 0/118 (0.0%) |
| gain_-12dB | dysarthric | 90 | 77/90 (85.6%) | 47/90 (52.2%) | 0/90 (0.0%) |
| gain_-24dB | control | 118 | 113/118 (95.8%) | 85/118 (72.0%) | 0/118 (0.0%) |
| gain_-24dB | dysarthric | 90 | 75/90 (83.3%) | 45/90 (50.0%) | 0/90 (0.0%) |
| gain_-36dB | control | 118 | 108/118 (91.5%) | 66/118 (55.9%) | 0/118 (0.0%) |
| gain_-36dB | dysarthric | 90 | 74/90 (82.2%) | 41/90 (45.6%) | 0/90 (0.0%) |
| lowpass_1500 | control | 118 | 102/118 (86.4%) | 68/118 (57.6%) | 0/118 (0.0%) |
| lowpass_1500 | dysarthric | 90 | 71/90 (78.9%) | 34/90 (37.8%) | 0/90 (0.0%) |
| noise_musan_snr0 | control | 118 | 106/118 (89.8%) | 72/118 (61.0%) | 0/118 (0.0%) |
| noise_musan_snr0 | dysarthric | 90 | 69/90 (76.7%) | 38/90 (42.2%) | 0/90 (0.0%) |
| noise_musan_snr10 | control | 118 | 110/118 (93.2%) | 84/118 (71.2%) | 0/118 (0.0%) |
| noise_musan_snr10 | dysarthric | 90 | 75/90 (83.3%) | 43/90 (47.8%) | 0/90 (0.0%) |
| noise_musan_snr20 | control | 118 | 113/118 (95.8%) | 86/118 (72.9%) | 0/118 (0.0%) |
| noise_musan_snr20 | dysarthric | 90 | 76/90 (84.4%) | 47/90 (52.2%) | 0/90 (0.0%) |
| noise_musan_snr5 | control | 118 | 112/118 (94.9%) | 82/118 (69.5%) | 0/118 (0.0%) |
| noise_musan_snr5 | dysarthric | 90 | 73/90 (81.1%) | 43/90 (47.8%) | 0/90 (0.0%) |
| reverb_rt60_0.3s | control | 118 | 112/118 (94.9%) | 83/118 (70.3%) | 0/118 (0.0%) |
| reverb_rt60_0.3s | dysarthric | 90 | 75/90 (83.3%) | 43/90 (47.8%) | 0/90 (0.0%) |
| reverb_rt60_0.6s | control | 118 | 102/118 (86.4%) | 73/118 (61.9%) | 0/118 (0.0%) |
| reverb_rt60_0.6s | dysarthric | 90 | 70/90 (77.8%) | 36/90 (40.0%) | 1/90 (1.1%) |
| tempo_0.7x | control | 118 | 106/118 (89.8%) | 67/118 (56.8%) | 0/118 (0.0%) |
| tempo_0.7x | dysarthric | 90 | 64/90 (71.1%) | 32/90 (35.6%) | 0/90 (0.0%) |
| tempo_0.85x | control | 118 | 106/118 (89.8%) | 70/118 (59.3%) | 1/118 (0.8%) |
| tempo_0.85x | dysarthric | 90 | 63/90 (70.0%) | 32/90 (35.6%) | 0/90 (0.0%) |

## Negatives by source (test)

| Model / method | Source | Label | False accepts |
|---|---|---|---|
| parakeet-ctc-0.6b/text | esc50 | non_speech | 0/63 (0.0%) [0.0–5.7] ≤4.8% |
| parakeet-ctc-0.6b/text | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| parakeet-ctc-0.6b/text | librispeech | unknown_speech | 0/112 (0.0%) [0.0–3.3] ≤2.7% |
| parakeet-ctc-0.6b/text | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| parakeet-ctc-0.6b/text | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| parakeet-ctc-0.6b/text | torgo | unknown_speech | 0/287 (0.0%) [0.0–1.3] ≤1.0% |
| parakeet-ctc-0.6b/text | tts (sanity) | hard_negative | 0/119 (0.0%) [0.0–3.1] ≤2.5% |
| parakeet-ctc-0.6b/ctc_sum | esc50 | non_speech | 0/63 (0.0%) [0.0–5.7] ≤4.8% |
| parakeet-ctc-0.6b/ctc_sum | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| parakeet-ctc-0.6b/ctc_sum | librispeech | unknown_speech | 0/112 (0.0%) [0.0–3.3] ≤2.7% |
| parakeet-ctc-0.6b/ctc_sum | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| parakeet-ctc-0.6b/ctc_sum | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| parakeet-ctc-0.6b/ctc_sum | torgo | unknown_speech | 1/287 (0.3%) [0.1–1.9] |
| parakeet-ctc-0.6b/ctc_sum | tts (sanity) | hard_negative | 0/119 (0.0%) [0.0–3.1] ≤2.5% |
| parakeet-ctc-0.6b/ctc_norm | esc50 | non_speech | 1/63 (1.6%) [0.3–8.5] |
| parakeet-ctc-0.6b/ctc_norm | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| parakeet-ctc-0.6b/ctc_norm | librispeech | unknown_speech | 0/112 (0.0%) [0.0–3.3] ≤2.7% |
| parakeet-ctc-0.6b/ctc_norm | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| parakeet-ctc-0.6b/ctc_norm | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| parakeet-ctc-0.6b/ctc_norm | torgo | unknown_speech | 0/287 (0.0%) [0.0–1.3] ≤1.0% |
| parakeet-ctc-0.6b/ctc_norm | tts (sanity) | hard_negative | 18/119 (15.1%) [9.8–22.6] |
| parakeet-ctc-0.6b/ctc_llr | esc50 | non_speech | 1/63 (1.6%) [0.3–8.5] |
| parakeet-ctc-0.6b/ctc_llr | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| parakeet-ctc-0.6b/ctc_llr | librispeech | unknown_speech | 0/112 (0.0%) [0.0–3.3] ≤2.7% |
| parakeet-ctc-0.6b/ctc_llr | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| parakeet-ctc-0.6b/ctc_llr | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| parakeet-ctc-0.6b/ctc_llr | torgo | unknown_speech | 0/287 (0.0%) [0.0–1.3] ≤1.0% |
| parakeet-ctc-0.6b/ctc_llr | tts (sanity) | hard_negative | 0/119 (0.0%) [0.0–3.1] ≤2.5% |
| parakeet-tdt-0.6b-v2/text | esc50 | non_speech | 0/63 (0.0%) [0.0–5.7] ≤4.8% |
| parakeet-tdt-0.6b-v2/text | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| parakeet-tdt-0.6b-v2/text | librispeech | unknown_speech | 0/112 (0.0%) [0.0–3.3] ≤2.7% |
| parakeet-tdt-0.6b-v2/text | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| parakeet-tdt-0.6b-v2/text | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| parakeet-tdt-0.6b-v2/text | torgo | unknown_speech | 8/287 (2.8%) [1.4–5.4] |
| parakeet-tdt-0.6b-v2/text | tts (sanity) | hard_negative | 49/119 (41.2%) [32.7–50.2] |
| nemotron-streaming-0.6b/text | esc50 | non_speech | 0/63 (0.0%) [0.0–5.7] ≤4.8% |
| nemotron-streaming-0.6b/text | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| nemotron-streaming-0.6b/text | librispeech | unknown_speech | 0/112 (0.0%) [0.0–3.3] ≤2.7% |
| nemotron-streaming-0.6b/text | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| nemotron-streaming-0.6b/text | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| nemotron-streaming-0.6b/text | torgo | unknown_speech | 0/287 (0.0%) [0.0–1.3] ≤1.0% |
| nemotron-streaming-0.6b/text | tts (sanity) | hard_negative | 0/119 (0.0%) [0.0–3.1] ≤2.5% |
| nemotron-streaming-0.6b-160ms/text | esc50 | non_speech | 0/63 (0.0%) [0.0–5.7] ≤4.8% |
| nemotron-streaming-0.6b-160ms/text | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| nemotron-streaming-0.6b-160ms/text | librispeech | unknown_speech | 1/112 (0.9%) [0.2–4.9] |
| nemotron-streaming-0.6b-160ms/text | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| nemotron-streaming-0.6b-160ms/text | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| nemotron-streaming-0.6b-160ms/text | torgo | unknown_speech | 4/287 (1.4%) [0.5–3.5] |
| nemotron-streaming-0.6b-160ms/text | tts (sanity) | hard_negative | 50/119 (42.0%) [33.5–51.0] |
| nemotron-streaming-0.6b-560ms/text | esc50 | non_speech | 0/63 (0.0%) [0.0–5.7] ≤4.8% |
| nemotron-streaming-0.6b-560ms/text | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| nemotron-streaming-0.6b-560ms/text | librispeech | unknown_speech | 0/112 (0.0%) [0.0–3.3] ≤2.7% |
| nemotron-streaming-0.6b-560ms/text | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| nemotron-streaming-0.6b-560ms/text | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| nemotron-streaming-0.6b-560ms/text | torgo | unknown_speech | 0/287 (0.0%) [0.0–1.3] ≤1.0% |
| nemotron-streaming-0.6b-560ms/text | tts (sanity) | hard_negative | 17/119 (14.3%) [9.1–21.7] |
| whisper-large-v3-turbo/text | esc50 | non_speech | 0/63 (0.0%) [0.0–5.7] ≤4.8% |
| whisper-large-v3-turbo/text | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| whisper-large-v3-turbo/text | librispeech | unknown_speech | 0/112 (0.0%) [0.0–3.3] ≤2.7% |
| whisper-large-v3-turbo/text | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| whisper-large-v3-turbo/text | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| whisper-large-v3-turbo/text | torgo | unknown_speech | 1/287 (0.3%) [0.1–1.9] |
| whisper-large-v3-turbo/text | tts (sanity) | hard_negative | 17/119 (14.3%) [9.1–21.7] |
| whisper-large-v2/text | esc50 | non_speech | 1/63 (1.6%) [0.3–8.5] |
| whisper-large-v2/text | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| whisper-large-v2/text | librispeech | unknown_speech | 0/112 (0.0%) [0.0–3.3] ≤2.7% |
| whisper-large-v2/text | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| whisper-large-v2/text | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| whisper-large-v2/text | torgo | unknown_speech | 0/287 (0.0%) [0.0–1.3] ≤1.0% |
| whisper-large-v2/text | tts (sanity) | hard_negative | 17/119 (14.3%) [9.1–21.7] |
| whisper-small/text | esc50 | non_speech | 0/63 (0.0%) [0.0–5.7] ≤4.8% |
| whisper-small/text | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| whisper-small/text | librispeech | unknown_speech | 0/112 (0.0%) [0.0–3.3] ≤2.7% |
| whisper-small/text | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| whisper-small/text | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| whisper-small/text | torgo | unknown_speech | 1/287 (0.3%) [0.1–1.9] |
| whisper-small/text | tts (sanity) | hard_negative | 17/119 (14.3%) [9.1–21.7] |

## WER / CER of the free transcript

| Model | Source | WER | CER | n |
|---|---|---|---|---|
| parakeet-ctc-0.6b | librispeech | 3.1% | 0.9% | 112 |
| parakeet-ctc-0.6b | torgo | 17.2% | 10.1% | 702 |
| parakeet-ctc-0.6b | tts | 1.7% | 1.0% | 459 |
| parakeet-tdt-0.6b-v2 | librispeech | 2.2% | 0.6% | 112 |
| parakeet-tdt-0.6b-v2 | torgo | 12.5% | 7.7% | 702 |
| parakeet-tdt-0.6b-v2 | tts | 2.1% | 1.6% | 459 |
| nemotron-streaming-0.6b | librispeech | 3.0% | 1.0% | 112 |
| nemotron-streaming-0.6b | torgo | 12.0% | 7.9% | 702 |
| nemotron-streaming-0.6b | tts | 2.1% | 0.9% | 459 |
| nemotron-streaming-0.6b-160ms | librispeech | 3.4% | 1.1% | 112 |
| nemotron-streaming-0.6b-160ms | torgo | 14.4% | 10.3% | 702 |
| nemotron-streaming-0.6b-160ms | tts | 1.9% | 1.6% | 459 |
| nemotron-streaming-0.6b-560ms | librispeech | 3.3% | 1.1% | 112 |
| nemotron-streaming-0.6b-560ms | torgo | 12.6% | 8.3% | 702 |
| nemotron-streaming-0.6b-560ms | tts | 2.0% | 1.3% | 459 |
| whisper-large-v3-turbo | librispeech | 2.7% | 1.2% | 112 |
| whisper-large-v3-turbo | torgo | 11.2% | 6.1% | 702 |
| whisper-large-v3-turbo | tts | 1.9% | 0.9% | 459 |
| whisper-large-v2 | librispeech | 3.5% | 1.4% | 112 |
| whisper-large-v2 | torgo | 15.2% | 10.7% | 702 |
| whisper-large-v2 | tts | 8.7% | 6.0% | 459 |
| whisper-small | librispeech | 4.3% | 1.7% | 112 |
| whisper-small | torgo | 16.5% | 16.9% | 702 |
| whisper-small | tts | 2.2% | 1.1% | 459 |

Text normalised (lowercase, contractions expanded, punctuation removed) on both sides; empty hypotheses count as deletions.

## Latency (batch 1, warm) and resources

| Model | Params | Load | Peak torch alloc | GPU used (device) | VAD p50 | ASR p50 / p95 / max | Scoring p50 / p95 | Total p50 / p95 / max | RTFx (median) |
|---|---:|---:|---:|---:|---:|---|---|---|---:|
| parakeet-ctc-0.6b | 608.8M | 2.4 s | 1.285 GB | 4.08 GB | 13.6 ms | 13 / 17 / 273 ms | 0 / 1 ms | 27 / 46 / 306 ms | 90.9 |
| parakeet-tdt-0.6b-v2 | 617.8M | 7.5 s | 4.988 GB | 5.7 GB | 13.6 ms | 16 / 19 / 45 ms | — | 30 / 48 / 184 ms | 81.7 |
| nemotron-streaming-0.6b | 618.1M | 7.2 s | 4.99 GB | 5.79 GB | 13.6 ms | 17 / 20 / 47 ms | — | 30 / 49 / 190 ms | 81.2 |
| nemotron-streaming-0.6b-160ms | 618.1M | 7.5 s | 4.99 GB | 5.51 GB | 13.6 ms | 17 / 20 / 51 ms | — | 30 / 48 / 192 ms | 80.5 |
| nemotron-streaming-0.6b-560ms | 618.1M | 7.2 s | 4.99 GB | 5.51 GB | 13.6 ms | 17 / 20 / 52 ms | — | 31 / 48 / 194 ms | 80.5 |
| whisper-large-v3-turbo | 808.9M | 0.9 s | 0.0 GB | 4.45 GB | 13.6 ms | 42 / 63 / 373 ms | — | 56 / 88 / 382 ms | 42.2 |
| whisper-large-v2 | 1543.3M | 10.9 s | 0.0 GB | 7.7 GB | 13.6 ms | 75 / 136 / 1607 ms | — | 88 / 167 / 1614 ms | 26.1 |
| whisper-small | 241.7M | 0.5 s | 0.0 GB | 4.45 GB | 13.6 ms | 27 / 49 / 624 ms | — | 42 / 76 / 637 ms | 56.1 |

'Peak torch alloc' is PyTorch's allocator (0 for CTranslate2/faster-whisper); 'GPU used (device)' is the whole device at the end of the model's run, including its CUDA context.

Total = VAD + ASR + phrase scoring (scoring is the forced pass over all phrases; the text method skips it). Network, capture and endpointing are not included.

| Model | Checkpoint | Revision | Device |
|---|---|---|---|
| parakeet-ctc-0.6b | nvidia/parakeet-ctc-0.6b | `ad09ba1cc627` | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| parakeet-tdt-0.6b-v2 | nvidia/parakeet-tdt-0.6b-v2 | `ae9ad07059c7` | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| nemotron-streaming-0.6b | nvidia/nemotron-speech-streaming-en-0.6b | `ebe59e5a8171` | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| nemotron-streaming-0.6b-160ms | nvidia/nemotron-speech-streaming-en-0.6b | `ebe59e5a8171` | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| nemotron-streaming-0.6b-560ms | nvidia/nemotron-speech-streaming-en-0.6b | `ebe59e5a8171` | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| whisper-large-v3-turbo | mobiuslabsgmbh/faster-whisper-large-v3-turbo | `0a363e9161cb` | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| whisper-large-v2 | Systran/faster-whisper-large-v2 | `f0fe81560cb8` | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| whisper-small | Systran/faster-whisper-small | `536b0662742c` | NVIDIA RTX PRO 6000 Blackwell Server Edition |

## Thresholds (frozen from calib)

Rows ending in a name (e.g. · balanced) are extra operating points for the live UI; every number above uses the headline point (false accepts ≤ 1% per negative type on calib).

| Model / method · task | VAD thr | abs | margin | Calib accepted correct | Calib FA by type | Feasible |
|---|---|---|---|---|---|---|
| parakeet-ctc-0.6b/text · phrases20 | 0.2 | -inf | 0.375 | 97.7% (n=220, TTS) | unknown_speech 0.0% (n=201), non_speech 0.0% (n=118), silence 0.0% (n=120) | yes |
| parakeet-ctc-0.6b/text · phrases20 · balanced | 0.2 | 0.425 | 0.087 | 99.1% (n=220, TTS) | unknown_speech 1.5% (n=201), non_speech 0.8% (n=118), silence 0.0% (n=120) | yes |
| parakeet-ctc-0.6b/text · torgo_words | 0.2 | -inf | 1.000 | 12.4% (n=1512) | unknown_speech 2.2% (n=93), non_speech 0.0% (n=118), silence 0.0% (n=120) | NO |
| parakeet-ctc-0.6b/text · torgo_words · balanced | 0.2 | -inf | 1.000 | 12.4% (n=1512) | unknown_speech 2.2% (n=93), non_speech 0.0% (n=118), silence 0.0% (n=120) | yes |
| parakeet-ctc-0.6b/ctc_sum · phrases20 | 0.2 | -31.398 | 6.548 | 100.0% (n=220, TTS) | unknown_speech 0.0% (n=201), non_speech 0.0% (n=118), silence 0.0% (n=120) | yes |
| parakeet-ctc-0.6b/ctc_sum · phrases20 · balanced | 0.2 | -31.398 | 6.548 | 100.0% (n=220, TTS) | unknown_speech 0.0% (n=201), non_speech 0.0% (n=118), silence 0.0% (n=120) | yes |
| parakeet-ctc-0.6b/ctc_sum · torgo_words | 0.2 | -inf | 11.500 | 18.7% (n=1512) | unknown_speech 0.0% (n=93), non_speech 0.0% (n=118), silence 0.0% (n=120) | yes |
| parakeet-ctc-0.6b/ctc_sum · torgo_words · balanced | 0.2 | -0.114 | 3.000 | 42.1% (n=1512) | unknown_speech 4.3% (n=93), non_speech 0.0% (n=118), silence 0.0% (n=120) | yes |
| parakeet-ctc-0.6b/ctc_norm · phrases20 | 0.6 | -12.422 | 2.503 | 99.1% (n=220, TTS) | unknown_speech 0.0% (n=201), non_speech 0.8% (n=118), silence 0.0% (n=120) | yes |
| parakeet-ctc-0.6b/ctc_norm · phrases20 · balanced | 0.3 | -29.981 | 2.503 | 99.5% (n=220, TTS) | unknown_speech 1.5% (n=201), non_speech 1.7% (n=118), silence 0.0% (n=120) | yes |
| parakeet-ctc-0.6b/ctc_norm · torgo_words | 0.2 | -inf | 11.500 | 18.7% (n=1512) | unknown_speech 0.0% (n=93), non_speech 0.0% (n=118), silence 0.0% (n=120) | yes |
| parakeet-ctc-0.6b/ctc_norm · torgo_words · balanced | 0.2 | -0.114 | 3.000 | 42.1% (n=1512) | unknown_speech 4.3% (n=93), non_speech 0.0% (n=118), silence 0.0% (n=120) | yes |
| parakeet-ctc-0.6b/ctc_llr · phrases20 | 0.6 | -4.297 | 1.103 | 99.5% (n=220, TTS) | unknown_speech 0.5% (n=201), non_speech 0.8% (n=118), silence 0.0% (n=120) | yes |
| parakeet-ctc-0.6b/ctc_llr · phrases20 · balanced | 0.3 | -4.297 | 0.658 | 100.0% (n=220, TTS) | unknown_speech 1.0% (n=201), non_speech 1.7% (n=118), silence 0.0% (n=120) | yes |
| parakeet-ctc-0.6b/ctc_llr · torgo_words | 0.2 | -inf | 11.500 | 18.7% (n=1512) | unknown_speech 0.0% (n=93), non_speech 0.0% (n=118), silence 0.0% (n=120) | yes |
| parakeet-ctc-0.6b/ctc_llr · torgo_words · balanced | 0.2 | -inf | 8.100 | 34.9% (n=1512) | unknown_speech 3.2% (n=93), non_speech 0.0% (n=118), silence 0.0% (n=120) | yes |
| parakeet-tdt-0.6b-v2/text · phrases20 | 0.2 | 0.379 | 0.023 | 95.5% (n=220, TTS) | unknown_speech 1.0% (n=201), non_speech 0.0% (n=118), silence 0.0% (n=120) | yes |
| parakeet-tdt-0.6b-v2/text · phrases20 · balanced | 0.2 | 0.379 | 0.023 | 95.5% (n=220, TTS) | unknown_speech 1.0% (n=201), non_speech 0.0% (n=118), silence 0.0% (n=120) | yes |
| parakeet-tdt-0.6b-v2/text · torgo_words | 0.2 | -inf | 0.667 | 42.8% (n=1512) | unknown_speech 0.0% (n=93), non_speech 0.0% (n=118), silence 0.0% (n=120) | yes |
| parakeet-tdt-0.6b-v2/text · torgo_words · balanced | 0.2 | 0.400 | 0.333 | 60.4% (n=1512) | unknown_speech 4.3% (n=93), non_speech 0.0% (n=118), silence 0.0% (n=120) | yes |
| nemotron-streaming-0.6b/text · phrases20 | 0.2 | 0.667 | 0.000 | 98.2% (n=220, TTS) | unknown_speech 0.0% (n=201), non_speech 0.0% (n=118), silence 0.0% (n=120) | yes |
| nemotron-streaming-0.6b/text · phrases20 · balanced | 0.2 | 0.315 | 0.036 | 99.1% (n=220, TTS) | unknown_speech 2.0% (n=201), non_speech 0.0% (n=118), silence 0.0% (n=120) | yes |
| nemotron-streaming-0.6b/text · torgo_words | 0.2 | -inf | 0.750 | 35.1% (n=1512) | unknown_speech 1.1% (n=93), non_speech 0.0% (n=118), silence 0.0% (n=120) | NO |
| nemotron-streaming-0.6b/text · torgo_words · balanced | 0.2 | -inf | 0.500 | 77.7% (n=1512) | unknown_speech 3.2% (n=93), non_speech 0.0% (n=118), silence 0.0% (n=120) | yes |
| nemotron-streaming-0.6b-160ms/text · phrases20 | 0.2 | 0.361 | 0.071 | 95.0% (n=220, TTS) | unknown_speech 1.0% (n=201), non_speech 0.0% (n=118), silence 0.0% (n=120) | yes |
| nemotron-streaming-0.6b-160ms/text · phrases20 · balanced | 0.2 | 0.297 | 0.071 | 95.5% (n=220, TTS) | unknown_speech 3.5% (n=201), non_speech 0.0% (n=118), silence 0.0% (n=120) | yes |
| nemotron-streaming-0.6b-160ms/text · torgo_words | 0.2 | -inf | 0.750 | 32.1% (n=1512) | unknown_speech 1.1% (n=93), non_speech 0.0% (n=118), silence 0.0% (n=120) | NO |
| nemotron-streaming-0.6b-160ms/text · torgo_words · balanced | 0.2 | -inf | 0.500 | 70.2% (n=1512) | unknown_speech 3.2% (n=93), non_speech 0.0% (n=118), silence 0.0% (n=120) | yes |
| nemotron-streaming-0.6b-560ms/text · phrases20 | 0.2 | 0.515 | 0.000 | 97.3% (n=220, TTS) | unknown_speech 0.0% (n=201), non_speech 0.0% (n=118), silence 0.0% (n=120) | yes |
| nemotron-streaming-0.6b-560ms/text · phrases20 · balanced | 0.2 | 0.515 | 0.000 | 97.3% (n=220, TTS) | unknown_speech 0.0% (n=201), non_speech 0.0% (n=118), silence 0.0% (n=120) | yes |
| nemotron-streaming-0.6b-560ms/text · torgo_words | 0.2 | -inf | 0.750 | 35.4% (n=1512) | unknown_speech 1.1% (n=93), non_speech 0.0% (n=118), silence 0.0% (n=120) | NO |
| nemotron-streaming-0.6b-560ms/text · torgo_words · balanced | 0.2 | -inf | 0.500 | 76.7% (n=1512) | unknown_speech 3.2% (n=93), non_speech 0.0% (n=118), silence 0.0% (n=120) | yes |
| whisper-large-v3-turbo/text · phrases20 | 0.4 | -inf | 0.208 | 98.6% (n=220, TTS) | unknown_speech 0.5% (n=201), non_speech 0.8% (n=118), silence 0.0% (n=120) | yes |
| whisper-large-v3-turbo/text · phrases20 · balanced | 0.4 | 0.333 | 0.031 | 99.1% (n=220, TTS) | unknown_speech 2.5% (n=201), non_speech 1.7% (n=118), silence 0.0% (n=120) | yes |
| whisper-large-v3-turbo/text · torgo_words | 0.2 | -inf | 1.000 | 6.9% (n=1512) | unknown_speech 0.0% (n=93), non_speech 0.0% (n=118), silence 0.0% (n=120) | yes |
| whisper-large-v3-turbo/text · torgo_words · balanced | 0.2 | -inf | 0.500 | 68.3% (n=1512) | unknown_speech 4.3% (n=93), non_speech 0.0% (n=118), silence 0.0% (n=120) | yes |
| whisper-large-v2/text · phrases20 | 0.2 | 0.500 | 0.000 | 97.3% (n=220, TTS) | unknown_speech 0.0% (n=201), non_speech 0.8% (n=118), silence 0.0% (n=120) | yes |
| whisper-large-v2/text · phrases20 · balanced | 0.2 | 0.286 | 0.036 | 97.7% (n=220, TTS) | unknown_speech 4.0% (n=201), non_speech 3.4% (n=118), silence 0.0% (n=120) | yes |
| whisper-large-v2/text · torgo_words | 0.2 | -inf | 0.667 | 38.2% (n=1512) | unknown_speech 0.0% (n=93), non_speech 0.0% (n=118), silence 0.0% (n=120) | yes |
| whisper-large-v2/text · torgo_words · balanced | 0.2 | -inf | 0.500 | 58.7% (n=1512) | unknown_speech 2.2% (n=93), non_speech 0.0% (n=118), silence 0.0% (n=120) | yes |
| whisper-small/text · phrases20 | 0.6 | -inf | 0.208 | 97.7% (n=220, TTS) | unknown_speech 0.5% (n=201), non_speech 0.8% (n=118), silence 0.0% (n=120) | yes |
| whisper-small/text · phrases20 · balanced | 0.2 | 0.333 | 0.033 | 98.6% (n=220, TTS) | unknown_speech 3.5% (n=201), non_speech 3.4% (n=118), silence 0.0% (n=120) | yes |
| whisper-small/text · torgo_words | 0.2 | -inf | 0.667 | 45.5% (n=1512) | unknown_speech 0.0% (n=93), non_speech 0.0% (n=118), silence 0.0% (n=120) | yes |
| whisper-small/text · torgo_words · balanced | 0.2 | 0.500 | 0.333 | 68.1% (n=1512) | unknown_speech 4.3% (n=93), non_speech 0.0% (n=118), silence 0.0% (n=120) | yes |

## Which model wins and why (measured numbers only)

Best on calib: **parakeet-ctc-0.6b/ctc_sum**. Without real human recordings of the 20 phrases there is no headline result yet; the comparison below this line is about plumbing, not accuracy.

## Context from literature (not our measurements)

Protocols differ from ours (open transcription vs a 20-phrase closed set with abstention), so their numbers are not comparable to the tables above and are not copied here.

- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — English WER on public test sets (incl. LibriSpeech test-clean) under the leaderboard's own text normaliser; open transcription, no abstention.
- [SAP 2025 challenge baselines](https://arxiv.org/html/2507.22047v1) — Whisper sizes on Speech Accessibility Project dysarthric speech; open-vocabulary WER on the challenge splits.
- [Takahashi et al., Interspeech 2025](https://www.isca-archive.org/interspeech_2025/takahashi25_interspeech.pdf) — Parakeet-TDT vs Whisper fine-tuned on SAP dysarthric speech; WER on SAP evaluation data.
- [Whisper non-speech hallucination + VAD](https://arxiv.org/pdf/2501.11378) — Hallucination rates of Whisper on non-speech audio and the effect of VAD gating.
- [Calm-Whisper, Interspeech 2025](https://www.isca-archive.org/interspeech_2025/wang25b_interspeech.pdf) — Non-speech hallucination rates and a mitigation by modifying attention heads.
- [Cross-model hallucination, Findings ACL 2025](https://aclanthology.org/2025.findings-acl.1190.pdf) — Hallucination behaviour across Whisper and transducer (RNN-T/TDT) models.
- [de Groot et al., Interspeech 2025](https://www.isca-archive.org/interspeech_2025/degroot25_interspeech.pdf) — Speech enhancement degrading dysarthric ASR on TORGO/UASpeech.
- [Mawalim et al., Interspeech 2024](https://www.isca-archive.org/interspeech_2024/mawalim24_interspeech.pdf) — DeepFilterNet3 enhancement evaluated on downstream ASR.
- [Latent Phrase Matching (CMU 2023)](https://www.cs.cmu.edu/~jbigham/pubs/pdfs/2023/lpm-dysarthric.pdf) — Few-shot, per-user template matching of dysarthric phrases (closed set, enrollment).
- [Personalized vs normative dysarthric ASR, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1701.pdf) — Speaker-dependent vs speaker-independent models on TORGO.
- [Cross-etiology transfer SAP to TORGO](https://arxiv.org/pdf/2501.14994v1) — Training on SAP and testing on TORGO speakers (cross-etiology generalisation).
