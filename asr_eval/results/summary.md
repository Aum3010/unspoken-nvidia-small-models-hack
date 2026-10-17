# ASR evaluation summary

Generated 2026-10-17 21:14 · config `eval.yaml` · **test split only** (thresholds tuned on calib).

> ⚠️ No threshold met the calib false-accept target for: parakeet-ctc-0.6b/text|torgo_words, nemotron-streaming-0.6b/text|torgo_words, nemotron-streaming-0.6b-160ms/text|torgo_words, nemotron-streaming-0.6b-560ms/text|torgo_words (strictest point used; see Thresholds).

## Headline: 20 phrases, real recordings, not degraded

| Model / method | Top-1 rank | Top-3 rank | Accepted correct | Wrong accept | Abstain | FA hard-neg | FA unknown speech | FA non-speech | FA silence |
|---|---|---|---|---|---|---|---|---|---|
| parakeet-ctc-0.6b/text | 112/156 (71.8%) | 126/156 (80.8%) | 81/156 (51.9%) | 1/156 (0.6%) | 74/156 (47.4%) | 1/31 (3.2%) [0.6–16.2] | 0/430 (0.0%) [0.0–0.9] ≤0.7% | 1/191 (0.5%) [0.1–2.9] | 0/184 (0.0%) [0.0–2.0] ≤1.6% |
| parakeet-ctc-0.6b/ctc_sum | 106/156 (67.9%) | 114/156 (73.1%) | 94/156 (60.3%) | 9/156 (5.8%) | 53/156 (34.0%) | 4/31 (12.9%) [5.1–28.9] | 1/430 (0.2%) [0.0–1.3] | 2/191 (1.0%) [0.3–3.7] | 0/184 (0.0%) [0.0–2.0] ≤1.6% |
| parakeet-ctc-0.6b/ctc_norm | 121/156 (77.6%) | 131/156 (84.0%) | 100/156 (64.1%) | 6/156 (3.8%) | 50/156 (32.1%) | 3/31 (9.7%) [3.3–24.9] | 1/430 (0.2%) [0.0–1.3] | 3/191 (1.6%) [0.5–4.5] | 0/184 (0.0%) [0.0–2.0] ≤1.6% |
| parakeet-ctc-0.6b/ctc_llr | 112/156 (71.8%) | 125/156 (80.1%) | 98/156 (62.8%) | 9/156 (5.8%) | 49/156 (31.4%) | 3/31 (9.7%) [3.3–24.9] | 4/430 (0.9%) [0.4–2.4] | 2/191 (1.0%) [0.3–3.7] | 0/184 (0.0%) [0.0–2.0] ≤1.6% |
| parakeet-tdt-0.6b-v2/text | 114/156 (73.1%) | 126/156 (80.8%) | 89/156 (57.1%) | 2/156 (1.3%) | 65/156 (41.7%) | 0/31 (0.0%) [0.0–11.0] ≤9.7% | 0/430 (0.0%) [0.0–0.9] ≤0.7% | 0/191 (0.0%) [0.0–2.0] ≤1.6% | 0/184 (0.0%) [0.0–2.0] ≤1.6% |
| nemotron-streaming-0.6b/text | 110/156 (70.5%) | 123/156 (78.8%) | 87/156 (55.8%) | 1/156 (0.6%) | 68/156 (43.6%) | 1/31 (3.2%) [0.6–16.2] | 0/430 (0.0%) [0.0–0.9] ≤0.7% | 0/191 (0.0%) [0.0–2.0] ≤1.6% | 0/184 (0.0%) [0.0–2.0] ≤1.6% |
| nemotron-streaming-0.6b-160ms/text | 83/156 (53.2%) | 99/156 (63.5%) | 60/156 (38.5%) | 0/156 (0.0%) | 96/156 (61.5%) | 2/31 (6.5%) [1.8–20.7] | 0/430 (0.0%) [0.0–0.9] ≤0.7% | 1/191 (0.5%) [0.1–2.9] | 0/184 (0.0%) [0.0–2.0] ≤1.6% |
| nemotron-streaming-0.6b-560ms/text | 108/156 (69.2%) | 118/156 (75.6%) | 84/156 (53.8%) | 1/156 (0.6%) | 71/156 (45.5%) | 3/31 (9.7%) [3.3–24.9] | 0/430 (0.0%) [0.0–0.9] ≤0.7% | 0/191 (0.0%) [0.0–2.0] ≤1.6% | 0/184 (0.0%) [0.0–2.0] ≤1.6% |
| whisper-large-v3-turbo/text | 116/156 (74.4%) | 129/156 (82.7%) | 93/156 (59.6%) | 2/156 (1.3%) | 61/156 (39.1%) | 2/31 (6.5%) [1.8–20.7] | 0/430 (0.0%) [0.0–0.9] ≤0.7% | 0/191 (0.0%) [0.0–2.0] ≤1.6% | 0/184 (0.0%) [0.0–2.0] ≤1.6% |
| whisper-small/text | 116/156 (74.4%) | 128/156 (82.1%) | 94/156 (60.3%) | 2/156 (1.3%) | 60/156 (38.5%) | 1/31 (3.2%) [0.6–16.2] | 0/430 (0.0%) [0.0–0.9] ≤0.7% | 0/191 (0.0%) [0.0–2.0] ≤1.6% | 0/184 (0.0%) [0.0–2.0] ≤1.6% |

FA = false accept (a phrase proposed on a negative clip): k/n (rate) [95% Wilson CI]; ≤x% = rule-of-three upper bound when k = 0. Ranking columns are before abstention; 'Accepted correct' is correct candidates over all positives.

## Per condition (nemotron-streaming-0.6b/text)

### Real recordings

| Condition | n | Top-1 rank | Accepted correct | Wrong accept | Abstain |
|---|---:|---|---|---|---|
| muffled | 15 | 3/15 (20.0%) | 2/15 (13.3%) | 0/15 (0.0%) | 13/15 (86.7%) |
| noisy | 40 | 18/40 (45.0%) | 6/40 (15.0%) | 0/40 (0.0%) | 34/40 (85.0%) |
| normal | 80 | 76/80 (95.0%) | 68/80 (85.0%) | 0/80 (0.0%) | 12/80 (15.0%) |
| whisper | 21 | 13/21 (61.9%) | 11/21 (52.4%) | 1/21 (4.8%) | 9/21 (42.9%) |

### Degraded copies of real recordings (synthetic degradation)

| Condition | n | Top-1 rank | Accepted correct | Wrong accept | Abstain |
|---|---:|---|---|---|---|
| bandpass_300-3400 | 80 | 76/80 (95.0%) | 65/80 (81.2%) | 0/80 (0.0%) | 15/80 (18.8%) |
| gain_-12dB | 80 | 76/80 (95.0%) | 68/80 (85.0%) | 0/80 (0.0%) | 12/80 (15.0%) |
| gain_-24dB | 80 | 76/80 (95.0%) | 67/80 (83.8%) | 0/80 (0.0%) | 13/80 (16.2%) |
| gain_-36dB | 80 | 76/80 (95.0%) | 62/80 (77.5%) | 0/80 (0.0%) | 18/80 (22.5%) |
| lowpass_1500 | 80 | 69/80 (86.2%) | 52/80 (65.0%) | 0/80 (0.0%) | 28/80 (35.0%) |
| noise_musan_snr0 | 80 | 70/80 (87.5%) | 52/80 (65.0%) | 0/80 (0.0%) | 28/80 (35.0%) |
| noise_musan_snr10 | 80 | 75/80 (93.8%) | 62/80 (77.5%) | 0/80 (0.0%) | 18/80 (22.5%) |
| noise_musan_snr20 | 80 | 76/80 (95.0%) | 66/80 (82.5%) | 0/80 (0.0%) | 14/80 (17.5%) |
| noise_musan_snr5 | 80 | 70/80 (87.5%) | 58/80 (72.5%) | 0/80 (0.0%) | 22/80 (27.5%) |
| reverb_rt60_0.3s | 80 | 73/80 (91.2%) | 53/80 (66.2%) | 0/80 (0.0%) | 27/80 (33.8%) |
| reverb_rt60_0.6s | 80 | 67/80 (83.8%) | 49/80 (61.3%) | 1/80 (1.2%) | 30/80 (37.5%) |
| tempo_0.7x | 80 | 55/80 (68.8%) | 40/80 (50.0%) | 2/80 (2.5%) | 38/80 (47.5%) |
| tempo_0.85x | 80 | 56/80 (70.0%) | 38/80 (47.5%) | 0/80 (0.0%) | 42/80 (52.5%) |

### TTS — sanity only

| Condition | n | Top-1 rank | Accepted correct | Wrong accept | Abstain |
|---|---:|---|---|---|---|
| tts_clean | 340 | 336/340 (98.8%) | 335/340 (98.5%) | 0/340 (0.0%) | 5/340 (1.5%) |

## TORGO surrogate closed-vocabulary task: dysarthric vs control

Vocabulary (config): yes, no, up, down, left, right, go, one, two, three

| Model / method | Group | Top-1 rank | Accepted correct | Wrong accept | FA other words | WER / CER (all TORGO clips) |
|---|---|---|---|---|---|---|
| parakeet-ctc-0.6b/text | control | 103/118 (87.3%) | 24/118 (20.3%) | 0/118 (0.0%) | 1/141 (0.7%) [0.1–3.9] | 9.4% / 5.3% (n=461) |
| parakeet-ctc-0.6b/text | dysarthric | 73/90 (81.1%) | 12/90 (13.3%) | 1/90 (1.1%) | 5/69 (7.2%) [3.1–15.9] | 29.8% / 19.0% (n=255) |
| parakeet-ctc-0.6b/ctc_sum | control | 103/118 (87.3%) | 74/118 (62.7%) | 0/118 (0.0%) | 2/141 (1.4%) [0.4–5.0] | 9.4% / 5.3% (n=461) |
| parakeet-ctc-0.6b/ctc_sum | dysarthric | 77/90 (85.6%) | 42/90 (46.7%) | 0/90 (0.0%) | 1/69 (1.4%) [0.3–7.8] | 29.8% / 19.0% (n=255) |
| parakeet-ctc-0.6b/ctc_norm | control | 103/118 (87.3%) | 74/118 (62.7%) | 0/118 (0.0%) | 2/141 (1.4%) [0.4–5.0] | 9.4% / 5.3% (n=461) |
| parakeet-ctc-0.6b/ctc_norm | dysarthric | 77/90 (85.6%) | 42/90 (46.7%) | 0/90 (0.0%) | 1/69 (1.4%) [0.3–7.8] | 29.8% / 19.0% (n=255) |
| parakeet-ctc-0.6b/ctc_llr | control | 103/118 (87.3%) | 62/118 (52.5%) | 0/118 (0.0%) | 3/141 (2.1%) [0.7–6.1] | 9.4% / 5.3% (n=461) |
| parakeet-ctc-0.6b/ctc_llr | dysarthric | 77/90 (85.6%) | 32/90 (35.6%) | 0/90 (0.0%) | 0/69 (0.0%) [0.0–5.3] ≤4.3% | 29.8% / 19.0% (n=255) |
| parakeet-tdt-0.6b-v2/text | control | 114/118 (96.6%) | 22/118 (18.6%) | 0/118 (0.0%) | 0/141 (0.0%) [0.0–2.7] ≤2.1% | 5.7% / 2.6% (n=461) |
| parakeet-tdt-0.6b-v2/text | dysarthric | 74/90 (82.2%) | 6/90 (6.7%) | 0/90 (0.0%) | 0/69 (0.0%) [0.0–5.3] ≤4.3% | 26.4% / 18.1% (n=255) |
| nemotron-streaming-0.6b/text | control | 118/118 (100.0%) | 63/118 (53.4%) | 0/118 (0.0%) | 1/141 (0.7%) [0.1–3.9] | 5.3% / 2.5% (n=461) |
| nemotron-streaming-0.6b/text | dysarthric | 84/90 (93.3%) | 36/90 (40.0%) | 1/90 (1.1%) | 0/69 (0.0%) [0.0–5.3] ≤4.3% | 27.9% / 19.8% (n=255) |
| nemotron-streaming-0.6b-160ms/text | control | 112/118 (94.9%) | 87/118 (73.7%) | 0/118 (0.0%) | 2/141 (1.4%) [0.4–5.0] | 6.8% / 4.4% (n=461) |
| nemotron-streaming-0.6b-160ms/text | dysarthric | 76/90 (84.4%) | 52/90 (57.8%) | 0/90 (0.0%) | 1/69 (1.4%) [0.3–7.8] | 29.5% / 22.3% (n=255) |
| nemotron-streaming-0.6b-560ms/text | control | 118/118 (100.0%) | 62/118 (52.5%) | 0/118 (0.0%) | 1/141 (0.7%) [0.1–3.9] | 5.8% / 3.0% (n=461) |
| nemotron-streaming-0.6b-560ms/text | dysarthric | 84/90 (93.3%) | 36/90 (40.0%) | 1/90 (1.1%) | 0/69 (0.0%) [0.0–5.3] ≤4.3% | 29.0% / 20.6% (n=255) |
| whisper-large-v3-turbo/text | control | 108/118 (91.5%) | 14/118 (11.9%) | 0/118 (0.0%) | 0/141 (0.0%) [0.0–2.7] ≤2.1% | 5.0% / 2.7% (n=461) |
| whisper-large-v3-turbo/text | dysarthric | 82/90 (91.1%) | 5/90 (5.6%) | 0/90 (0.0%) | 0/69 (0.0%) [0.0–5.3] ≤4.3% | 24.0% / 14.5% (n=255) |
| whisper-small/text | control | 112/118 (94.9%) | 87/118 (73.7%) | 0/118 (0.0%) | 1/141 (0.7%) [0.1–3.9] | 6.9% / 3.4% (n=461) |
| whisper-small/text | dysarthric | 79/90 (87.8%) | 46/90 (51.1%) | 0/90 (0.0%) | 0/69 (0.0%) [0.0–5.3] ≤4.3% | 44.7% / 48.1% (n=255) |

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
| parakeet-ctc-0.6b/text | librispeech | unknown_speech | 0/127 (0.0%) [0.0–2.9] ≤2.4% |
| parakeet-ctc-0.6b/text | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| parakeet-ctc-0.6b/text | own | hard_negative | 1/31 (3.2%) [0.6–16.2] |
| parakeet-ctc-0.6b/text | own | non_speech | 1/9 (11.1%) [2.0–43.5] |
| parakeet-ctc-0.6b/text | own | silence | 0/4 (0.0%) [0.0–49.0] ≤75.0% |
| parakeet-ctc-0.6b/text | own | unknown_speech | 0/5 (0.0%) [0.0–43.4] ≤60.0% |
| parakeet-ctc-0.6b/text | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| parakeet-ctc-0.6b/text | torgo | unknown_speech | 0/298 (0.0%) [0.0–1.3] ≤1.0% |
| parakeet-ctc-0.6b/text | tts (sanity) | hard_negative | 0/119 (0.0%) [0.0–3.1] ≤2.5% |
| parakeet-ctc-0.6b/ctc_sum | esc50 | non_speech | 1/63 (1.6%) [0.3–8.5] |
| parakeet-ctc-0.6b/ctc_sum | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| parakeet-ctc-0.6b/ctc_sum | librispeech | unknown_speech | 0/127 (0.0%) [0.0–2.9] ≤2.4% |
| parakeet-ctc-0.6b/ctc_sum | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| parakeet-ctc-0.6b/ctc_sum | own | hard_negative | 4/31 (12.9%) [5.1–28.9] |
| parakeet-ctc-0.6b/ctc_sum | own | non_speech | 1/9 (11.1%) [2.0–43.5] |
| parakeet-ctc-0.6b/ctc_sum | own | silence | 0/4 (0.0%) [0.0–49.0] ≤75.0% |
| parakeet-ctc-0.6b/ctc_sum | own | unknown_speech | 1/5 (20.0%) [3.6–62.4] |
| parakeet-ctc-0.6b/ctc_sum | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| parakeet-ctc-0.6b/ctc_sum | torgo | unknown_speech | 0/298 (0.0%) [0.0–1.3] ≤1.0% |
| parakeet-ctc-0.6b/ctc_sum | tts (sanity) | hard_negative | 0/119 (0.0%) [0.0–3.1] ≤2.5% |
| parakeet-ctc-0.6b/ctc_norm | esc50 | non_speech | 1/63 (1.6%) [0.3–8.5] |
| parakeet-ctc-0.6b/ctc_norm | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| parakeet-ctc-0.6b/ctc_norm | librispeech | unknown_speech | 0/127 (0.0%) [0.0–2.9] ≤2.4% |
| parakeet-ctc-0.6b/ctc_norm | musan | non_speech | 1/90 (1.1%) [0.2–6.0] |
| parakeet-ctc-0.6b/ctc_norm | own | hard_negative | 3/31 (9.7%) [3.3–24.9] |
| parakeet-ctc-0.6b/ctc_norm | own | non_speech | 1/9 (11.1%) [2.0–43.5] |
| parakeet-ctc-0.6b/ctc_norm | own | silence | 0/4 (0.0%) [0.0–49.0] ≤75.0% |
| parakeet-ctc-0.6b/ctc_norm | own | unknown_speech | 1/5 (20.0%) [3.6–62.4] |
| parakeet-ctc-0.6b/ctc_norm | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| parakeet-ctc-0.6b/ctc_norm | torgo | unknown_speech | 0/298 (0.0%) [0.0–1.3] ≤1.0% |
| parakeet-ctc-0.6b/ctc_norm | tts (sanity) | hard_negative | 0/119 (0.0%) [0.0–3.1] ≤2.5% |
| parakeet-ctc-0.6b/ctc_llr | esc50 | non_speech | 1/63 (1.6%) [0.3–8.5] |
| parakeet-ctc-0.6b/ctc_llr | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| parakeet-ctc-0.6b/ctc_llr | librispeech | unknown_speech | 0/127 (0.0%) [0.0–2.9] ≤2.4% |
| parakeet-ctc-0.6b/ctc_llr | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| parakeet-ctc-0.6b/ctc_llr | own | hard_negative | 3/31 (9.7%) [3.3–24.9] |
| parakeet-ctc-0.6b/ctc_llr | own | non_speech | 1/9 (11.1%) [2.0–43.5] |
| parakeet-ctc-0.6b/ctc_llr | own | silence | 0/4 (0.0%) [0.0–49.0] ≤75.0% |
| parakeet-ctc-0.6b/ctc_llr | own | unknown_speech | 2/5 (40.0%) [11.8–76.9] |
| parakeet-ctc-0.6b/ctc_llr | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| parakeet-ctc-0.6b/ctc_llr | torgo | unknown_speech | 2/298 (0.7%) [0.2–2.4] |
| parakeet-ctc-0.6b/ctc_llr | tts (sanity) | hard_negative | 0/119 (0.0%) [0.0–3.1] ≤2.5% |
| parakeet-tdt-0.6b-v2/text | esc50 | non_speech | 0/63 (0.0%) [0.0–5.7] ≤4.8% |
| parakeet-tdt-0.6b-v2/text | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| parakeet-tdt-0.6b-v2/text | librispeech | unknown_speech | 0/127 (0.0%) [0.0–2.9] ≤2.4% |
| parakeet-tdt-0.6b-v2/text | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| parakeet-tdt-0.6b-v2/text | own | hard_negative | 0/31 (0.0%) [0.0–11.0] ≤9.7% |
| parakeet-tdt-0.6b-v2/text | own | non_speech | 0/9 (0.0%) [0.0–29.9] ≤33.3% |
| parakeet-tdt-0.6b-v2/text | own | silence | 0/4 (0.0%) [0.0–49.0] ≤75.0% |
| parakeet-tdt-0.6b-v2/text | own | unknown_speech | 0/5 (0.0%) [0.0–43.4] ≤60.0% |
| parakeet-tdt-0.6b-v2/text | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| parakeet-tdt-0.6b-v2/text | torgo | unknown_speech | 0/298 (0.0%) [0.0–1.3] ≤1.0% |
| parakeet-tdt-0.6b-v2/text | tts (sanity) | hard_negative | 0/119 (0.0%) [0.0–3.1] ≤2.5% |
| nemotron-streaming-0.6b/text | esc50 | non_speech | 0/63 (0.0%) [0.0–5.7] ≤4.8% |
| nemotron-streaming-0.6b/text | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| nemotron-streaming-0.6b/text | librispeech | unknown_speech | 0/127 (0.0%) [0.0–2.9] ≤2.4% |
| nemotron-streaming-0.6b/text | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| nemotron-streaming-0.6b/text | own | hard_negative | 1/31 (3.2%) [0.6–16.2] |
| nemotron-streaming-0.6b/text | own | non_speech | 0/9 (0.0%) [0.0–29.9] ≤33.3% |
| nemotron-streaming-0.6b/text | own | silence | 0/4 (0.0%) [0.0–49.0] ≤75.0% |
| nemotron-streaming-0.6b/text | own | unknown_speech | 0/5 (0.0%) [0.0–43.4] ≤60.0% |
| nemotron-streaming-0.6b/text | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| nemotron-streaming-0.6b/text | torgo | unknown_speech | 0/298 (0.0%) [0.0–1.3] ≤1.0% |
| nemotron-streaming-0.6b/text | tts (sanity) | hard_negative | 0/119 (0.0%) [0.0–3.1] ≤2.5% |
| nemotron-streaming-0.6b-160ms/text | esc50 | non_speech | 0/63 (0.0%) [0.0–5.7] ≤4.8% |
| nemotron-streaming-0.6b-160ms/text | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| nemotron-streaming-0.6b-160ms/text | librispeech | unknown_speech | 0/127 (0.0%) [0.0–2.9] ≤2.4% |
| nemotron-streaming-0.6b-160ms/text | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| nemotron-streaming-0.6b-160ms/text | own | hard_negative | 2/31 (6.5%) [1.8–20.7] |
| nemotron-streaming-0.6b-160ms/text | own | non_speech | 1/9 (11.1%) [2.0–43.5] |
| nemotron-streaming-0.6b-160ms/text | own | silence | 0/4 (0.0%) [0.0–49.0] ≤75.0% |
| nemotron-streaming-0.6b-160ms/text | own | unknown_speech | 0/5 (0.0%) [0.0–43.4] ≤60.0% |
| nemotron-streaming-0.6b-160ms/text | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| nemotron-streaming-0.6b-160ms/text | torgo | unknown_speech | 0/298 (0.0%) [0.0–1.3] ≤1.0% |
| nemotron-streaming-0.6b-160ms/text | tts (sanity) | hard_negative | 0/119 (0.0%) [0.0–3.1] ≤2.5% |
| nemotron-streaming-0.6b-560ms/text | esc50 | non_speech | 0/63 (0.0%) [0.0–5.7] ≤4.8% |
| nemotron-streaming-0.6b-560ms/text | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| nemotron-streaming-0.6b-560ms/text | librispeech | unknown_speech | 0/127 (0.0%) [0.0–2.9] ≤2.4% |
| nemotron-streaming-0.6b-560ms/text | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| nemotron-streaming-0.6b-560ms/text | own | hard_negative | 3/31 (9.7%) [3.3–24.9] |
| nemotron-streaming-0.6b-560ms/text | own | non_speech | 0/9 (0.0%) [0.0–29.9] ≤33.3% |
| nemotron-streaming-0.6b-560ms/text | own | silence | 0/4 (0.0%) [0.0–49.0] ≤75.0% |
| nemotron-streaming-0.6b-560ms/text | own | unknown_speech | 0/5 (0.0%) [0.0–43.4] ≤60.0% |
| nemotron-streaming-0.6b-560ms/text | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| nemotron-streaming-0.6b-560ms/text | torgo | unknown_speech | 0/298 (0.0%) [0.0–1.3] ≤1.0% |
| nemotron-streaming-0.6b-560ms/text | tts (sanity) | hard_negative | 0/119 (0.0%) [0.0–3.1] ≤2.5% |
| whisper-large-v3-turbo/text | esc50 | non_speech | 0/63 (0.0%) [0.0–5.7] ≤4.8% |
| whisper-large-v3-turbo/text | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| whisper-large-v3-turbo/text | librispeech | unknown_speech | 0/127 (0.0%) [0.0–2.9] ≤2.4% |
| whisper-large-v3-turbo/text | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| whisper-large-v3-turbo/text | own | hard_negative | 2/31 (6.5%) [1.8–20.7] |
| whisper-large-v3-turbo/text | own | non_speech | 0/9 (0.0%) [0.0–29.9] ≤33.3% |
| whisper-large-v3-turbo/text | own | silence | 0/4 (0.0%) [0.0–49.0] ≤75.0% |
| whisper-large-v3-turbo/text | own | unknown_speech | 0/5 (0.0%) [0.0–43.4] ≤60.0% |
| whisper-large-v3-turbo/text | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| whisper-large-v3-turbo/text | torgo | unknown_speech | 0/298 (0.0%) [0.0–1.3] ≤1.0% |
| whisper-large-v3-turbo/text | tts (sanity) | hard_negative | 0/119 (0.0%) [0.0–3.1] ≤2.5% |
| whisper-small/text | esc50 | non_speech | 0/63 (0.0%) [0.0–5.7] ≤4.8% |
| whisper-small/text | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| whisper-small/text | librispeech | unknown_speech | 0/127 (0.0%) [0.0–2.9] ≤2.4% |
| whisper-small/text | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| whisper-small/text | own | hard_negative | 1/31 (3.2%) [0.6–16.2] |
| whisper-small/text | own | non_speech | 0/9 (0.0%) [0.0–29.9] ≤33.3% |
| whisper-small/text | own | silence | 0/4 (0.0%) [0.0–49.0] ≤75.0% |
| whisper-small/text | own | unknown_speech | 0/5 (0.0%) [0.0–43.4] ≤60.0% |
| whisper-small/text | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| whisper-small/text | torgo | unknown_speech | 0/298 (0.0%) [0.0–1.3] ≤1.0% |
| whisper-small/text | tts (sanity) | hard_negative | 0/119 (0.0%) [0.0–3.1] ≤2.5% |

## WER / CER of the free transcript

| Model | Source | WER | CER | n |
|---|---|---|---|---|
| parakeet-ctc-0.6b | librispeech | 2.5% | 0.7% | 127 |
| parakeet-ctc-0.6b | own | 43.6% | 35.3% | 187 |
| parakeet-ctc-0.6b | torgo | 16.3% | 9.8% | 716 |
| parakeet-ctc-0.6b | tts | 1.7% | 1.0% | 459 |
| parakeet-tdt-0.6b-v2 | librispeech | 2.0% | 0.5% | 127 |
| parakeet-tdt-0.6b-v2 | own | 39.6% | 31.8% | 187 |
| parakeet-tdt-0.6b-v2 | torgo | 12.7% | 7.7% | 716 |
| parakeet-tdt-0.6b-v2 | tts | 2.1% | 1.6% | 459 |
| nemotron-streaming-0.6b | librispeech | 2.3% | 0.6% | 127 |
| nemotron-streaming-0.6b | own | 39.1% | 32.8% | 187 |
| nemotron-streaming-0.6b | torgo | 13.0% | 8.1% | 716 |
| nemotron-streaming-0.6b | tts | 2.1% | 0.9% | 459 |
| nemotron-streaming-0.6b-160ms | librispeech | 2.5% | 0.7% | 127 |
| nemotron-streaming-0.6b-160ms | own | 54.6% | 49.4% | 187 |
| nemotron-streaming-0.6b-160ms | torgo | 14.5% | 10.3% | 716 |
| nemotron-streaming-0.6b-160ms | tts | 1.9% | 1.6% | 459 |
| nemotron-streaming-0.6b-560ms | librispeech | 2.5% | 0.7% | 127 |
| nemotron-streaming-0.6b-560ms | own | 40.1% | 34.5% | 187 |
| nemotron-streaming-0.6b-560ms | torgo | 13.7% | 8.7% | 716 |
| nemotron-streaming-0.6b-560ms | tts | 2.0% | 1.3% | 459 |
| whisper-large-v3-turbo | librispeech | 2.3% | 1.0% | 127 |
| whisper-large-v3-turbo | own | 36.3% | 29.4% | 187 |
| whisper-large-v3-turbo | torgo | 11.4% | 6.5% | 716 |
| whisper-large-v3-turbo | tts | 1.9% | 0.9% | 459 |
| whisper-small | librispeech | 4.0% | 1.7% | 127 |
| whisper-small | own | 49.0% | 43.3% | 187 |
| whisper-small | torgo | 19.7% | 18.0% | 716 |
| whisper-small | tts | 2.2% | 1.1% | 459 |

Text normalised (lowercase, contractions expanded, punctuation removed) on both sides; empty hypotheses count as deletions.

## Latency (batch 1, warm) and resources

| Model | Params | Load | Peak torch alloc | GPU used (device) | VAD p50 | ASR p50 / p95 / max | Scoring p50 / p95 | Total p50 / p95 / max | RTFx (median) |
|---|---:|---:|---:|---:|---:|---|---|---|---:|
| parakeet-ctc-0.6b | 608.8M | 2.3 s | 1.284 GB | 8.06 GB | 12.8 ms | 12 / 15 / 190 ms | 0 / 1 ms | 26 / 44 / 206 ms | 88.4 |
| parakeet-tdt-0.6b-v2 | 617.8M | 7.2 s | 4.988 GB | 9.67 GB | 12.8 ms | 16 / 19 / 43 ms | — | 29 / 46 / 184 ms | 78.7 |
| nemotron-streaming-0.6b | 618.1M | 7.5 s | 4.99 GB | 9.76 GB | 12.8 ms | 17 / 20 / 47 ms | — | 29 / 47 / 195 ms | 77.4 |
| nemotron-streaming-0.6b-160ms | 618.1M | 7.7 s | 4.99 GB | 9.76 GB | 12.8 ms | 17 / 20 / 45 ms | — | 29 / 47 / 192 ms | 77.3 |
| nemotron-streaming-0.6b-560ms | 618.1M | 7.5 s | 4.99 GB | 9.76 GB | 12.8 ms | 17 / 20 / 41 ms | — | 29 / 47 / 189 ms | 77.5 |
| whisper-large-v3-turbo | 808.9M | 0.9 s | 0.0 GB | 8.74 GB | 12.8 ms | 42 / 52 / 372 ms | — | 55 / 78 / 382 ms | 40.9 |
| whisper-small | 241.7M | 0.5 s | 0.0 GB | 7.23 GB | 12.8 ms | 28 / 47 / 622 ms | — | 42 / 73 / 634 ms | 53.0 |

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
| whisper-small | Systran/faster-whisper-small | `536b0662742c` | NVIDIA RTX PRO 6000 Blackwell Server Edition |

## Thresholds (frozen from calib)

Rows ending in a name (e.g. · balanced) are extra operating points for the live UI; every number above uses the headline point (false accepts ≤ 1% per negative type on calib).

| Model / method · task | VAD thr | abs | margin | Calib accepted correct | Calib FA by type | Feasible |
|---|---|---|---|---|---|---|
| parakeet-ctc-0.6b/text · phrases20 | 0.2 | 0.722 | 0.000 | 77.1% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| parakeet-ctc-0.6b/text · phrases20 · balanced | 0.2 | 0.625 | 0.000 | 80.9% (n=586) | hard_negative 0.0% (n=14), unknown_speech 1.1% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| parakeet-ctc-0.6b/text · torgo_words | 0.2 | -inf | 1.000 | 12.4% (n=1512) | unknown_speech 8.9% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | NO |
| parakeet-ctc-0.6b/text · torgo_words · balanced | 0.2 | -inf | 1.000 | 12.4% (n=1512) | unknown_speech 8.9% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | NO |
| parakeet-ctc-0.6b/ctc_sum · phrases20 | 0.6 | -24.206 | 2.503 | 85.0% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.6% (n=177), non_speech 0.8% (n=122), silence 0.0% (n=122) | yes |
| parakeet-ctc-0.6b/ctc_sum · phrases20 · balanced | 0.5 | -24.206 | 0.000 | 86.7% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.6% (n=177), non_speech 4.1% (n=122), silence 0.0% (n=122) | yes |
| parakeet-ctc-0.6b/ctc_sum · torgo_words | 0.2 | -0.026 | 0.000 | 35.4% (n=1512) | unknown_speech 0.0% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| parakeet-ctc-0.6b/ctc_sum · torgo_words · balanced | 0.2 | -0.337 | 6.500 | 40.5% (n=1512) | unknown_speech 4.4% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| parakeet-ctc-0.6b/ctc_norm · phrases20 | 0.6 | -5.196 | 0.000 | 89.6% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.6% (n=177), non_speech 0.8% (n=122), silence 0.0% (n=122) | yes |
| parakeet-ctc-0.6b/ctc_norm · phrases20 · balanced | 0.3 | -6.225 | 0.000 | 90.8% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.6% (n=177), non_speech 4.1% (n=122), silence 0.0% (n=122) | yes |
| parakeet-ctc-0.6b/ctc_norm · torgo_words | 0.2 | -0.026 | 0.000 | 35.4% (n=1512) | unknown_speech 0.0% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| parakeet-ctc-0.6b/ctc_norm · torgo_words · balanced | 0.2 | -0.337 | 6.500 | 40.5% (n=1512) | unknown_speech 4.4% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| parakeet-ctc-0.6b/ctc_llr · phrases20 | 0.3 | -3.442 | 0.550 | 86.7% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.8% (n=122), silence 0.0% (n=122) | yes |
| parakeet-ctc-0.6b/ctc_llr · phrases20 · balanced | 0.2 | -3.442 | 0.048 | 87.7% (n=586) | hard_negative 0.0% (n=14), unknown_speech 4.0% (n=177), non_speech 2.5% (n=122), silence 0.0% (n=122) | yes |
| parakeet-ctc-0.6b/ctc_llr · torgo_words | 0.2 | -inf | 11.500 | 18.7% (n=1512) | unknown_speech 0.0% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| parakeet-ctc-0.6b/ctc_llr · torgo_words · balanced | 0.2 | -inf | 8.001 | 35.1% (n=1512) | unknown_speech 2.2% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| parakeet-tdt-0.6b-v2/text · phrases20 | 0.2 | 0.667 | 0.000 | 87.2% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| parakeet-tdt-0.6b-v2/text · phrases20 · balanced | 0.2 | 0.667 | 0.000 | 87.2% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| parakeet-tdt-0.6b-v2/text · torgo_words | 0.2 | -inf | 1.000 | 8.2% (n=1512) | unknown_speech 0.0% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| parakeet-tdt-0.6b-v2/text · torgo_words · balanced | 0.2 | 0.500 | 0.333 | 60.4% (n=1512) | unknown_speech 3.3% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b/text · phrases20 | 0.2 | 0.625 | 0.000 | 92.8% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b/text · phrases20 · balanced | 0.2 | 0.625 | 0.000 | 92.8% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b/text · torgo_words | 0.2 | -inf | 0.750 | 35.1% (n=1512) | unknown_speech 2.2% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | NO |
| nemotron-streaming-0.6b/text · torgo_words · balanced | 0.2 | -inf | 0.500 | 77.7% (n=1512) | unknown_speech 4.4% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b-160ms/text · phrases20 | 0.2 | 0.625 | 0.000 | 87.5% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b-160ms/text · phrases20 · balanced | 0.2 | 0.625 | 0.000 | 87.5% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b-160ms/text · torgo_words | 0.2 | -inf | 0.667 | 51.7% (n=1512) | unknown_speech 1.1% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | NO |
| nemotron-streaming-0.6b-160ms/text · torgo_words · balanced | 0.2 | -inf | 0.500 | 70.2% (n=1512) | unknown_speech 2.2% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b-560ms/text · phrases20 | 0.2 | 0.625 | 0.000 | 92.3% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b-560ms/text · phrases20 · balanced | 0.2 | 0.625 | 0.000 | 92.3% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b-560ms/text · torgo_words | 0.2 | -inf | 0.750 | 35.4% (n=1512) | unknown_speech 1.1% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | NO |
| nemotron-streaming-0.6b-560ms/text · torgo_words · balanced | 0.2 | -inf | 0.500 | 76.7% (n=1512) | unknown_speech 3.3% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| whisper-large-v3-turbo/text · phrases20 | 0.2 | 0.714 | 0.000 | 91.3% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| whisper-large-v3-turbo/text · phrases20 · balanced | 0.2 | 0.714 | 0.000 | 91.3% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| whisper-large-v3-turbo/text · torgo_words | 0.2 | -inf | 1.000 | 6.9% (n=1512) | unknown_speech 0.0% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| whisper-large-v3-turbo/text · torgo_words · balanced | 0.2 | 0.400 | 0.333 | 69.5% (n=1512) | unknown_speech 4.4% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| whisper-small/text · phrases20 | 0.2 | 0.636 | 0.000 | 86.9% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.6% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| whisper-small/text · phrases20 · balanced | 0.2 | 0.636 | 0.000 | 86.9% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.6% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| whisper-small/text · torgo_words | 0.2 | -inf | 0.667 | 45.5% (n=1512) | unknown_speech 0.0% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| whisper-small/text · torgo_words · balanced | 0.2 | 0.667 | 0.000 | 67.7% (n=1512) | unknown_speech 4.4% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |

## Which model wins and why (measured numbers only)

**nemotron-streaming-0.6b/text** had the highest calib objective among configurations meeting the calib false-accept target. On test real recordings it accepted the correct phrase for 87/156 (55.8%) of positives, wrongly accepted 1/156 (0.6%), and false-accepted hard negative 1/31 (3.2%), unknown speech 0/430 (0.0%), non speech 0/191 (0.0%), silence 0/4 (0.0%).

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
