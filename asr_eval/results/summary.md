# ASR evaluation summary

Generated 2026-10-17 22:42 · config `eval.yaml` · **test split only** (thresholds tuned on calib).

> ⚠️ No threshold met the calib false-accept target for: parakeet-ctc-0.6b/text|torgo_words, nemotron-streaming-0.6b/text|torgo_words, nemotron-streaming-0.6b-160ms/text|torgo_words, nemotron-streaming-0.6b-560ms/text|torgo_words (strictest point used; see Thresholds).

## Headline: 20 phrases, real recordings, not degraded

| Model / method | Top-1 rank | Top-3 rank | Accepted correct | Wrong accept | Abstain | FA hard-neg | FA unknown speech | FA non-speech | FA silence |
|---|---|---|---|---|---|---|---|---|---|
| parakeet-ctc-0.6b/text | 112/156 (71.8%) | 126/156 (80.8%) | 81/156 (51.9%) | 1/156 (0.6%) | 74/156 (47.4%) | 1/31 (3.2%) [0.6–16.2] | 0/430 (0.0%) [0.0–0.9] ≤0.7% | 1/191 (0.5%) [0.1–2.9] | 0/184 (0.0%) [0.0–2.0] ≤1.6% |
| parakeet-ctc-0.6b/forced_sum | 106/156 (67.9%) | 114/156 (73.1%) | 97/156 (62.2%) | 12/156 (7.7%) | 47/156 (30.1%) | 5/31 (16.1%) [7.1–32.6] | 2/430 (0.5%) [0.1–1.7] | 2/191 (1.0%) [0.3–3.7] | 0/184 (0.0%) [0.0–2.0] ≤1.6% |
| parakeet-ctc-0.6b/forced_norm | 121/156 (77.6%) | 131/156 (84.0%) | 106/156 (67.9%) | 7/156 (4.5%) | 43/156 (27.6%) | 4/31 (12.9%) [5.1–28.9] | 0/430 (0.0%) [0.0–0.9] ≤0.7% | 3/191 (1.6%) [0.5–4.5] | 0/184 (0.0%) [0.0–2.0] ≤1.6% |
| parakeet-ctc-0.6b/forced_llr | 112/156 (71.8%) | 125/156 (80.1%) | 100/156 (64.1%) | 8/156 (5.1%) | 48/156 (30.8%) | 4/31 (12.9%) [5.1–28.9] | 2/430 (0.5%) [0.1–1.7] | 3/191 (1.6%) [0.5–4.5] | 0/184 (0.0%) [0.0–2.0] ≤1.6% |
| parakeet-tdt-0.6b-v2/text | 114/156 (73.1%) | 126/156 (80.8%) | 89/156 (57.1%) | 2/156 (1.3%) | 65/156 (41.7%) | 0/31 (0.0%) [0.0–11.0] ≤9.7% | 0/430 (0.0%) [0.0–0.9] ≤0.7% | 0/191 (0.0%) [0.0–2.0] ≤1.6% | 0/184 (0.0%) [0.0–2.0] ≤1.6% |
| nemotron-streaming-0.6b/text | 110/156 (70.5%) | 123/156 (78.8%) | 87/156 (55.8%) | 1/156 (0.6%) | 68/156 (43.6%) | 1/31 (3.2%) [0.6–16.2] | 0/430 (0.0%) [0.0–0.9] ≤0.7% | 0/191 (0.0%) [0.0–2.0] ≤1.6% | 0/184 (0.0%) [0.0–2.0] ≤1.6% |
| nemotron-streaming-0.6b/forced_sum | 123/156 (78.8%) | 129/156 (82.7%) | 97/156 (62.2%) | 13/156 (8.3%) | 46/156 (29.5%) | 1/31 (3.2%) [0.6–16.2] | 4/430 (0.9%) [0.4–2.4] | 1/191 (0.5%) [0.1–2.9] | 1/184 (0.5%) [0.1–3.0] |
| nemotron-streaming-0.6b/forced_norm | 132/156 (84.6%) | 140/156 (89.7%) | 107/156 (68.6%) | 7/156 (4.5%) | 42/156 (26.9%) | 2/31 (6.5%) [1.8–20.7] | 1/430 (0.2%) [0.0–1.3] | 1/191 (0.5%) [0.1–2.9] | 2/184 (1.1%) [0.3–3.9] |
| nemotron-streaming-0.6b/forced_llr | 129/156 (82.7%) | 136/156 (87.2%) | 89/156 (57.1%) | 1/156 (0.6%) | 66/156 (42.3%) | 0/31 (0.0%) [0.0–11.0] ≤9.7% | 0/430 (0.0%) [0.0–0.9] ≤0.7% | 0/191 (0.0%) [0.0–2.0] ≤1.6% | 0/184 (0.0%) [0.0–2.0] ≤1.6% |
| nemotron-3.5-asr-0.6b/text | 80/156 (51.3%) | 100/156 (64.1%) | 58/156 (37.2%) | 1/156 (0.6%) | 97/156 (62.2%) | 1/31 (3.2%) [0.6–16.2] | 0/430 (0.0%) [0.0–0.9] ≤0.7% | 0/191 (0.0%) [0.0–2.0] ≤1.6% | 0/184 (0.0%) [0.0–2.0] ≤1.6% |
| nemotron-3.5-asr-0.6b/forced_sum | 97/156 (62.2%) | 110/156 (70.5%) | 72/156 (46.2%) | 15/156 (9.6%) | 69/156 (44.2%) | 4/31 (12.9%) [5.1–28.9] | 2/430 (0.5%) [0.1–1.7] | 2/191 (1.0%) [0.3–3.7] | 0/184 (0.0%) [0.0–2.0] ≤1.6% |
| nemotron-3.5-asr-0.6b/forced_norm | 107/156 (68.6%) | 114/156 (73.1%) | 85/156 (54.5%) | 6/156 (3.8%) | 65/156 (41.7%) | 6/31 (19.4%) [9.2–36.3] | 1/430 (0.2%) [0.0–1.3] | 1/191 (0.5%) [0.1–2.9] | 0/184 (0.0%) [0.0–2.0] ≤1.6% |
| nemotron-3.5-asr-0.6b/forced_llr | 98/156 (62.8%) | 111/156 (71.2%) | 57/156 (36.5%) | 3/156 (1.9%) | 96/156 (61.5%) | 1/31 (3.2%) [0.6–16.2] | 2/430 (0.5%) [0.1–1.7] | 2/191 (1.0%) [0.3–3.7] | 0/184 (0.0%) [0.0–2.0] ≤1.6% |
| nemotron-streaming-0.6b-160ms/text | 83/156 (53.2%) | 99/156 (63.5%) | 60/156 (38.5%) | 0/156 (0.0%) | 96/156 (61.5%) | 2/31 (6.5%) [1.8–20.7] | 0/430 (0.0%) [0.0–0.9] ≤0.7% | 1/191 (0.5%) [0.1–2.9] | 0/184 (0.0%) [0.0–2.0] ≤1.6% |
| nemotron-streaming-0.6b-560ms/text | 108/156 (69.2%) | 118/156 (75.6%) | 84/156 (53.8%) | 1/156 (0.6%) | 71/156 (45.5%) | 3/31 (9.7%) [3.3–24.9] | 0/430 (0.0%) [0.0–0.9] ≤0.7% | 0/191 (0.0%) [0.0–2.0] ≤1.6% | 0/184 (0.0%) [0.0–2.0] ≤1.6% |
| nemotron-streaming-0.6b-560ms/forced_sum | 120/156 (76.9%) | 127/156 (81.4%) | 97/156 (62.2%) | 17/156 (10.9%) | 42/156 (26.9%) | 3/31 (9.7%) [3.3–24.9] | 3/430 (0.7%) [0.2–2.0] | 2/191 (1.0%) [0.3–3.7] | 1/184 (0.5%) [0.1–3.0] |
| nemotron-streaming-0.6b-560ms/forced_norm | 129/156 (82.7%) | 139/156 (89.1%) | 107/156 (68.6%) | 12/156 (7.7%) | 37/156 (23.7%) | 5/31 (16.1%) [7.1–32.6] | 1/430 (0.2%) [0.0–1.3] | 2/191 (1.0%) [0.3–3.7] | 2/184 (1.1%) [0.3–3.9] |
| nemotron-streaming-0.6b-560ms/forced_llr | 124/156 (79.5%) | 135/156 (86.5%) | 84/156 (53.8%) | 2/156 (1.3%) | 70/156 (44.9%) | 1/31 (3.2%) [0.6–16.2] | 0/430 (0.0%) [0.0–0.9] ≤0.7% | 0/191 (0.0%) [0.0–2.0] ≤1.6% | 0/184 (0.0%) [0.0–2.0] ≤1.6% |
| whisper-large-v3-turbo/text | 116/156 (74.4%) | 129/156 (82.7%) | 93/156 (59.6%) | 2/156 (1.3%) | 61/156 (39.1%) | 2/31 (6.5%) [1.8–20.7] | 0/430 (0.0%) [0.0–0.9] ≤0.7% | 0/191 (0.0%) [0.0–2.0] ≤1.6% | 0/184 (0.0%) [0.0–2.0] ≤1.6% |
| whisper-small/text | 116/156 (74.4%) | 128/156 (82.1%) | 94/156 (60.3%) | 2/156 (1.3%) | 60/156 (38.5%) | 1/31 (3.2%) [0.6–16.2] | 0/430 (0.0%) [0.0–0.9] ≤0.7% | 0/191 (0.0%) [0.0–2.0] ≤1.6% | 0/184 (0.0%) [0.0–2.0] ≤1.6% |

FA = false accept (a phrase proposed on a negative clip): k/n (rate) [95% Wilson CI]; ≤x% = rule-of-three upper bound when k = 0. Ranking columns are before abstention; 'Accepted correct' is correct candidates over all positives.

## Per condition (nemotron-streaming-0.6b/forced_norm)

### Real recordings

| Condition | n | Top-1 rank | Accepted correct | Wrong accept | Abstain |
|---|---:|---|---|---|---|
| muffled | 15 | 8/15 (53.3%) | 5/15 (33.3%) | 0/15 (0.0%) | 10/15 (66.7%) |
| noisy | 40 | 31/40 (77.5%) | 19/40 (47.5%) | 3/40 (7.5%) | 18/40 (45.0%) |
| normal | 80 | 80/80 (100.0%) | 72/80 (90.0%) | 0/80 (0.0%) | 8/80 (10.0%) |
| whisper | 21 | 13/21 (61.9%) | 11/21 (52.4%) | 4/21 (19.0%) | 6/21 (28.6%) |

### Degraded copies of real recordings (synthetic degradation)

| Condition | n | Top-1 rank | Accepted correct | Wrong accept | Abstain |
|---|---:|---|---|---|---|
| bandpass_300-3400 | 80 | 80/80 (100.0%) | 72/80 (90.0%) | 0/80 (0.0%) | 8/80 (10.0%) |
| gain_-12dB | 80 | 80/80 (100.0%) | 70/80 (87.5%) | 0/80 (0.0%) | 10/80 (12.5%) |
| gain_-24dB | 80 | 80/80 (100.0%) | 71/80 (88.8%) | 0/80 (0.0%) | 9/80 (11.2%) |
| gain_-36dB | 80 | 80/80 (100.0%) | 68/80 (85.0%) | 0/80 (0.0%) | 12/80 (15.0%) |
| lowpass_1500 | 80 | 79/80 (98.8%) | 68/80 (85.0%) | 0/80 (0.0%) | 12/80 (15.0%) |
| noise_musan_snr0 | 80 | 77/80 (96.2%) | 66/80 (82.5%) | 1/80 (1.2%) | 13/80 (16.2%) |
| noise_musan_snr10 | 80 | 80/80 (100.0%) | 70/80 (87.5%) | 0/80 (0.0%) | 10/80 (12.5%) |
| noise_musan_snr20 | 80 | 80/80 (100.0%) | 73/80 (91.2%) | 0/80 (0.0%) | 7/80 (8.8%) |
| noise_musan_snr5 | 80 | 79/80 (98.8%) | 67/80 (83.8%) | 0/80 (0.0%) | 13/80 (16.2%) |
| reverb_rt60_0.3s | 80 | 78/80 (97.5%) | 67/80 (83.8%) | 1/80 (1.2%) | 12/80 (15.0%) |
| reverb_rt60_0.6s | 80 | 73/80 (91.2%) | 61/80 (76.2%) | 4/80 (5.0%) | 15/80 (18.8%) |
| tempo_0.7x | 80 | 74/80 (92.5%) | 58/80 (72.5%) | 2/80 (2.5%) | 20/80 (25.0%) |
| tempo_0.85x | 80 | 74/80 (92.5%) | 60/80 (75.0%) | 1/80 (1.2%) | 19/80 (23.8%) |

### TTS — sanity only

| Condition | n | Top-1 rank | Accepted correct | Wrong accept | Abstain |
|---|---:|---|---|---|---|
| tts_clean | 340 | 340/340 (100.0%) | 340/340 (100.0%) | 0/340 (0.0%) | 0/340 (0.0%) |

## TORGO surrogate closed-vocabulary task: dysarthric vs control

Vocabulary (config): yes, no, up, down, left, right, go, one, two, three

| Model / method | Group | Top-1 rank | Accepted correct | Wrong accept | FA other words | WER / CER (all TORGO clips) |
|---|---|---|---|---|---|---|
| parakeet-ctc-0.6b/text | control | 103/118 (87.3%) | 24/118 (20.3%) | 0/118 (0.0%) | 1/141 (0.7%) [0.1–3.9] | 9.4% / 5.3% (n=461) |
| parakeet-ctc-0.6b/text | dysarthric | 73/90 (81.1%) | 12/90 (13.3%) | 1/90 (1.1%) | 5/69 (7.2%) [3.1–15.9] | 29.8% / 19.0% (n=255) |
| parakeet-ctc-0.6b/forced_sum | control | 103/118 (87.3%) | 74/118 (62.7%) | 0/118 (0.0%) | 2/141 (1.4%) [0.4–5.0] | 9.4% / 5.3% (n=461) |
| parakeet-ctc-0.6b/forced_sum | dysarthric | 77/90 (85.6%) | 42/90 (46.7%) | 0/90 (0.0%) | 1/69 (1.4%) [0.3–7.8] | 29.8% / 19.0% (n=255) |
| parakeet-ctc-0.6b/forced_norm | control | 103/118 (87.3%) | 74/118 (62.7%) | 0/118 (0.0%) | 2/141 (1.4%) [0.4–5.0] | 9.4% / 5.3% (n=461) |
| parakeet-ctc-0.6b/forced_norm | dysarthric | 77/90 (85.6%) | 42/90 (46.7%) | 0/90 (0.0%) | 1/69 (1.4%) [0.3–7.8] | 29.8% / 19.0% (n=255) |
| parakeet-ctc-0.6b/forced_llr | control | 103/118 (87.3%) | 62/118 (52.5%) | 0/118 (0.0%) | 3/141 (2.1%) [0.7–6.1] | 9.4% / 5.3% (n=461) |
| parakeet-ctc-0.6b/forced_llr | dysarthric | 77/90 (85.6%) | 32/90 (35.6%) | 0/90 (0.0%) | 0/69 (0.0%) [0.0–5.3] ≤4.3% | 29.8% / 19.0% (n=255) |
| parakeet-tdt-0.6b-v2/text | control | 114/118 (96.6%) | 22/118 (18.6%) | 0/118 (0.0%) | 0/141 (0.0%) [0.0–2.7] ≤2.1% | 5.7% / 2.6% (n=461) |
| parakeet-tdt-0.6b-v2/text | dysarthric | 74/90 (82.2%) | 6/90 (6.7%) | 0/90 (0.0%) | 0/69 (0.0%) [0.0–5.3] ≤4.3% | 26.4% / 18.1% (n=255) |
| nemotron-streaming-0.6b/text | control | 118/118 (100.0%) | 63/118 (53.4%) | 0/118 (0.0%) | 1/141 (0.7%) [0.1–3.9] | 5.3% / 2.5% (n=461) |
| nemotron-streaming-0.6b/text | dysarthric | 84/90 (93.3%) | 36/90 (40.0%) | 1/90 (1.1%) | 0/69 (0.0%) [0.0–5.3] ≤4.3% | 27.9% / 19.8% (n=255) |
| nemotron-streaming-0.6b/forced_sum | control | 118/118 (100.0%) | 114/118 (96.6%) | 0/118 (0.0%) | 1/141 (0.7%) [0.1–3.9] | 5.3% / 2.5% (n=461) |
| nemotron-streaming-0.6b/forced_sum | dysarthric | 88/90 (97.8%) | 74/90 (82.2%) | 0/90 (0.0%) | 0/69 (0.0%) [0.0–5.3] ≤4.3% | 27.9% / 19.8% (n=255) |
| nemotron-streaming-0.6b/forced_norm | control | 118/118 (100.0%) | 109/118 (92.4%) | 0/118 (0.0%) | 1/141 (0.7%) [0.1–3.9] | 5.3% / 2.5% (n=461) |
| nemotron-streaming-0.6b/forced_norm | dysarthric | 88/90 (97.8%) | 65/90 (72.2%) | 0/90 (0.0%) | 0/69 (0.0%) [0.0–5.3] ≤4.3% | 27.9% / 19.8% (n=255) |
| nemotron-streaming-0.6b/forced_llr | control | 118/118 (100.0%) | 89/118 (75.4%) | 0/118 (0.0%) | 0/141 (0.0%) [0.0–2.7] ≤2.1% | 5.3% / 2.5% (n=461) |
| nemotron-streaming-0.6b/forced_llr | dysarthric | 88/90 (97.8%) | 55/90 (61.1%) | 0/90 (0.0%) | 1/69 (1.4%) [0.3–7.8] | 27.9% / 19.8% (n=255) |
| nemotron-3.5-asr-0.6b/text | control | 104/118 (88.1%) | 58/118 (49.2%) | 0/118 (0.0%) | 0/141 (0.0%) [0.0–2.7] ≤2.1% | 11.4% / 7.3% (n=461) |
| nemotron-3.5-asr-0.6b/text | dysarthric | 76/90 (84.4%) | 33/90 (36.7%) | 0/90 (0.0%) | 0/69 (0.0%) [0.0–5.3] ≤4.3% | 39.2% / 29.8% (n=255) |
| nemotron-3.5-asr-0.6b/forced_sum | control | 118/118 (100.0%) | 28/118 (23.7%) | 0/118 (0.0%) | 0/141 (0.0%) [0.0–2.7] ≤2.1% | 11.4% / 7.3% (n=461) |
| nemotron-3.5-asr-0.6b/forced_sum | dysarthric | 83/90 (92.2%) | 22/90 (24.4%) | 0/90 (0.0%) | 1/69 (1.4%) [0.3–7.8] | 39.2% / 29.8% (n=255) |
| nemotron-3.5-asr-0.6b/forced_norm | control | 118/118 (100.0%) | 49/118 (41.5%) | 0/118 (0.0%) | 0/141 (0.0%) [0.0–2.7] ≤2.1% | 11.4% / 7.3% (n=461) |
| nemotron-3.5-asr-0.6b/forced_norm | dysarthric | 81/90 (90.0%) | 36/90 (40.0%) | 0/90 (0.0%) | 1/69 (1.4%) [0.3–7.8] | 39.2% / 29.8% (n=255) |
| nemotron-3.5-asr-0.6b/forced_llr | control | 118/118 (100.0%) | 7/118 (5.9%) | 0/118 (0.0%) | 0/141 (0.0%) [0.0–2.7] ≤2.1% | 11.4% / 7.3% (n=461) |
| nemotron-3.5-asr-0.6b/forced_llr | dysarthric | 81/90 (90.0%) | 5/90 (5.6%) | 0/90 (0.0%) | 0/69 (0.0%) [0.0–5.3] ≤4.3% | 39.2% / 29.8% (n=255) |
| nemotron-streaming-0.6b-160ms/text | control | 112/118 (94.9%) | 87/118 (73.7%) | 0/118 (0.0%) | 2/141 (1.4%) [0.4–5.0] | 6.8% / 4.4% (n=461) |
| nemotron-streaming-0.6b-160ms/text | dysarthric | 76/90 (84.4%) | 52/90 (57.8%) | 0/90 (0.0%) | 1/69 (1.4%) [0.3–7.8] | 29.5% / 22.3% (n=255) |
| nemotron-streaming-0.6b-560ms/text | control | 118/118 (100.0%) | 62/118 (52.5%) | 0/118 (0.0%) | 1/141 (0.7%) [0.1–3.9] | 5.8% / 3.0% (n=461) |
| nemotron-streaming-0.6b-560ms/text | dysarthric | 84/90 (93.3%) | 36/90 (40.0%) | 1/90 (1.1%) | 0/69 (0.0%) [0.0–5.3] ≤4.3% | 29.0% / 20.6% (n=255) |
| nemotron-streaming-0.6b-560ms/forced_sum | control | 118/118 (100.0%) | 111/118 (94.1%) | 0/118 (0.0%) | 1/141 (0.7%) [0.1–3.9] | 5.8% / 3.0% (n=461) |
| nemotron-streaming-0.6b-560ms/forced_sum | dysarthric | 88/90 (97.8%) | 73/90 (81.1%) | 0/90 (0.0%) | 0/69 (0.0%) [0.0–5.3] ≤4.3% | 29.0% / 20.6% (n=255) |
| nemotron-streaming-0.6b-560ms/forced_norm | control | 118/118 (100.0%) | 108/118 (91.5%) | 0/118 (0.0%) | 1/141 (0.7%) [0.1–3.9] | 5.8% / 3.0% (n=461) |
| nemotron-streaming-0.6b-560ms/forced_norm | dysarthric | 88/90 (97.8%) | 70/90 (77.8%) | 0/90 (0.0%) | 2/69 (2.9%) [0.8–10.0] | 29.0% / 20.6% (n=255) |
| nemotron-streaming-0.6b-560ms/forced_llr | control | 118/118 (100.0%) | 81/118 (68.6%) | 0/118 (0.0%) | 0/141 (0.0%) [0.0–2.7] ≤2.1% | 5.8% / 3.0% (n=461) |
| nemotron-streaming-0.6b-560ms/forced_llr | dysarthric | 88/90 (97.8%) | 53/90 (58.9%) | 0/90 (0.0%) | 1/69 (1.4%) [0.3–7.8] | 29.0% / 20.6% (n=255) |
| whisper-large-v3-turbo/text | control | 108/118 (91.5%) | 14/118 (11.9%) | 0/118 (0.0%) | 0/141 (0.0%) [0.0–2.7] ≤2.1% | 5.0% / 2.7% (n=461) |
| whisper-large-v3-turbo/text | dysarthric | 82/90 (91.1%) | 5/90 (5.6%) | 0/90 (0.0%) | 0/69 (0.0%) [0.0–5.3] ≤4.3% | 24.0% / 14.5% (n=255) |
| whisper-small/text | control | 112/118 (94.9%) | 87/118 (73.7%) | 0/118 (0.0%) | 1/141 (0.7%) [0.1–3.9] | 6.9% / 3.4% (n=461) |
| whisper-small/text | dysarthric | 79/90 (87.8%) | 46/90 (51.1%) | 0/90 (0.0%) | 0/69 (0.0%) [0.0–5.3] ≤4.3% | 44.7% / 48.1% (n=255) |

### TORGO words by condition (nemotron-streaming-0.6b/forced_sum, best on calib for this task; degraded rows are synthetic)

| Condition | Group | n | Top-1 rank | Accepted correct | Wrong accept |
|---|---|---:|---|---|---|
| bandpass_300-3400 | control | 118 | 118/118 (100.0%) | 114/118 (96.6%) | 0/118 (0.0%) |
| bandpass_300-3400 | dysarthric | 90 | 88/90 (97.8%) | 75/90 (83.3%) | 0/90 (0.0%) |
| clean | control | 118 | 118/118 (100.0%) | 114/118 (96.6%) | 0/118 (0.0%) |
| clean | dysarthric | 90 | 88/90 (97.8%) | 74/90 (82.2%) | 0/90 (0.0%) |
| gain_-12dB | control | 118 | 118/118 (100.0%) | 113/118 (95.8%) | 0/118 (0.0%) |
| gain_-12dB | dysarthric | 90 | 88/90 (97.8%) | 74/90 (82.2%) | 0/90 (0.0%) |
| gain_-24dB | control | 118 | 118/118 (100.0%) | 107/118 (90.7%) | 0/118 (0.0%) |
| gain_-24dB | dysarthric | 90 | 88/90 (97.8%) | 72/90 (80.0%) | 0/90 (0.0%) |
| gain_-36dB | control | 118 | 118/118 (100.0%) | 83/118 (70.3%) | 0/118 (0.0%) |
| gain_-36dB | dysarthric | 90 | 87/90 (96.7%) | 63/90 (70.0%) | 0/90 (0.0%) |
| lowpass_1500 | control | 118 | 118/118 (100.0%) | 108/118 (91.5%) | 0/118 (0.0%) |
| lowpass_1500 | dysarthric | 90 | 87/90 (96.7%) | 72/90 (80.0%) | 0/90 (0.0%) |
| noise_musan_snr0 | control | 118 | 118/118 (100.0%) | 102/118 (86.4%) | 0/118 (0.0%) |
| noise_musan_snr0 | dysarthric | 90 | 89/90 (98.9%) | 61/90 (67.8%) | 0/90 (0.0%) |
| noise_musan_snr10 | control | 118 | 118/118 (100.0%) | 111/118 (94.1%) | 0/118 (0.0%) |
| noise_musan_snr10 | dysarthric | 90 | 89/90 (98.9%) | 68/90 (75.6%) | 0/90 (0.0%) |
| noise_musan_snr20 | control | 118 | 118/118 (100.0%) | 111/118 (94.1%) | 0/118 (0.0%) |
| noise_musan_snr20 | dysarthric | 90 | 87/90 (96.7%) | 69/90 (76.7%) | 0/90 (0.0%) |
| noise_musan_snr5 | control | 118 | 118/118 (100.0%) | 108/118 (91.5%) | 0/118 (0.0%) |
| noise_musan_snr5 | dysarthric | 90 | 88/90 (97.8%) | 67/90 (74.4%) | 0/90 (0.0%) |
| reverb_rt60_0.3s | control | 118 | 118/118 (100.0%) | 104/118 (88.1%) | 0/118 (0.0%) |
| reverb_rt60_0.3s | dysarthric | 90 | 88/90 (97.8%) | 68/90 (75.6%) | 0/90 (0.0%) |
| reverb_rt60_0.6s | control | 118 | 117/118 (99.2%) | 96/118 (81.4%) | 0/118 (0.0%) |
| reverb_rt60_0.6s | dysarthric | 90 | 86/90 (95.6%) | 60/90 (66.7%) | 1/90 (1.1%) |
| tempo_0.7x | control | 118 | 118/118 (100.0%) | 94/118 (79.7%) | 0/118 (0.0%) |
| tempo_0.7x | dysarthric | 90 | 82/90 (91.1%) | 52/90 (57.8%) | 1/90 (1.1%) |
| tempo_0.85x | control | 118 | 118/118 (100.0%) | 99/118 (83.9%) | 0/118 (0.0%) |
| tempo_0.85x | dysarthric | 90 | 84/90 (93.3%) | 61/90 (67.8%) | 1/90 (1.1%) |

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
| parakeet-ctc-0.6b/forced_sum | esc50 | non_speech | 0/63 (0.0%) [0.0–5.7] ≤4.8% |
| parakeet-ctc-0.6b/forced_sum | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| parakeet-ctc-0.6b/forced_sum | librispeech | unknown_speech | 0/127 (0.0%) [0.0–2.9] ≤2.4% |
| parakeet-ctc-0.6b/forced_sum | musan | non_speech | 1/90 (1.1%) [0.2–6.0] |
| parakeet-ctc-0.6b/forced_sum | own | hard_negative | 5/31 (16.1%) [7.1–32.6] |
| parakeet-ctc-0.6b/forced_sum | own | non_speech | 1/9 (11.1%) [2.0–43.5] |
| parakeet-ctc-0.6b/forced_sum | own | silence | 0/4 (0.0%) [0.0–49.0] ≤75.0% |
| parakeet-ctc-0.6b/forced_sum | own | unknown_speech | 2/5 (40.0%) [11.8–76.9] |
| parakeet-ctc-0.6b/forced_sum | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| parakeet-ctc-0.6b/forced_sum | torgo | unknown_speech | 0/298 (0.0%) [0.0–1.3] ≤1.0% |
| parakeet-ctc-0.6b/forced_sum | tts (sanity) | hard_negative | 0/119 (0.0%) [0.0–3.1] ≤2.5% |
| parakeet-ctc-0.6b/forced_norm | esc50 | non_speech | 1/63 (1.6%) [0.3–8.5] |
| parakeet-ctc-0.6b/forced_norm | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| parakeet-ctc-0.6b/forced_norm | librispeech | unknown_speech | 0/127 (0.0%) [0.0–2.9] ≤2.4% |
| parakeet-ctc-0.6b/forced_norm | musan | non_speech | 1/90 (1.1%) [0.2–6.0] |
| parakeet-ctc-0.6b/forced_norm | own | hard_negative | 4/31 (12.9%) [5.1–28.9] |
| parakeet-ctc-0.6b/forced_norm | own | non_speech | 1/9 (11.1%) [2.0–43.5] |
| parakeet-ctc-0.6b/forced_norm | own | silence | 0/4 (0.0%) [0.0–49.0] ≤75.0% |
| parakeet-ctc-0.6b/forced_norm | own | unknown_speech | 0/5 (0.0%) [0.0–43.4] ≤60.0% |
| parakeet-ctc-0.6b/forced_norm | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| parakeet-ctc-0.6b/forced_norm | torgo | unknown_speech | 0/298 (0.0%) [0.0–1.3] ≤1.0% |
| parakeet-ctc-0.6b/forced_norm | tts (sanity) | hard_negative | 0/119 (0.0%) [0.0–3.1] ≤2.5% |
| parakeet-ctc-0.6b/forced_llr | esc50 | non_speech | 2/63 (3.2%) [0.9–10.9] |
| parakeet-ctc-0.6b/forced_llr | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| parakeet-ctc-0.6b/forced_llr | librispeech | unknown_speech | 0/127 (0.0%) [0.0–2.9] ≤2.4% |
| parakeet-ctc-0.6b/forced_llr | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| parakeet-ctc-0.6b/forced_llr | own | hard_negative | 4/31 (12.9%) [5.1–28.9] |
| parakeet-ctc-0.6b/forced_llr | own | non_speech | 1/9 (11.1%) [2.0–43.5] |
| parakeet-ctc-0.6b/forced_llr | own | silence | 0/4 (0.0%) [0.0–49.0] ≤75.0% |
| parakeet-ctc-0.6b/forced_llr | own | unknown_speech | 2/5 (40.0%) [11.8–76.9] |
| parakeet-ctc-0.6b/forced_llr | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| parakeet-ctc-0.6b/forced_llr | torgo | unknown_speech | 0/298 (0.0%) [0.0–1.3] ≤1.0% |
| parakeet-ctc-0.6b/forced_llr | tts (sanity) | hard_negative | 0/119 (0.0%) [0.0–3.1] ≤2.5% |
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
| nemotron-streaming-0.6b/forced_sum | esc50 | non_speech | 1/63 (1.6%) [0.3–8.5] |
| nemotron-streaming-0.6b/forced_sum | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| nemotron-streaming-0.6b/forced_sum | librispeech | unknown_speech | 0/127 (0.0%) [0.0–2.9] ≤2.4% |
| nemotron-streaming-0.6b/forced_sum | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| nemotron-streaming-0.6b/forced_sum | own | hard_negative | 1/31 (3.2%) [0.6–16.2] |
| nemotron-streaming-0.6b/forced_sum | own | non_speech | 0/9 (0.0%) [0.0–29.9] ≤33.3% |
| nemotron-streaming-0.6b/forced_sum | own | silence | 1/4 (25.0%) [4.6–69.9] |
| nemotron-streaming-0.6b/forced_sum | own | unknown_speech | 1/5 (20.0%) [3.6–62.4] |
| nemotron-streaming-0.6b/forced_sum | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| nemotron-streaming-0.6b/forced_sum | torgo | unknown_speech | 3/298 (1.0%) [0.3–2.9] |
| nemotron-streaming-0.6b/forced_sum | tts (sanity) | hard_negative | 0/119 (0.0%) [0.0–3.1] ≤2.5% |
| nemotron-streaming-0.6b/forced_norm | esc50 | non_speech | 1/63 (1.6%) [0.3–8.5] |
| nemotron-streaming-0.6b/forced_norm | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| nemotron-streaming-0.6b/forced_norm | librispeech | unknown_speech | 0/127 (0.0%) [0.0–2.9] ≤2.4% |
| nemotron-streaming-0.6b/forced_norm | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| nemotron-streaming-0.6b/forced_norm | own | hard_negative | 2/31 (6.5%) [1.8–20.7] |
| nemotron-streaming-0.6b/forced_norm | own | non_speech | 0/9 (0.0%) [0.0–29.9] ≤33.3% |
| nemotron-streaming-0.6b/forced_norm | own | silence | 2/4 (50.0%) [15.0–85.0] |
| nemotron-streaming-0.6b/forced_norm | own | unknown_speech | 0/5 (0.0%) [0.0–43.4] ≤60.0% |
| nemotron-streaming-0.6b/forced_norm | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| nemotron-streaming-0.6b/forced_norm | torgo | unknown_speech | 1/298 (0.3%) [0.1–1.9] |
| nemotron-streaming-0.6b/forced_norm | tts (sanity) | hard_negative | 0/119 (0.0%) [0.0–3.1] ≤2.5% |
| nemotron-streaming-0.6b/forced_llr | esc50 | non_speech | 0/63 (0.0%) [0.0–5.7] ≤4.8% |
| nemotron-streaming-0.6b/forced_llr | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| nemotron-streaming-0.6b/forced_llr | librispeech | unknown_speech | 0/127 (0.0%) [0.0–2.9] ≤2.4% |
| nemotron-streaming-0.6b/forced_llr | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| nemotron-streaming-0.6b/forced_llr | own | hard_negative | 0/31 (0.0%) [0.0–11.0] ≤9.7% |
| nemotron-streaming-0.6b/forced_llr | own | non_speech | 0/9 (0.0%) [0.0–29.9] ≤33.3% |
| nemotron-streaming-0.6b/forced_llr | own | silence | 0/4 (0.0%) [0.0–49.0] ≤75.0% |
| nemotron-streaming-0.6b/forced_llr | own | unknown_speech | 0/5 (0.0%) [0.0–43.4] ≤60.0% |
| nemotron-streaming-0.6b/forced_llr | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| nemotron-streaming-0.6b/forced_llr | torgo | unknown_speech | 0/298 (0.0%) [0.0–1.3] ≤1.0% |
| nemotron-streaming-0.6b/forced_llr | tts (sanity) | hard_negative | 0/119 (0.0%) [0.0–3.1] ≤2.5% |
| nemotron-3.5-asr-0.6b/text | esc50 | non_speech | 0/63 (0.0%) [0.0–5.7] ≤4.8% |
| nemotron-3.5-asr-0.6b/text | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| nemotron-3.5-asr-0.6b/text | librispeech | unknown_speech | 0/127 (0.0%) [0.0–2.9] ≤2.4% |
| nemotron-3.5-asr-0.6b/text | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| nemotron-3.5-asr-0.6b/text | own | hard_negative | 1/31 (3.2%) [0.6–16.2] |
| nemotron-3.5-asr-0.6b/text | own | non_speech | 0/9 (0.0%) [0.0–29.9] ≤33.3% |
| nemotron-3.5-asr-0.6b/text | own | silence | 0/4 (0.0%) [0.0–49.0] ≤75.0% |
| nemotron-3.5-asr-0.6b/text | own | unknown_speech | 0/5 (0.0%) [0.0–43.4] ≤60.0% |
| nemotron-3.5-asr-0.6b/text | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| nemotron-3.5-asr-0.6b/text | torgo | unknown_speech | 0/298 (0.0%) [0.0–1.3] ≤1.0% |
| nemotron-3.5-asr-0.6b/text | tts (sanity) | hard_negative | 0/119 (0.0%) [0.0–3.1] ≤2.5% |
| nemotron-3.5-asr-0.6b/forced_sum | esc50 | non_speech | 2/63 (3.2%) [0.9–10.9] |
| nemotron-3.5-asr-0.6b/forced_sum | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| nemotron-3.5-asr-0.6b/forced_sum | librispeech | unknown_speech | 0/127 (0.0%) [0.0–2.9] ≤2.4% |
| nemotron-3.5-asr-0.6b/forced_sum | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| nemotron-3.5-asr-0.6b/forced_sum | own | hard_negative | 4/31 (12.9%) [5.1–28.9] |
| nemotron-3.5-asr-0.6b/forced_sum | own | non_speech | 0/9 (0.0%) [0.0–29.9] ≤33.3% |
| nemotron-3.5-asr-0.6b/forced_sum | own | silence | 0/4 (0.0%) [0.0–49.0] ≤75.0% |
| nemotron-3.5-asr-0.6b/forced_sum | own | unknown_speech | 1/5 (20.0%) [3.6–62.4] |
| nemotron-3.5-asr-0.6b/forced_sum | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| nemotron-3.5-asr-0.6b/forced_sum | torgo | unknown_speech | 1/298 (0.3%) [0.1–1.9] |
| nemotron-3.5-asr-0.6b/forced_sum | tts (sanity) | hard_negative | 0/119 (0.0%) [0.0–3.1] ≤2.5% |
| nemotron-3.5-asr-0.6b/forced_norm | esc50 | non_speech | 1/63 (1.6%) [0.3–8.5] |
| nemotron-3.5-asr-0.6b/forced_norm | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| nemotron-3.5-asr-0.6b/forced_norm | librispeech | unknown_speech | 0/127 (0.0%) [0.0–2.9] ≤2.4% |
| nemotron-3.5-asr-0.6b/forced_norm | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| nemotron-3.5-asr-0.6b/forced_norm | own | hard_negative | 6/31 (19.4%) [9.2–36.3] |
| nemotron-3.5-asr-0.6b/forced_norm | own | non_speech | 0/9 (0.0%) [0.0–29.9] ≤33.3% |
| nemotron-3.5-asr-0.6b/forced_norm | own | silence | 0/4 (0.0%) [0.0–49.0] ≤75.0% |
| nemotron-3.5-asr-0.6b/forced_norm | own | unknown_speech | 0/5 (0.0%) [0.0–43.4] ≤60.0% |
| nemotron-3.5-asr-0.6b/forced_norm | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| nemotron-3.5-asr-0.6b/forced_norm | torgo | unknown_speech | 1/298 (0.3%) [0.1–1.9] |
| nemotron-3.5-asr-0.6b/forced_norm | tts (sanity) | hard_negative | 0/119 (0.0%) [0.0–3.1] ≤2.5% |
| nemotron-3.5-asr-0.6b/forced_llr | esc50 | non_speech | 2/63 (3.2%) [0.9–10.9] |
| nemotron-3.5-asr-0.6b/forced_llr | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| nemotron-3.5-asr-0.6b/forced_llr | librispeech | unknown_speech | 0/127 (0.0%) [0.0–2.9] ≤2.4% |
| nemotron-3.5-asr-0.6b/forced_llr | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| nemotron-3.5-asr-0.6b/forced_llr | own | hard_negative | 1/31 (3.2%) [0.6–16.2] |
| nemotron-3.5-asr-0.6b/forced_llr | own | non_speech | 0/9 (0.0%) [0.0–29.9] ≤33.3% |
| nemotron-3.5-asr-0.6b/forced_llr | own | silence | 0/4 (0.0%) [0.0–49.0] ≤75.0% |
| nemotron-3.5-asr-0.6b/forced_llr | own | unknown_speech | 1/5 (20.0%) [3.6–62.4] |
| nemotron-3.5-asr-0.6b/forced_llr | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| nemotron-3.5-asr-0.6b/forced_llr | torgo | unknown_speech | 1/298 (0.3%) [0.1–1.9] |
| nemotron-3.5-asr-0.6b/forced_llr | tts (sanity) | hard_negative | 0/119 (0.0%) [0.0–3.1] ≤2.5% |
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
| nemotron-streaming-0.6b-560ms/forced_sum | esc50 | non_speech | 2/63 (3.2%) [0.9–10.9] |
| nemotron-streaming-0.6b-560ms/forced_sum | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| nemotron-streaming-0.6b-560ms/forced_sum | librispeech | unknown_speech | 0/127 (0.0%) [0.0–2.9] ≤2.4% |
| nemotron-streaming-0.6b-560ms/forced_sum | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| nemotron-streaming-0.6b-560ms/forced_sum | own | hard_negative | 3/31 (9.7%) [3.3–24.9] |
| nemotron-streaming-0.6b-560ms/forced_sum | own | non_speech | 0/9 (0.0%) [0.0–29.9] ≤33.3% |
| nemotron-streaming-0.6b-560ms/forced_sum | own | silence | 1/4 (25.0%) [4.6–69.9] |
| nemotron-streaming-0.6b-560ms/forced_sum | own | unknown_speech | 2/5 (40.0%) [11.8–76.9] |
| nemotron-streaming-0.6b-560ms/forced_sum | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| nemotron-streaming-0.6b-560ms/forced_sum | torgo | unknown_speech | 1/298 (0.3%) [0.1–1.9] |
| nemotron-streaming-0.6b-560ms/forced_sum | tts (sanity) | hard_negative | 0/119 (0.0%) [0.0–3.1] ≤2.5% |
| nemotron-streaming-0.6b-560ms/forced_norm | esc50 | non_speech | 2/63 (3.2%) [0.9–10.9] |
| nemotron-streaming-0.6b-560ms/forced_norm | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| nemotron-streaming-0.6b-560ms/forced_norm | librispeech | unknown_speech | 0/127 (0.0%) [0.0–2.9] ≤2.4% |
| nemotron-streaming-0.6b-560ms/forced_norm | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| nemotron-streaming-0.6b-560ms/forced_norm | own | hard_negative | 5/31 (16.1%) [7.1–32.6] |
| nemotron-streaming-0.6b-560ms/forced_norm | own | non_speech | 0/9 (0.0%) [0.0–29.9] ≤33.3% |
| nemotron-streaming-0.6b-560ms/forced_norm | own | silence | 2/4 (50.0%) [15.0–85.0] |
| nemotron-streaming-0.6b-560ms/forced_norm | own | unknown_speech | 0/5 (0.0%) [0.0–43.4] ≤60.0% |
| nemotron-streaming-0.6b-560ms/forced_norm | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| nemotron-streaming-0.6b-560ms/forced_norm | torgo | unknown_speech | 1/298 (0.3%) [0.1–1.9] |
| nemotron-streaming-0.6b-560ms/forced_norm | tts (sanity) | hard_negative | 0/119 (0.0%) [0.0–3.1] ≤2.5% |
| nemotron-streaming-0.6b-560ms/forced_llr | esc50 | non_speech | 0/63 (0.0%) [0.0–5.7] ≤4.8% |
| nemotron-streaming-0.6b-560ms/forced_llr | generated | silence | 0/180 (0.0%) [0.0–2.1] ≤1.7% |
| nemotron-streaming-0.6b-560ms/forced_llr | librispeech | unknown_speech | 0/127 (0.0%) [0.0–2.9] ≤2.4% |
| nemotron-streaming-0.6b-560ms/forced_llr | musan | non_speech | 0/90 (0.0%) [0.0–4.1] ≤3.3% |
| nemotron-streaming-0.6b-560ms/forced_llr | own | hard_negative | 1/31 (3.2%) [0.6–16.2] |
| nemotron-streaming-0.6b-560ms/forced_llr | own | non_speech | 0/9 (0.0%) [0.0–29.9] ≤33.3% |
| nemotron-streaming-0.6b-560ms/forced_llr | own | silence | 0/4 (0.0%) [0.0–49.0] ≤75.0% |
| nemotron-streaming-0.6b-560ms/forced_llr | own | unknown_speech | 0/5 (0.0%) [0.0–43.4] ≤60.0% |
| nemotron-streaming-0.6b-560ms/forced_llr | speech_commands | non_speech | 0/29 (0.0%) [0.0–11.7] ≤10.3% |
| nemotron-streaming-0.6b-560ms/forced_llr | torgo | unknown_speech | 0/298 (0.0%) [0.0–1.3] ≤1.0% |
| nemotron-streaming-0.6b-560ms/forced_llr | tts (sanity) | hard_negative | 0/119 (0.0%) [0.0–3.1] ≤2.5% |
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
| nemotron-3.5-asr-0.6b | librispeech | 3.6% | 1.1% | 127 |
| nemotron-3.5-asr-0.6b | own | 63.2% | 53.9% | 187 |
| nemotron-3.5-asr-0.6b | torgo | 20.8% | 14.7% | 716 |
| nemotron-3.5-asr-0.6b | tts | 2.2% | 1.3% | 459 |
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
| nemotron-streaming-0.6b | 618.1M | 7.7 s | 7.049 GB | 21.02 GB | 12.8 ms | 17 / 20 / 46 ms | 13 / 48 ms | 43 / 94 / 2920 ms | 49.8 |
| nemotron-3.5-asr-0.6b | 638.0M | 39.7 s | 9.678 GB | 28.7 GB | 12.8 ms | 17 / 22 / 52 ms | 17 / 214 ms | 51 / 263 / 6698 ms | 41.8 |
| nemotron-streaming-0.6b-160ms | 618.1M | 7.7 s | 4.99 GB | 9.76 GB | 12.8 ms | 17 / 20 / 45 ms | — | 29 / 47 / 192 ms | 77.3 |
| nemotron-streaming-0.6b-560ms | 618.1M | 7.8 s | 34.118 GB | 90.62 GB | 12.8 ms | 17 / 20 / 45 ms | 13 / 48 ms | 43 / 94 / 4298 ms | 49.9 |
| whisper-large-v3-turbo | 808.9M | 0.9 s | 0.0 GB | 8.74 GB | 12.8 ms | 42 / 52 / 372 ms | — | 55 / 78 / 382 ms | 40.9 |
| whisper-small | 241.7M | 0.5 s | 0.0 GB | 7.23 GB | 12.8 ms | 28 / 47 / 622 ms | — | 42 / 73 / 634 ms | 53.0 |

'Peak torch alloc' is PyTorch's allocator (0 for CTranslate2/faster-whisper); 'GPU used (device)' is the whole device at the end of the model's run, including its CUDA context.

Total = VAD + ASR + phrase scoring (scoring is the forced pass over all phrases; the text method skips it). Network, capture and endpointing are not included.

| Model | Checkpoint | Revision | Device |
|---|---|---|---|
| parakeet-ctc-0.6b | nvidia/parakeet-ctc-0.6b | `ad09ba1cc627` | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| parakeet-tdt-0.6b-v2 | nvidia/parakeet-tdt-0.6b-v2 | `ae9ad07059c7` | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| nemotron-streaming-0.6b | nvidia/nemotron-speech-streaming-en-0.6b | `ebe59e5a8171` | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| nemotron-3.5-asr-0.6b | nvidia/nemotron-3.5-asr-streaming-0.6b | `ea30d66debe3` | NVIDIA RTX PRO 6000 Blackwell Server Edition |
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
| parakeet-ctc-0.6b/forced_sum · phrases20 | 0.2 | -24.206 | 2.503 | 85.5% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.6% (n=177), non_speech 0.8% (n=122), silence 0.0% (n=122) | yes |
| parakeet-ctc-0.6b/forced_sum · phrases20 · balanced | 0.3 | -24.206 | 0.000 | 86.9% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.6% (n=177), non_speech 1.6% (n=122), silence 0.0% (n=122) | yes |
| parakeet-ctc-0.6b/forced_sum · torgo_words | 0.2 | -0.026 | 0.000 | 35.4% (n=1512) | unknown_speech 0.0% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| parakeet-ctc-0.6b/forced_sum · torgo_words · balanced | 0.2 | -0.337 | 6.500 | 40.5% (n=1512) | unknown_speech 4.4% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| parakeet-ctc-0.6b/forced_norm · phrases20 | 0.2 | -5.463 | 0.000 | 90.1% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.6% (n=177), non_speech 0.8% (n=122), silence 0.0% (n=122) | yes |
| parakeet-ctc-0.6b/forced_norm · phrases20 · balanced | 0.2 | -6.225 | 0.000 | 90.6% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.6% (n=177), non_speech 1.6% (n=122), silence 0.0% (n=122) | yes |
| parakeet-ctc-0.6b/forced_norm · torgo_words | 0.2 | -0.026 | 0.000 | 35.4% (n=1512) | unknown_speech 0.0% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| parakeet-ctc-0.6b/forced_norm · torgo_words · balanced | 0.2 | -0.337 | 6.500 | 40.5% (n=1512) | unknown_speech 4.4% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| parakeet-ctc-0.6b/forced_llr · phrases20 | 0.2 | -4.113 | 0.864 | 87.0% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.6% (n=177), non_speech 0.8% (n=122), silence 0.0% (n=122) | yes |
| parakeet-ctc-0.6b/forced_llr · phrases20 · balanced | 0.2 | -3.442 | 0.048 | 87.7% (n=586) | hard_negative 0.0% (n=14), unknown_speech 4.0% (n=177), non_speech 1.6% (n=122), silence 0.0% (n=122) | yes |
| parakeet-ctc-0.6b/forced_llr · torgo_words | 0.2 | -inf | 11.500 | 18.7% (n=1512) | unknown_speech 0.0% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| parakeet-ctc-0.6b/forced_llr · torgo_words · balanced | 0.2 | -inf | 8.001 | 35.1% (n=1512) | unknown_speech 2.2% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| parakeet-tdt-0.6b-v2/text · phrases20 | 0.2 | 0.667 | 0.000 | 87.2% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| parakeet-tdt-0.6b-v2/text · phrases20 · balanced | 0.2 | 0.667 | 0.000 | 87.2% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| parakeet-tdt-0.6b-v2/text · torgo_words | 0.2 | -inf | 1.000 | 8.2% (n=1512) | unknown_speech 0.0% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| parakeet-tdt-0.6b-v2/text · torgo_words · balanced | 0.2 | 0.500 | 0.333 | 60.4% (n=1512) | unknown_speech 3.3% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b/text · phrases20 | 0.2 | 0.625 | 0.000 | 92.8% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b/text · phrases20 · balanced | 0.2 | 0.625 | 0.000 | 92.8% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b/text · torgo_words | 0.2 | -inf | 0.750 | 35.1% (n=1512) | unknown_speech 2.2% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | NO |
| nemotron-streaming-0.6b/text · torgo_words · balanced | 0.2 | -inf | 0.500 | 77.7% (n=1512) | unknown_speech 4.4% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b/forced_sum · phrases20 | 0.2 | -12.552 | 0.847 | 96.2% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b/forced_sum · phrases20 · balanced | 0.2 | -15.387 | 0.000 | 96.6% (n=586) | hard_negative 0.0% (n=14), unknown_speech 2.3% (n=177), non_speech 0.8% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b/forced_sum · torgo_words | 0.2 | -0.668 | 0.000 | 72.2% (n=1512) | unknown_speech 0.0% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b/forced_sum · torgo_words · balanced | 0.2 | -1.731 | 0.000 | 79.0% (n=1512) | unknown_speech 3.3% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b/forced_norm · phrases20 | 0.2 | -2.186 | 0.000 | 96.6% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b/forced_norm · phrases20 · balanced | 0.2 | -2.655 | 0.000 | 96.8% (n=586) | hard_negative 0.0% (n=14), unknown_speech 1.7% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b/forced_norm · torgo_words | 0.2 | -0.148 | 0.000 | 67.1% (n=1512) | unknown_speech 0.0% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b/forced_norm · torgo_words · balanced | 0.2 | -0.686 | 0.617 | 79.7% (n=1512) | unknown_speech 4.4% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b/forced_llr · phrases20 | 0.2 | -inf | 1.180 | 94.4% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.8% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b/forced_llr · phrases20 · balanced | 0.2 | -1.155 | 0.405 | 95.9% (n=586) | hard_negative 0.0% (n=14), unknown_speech 1.7% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b/forced_llr · torgo_words | 0.2 | -inf | 2.150 | 50.5% (n=1512) | unknown_speech 0.0% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b/forced_llr · torgo_words · balanced | 0.2 | -inf | 1.163 | 78.3% (n=1512) | unknown_speech 4.4% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-3.5-asr-0.6b/text · phrases20 | 0.2 | 0.625 | 0.000 | 78.8% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-3.5-asr-0.6b/text · phrases20 · balanced | 0.2 | 0.625 | 0.000 | 78.8% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-3.5-asr-0.6b/text · torgo_words | 0.2 | -inf | 0.750 | 28.0% (n=1512) | unknown_speech 0.0% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-3.5-asr-0.6b/text · torgo_words · balanced | 0.2 | 0.750 | 0.000 | 64.3% (n=1512) | unknown_speech 3.3% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-3.5-asr-0.6b/forced_sum · phrases20 | 0.3 | -14.493 | 0.000 | 85.7% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.6% (n=177), non_speech 0.8% (n=122), silence 0.0% (n=122) | yes |
| nemotron-3.5-asr-0.6b/forced_sum · phrases20 · balanced | 0.3 | -14.493 | 0.000 | 85.7% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.6% (n=177), non_speech 0.8% (n=122), silence 0.0% (n=122) | yes |
| nemotron-3.5-asr-0.6b/forced_sum · torgo_words | 0.2 | -1.787 | 0.000 | 24.3% (n=1512) | unknown_speech 0.0% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-3.5-asr-0.6b/forced_sum · torgo_words · balanced | 0.2 | -5.617 | 0.000 | 69.8% (n=1512) | unknown_speech 4.4% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-3.5-asr-0.6b/forced_norm · phrases20 | 0.2 | -2.448 | 0.000 | 87.2% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-3.5-asr-0.6b/forced_norm · phrases20 · balanced | 0.2 | -2.448 | 0.000 | 87.2% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-3.5-asr-0.6b/forced_norm · torgo_words | 0.2 | -0.882 | 2.033 | 37.2% (n=1512) | unknown_speech 0.0% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-3.5-asr-0.6b/forced_norm · torgo_words · balanced | 0.2 | -1.908 | 0.699 | 69.0% (n=1512) | unknown_speech 3.3% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-3.5-asr-0.6b/forced_llr · phrases20 | 0.2 | -0.033 | 0.468 | 79.7% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.6% (n=177), non_speech 0.8% (n=122), silence 0.0% (n=122) | yes |
| nemotron-3.5-asr-0.6b/forced_llr · phrases20 · balanced | 0.2 | -0.033 | 0.072 | 80.7% (n=586) | hard_negative 0.0% (n=14), unknown_speech 2.8% (n=177), non_speech 1.6% (n=122), silence 0.0% (n=122) | yes |
| nemotron-3.5-asr-0.6b/forced_llr · torgo_words | 0.2 | 3.348 | 0.000 | 5.8% (n=1512) | unknown_speech 0.0% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-3.5-asr-0.6b/forced_llr · torgo_words · balanced | 0.2 | -inf | 0.989 | 69.5% (n=1512) | unknown_speech 4.4% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b-160ms/text · phrases20 | 0.2 | 0.625 | 0.000 | 87.5% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b-160ms/text · phrases20 · balanced | 0.2 | 0.625 | 0.000 | 87.5% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b-160ms/text · torgo_words | 0.2 | -inf | 0.667 | 51.7% (n=1512) | unknown_speech 1.1% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | NO |
| nemotron-streaming-0.6b-160ms/text · torgo_words · balanced | 0.2 | -inf | 0.500 | 70.2% (n=1512) | unknown_speech 2.2% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b-560ms/text · phrases20 | 0.2 | 0.625 | 0.000 | 92.3% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b-560ms/text · phrases20 · balanced | 0.2 | 0.625 | 0.000 | 92.3% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b-560ms/text · torgo_words | 0.2 | -inf | 0.750 | 35.4% (n=1512) | unknown_speech 1.1% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | NO |
| nemotron-streaming-0.6b-560ms/text · torgo_words · balanced | 0.2 | -inf | 0.500 | 76.7% (n=1512) | unknown_speech 3.3% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b-560ms/forced_sum · phrases20 | 0.2 | -12.453 | 0.000 | 95.9% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.8% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b-560ms/forced_sum · phrases20 · balanced | 0.2 | -15.521 | 0.000 | 96.1% (n=586) | hard_negative 0.0% (n=14), unknown_speech 1.7% (n=177), non_speech 0.8% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b-560ms/forced_sum · torgo_words | 0.2 | -0.684 | 0.000 | 72.1% (n=1512) | unknown_speech 0.0% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b-560ms/forced_sum · torgo_words · balanced | 0.2 | -2.809 | 0.000 | 80.6% (n=1512) | unknown_speech 4.4% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b-560ms/forced_norm · phrases20 | 0.2 | -2.420 | 0.000 | 95.9% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b-560ms/forced_norm · phrases20 · balanced | 0.2 | -2.680 | 0.000 | 96.1% (n=586) | hard_negative 0.0% (n=14), unknown_speech 1.7% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b-560ms/forced_norm · torgo_words | 0.2 | -0.575 | 1.438 | 69.0% (n=1512) | unknown_speech 0.0% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b-560ms/forced_norm · torgo_words · balanced | 0.2 | -0.759 | 0.408 | 79.8% (n=1512) | unknown_speech 4.4% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b-560ms/forced_llr · phrases20 | 0.2 | -inf | 1.202 | 93.5% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.8% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b-560ms/forced_llr · phrases20 · balanced | 0.2 | -1.155 | 0.405 | 95.4% (n=586) | hard_negative 0.0% (n=14), unknown_speech 2.8% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b-560ms/forced_llr · torgo_words | 0.2 | -inf | 2.185 | 44.3% (n=1512) | unknown_speech 0.0% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| nemotron-streaming-0.6b-560ms/forced_llr · torgo_words · balanced | 0.2 | -0.132 | 0.000 | 78.2% (n=1512) | unknown_speech 4.4% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| whisper-large-v3-turbo/text · phrases20 | 0.2 | 0.714 | 0.000 | 91.3% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| whisper-large-v3-turbo/text · phrases20 · balanced | 0.2 | 0.714 | 0.000 | 91.3% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.0% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| whisper-large-v3-turbo/text · torgo_words | 0.2 | -inf | 1.000 | 6.9% (n=1512) | unknown_speech 0.0% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| whisper-large-v3-turbo/text · torgo_words · balanced | 0.2 | 0.400 | 0.333 | 69.5% (n=1512) | unknown_speech 4.4% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| whisper-small/text · phrases20 | 0.2 | 0.636 | 0.000 | 86.9% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.6% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| whisper-small/text · phrases20 · balanced | 0.2 | 0.636 | 0.000 | 86.9% (n=586) | hard_negative 0.0% (n=14), unknown_speech 0.6% (n=177), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| whisper-small/text · torgo_words | 0.2 | -inf | 0.667 | 45.5% (n=1512) | unknown_speech 0.0% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |
| whisper-small/text · torgo_words · balanced | 0.2 | 0.667 | 0.000 | 67.7% (n=1512) | unknown_speech 4.4% (n=90), non_speech 0.0% (n=122), silence 0.0% (n=122) | yes |

## Which model wins and why (measured numbers only)

**nemotron-streaming-0.6b/forced_norm** had the highest calib objective among configurations meeting the calib false-accept target. On test real recordings it accepted the correct phrase for 107/156 (68.6%) of positives, wrongly accepted 7/156 (4.5%), and false-accepted hard negative 2/31 (6.5%), unknown speech 1/430 (0.2%), non speech 1/191 (0.5%), silence 2/4 (50.0%).

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
