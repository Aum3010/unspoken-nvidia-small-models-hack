# 20 worst errors (nemotron-streaming-0.6b/forced_norm, test split)

Most confident first: false accepts, then wrong accepts, then missed positives.

| Kind | Path | Source / condition | Truth | Decision | Score | Transcript |
|---|---|---|---|---|---|---|
| false accept | `raw/torgo/MC01/Session3/wav_arrayMic/0175.wav` | torgo / clean | (unknown_speech) lip | left | -0.094 | Left. |
| wrong accept | `degraded/torgo/raw__torgo__F03__Session3__wav_arrayMic/reverb_rt60_0.6s/0145.wav` | torgo / reverb_rt60_0.6s | left | right | -0.115 | Right. |
| wrong accept | `degraded/own/own/reverb_rt60_0.6s/p01_normal_hot_1.wav` | own / reverb_rt60_0.6s | hot | tired | -0.663 | I'm too much. |
| wrong accept | `own/p01_whisper_breathe_1.wav` | own / whisper | breathe | suction | -0.857 | I need a suction. |
| wrong accept | `degraded/own/own/tempo_0.7x/p01_normal_tired_2.wav` | own / tempo_0.7x | tired | scared | -0.961 | I'm |
| wrong accept | `degraded/own/own/reverb_rt60_0.6s/p01_normal_pain_2.wav` | own / reverb_rt60_0.6s | pain | scared | -1.214 | And pay. |
| wrong accept | `degraded/own/own/tempo_0.7x/p01_normal_time_2.wav` | own / tempo_0.7x | time | tired | -1.379 | No time. |
| wrong accept | `own/p01_noisy_sick_1.wav` | own / noisy | sick | scared | -1.459 | And so. |
| false accept | `own/p01_noisy_neg_hn_time_close_1.wav` | own / noisy | (hard_negative) what time does it close | time | -1.527 | Time does it close? |
| wrong accept | `own/p01_whisper_hot_1.wav` | own / whisper | hot | tired | -1.527 | I told. |
| wrong accept | `degraded/own/own/noise_musan_snr0/p01_normal_help_2.wav` | own / noise_musan_snr0 | help | scared | -1.546 | How to |
| false accept | `own/p03_normal_neg_silence_1.wav` | own / normal | (silence) nan | no | -1.612 | I |
| wrong accept | `own/p01_whisper_suction_1.wav` | own / whisper | suction | cold | -1.685 | I am gone. |
| wrong accept | `degraded/own/own/reverb_rt60_0.6s/p01_normal_nurse_1.wav` | own / reverb_rt60_0.6s | nurse | tired | -1.689 | Alt. |
| wrong accept | `own/p01_whisper_light_1.wav` | own / whisper | light | tired | -1.794 | What time? |
| wrong accept | `degraded/own/own/reverb_rt60_0.3s/p01_normal_time_2.wav` | own / reverb_rt60_0.3s | time | tired | -1.889 | What time? |
| false accept | `own/p01_whisper_neg_hn_time_close_1.wav` | own / whisper | (hard_negative) what time does it close | time | -1.901 | What time does it |
| wrong accept | `own/p01_noisy_pain_2.wav` | own / noisy | pain | tired | -1.901 | M ten. |
| wrong accept | `own/p01_noisy_suction_1.wav` | own / noisy | suction | scared | -1.957 | And it's |
| wrong accept | `degraded/own/own/reverb_rt60_0.6s/p01_normal_time_2.wav` | own / reverb_rt60_0.6s | time | tired | -2.036 | What time? |
