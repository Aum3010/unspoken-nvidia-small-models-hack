# 20 worst errors (nemotron-streaming-0.6b/text, test split)

Most confident first: false accepts, then wrong accepts, then missed positives.

| Kind | Path | Source / condition | Truth | Decision | Score | Transcript |
|---|---|---|---|---|---|---|
| false accept | `raw/torgo/MC01/Session3/wav_arrayMic/0175.wav` | torgo / clean | (unknown_speech) lip | left | 1.000 | Left. |
| wrong accept | `degraded/torgo/raw__torgo__M05__Session1__wav_arrayMic/noise_musan_snr10/0032.wav` | torgo / noise_musan_snr10 | left | yes | 1.000 | Yes. |
| wrong accept | `degraded/torgo/raw__torgo__M05__Session1__wav_arrayMic/gain_-12dB/0032.wav` | torgo / gain_-12dB | left | yes | 1.000 | Yes. |
| wrong accept | `degraded/torgo/raw__torgo__F03__Session3__wav_arrayMic/reverb_rt60_0.6s/0145.wav` | torgo / reverb_rt60_0.6s | left | right | 1.000 | Right. |
| wrong accept | `raw/torgo/M05/Session1/wav_arrayMic/0032.wav` | torgo / clean | left | yes | 1.000 | Yes. |
| wrong accept | `degraded/torgo/raw__torgo__M05__Session1__wav_arrayMic/lowpass_1500/0032.wav` | torgo / lowpass_1500 | left | yes | 1.000 | Yes. |
| wrong accept | `degraded/torgo/raw__torgo__M05__Session1__wav_arrayMic/bandpass_300-3400/0032.wav` | torgo / bandpass_300-3400 | left | yes | 1.000 | Yes. |
| wrong accept | `degraded/torgo/raw__torgo__M05__Session1__wav_arrayMic/tempo_0.7x/0032.wav` | torgo / tempo_0.7x | left | yes | 1.000 | Yes. |
| wrong accept | `own/p01_whisper_breathe_1.wav` | own / whisper | breathe | suction | 0.909 | I need a suction. |
| false accept | `own/p01_whisper_neg_hn_time_close_1.wav` | own / whisper | (hard_negative) what time does it close | time | 0.818 | What time does it |
| wrong accept | `degraded/own/own/reverb_rt60_0.6s/p01_normal_cold_2.wav` | own / reverb_rt60_0.6s | cold | no | 0.667 | Unco. |
| wrong accept | `degraded/own/own/tempo_0.7x/p01_normal_nurse_1.wav` | own / tempo_0.7x | nurse | no | 0.667 | Oh no. |
| wrong accept | `degraded/own/own/tempo_0.7x/p03_normal_stop_2.wav` | own / tempo_0.7x | stop | yes | 0.667 | Yep. |
| missed (abstained) | `degraded/torgo/raw__torgo__MC04__Session1__wav_arrayMic/reverb_rt60_0.3s/0481.wav` | torgo / reverb_rt60_0.3s | up | unknown | 1.000 | Up. |
| missed (abstained) | `degraded/torgo/raw__torgo__MC04__Session2__wav_arrayMic/reverb_rt60_0.3s/0770.wav` | torgo / reverb_rt60_0.3s | three | no_speech | 1.000 | Three. |
| missed (abstained) | `degraded/torgo/raw__torgo__MC04__Session1__wav_arrayMic/tempo_0.85x/0481.wav` | torgo / tempo_0.85x | up | unknown | 1.000 | Up. |
| missed (abstained) | `degraded/torgo/raw__torgo__MC04__Session2__wav_arrayMic/reverb_rt60_0.6s/0725.wav` | torgo / reverb_rt60_0.6s | two | no_speech | 1.000 | Two. |
| missed (abstained) | `degraded/torgo/raw__torgo__MC04__Session2__wav_arrayMic/tempo_0.7x/0725.wav` | torgo / tempo_0.7x | two | no_speech | 1.000 | Two. |
| missed (abstained) | `degraded/torgo/raw__torgo__MC04__Session2__wav_arrayMic/noise_musan_snr5/0725.wav` | torgo / noise_musan_snr5 | two | no_speech | 1.000 | Two. |
| missed (abstained) | `degraded/torgo/raw__torgo__MC04__Session2__wav_arrayMic/gain_-24dB/0725.wav` | torgo / gain_-24dB | two | no_speech | 1.000 | Two. |
