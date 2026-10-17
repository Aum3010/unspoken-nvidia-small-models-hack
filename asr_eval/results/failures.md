# 20 worst errors (parakeet-ctc-0.6b/ctc_sum, test split)

Most confident first: false accepts, then wrong accepts, then missed positives.

| Kind | Path | Source / condition | Truth | Decision | Score | Transcript |
|---|---|---|---|---|---|---|
| wrong accept | `degraded/torgo/raw__torgo__M01__Session1__wav_arrayMic/gain_-36dB/0086.wav` | torgo / gain_-36dB | down | no | -0.002 | no |
| false accept | `raw/torgo/MC01/Session2/wav_arrayMic/0200.wav` | torgo / clean | (unknown_speech) ride | right | -2.756 | ride |
| false accept | `raw/torgo/M05/Session2/wav_headMic/0147.wav` | torgo / clean | (unknown_speech) alfalfa is healthy for you | help | -30.546 | i sounds like it helpalthy for you |
| missed (abstained) | `degraded/torgo/raw__torgo__MC01__Session2__wav_arrayMic/reverb_rt60_0.6s/0264.wav` | torgo / reverb_rt60_0.6s | three | unknown | -0.000 | three |
| missed (abstained) | `degraded/torgo/raw__torgo__MC01__Session2__wav_arrayMic/noise_musan_snr20/0264.wav` | torgo / noise_musan_snr20 | three | unknown | -0.000 | three |
| missed (abstained) | `degraded/torgo/raw__torgo__MC04__Session1__wav_arrayMic/gain_-24dB/0487.wav` | torgo / gain_-24dB | two | unknown | -0.000 | two |
| missed (abstained) | `degraded/torgo/raw__torgo__MC01__Session3__wav_arrayMic/reverb_rt60_0.3s/0061.wav` | torgo / reverb_rt60_0.3s | two | unknown | -0.000 | two |
| missed (abstained) | `degraded/torgo/raw__torgo__MC04__Session1__wav_arrayMic/gain_-12dB/0487.wav` | torgo / gain_-12dB | two | unknown | -0.000 | two |
| missed (abstained) | `degraded/torgo/raw__torgo__MC01__Session2__wav_arrayMic/gain_-12dB/0101.wav` | torgo / gain_-12dB | two | unknown | -0.000 | two |
| missed (abstained) | `degraded/torgo/raw__torgo__MC04__Session1__wav_arrayMic/lowpass_1500/0230.wav` | torgo / lowpass_1500 | three | unknown | -0.000 | three |
| missed (abstained) | `degraded/torgo/raw__torgo__MC01__Session2__wav_arrayMic/noise_musan_snr20/0101.wav` | torgo / noise_musan_snr20 | two | unknown | -0.000 | two |
| missed (abstained) | `degraded/torgo/raw__torgo__MC01__Session2__wav_arrayMic/noise_musan_snr10/0101.wav` | torgo / noise_musan_snr10 | two | unknown | -0.000 | two |
| missed (abstained) | `raw/torgo/MC01/Session2/wav_arrayMic/0101.wav` | torgo / clean | two | unknown | -0.000 | two |
| missed (abstained) | `degraded/torgo/raw__torgo__MC04__Session1__wav_arrayMic/bandpass_300-3400/0487.wav` | torgo / bandpass_300-3400 | two | unknown | -0.000 | two |
| missed (abstained) | `degraded/torgo/raw__torgo__MC04__Session1__wav_arrayMic/noise_musan_snr10/0102.wav` | torgo / noise_musan_snr10 | two | unknown | -0.000 | two |
| missed (abstained) | `degraded/torgo/raw__torgo__MC04__Session1__wav_arrayMic/noise_musan_snr5/0487.wav` | torgo / noise_musan_snr5 | two | unknown | -0.000 | two |
| missed (abstained) | `degraded/torgo/raw__torgo__MC04__Session1__wav_arrayMic/gain_-36dB/0102.wav` | torgo / gain_-36dB | two | unknown | -0.000 | two |
| missed (abstained) | `degraded/torgo/raw__torgo__MC04__Session1__wav_arrayMic/gain_-24dB/0524.wav` | torgo / gain_-24dB | three | unknown | -0.000 | three |
| missed (abstained) | `degraded/torgo/raw__torgo__MC04__Session1__wav_arrayMic/noise_musan_snr10/0487.wav` | torgo / noise_musan_snr10 | two | unknown | -0.000 | two |
| missed (abstained) | `degraded/torgo/raw__torgo__MC04__Session1__wav_arrayMic/noise_musan_snr0/0102.wav` | torgo / noise_musan_snr0 | two | unknown | -0.000 | two |
