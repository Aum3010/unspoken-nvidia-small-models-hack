# Environment

Generated 2026-10-17 18:47:57 UTC

## Machine

- Host OS: Linux-6.8.0-90-generic-x86_64-with-glibc2.35
- Python: 3.10.12 (/workspace/venv-asr/bin/python)
- CPU: x86_64 · 16 logical cores
- RAM: 152.1 GB
- Disk free at data dir: 634 GB of 765 GB
- Git commit: 
- HF_HOME: /workspace/hf · HF_TOKEN set: no

## GPU

- torch 2.11.0+cu128, CUDA runtime 12.8, cuDNN 91900
- cuda:0 NVIDIA RTX PRO 6000 Blackwell Server Edition · 102.0 GB · compute capability 12.0
- nvidia-smi: NVIDIA RTX PRO 6000 Blackwell Server Edition, 580.126.09, 97887 MiB

```
Sat Oct 17 18:47:58 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX PRO 6000 Blac...    On  |   00000000:01:00.0 Off |                    0 |
| N/A   41C    P0            104W /  600W |   19802MiB /  97887MiB |      7%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A          145980    C+G   .venv/bin/python                       2183MiB |
|    0   N/A  N/A          150205    C+G   .venv/bin/python                       2183MiB |
|    0   N/A  N/A          153781    C+G   .venv/bin/python                       2183MiB |
|    0   N/A  N/A          153920    C+G   .venv/bin/python                       2183MiB |
|    0   N/A  N/A          154247    C+G   .venv/bin/python                       2183MiB |
|    0   N/A  N/A          155363    C+G   .venv/bin/python                       2183MiB |
|    0   N/A  N/A          172570    C+G   .venv/bin/python                       2183MiB |
|    0   N/A  N/A          176271    C+G   .venv/bin/python                       2115MiB |
|    0   N/A  N/A          180035    C+G   .venv/bin/python                       2183MiB |
+-----------------------------------------------------------------------------------------+
```

## Packages

| Package | Version |
|---|---|
| torch | 2.11.0+cu128 |
| torchaudio | 2.11.0+cu128 |
| transformers | 5.20.1 |
| nemo_toolkit | 3.0.0 |
| faster-whisper | 1.2.1 |
| ctranslate2 | 4.8.3 |
| silero-vad | 6.2.3 |
| librosa | 0.11.0 |
| soundfile | 0.14.0 |
| rapidfuzz | 3.14.5 |
| pronouncing | 0.3.0 |
| jiwer | 4.0.0 |
| pandas | 2.3.3 |
| numpy | 2.2.6 |
| scipy | 1.15.3 |
| kokoro | 0.9.4 |
| sherpa-onnx | 1.13.8 |

Full pin list: `requirements.lock` (present).

## Datasets

| Dataset | Source | Licence | Notes |
|---|---|---|---|
| torgo | https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html | Academic / non-profit research use; cite Rudzicz, Namasivayam & Wolff (2012), LRE 46(4) | already downloaded |
| speech_commands | https://www.tensorflow.org/datasets/catalog/speech_commands | CC BY 4.0 | already downloaded |
| librispeech | https://www.openslr.org/12/ | CC BY 4.0 | already downloaded |
| esc50 | https://github.com/karolpiczak/ESC-50 | CC BY-NC 3.0 | already downloaded |
| musan | https://www.openslr.org/17/ | CC BY 4.0 (per OpenSLR page; individual files carry their own attributions) | 934 files extracted 2026-10-17 |
