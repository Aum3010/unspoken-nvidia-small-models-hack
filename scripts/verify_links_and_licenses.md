# Link & licence verification

Checked 2026-10-16 with `curl -L` (browser UA). Every source link in plan.md was checked; status below.

| HTTP | URL |
|---|---|
| 200 | https://arxiv.org/abs/2111.01740 |
| 200 | https://arxiv.org/abs/2205.05474 |
| 200 | https://arxiv.org/abs/2509.25275 |
| 200 | https://arxiv.org/pdf/2509.25275 |
| 200 | https://bnci-horizon-2020.eu/database/data-sets |
| 200 | https://brainflow.readthedocs.io/en/stable/ |
| 200 | https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker |
| 200 | https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker |
| 200 | https://docs.vllm.ai/en/latest/ |
| 200 | https://github.com/Isti23/bigp3bci-eeg-bci |
| 200 | https://github.com/NVIDIA-NeMo/gym |
| 200 | https://github.com/Rikorose/DeepFilterNet |
| 200 | https://github.com/SYSTRAN/faster-whisper |
| 200 | https://github.com/amanvirparhar/chaplin |
| 200 | https://github.com/amanvirparhar/chaplin.git |
| 200 | https://github.com/brainflow-dev/brainflow |
| 200 | https://github.com/brownhci/webgazer |
| 200 | https://github.com/facebookresearch/av_hubert |
| 200 | https://github.com/facebookresearch/brain2qwerty |
| 200 | https://github.com/google-ai-edge/mediapipe |
| 200 | https://github.com/hexgrad/kokoro |
| 200 | https://github.com/jeremyipark/vision-demos |
| 200 | https://github.com/loganngarcia/chaplin-ui |
| 200 | https://github.com/loganngarcia/chaplin-ui.git |
| 200 | https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages |
| 200 | https://github.com/mpc001/auto_avsr |
| 200 | https://github.com/mpc001/auto_avsr/tree/main/tutorials |
| 200 | https://github.com/openai/whisper |
| 200 | https://github.com/rhasspy/piper |
| 200 | https://github.com/weichkai/p300-speller-bci |
| 200 | https://github.com/weichkai/p300-speller-bci.git |
| 200 | https://github.com/xiph/rnnoise |
| 200 | https://github.com/yochaiye/LipVoicer |
| 200 | https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 |
| 200 | https://huggingface.co/hexgrad/Kokoro-82M |
| 200 | https://huggingface.co/rhasspy/piper-voices |
| 200 | https://lipvoicer.github.io/ |
| 200 | https://mne.tools/stable/index.html |
| 200 | https://moabb.neurotechx.com/ |
| 200 | https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_008.html |
| 200 | https://mozilladatacollective.com/datasets/cmpe4rs9800r2nv07qpq8972c |
| 200 | https://physionet.org/content/bigp3bci/1.0.0/ |
| 200 | https://physionet.org/files/bigp3bci/1.0.0/ |
| 200 | https://pmc.ncbi.nlm.nih.gov/articles/PMC9844948/ |
| 200 | https://talkbank.org/aphasia/ |
| 200 | https://talkbank.org/aphasia/GR/ |
| 200 | https://voicebridgedemo.github.io/ |
| 200 | https://webgazer.cs.brown.edu/ |
| 200 | https://www.nature.com/articles/s41467-025-68228-9 |
| 200 | https://www.nature.com/articles/s41593-026-02446-2 |
| 200 | https://www.nemar.org/dataset/nm000169?v=v1.0.1 |
| 200 | https://youtu.be/_famGVaem-8 |
| 403 | https://zenodo.org/records/3625687 |

Zenodo returns 403 to scripted page requests, but its API and file downloads work (that is how GRID was fetched: `https://zenodo.org/api/records/3625687`, licence CC BY 4.0).

## Licences confirmed from the artefacts themselves

- Chaplin: `third_party/chaplin/LICENSE`: MIT
- p300-speller-bci: `LICENSE`: MIT
- WebGazer `package.json`: GPL-3.0-or-later (used unmodified, loaded as a separate script)
- @mediapipe/tasks-vision `package.json`: Apache-2.0
- BNCI2014-008: CC BY-NC-ND 4.0 (dataset description + MOABB page); not redistributed
- GRID: CC BY 4.0 (Zenodo record metadata)
- Nemotron 3.5 Lightning: OpenMDW-1.1 (NVIDIA model card)
- Kokoro-82M: Apache-2.0 (model card)
