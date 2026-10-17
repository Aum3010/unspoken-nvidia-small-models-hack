


# Unspoken — Technical Project Plan and Sourcebook

> **A voice for those who cannot speak.**  
> Adaptive, multimodal, patient-controlled communication using specialized small models for visual speech, residual audio, gaze/gestures, and recorded EEG, coordinated through a shared uncertainty and confirmation protocol.
>
> **Status:** design and source-research document; the model stack has **not** been demonstrated as one integrated system.  
> **Deployment target:** local inference and/or NVIDIA Brev, with the event's **≤33B total model-parameter** constraint.  

## 1. The problem, users, and product promise

Some patients retain awareness and the desire to communicate but cannot use ordinary speech. Relevant situations include intubation, neurological disease, severe dysarthria, aphonia, and motor impairments. **Different patients retain different abilities**: mouth movement, weak audible speech, gaze, head/hand movement, or intentional attention reflected in EEG signals.

Unspoken is not a mind reader and is not a medical diagnosis tool. It is an **assistive communication prototype**: find an accessible input, extract tentative communication candidates, make uncertainty visible, let the person clarify and explicitly confirm the intended message, and only then produce speech.

**One-line pitch:** *When speech disappears, communication shouldn't. Unspoken combines specialist AI models across lips, voice, eyes, and brain-signal research to help people express what they mean.*

**Product invariant:** **No unverified interpretation is spoken as the patient's intention.** A large model must never quietly convert ambiguous visual evidence into an asserted patient statement.

### User journeys

1. **Silent-mouth mode (live):** Person mouths a sentence into a webcam. Visual speech recognition proposes candidates; the user confirms or corrects via gaze or other accessible controls; confirmed text is spoken aloud.
2. **Residual-voice mode (live):** Person produces weak but usable speech. An audio enhancer and ASR extract tentative words; uncertain words are highlighted; the user confirms the final message.
3. **Eye-selection mode (live):** Person cannot articulate a phrase. A gaze-calibrated board allows large-target selection, yes/no answers, or spelling; a small coordinating model chooses the smallest useful *question*, not the patient's answer.
4. **Brain-signal research mode (recorded EEG only):** Run a real EEG classification pipeline against prerecorded P300 data, visualize stimulus-synchronized signal responses and intended selections, and route **decoded, explicitly marked replay results** into the same communication UI. **No live brain input is possible without EEG acquisition hardware.**

These modes are alternatives and can be combined where feasible. They are **not** equally available to all patients.

## 2. Scope and deliverables

### Core functions

- A polished browser front end: camera view, mouth crop and tracking overlays, candidate messages, a gaze-accessible confirmation panel, conversation history, and speech output.
- Backend service(s) for video-only speech recognition, audio enhancement/transcription, EEG replay, and a small instruction model coordinating **typed** tools.
- Parallel or event-driven specialist workers feeding standardized hypotheses into one arbitration/confirmation protocol.
- Honest source/mode badges: `LIVE CAMERA`, `LIVE MICROPHONE`, `RECORDED EEG`, `RESEARCH VIDEO`.
- A deterministic state machine that prevents TTS until explicit user confirmation.
- Fully working fallback: gaze/large-button communication should function if lip reading fails.
- Benchmarks with separate baselines and coordinated-system results, plus saved failure examples.
- For tracks purposes use these wheerverlanguage models or inference purposes, i have mac m5 max so run on metal versiion , i have 128 gb ram on my machine
#### Additional goal
​Small. Sharp. Specialized

​Build AI that punches above its weight—specialized small models, multi-agent systems all running on NVIDIA Brev or locally.



Multi-Agent Systems
​Focus: Many small models beating one big one.

​The Goal: Wire several small models together so they coordinate—dividing up work, passing context back and forth, building on each other's output—to accomplish something no single model could alone.

Primary competition: Track 2 — Multi-Agent Systems, focusing on heterogeneous models collaborating across vision, audio, gaze and EEG.



NVIDIA Nemotron 3.5 Lightning: Designated as the primary coordination and tool-calling model, with links to official checkpoints and deployment examples.



NVIDIA NeMo Gym bounty: A custom evaluation environment for communication recovery, agent coordination, tool failures and confirmation safety.



Red Hat vLLM bounty: Model serving architecture, deployment of our thing for fast latency.

### Deliberate exclusions

- Not a substitute for a nurse call system or medical emergency alarm.
- Not autonomously selecting or sending medical instructions.
- Not diagnosing pain, emotional state, neurological disease, or capacity to consent.
- Not restoring audio from a person who produces no sound: audio restoration requires a usable signal.
- Not claiming arbitrary thoughts can be reconstructed from non-invasive EEG.
- Not claiming real patient accuracy based on healthy-volunteer benchmark results.
- No face/voice cloning or unconsented use of identifiable patient footage.

## 3. High-level system design

```text
                  Unspoken communication UI
              ┌─────────────┴──────────────┐
          Live camera/mic              Research replay
              │                            │
      ┌───────┼──────────┐                 │
      │       │          │                 │
   Face &   Audio     Eye gaze        EEG recording +
   mouth    agent     / gesture       stimulus markers
   tracker    │       specialists           │
      │       │          │                   │
  Visual-  Enhancement  Selection        P300 classifier
  speech    + ASR       events                │
   model      │          │                   │
      └───────┴──────────┴───────────────────┘
                      │
              Standardized evidence
                 and hypotheses
                      │
           Small coordination model
           + deterministic policy
                      │
         Ambiguous? → Ask via accessible
          modality → user correction
                      │
            EXPLICIT CONFIRMATION
                      │
              Text → local TTS
```

**The valuable multi-agent behavior is active uncertainty resolution**, not just multiple models voting. Example: mouth decoder returns two plausible messages; coordinator notices conflict; gaze specialist offers an accessible binary choice; person selects one; system verifies that the choice maps to the presented candidate; TTS reads precisely the confirmed message.

### Architectural rules

- **Shared backbone is allowed:** Multiple task-specific agent roles may call the same 3–4B instruction model. Distinct physical modalities can use different small neural models. Count actual distinct deployed model weights conservatively under event rules; *quantization does not reduce parameter count*.
- **Keep deterministic responsibilities deterministic:** gaze dwell thresholds, timestamp alignment, spelling-board selection, state transitions, exact confirmation and TTS authorization are code, not LLM judgment.
- **Do not silently overwrite perception:** Retain `raw_vsr`, `raw_asr`, enhancement metadata, all user edits, and final confirmation independently.
- **No forced consensus:** If modalities disagree, the result is `ambiguous`, `unknown`, or `needs_clarification`, not a fabricated unified sentence.
- **Graceful degradation:** If camera, audio, or backend drops, preserve an accessible communication board and an immediately usable manual alternative.
- **Replay transparency:** Recorded EEG and clinical research video are labeled as such throughout the presentation and logs.

## 4. Shared typed interface and policy engine

Suggested Python/Pydantic/TypeScript contract:

```json
{
  "event_id": "uuid",
  "session_id": "demo-session",
  "modality": "lip|audio|gaze|gesture|eeg_replay",
  "source_mode": "live|prerecorded",
  "capture_time_ms": 0,
  "source_ref": "camera-clip-42 | physionet-session-path",
  "hypotheses": [
    {"text": "I need water", "score": 0.61},
    {"text": "I need a nurse", "score": 0.31}
  ],
  "quality_flags": ["occlusion", "low_signal", "out_of_distribution"],
  "provenance": {"checkpoint": "model/version", "preprocess": "pipeline/version"},
  "status": "proposed|ambiguous|confirmed|rejected"
}
```

`score` is a raw model score unless separately calibrated; **do not call it a probability or 61% certainty without calibration**. For gaze, use `selected_option_id`, dwell evidence, calibration quality and a human confirmation, rather than pretending gaze coordinates are language hypotheses. For EEG, preserve trial/stimulus IDs, epoch boundaries and target labels; reconstructed character selections are outcomes, not claims of free-form thought-reading.

The coordination model receives structured summaries, not full videos or raw EEG tensors. Its only allowed tool actions should be narrowly typed:

```text
propose_candidates(hypotheses, source_refs)
ask_confirmation(candidate_ids, input_method)
switch_input_method(method)
request_retake(reason)
append_confirmed_text(confirmation_event_id)
speak_confirmed_text(confirmation_event_id)
reset_or_undo(confirmed_segment_id)
```

**Policy gate (must be enforceable outside the model):**

```python
def speak_confirmed_text(message, audit_log):
    assert message.status == "confirmed"
    assert message.confirmation_method in {
        "gaze_dwell_confirmed", "deliberate_blink_confirmed",
        "accessible_button_confirmed", "keyboard_confirmed",
        "other_explicit_user_confirmed"
    }
    assert audit_log.includes(message.confirmation_event_id)
    return tts.synthesize(message.confirmed_text)
```

Important nuance: **a recorded P300 ground-truth label is not new, live patient consent.** Research replay output can be read aloud as an explicitly identified benchmark result, not presented as a currently speaking patient's message.

## 5. Modality A — Visual speech / lip movement

### First-choice runnable stack

1. **Chaplin** — [source repository](https://github.com/amanvirparhar/chaplin). An existing local silent-speech pipeline based on Auto-AVSR, with webcam capture and LLM text refinement. It documents a `setup.sh` to download checkpoints and a local Ollama backend. **Code to adapt:** camera recording, video preprocessing, VSR model wrapper, invocation and raw prediction capture.
2. **Chaplin-UI** — [web interface repository](https://github.com/loganngarcia/chaplin-ui). A community implementation exposing local recording/upload and transcription through a browser/FastAPI UI. **Code to adapt:** browser-to-Python video flow, stream/upload controls, backend endpoints. Use it as scaffolding rather than directly adopting its entire UI or inference policy.
3. **Auto-AVSR** — [primary framework](https://github.com/mpc001/auto_avsr); [mouth-crop and inference tutorials](https://github.com/mpc001/auto_avsr/tree/main/tutorials); [multilingual AVSR framework](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages). Auto-AVSR provides checkpoints and visual-speech feature/inference utilities. Its documented LRS3 VSR base checkpoint is **250M parameters** with **20.3% benchmark word error rate** under the listed training setup; *this is not an expected real-world, unfamiliar-webcam error rate*. Weights may carry separate training-dataset restrictions.
4. **AV-HuBERT** — [official research code](https://github.com/facebookresearch/av_hubert). Useful alternative for audio-visual speech representation and pretrained models. **Archived/read-only repository**; expect legacy fairseq/PyTorch dependency work. Use as a research fallback rather than assuming it is the quickest live webcam integration.
5. **LipVoicer** — [code](https://github.com/yochaiye/LipVoicer), [paper/demo](https://lipvoicer.github.io/). A distinct research system that **generates audio from silent video**, using lip-reading guidance and a diffusion/generative pipeline. It provides `download_checkpoints.py` and `inference_real_video.py`. Treat as an **optional media-generation experiment**, not the primary low-latency confirmation path and not verified for clinical voices.

**Fastest grounded starting point:** Chaplin's real webcam VSR and raw-prediction output; then separate its language-model polishing from the verified final message so a language model cannot secretly replace the patient's intended words.

### Pipeline

```text
webcam frames → face detection & mouth ROI tracking
→ standardized frame-rate/crop buffer
→ pretrained VSR → n-best decoded text
→ ambiguity / OOD tests → candidate UI → user confirmation
```

Capture requirements: normalized face angle, crop visibility score, timestamps, consistent fps according to the selected model, and a retake flow for off-screen face/occlusion. Do not describe chunked utterance processing as instantaneous token-level live lip-reading unless tested. Generic visual speech is intrinsically ambiguous (many phonemes have similar lip movements).

### Source inspirations, not drop-in lip readers

- [Jeremy Park's vision-demos](https://github.com/jeremyipark/vision-demos) — Apache-2.0 code illustrating pose/segmentation → temporal inference → user-facing feedback for sports and motion analysis. **Not itself a lip-reader, gaze tracker, or patient communication model.** The repository uses VLM Run-hosted models, so running those demos unchanged may violate the local/Brev inference intention. Reuse interaction and analysis patterns, not its cloud inference dependency.
- [MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker) — face and mouth landmarks, geometry, crop robustness; landmarks **do not decode words**.

## 6. Modality B — Residual/weak speech

### Models and code

- **Whisper Tiny/Base** — [official repository](https://github.com/openai/whisper). `tiny` ≈ **39M** and `base` ≈ **74M** parameters. Use a local model to transcribe any actually audible speech. The official repo documents `pip install -U openai-whisper` and `whisper audio.wav --model base`.
- **faster-whisper** — [optimized inference implementation](https://github.com/SYSTRAN/faster-whisper). CTranslate2 acceleration, int8/fp16 options, practical streaming/chunking wrappers. Same purpose: speech recognition, not restoration.
- **DeepFilterNet** — [code and pretrained denoiser](https://github.com/Rikorose/DeepFilterNet), [paper](https://arxiv.org/abs/2205.05474). Practical denoising to test before Whisper. The repo documents the `deepFilter` command on WAV input; also provides Python integration.
- **RNNoise** — [official code](https://github.com/xiph/rnnoise). Small RNN-based real-time noise suppression. A lightweight alternative if denoising matters more than general generative restoration.
- **VoiceBridge: General Speech Restoration with One-step Latent Bridge Models** — [paper PDF provided by the user](https://arxiv.org/pdf/2509.25275), [abstract](https://arxiv.org/abs/2509.25275), [author demo](https://voicebridgedemo.github.io/). Relevant research for restoring **degraded existing audio** at high fidelity (noise/reverberation/bandwidth). **I did not verify an official ready-to-run public code/checkpoint release**; do not make it a prerequisite. It is **not** an EEG decoder, lip-to-speech decoder, or a proven dysarthria treatment.

### Audio path

```text
mic / recorded low-volume speech
→ VAD + audio quality estimate
→ unprocessed Whisper + enhanced-audio Whisper (if enhancer helps)
→ compare evidence/word spans
→ mark disagreement → confirmation UI
```

Audio enhancement may alter phonetic content or hallucinate plausible-sounding detail. Show original-versus-enhanced audio and original-versus-enhanced ASR when useful. Never infer words that are absent from all usable input. For severe dysarthria, do not assume pretrained Whisper achieves acceptable accuracy without specific evaluation.

## 7. Modality C — Eyes, head, and gestures

### Building blocks

- **WebGazer.js** — [source](https://github.com/brownhci/webgazer), [project and demo](https://webgazer.cs.brown.edu/). Browser-side webcam gaze prediction; requires user calibration and quality checks. **This** is the practical gaze-selection reference, not raw MediaPipe landmarks alone.
- **MediaPipe Face Landmarker** — [documentation](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker). Face landmark, eyelid and face-pose signals; suitable for blinks, simple head gestures and crop-quality heuristics after testing.
- **MediaPipe Hand Landmarker** — [documentation](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker). For optional deliberate hand gestures. **Not usable for people who cannot move their hands.**
- **MediaPipe repository** — [source](https://github.com/google-ai-edge/mediapipe).

**Recommended interaction:** three–six large choices, visible dwell progress, explicit 'confirm', 'undo', 'none of these', and 'help' options. Avoid accidentally interpreting ordinary looking-around as consent. Track calibration confidence and suspend gaze selection when it is poor. Offer alternative head gesture, blink, keyboard, and/or large-button input.

**Implementation detail:** WebGazer returns estimated screen coordinates, not clinical-grade eye intention; use hysteresis, dwell timers, debouncing, and a deliberately separate confirmation action. Standard MediaPipe Face Landmarker is **not a calibrated screen-gaze estimator**. A deliberate blink recognizer also needs calibration and false-positive evaluation.

## 8. Modality D — Brain signals without EEG hardware

**Honest label:** `EEG RESEARCH REPLAY — previously recorded signal, newly run inference`.

**What is possible:** train/run a P300 target/non-target classifier on recorded EEG epochs, reconstruct the participant's selection under the original visual-stimulus protocol, and show the result feeding the shared communication UI. The inference can execute live **on prerecorded data**; neural signal acquisition is not live.

**What is not possible:** a judge invents an arbitrary thought and the system reads it without electrodes. P300 is an attention-based selection task with time-locked flashes, **not spontaneous imagined-speech decoding**.

### Code and datasets

1. **P300 Speller BCI** — [repository](https://github.com/weichkai/p300-speller-bci). Includes `examples/run_simulation.py`, an offline classifier, speller matrix visualization, sample files, and MOABB benchmark utilities. The **live acquisition script requires external EEG hardware**; the simulation does not. This is the simplest starting repo to extract a real replay interface.
2. **bigP3BCI** — [PhysioNet dataset](https://physionet.org/content/bigp3bci/1.0.0/), [dataset folder](https://physionet.org/files/bigp3bci/1.0.0/). Includes a diverse P300 participant collection, including participants with ALS and some synchronized eye-tracking recordings. **CC BY 4.0, public access**, but large (~44.6 GB uncompressed; ~10.3 GB ZIP). Select a small permitted subset/session rather than downloading everything indiscriminately. Read the dataset's event markers, stimulus protocol, and per-session metadata before building your decoder.
3. **EEG-Conformer P300 decoder** — [research implementation](https://github.com/Isti23/bigp3bci-eeg-bci). More ambitious sequence model for the bigP3BCI dataset, including a cross-session experiment. **Reported repository performance is not an expected result on your held-out session**; inspect splits and preprocessing before reuse.
4. **BNCI2014-008 (ALS P300 speller)** — [MOABB dataset page](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_008.html), [original catalog](https://bnci-horizon-2020.eu/database/data-sets), [NEMAR mirror/description](https://www.nemar.org/dataset/nm000169?v=v1.0.1). Eight ALS participants; eight EEG channels; usable for a focused clinical-context replay. **CC BY-NC-ND 4.0**; be especially careful about redistributing modified data/media or including raw recordings in an open repository.
5. **MOABB** — [documentation](https://moabb.neurotechx.com/): benchmark loaders/splits for EEG datasets. **MNE-Python** — [documentation](https://mne.tools/stable/index.html): filtering, epoching, event alignment, EEG visualization.
6. **BrainFlow** — [code](https://github.com/brainflow-dev/brainflow), [documentation](https://brainflow.readthedocs.io/en/stable/). Recording/playback abstraction if a board-like input stream is useful. Dataset files need proper format conversion and timestamp/event preservation; the playback API does not magically create new EEG measurements.
7. **Brain2Qwerty** — [Meta research code](https://github.com/facebookresearch/brain2qwerty). Research on typed sentence decoding from MEG/EEG; **not** a simple consumer-EEG thought-to-speech model. Heavy datasets and research infrastructure; background inspiration rather than selected P300 implementation.

### EEG pipeline

```text
selected participant EEG + timestamps + flash/target metadata
→ bandpass / artifact rejection
→ stimulus-aligned epochs
→ target vs non-target P300 scores
→ accumulate evidence across repeat flashes
→ select among letters / options
→ report held-out target accuracy and confidence quality
→ replay result visibly tagged as recorded study data
```

**Do not pre-fill predicted letters from known labels.** Ground truth may be used to calculate correctness *after* model inference, never to supply the predicted result. Keep original trial boundaries intact; avoid train/test leakage across repeated flashes from the same target trial. Ensure the visualized flashing matrix matches the recording's original stimulus timing and layout; a visually unrelated animation would not constitute EEG decoding.

**Stretch experiment:** For appropriately synchronized bigP3BCI sessions, compare EEG-only, gaze-only, and EEG+gaze selection on aligned recorded samples. Only do this when synchronized eye data and permission are actually present. Do not claim independently recorded webcam gaze and public EEG belong to the same patient/event.

## 9. Coordination and speech output

### Language/coordination model

- **Qwen3-4B-Instruct-2507** — [official model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), Apache-2.0. Suggested main orchestration model for structured tasks, candidate clarification and tool selection. For a locally reproducible alternative, Chaplin documents **Ollama `qwen3:4b`** ([Chaplin repo](https://github.com/amanvirparhar/chaplin)). They are different releases/packaging: pin the exact model used and test tool/JSON behavior.
- **vLLM** — [documentation](https://docs.vllm.ai/en/latest/), useful for serving compatible instruction models on a NVIDIA GPU. You can also run a local Ollama-compatible endpoint for the coordinator if simpler. Ensure the chosen model's chat-template/tool-call behavior is supported; enforce JSON schema validation and retries outside the model.
- **NVIDIA NeMo Gym** — [official repository](https://github.com/NVIDIA-NeMo/gym). Candidate environment for evaluation/verifier tasks, especially uncertainty resolution, tool correctness, and policy compliance. Avoid making it a prerequisite for the communication UI.

### Text-to-speech

- **Kokoro-82M** — [model weights](https://huggingface.co/hexgrad/Kokoro-82M), [official inference code](https://github.com/hexgrad/kokoro). Small (~82M) local TTS; model card lists Apache-2.0. Confirm voicepack and dependency licenses in the chosen packaging.
- **Piper TTS** — [Piper project source](https://github.com/rhasspy/piper) and [voice collection](https://huggingface.co/rhasspy/piper-voices). Local ONNX TTS alternative; check per-voice terms. If a linked repository has moved or archived, use its current maintained successor and record the pinned commit.

**TTS is an output renderer, never the authority on message contents.** No invented empathy, urgency, diagnosis, pronouns, medication instructions, or contextual expansions that were not explicitly verified. Use natural prosody but preserve exact confirmed text.

### Model-budget ledger (illustrative; verify selected files)

| Component | Example choice | Approx. distinct parameters | Notes |
|---|---|---:|---|
| Coordinator | Qwen3-4B-Instruct | 4.0B | Shared across logical agent roles |
| Visual speech | Auto-AVSR base | 0.25B | Pretrained VSR model zoo |
| Audio ASR | Whisper Base | 0.074B | Tiny alternative 0.039B |
| TTS | Kokoro | 0.082B | Confirm actual downloaded model |
| Eye/face/hand inference | WebGazer + MediaPipe | Check shipped weights | Multiple compact models/estimators |
| Audio enhancement | DeepFilterNet or RNNoise | Check checkpoint | Select one primary enhancer |
| P300 decoder | Classifier / EEG-Conformer | Check implementation | Trained or loaded separately |
| Optional LipVoicer | Several additional models | **Not budgeted** | Do not activate until all checkpoint weights counted |
| Optional VoiceBridge paper model | Research only | **Not deployed** | No verified official ready-to-run checkpoint |

The named baseline's **known** parameter subtotal is approximately **4.406B**, excluding unspecified eye/hand, enhancement, and EEG models. This is *not* a complete audited total. Sum every **distinct deployed model**, including auxiliary neural frontends/vocoders, according to event rules; MoE total parameters count, not just active parameters. Quantized weights reduce memory, **not** parameter count. Confirm how organizers treat identical shared weights instantiated in multiple services.

## 10. Data, evaluation and genuine patient evidence

### For engineering tests

| Resource | Link | What it tests | Caveat |
|---|---|---|---|
| LRS3 (via cited model repositories) | [Auto-AVSR](https://github.com/mpc001/auto_avsr) | Visual-speech benchmark | Dataset access/license constraints; broadcast speech ≠ hospital lip movements |
| GRID audiovisual corpus | [Zenodo](https://zenodo.org/records/3625687) | Controlled mouth-video/sentence decoding | Healthy speakers and restricted phrase grammar; verify redistribution rights |
| Mozilla Multilingual Audio Visual Speech | [Mozilla Data Collective](https://mozilladatacollective.com/datasets/cmpe4rs9800r2nv07qpq8972c) | Small aligned lip/audio tests | 100 samples / 16 volunteers, only 15 English; follow non-redistribution terms |
| bigP3BCI | [PhysioNet](https://physionet.org/content/bigp3bci/1.0.0/) | P300 classifier and possibly aligned EEG/gaze | Large dataset; proper split, attribution, stimulation metadata |
| BNCI2014-008 | [MOABB](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_008.html) | ALS participant P300 recordings | Noncommercial/no-derivatives terms; cannot make up unseen thoughts |
| New consensual teammate/judge samples | Locally recorded | Live unfamiliar-person accuracy and UI robustness | Explicit opt-in and opt-out; do not retain by default |

### Real patient research/videos: clinical motivation, **not automatically licensed training data**

- [Personalized One-Shot Lipreading for an ALS Patient](https://arxiv.org/abs/2111.01740) — [research demonstration video](https://youtu.be/_famGVaem-8). Highly relevant to the problem and personalization. Do not assume permission to crop/retrain on patient footage or redistribute it; show a link or obtain explicit reuse permission.
- [Two-stage visual speech recognition for intensive care patients](https://pmc.ncbi.nlm.nih.gov/articles/PMC9844948/) — research on ICU-related phrases with healthy participants; not proof of performance on actual patients.
- [Wearable intelligent throat enables natural speech in stroke patients with dysarthria](https://www.nature.com/articles/s41467-025-68228-9) — authentic stroke-patient research with supplementary videos. **Requires custom throat sensing hardware**; suitable for clinical motivation, not equivalent to webcam input. Verify supplementary-media rights and patient consent wording before embedding clips in a hackathon submission.
- [Simultaneous speech and gesture decoding for multimodal communication in paralysis](https://www.nature.com/articles/s41593-026-02446-2) — supplementary real-patient demonstrations using **implanted brain interfaces**. Motivational/research evidence only, not a webcam or scalp-EEG benchmark.
- [AphasiaBank](https://talkbank.org/aphasia/) — video-linked patient language research; controlled access. [Grand Rounds](https://talkbank.org/aphasia/GR/) materials have separate restrictions; **do not copy** them into the product without permission.

**Attribution rule:** A patient video being viewable online is not sufficient authorization to download, crop, transform, train on, or redistribute it. Check media-specific license, consent and restrictions. Prefer a short attributed link-out for clinical motivation and use your own opted-in footage for product performance tests.

### Proposed evaluation

**Per-modality**

- Visual speech: WER/CER on unseen speakers and phrases; face/mouth tracking failure rate; false-phrase rate on silence and non-speech mouth motion.
- Audio: WER on weak/degraded speech and paired clean references where ethically available; enhancer impact on ASR (better or worse); fabricated-word rate.
- Gaze/gestures: target-selection accuracy, wrong-selection rate, correction rate, robustness to head motion/lighting and calibration drift.
- EEG replay: P300 AUC, character/target selection accuracy, correct handling of event timing, performance on held-out sessions/participants.

**End-to-end**

- **Confirmed-message exactness:** spoken output matches the patient's explicitly confirmed string.
- **Successful communication rate:** user independently communicates the intended message, not just a plausible model guess.
- **Clarification efficiency:** attempts/selections per successful message; separate failures from successes.
- **Unsafe assertion rate:** unconfirmed or contradicted model text reaching TTS (target: **zero by construction**).
- **Comparative ablation:** VSR alone vs VSR+candidate clarification; audio alone vs audio+quality/adaptive selection; static AAC board vs coordinator-selected clarification (if fair).
- **Reliability:** camera loss, face occlusion, silence, wrong gaze, no correct candidate, incomplete EEG events, unreachable inference server.

**Evaluation discipline:** save and demonstrate some genuine failures; avoid reporting paper test-set WER as your demo's accuracy. Never leak participant-specific repeated trials into both train and test when evaluating EEG.

## 11. Frontend experience and presentation

### Home / modality picker

- Brand: **Unspoken** — *A voice for those who cannot speak.*
- Four modes: **Lips / Weak Speech / Eyes / Brain Signals (Recorded EEG Demo)**.
- Visible status for each subsystem: `Ready`, `Calibrate`, `Low confidence`, `Needs input`, `Replay only`, `Unavailable`.
- Accessibility-first controls: very large targets, no essential microtext, keyboard navigation, high contrast, explicit cancellation and backtracking.

### Central interaction

1. Webcam/microphone shows **real** live feed and a visual indication of which models are processing it.
2. Model emits **raw hypotheses**, separate from any language-model proposal.
3. Coordinator displays candidate interpretations and a transparent rationale such as *“Two similar lip patterns; please choose or reject.”*
4. Patient selects or corrects using gaze, deliberate gesture, or an accessible button.
5. Patient independently **confirms** final text.
6. TTS speaks the exact text, with an editable audit view of source modality, raw hypothesis, user correction, and confirmation.

### Two honest demo paths

**Interactive live demo**: An unfamiliar judge silently mouths a request. If decoder uncertain, the judge uses gaze or accessible input to select or spell; the voice renders the confirmed message. Then try low-volume audible speech, showing raw vs enhanced transcription and confirmation.

**Clinical research replay**: Feed **actual withheld, recorded EEG epochs** into a P300 classifier, show the original stimulus protocol with model prediction and ground truth, then present its decoded selection through the same interface labeled `RECORDED EEG`. This proves an engineering integration, **not** brain-to-speech acquisition from the judge.

Optional clinical-problem video is presented as independently published research footage with attribution and checked permissions, never as footage secretly processed by Unspoken.

## 12. Codebase outline

```text
unspoken/
├── README.md
├── plan.md
├── docs/
│   ├── MODEL_CARDS_AND_LICENSES.md
│   ├── DATASETS_AND_CONSENT.md
│   ├── EVALUATION.md
│   └── ARCHITECTURE.md
├── web/
│   ├── app/                      # camera + audio + gaze UI
│   ├── components/               # candidate, clarification, EEG replay
│   ├── lib/capture.ts            # media permission / timestamps
│   └── lib/gaze.ts               # WebGazer calibration + dwell
├── services/
│   ├── gateway/                  # WebSocket/API, session state
│   ├── vsr/                      # Chaplin/Auto-AVSR wrapper
│   ├── audio/                    # enhancement + Whisper
│   ├── vision/                   # MediaPipe face/hand signals
│   ├── eeg/                      # P300 preprocessing / replay / decode
│   ├── coordinator/              # compact instruction model + tools
│   └── tts/                      # Kokoro/Piper
├── shared/
│   ├── schemas/                  # strict typed messages
│   ├── confirmation.py           # deterministic authorization
│   └── provenance.py             # model/event/source metadata
├── evaluations/
│   ├── test_cases/
│   ├── modality_baselines/
│   ├── replay_heldout/
│   └── safety_tests/
└── scripts/
    ├── setup_models.sh
    ├── verify_weights.py
    ├── verify_links_and_licenses.md
    └── smoke_test.sh
```

Use a browser-based frontend, Python/FastAPI services, PyTorch for specialist inference, and a backend WebSocket for inference events. Run gaze and landmark tracking locally in browser where compatible; keep raw camera/voice on-device when feasible. If Brev is used for VSR/ASR, **explicitly disclose cloud processing**, require user opt-in, and never upload identifiable clinical recordings by default.

### Suggested staged implementation dependencies (not a schedule)

- **Foundation:** confirmed-text state machine; usable AAC board; simple local TTS; typed backend messages.
- **Live modalities:** gaze calibration/selection; visual-speech wrapper with raw output; weak-speech ASR and optional enhancement.
- **Coordination:** confidence-aware candidate UI, tools, confirmation, disagreement handling and fallbacks.
- **Research:** public EEG replay classifier with validated trial/stimulus metadata; optional EEG+gaze replay ablation.
- **Presentation/evaluation:** patient-context sources (permission-checked), held-out cases, source/model labels and metrics.

### Concrete starting commands and integration checklist

These are **upstream-documented starting points**, not a promise that every repository runs together without dependency isolation. Pin tested commits/checkpoints and capture any necessary adjustments in your own `README.md`.

**Chaplin (baseline silent lip-reading):**

```bash
git clone https://github.com/amanvirparhar/chaplin.git
cd chaplin
./setup.sh                       # upstream downloads VSR checkpoints
# Install Ollama and pull the documented model separately:
ollama pull qwen3:4b
uv run --with-requirements requirements.txt --python 3.12 \
  main.py config_filename=./configs/LRS3_V_WER19.1.ini detector=mediapipe
```

Expose its **raw transcription** as `vsr.propose(...)`, rather than typing the LLM-corrected text into the user's active window. If its dependency versions conflict with other workers, run Chaplin in its own virtual environment/service.

**Chaplin-UI (alternative browser-to-VSR skeleton):**

```bash
git clone https://github.com/loganngarcia/chaplin-ui.git
cd chaplin-ui
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./setup.sh
# Start a compatible local LLM endpoint separately; then:
./run_web.sh
```

Use primarily as an API/UI reference; Unspoken needs a different confirmation-first interaction and separate EEG/audio/gaze workers.

**EEG speller replay (no headset):**

```bash
git clone https://github.com/weichkai/p300-speller-bci.git
cd p300-speller-bci
pip install -r requirements.txt
python -m examples.run_simulation
python -m src.benchmark.visualize_speller
python -m src.benchmark.p300_benchmark
```

Its simulated stream demonstrates real prerecorded EEG processing. To use **ALS-specific** recordings, introduce a properly implemented MOABB/PhysioNet data adapter with exact event/stimulus mapping and re-evaluate the model. Do not claim that the bundled sample recording is necessarily an ALS patient's recording.

**Audio baseline:**

```bash
pip install -U openai-whisper
whisper sample.wav --model base
# Optional alternative implementation:
pip install faster-whisper
# Optional noise suppression baseline:
pip install deepfilternet
# On supported .wav input:
deepFilter sample.wav
```

Check ffmpeg and audio sample-rate prerequisites. Benchmark raw Whisper against enhanced+Whisper on the **same held-out utterances** and preserve both outputs. The DeepFilterNet command provides noise suppression, not recovery of silent or unintelligible speech.

**Linking agents:** each service needs an `/health`, an `/infer` or WebSocket endpoint, checkpoint metadata, quality/error states, and a contract test. The frontend must keep working when any one optional service is unavailable. The final `speak` action has to reference a confirmed message ID—not arbitrary string text generated by the coordinator.

## 13. Important risks and mitigation

| Risk | Mitigation |
|---|---|
| Lip reading fails on unfamiliar speakers | Evaluate unfamiliar volunteers; provide retake + eye board; don't pretend outputs are correct |
| Small LLM hallucinates plausible words | Keep raw model candidates; require user correction/confirmation; constrain tool calls and final text |
| Gaze drifts or false-selects | Calibration check, stable dwell, separate confirmation, large buttons, accessible fallback |
| Audio enhancement changes intended words | Parallel raw/enhanced ASR, uncertainty display, confirm text |
| EEG model overfits repeated trials | Hold out sessions/targets correctly, inspect leakage, show true held-out metrics |
| EEG replay misrepresented as thought reading | Constant `RECORDED EEG` label, original stimuli and target ground truth shown |
| Patient-identifying videos reused improperly | Respect per-asset consent/license, attribution and no automatic redistribution |
| Processing sensitive camera/mic streams | Explicit permissions, ephemeral storage, no default upload/recording |
| Unsupported clinical claims | Frame as assistive research prototype; no diagnostic/medical decision suggestions |
| 33B cap exceeded by auxiliary models | Inventory all distinct checkpoints, vocoders, vision bundles, and any MoE total weights |
| Separate repos don't integrate easily | Wrap each behind stable JSON/events; pin commits, checkpoints and preprocessing |

## 14. What makes this a compelling multi-agent system

The demonstration must show that specialist models do **different kinds of work**, and that cooperation overcomes a real limitation:

- Audio cannot help when a patient makes no sound.
- Lip reading loses information about visually indistinguishable phonemes.
- Gaze alone can select explicit options but is slow for arbitrary free-form sentences.
- EEG P300 may provide a separate intentional-selection pathway **only with compatible hardware**, and the hackathon branch uses recorded signals.
- The coordinating model chooses **how to ask**; the user alone decides **what they mean**.

**Primary hypothesis:** adaptive collaboration across incomplete, differently reliable modalities can reduce the effort required to communicate an explicitly confirmed message without increasing unverified statements.

The central wow moment is therefore: **a failed or ambiguous speech attempt is rescued by another modality**, and the patient still controls the final sentence.

## 15. Direct source index (verified pages; pin versions before cloning)

### Visual speech and computer vision

- Chaplin: https://github.com/amanvirparhar/chaplin
- Chaplin-UI: https://github.com/loganngarcia/chaplin-ui
- Auto-AVSR: https://github.com/mpc001/auto_avsr
- Multilingual VSR: https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages
- AV-HuBERT: https://github.com/facebookresearch/av_hubert
- LipVoicer: https://github.com/yochaiye/LipVoicer
- Jeremy Park vision demos: https://github.com/jeremyipark/vision-demos
- MediaPipe: https://github.com/google-ai-edge/mediapipe
- Face Landmarker: https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker
- Hand Landmarker: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker
- WebGazer: https://github.com/brownhci/webgazer

### Audio and voice

- Whisper: https://github.com/openai/whisper
- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- DeepFilterNet: https://github.com/Rikorose/DeepFilterNet
- RNNoise: https://github.com/xiph/rnnoise
- VoiceBridge paper: https://arxiv.org/pdf/2509.25275
- VoiceBridge project samples: https://voicebridgedemo.github.io/
- Kokoro: https://github.com/hexgrad/kokoro
- Kokoro weights: https://huggingface.co/hexgrad/Kokoro-82M
- Piper: https://github.com/rhasspy/piper

### EEG and neural communication

- P300 Speller BCI: https://github.com/weichkai/p300-speller-bci
- bigP3BCI: https://physionet.org/content/bigp3bci/1.0.0/
- EEG-Conformer P300: https://github.com/Isti23/bigp3bci-eeg-bci
- BNCI2014-008: https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_008.html
- NEMAR ALS P300: https://www.nemar.org/dataset/nm000169?v=v1.0.1
- MOABB documentation: https://moabb.neurotechx.com/
- MNE documentation: https://mne.tools/stable/index.html
- BrainFlow: https://github.com/brainflow-dev/brainflow
- Brain2Qwerty: https://github.com/facebookresearch/brain2qwerty

### Agent serving, evaluation, output

- Qwen3-4B Instruct: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- vLLM: https://docs.vllm.ai/en/latest/
- NeMo Gym: https://github.com/NVIDIA-NeMo/gym

### Data and real-patient research

- GRID: https://zenodo.org/records/3625687
- Mozilla aligned audio/visual dataset: https://mozilladatacollective.com/datasets/cmpe4rs9800r2nv07qpq8972c
- ALS patient one-shot lipreading: https://arxiv.org/abs/2111.01740
- ALS research demo video: https://youtu.be/_famGVaem-8
- ICU visual speech research: https://pmc.ncbi.nlm.nih.gov/articles/PMC9844948/
- Stroke intelligent throat: https://www.nature.com/articles/s41467-025-68228-9
- Paralysis speech/gesture BCI: https://www.nature.com/articles/s41593-026-02446-2
- AphasiaBank: https://talkbank.org/aphasia/

---

## Final design decision

Build **Unspoken**, not four disconnected demos. Deliver one accessible confirmation-first conversation interface and four real adapters: **live lip recognition, live residual audio, live gaze selection, and authentic prerecorded EEG inference**. Measure whether coordination can recover an intended message when the initial modality is inadequate. Treat all research results, patient footage, and model outputs with honest provenance.
