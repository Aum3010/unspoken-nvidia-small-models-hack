# Lip reading in Unspoken

The lip agent turns a short, silent mouth clip into **candidate messages**. The person then picks one and confirms it (gaze, blink, click or keyboard). Nothing the lip agent outputs is ever spoken without that confirmation (see `shared/confirmation.py`).

It has three modes. Your own recordings are tried first; the other two run only when a clip matches none of them:

| Mode | What it does | Needs | Default |
|---|---|---|---|
| **Free text** | Auto-AVSR beam search writes any sentence (top 5) | nothing | when nothing recorded matches (`UNSPOKEN_VSR_FREE_TEXT=0` turns it off) |
| **Personal enrollment** | Matches the clip against the user's own recordings of their phrases | recordings of each phrase in `data/enroll/` (3–5 takes) | on as soon as one phrase is recorded |
| **Phrase list** | Scores a fixed list of phrases against the clip | a text file of phrases | off (`UNSPOKEN_VSR_PHRASES`) |

**Recommendation from the measurements below:** personal enrollment is by far the most accurate mode (96–99% top-1 on MIRACL-VC1 vs 66–72% for the phrase list). Free text on unfamiliar speakers is mostly wrong (≈80% WER), which is why lip readings are only ever candidates.

---

## 1. Architecture

```
Browser (web/)                         Gateway :8000                     VSR service :8101 (services/vsr/)
──────────────                         ─────────────                     ──────────────────────────────────
MediaPipe Face Landmarker              POST /api/{sid}/capture/lip  ──►  POST /infer
  mouth box overlay, face hints          forwards the clip                 decode → face → mouth crop → quality gates
MediaRecorder (webm, ≤10 s)              adds Evidence to the session      Auto-AVSR encoder (once per clip)
  silent mouth clip  ──────────────►     runs the coordinator              ├─ enrollment match   (if recordings exist)
                                         pushes candidates (WebSocket)     ├─ phrase-list score  (if enabled)
recording (curl / API)                                                     └─ beam search, top 5 free text
  record phrase ───────────────────►   POST /api/enroll             ──►  POST /enroll   (store features)
                                       GET/DELETE /api/enroll       ──►  GET/DELETE /enroll
```

Every mode shares **one encoder pass**. The video is encoded once, and enrollment matching, phrase scoring and beam search all reuse those features.

### Files

| File | Role |
|---|---|
| `services/vsr/app.py` | FastAPI service: preprocessing, quality gates, the three modes, enrollment endpoints |
| `services/vsr/enroll.py` | Enrollment store (features on disk) and nearest-neighbour matching |
| `services/vsr/phrases.py` | Phrase tokenizer, phrase-list scoring, DTW matcher |
| `services/vsr/patient_phrases.txt` | The voice agent's 20 phrases: phrases to record, and the optional phrase list |
| `services/vsr/negative_phrases.txt` | 7 near-misses that must become unknown, recordable as negatives |
| `services/gateway/app.py` | `/api/{sid}/capture/lip` and `/api/enroll` routes |
| `web/src/glass/sensors.ts` (`startLips`), `web/src/lib/face.ts` | Mouth-motion detection cuts each clip, live and when recording |
| `data/enroll/` | The recorded phrases (encoder features only, committed on this branch; see §5.2) |
| `scripts/finetune_vsr_grid.py` | Feature extraction, fine-tuning, free-text and phrase-list evaluation |
| `scripts/eval_enrollment_miracl.py` | Enrollment evaluation on MIRACL-VC1 |
| `scripts/brev_bench.sh` | The same benchmarks on an NVIDIA Brev GPU |
| `evaluations/modality_baselines/eval_vsr_grid.py` | Original baseline + still-face test (part of `run_evals.sh`) |

---

## 2. Model

| | |
|---|---|
| Model | **Auto-AVSR** visual-only, checkpoint `LRS3_V_WER19.1`, via the Chaplin wrapper (pinned commit `7aee1f8`) |
| Architecture | 3D-conv + ResNet front-end → 12-layer Conformer encoder (dim 768) → 6-layer Transformer decoder + CTC head, 5,049 SentencePiece tokens |
| Parameters | 250.4M (model) + 53.7M (subword language model) |
| Training data (upstream) | LRS3 (TED talks), plus pseudo-labelled data, as in the Auto-AVSR paper. Reported LRS3 WER ≈ 19% |
| Runtime | PyTorch on **MPS** (Apple Metal) on the Mac, **CUDA** on Brev (`UNSPOKEN_VSR_DEVICE`), CPU fallback |
| Not used | Chaplin's LLM "correction" step. No model may rewrite the words |

Both Auto-AVSR checkpoints count toward the 33B parameter budget (`docs/MODEL_CARDS_AND_LICENSES.md`). Enrollment and the phrase list add **no parameters**. The fine-tuned decoder replaces weights in place rather than adding any.

---

## 3. How a clip is processed (`POST /infer`)

1. **Decode.** PyAV decodes the browser webm/mp4 and resamples it to **25 fps**, the model's training rate.
2. **Gate: length.** Under 0.6 s (15 frames) gives `too_short` and stops. Over 12 s gives `too_long` (processed, but flagged).
3. **Find the face.** MediaPipe face detection, full-range model first, short-range as a fallback.
   - No face in any frame gives `no_face` and stops.
   - Face in under 90% of frames gives `partial_face`.
4. **Crop the mouth.** Landmarks are smoothed over a 12-frame window, the face is similarity-warped to a mean face, and a 96×96 grayscale mouth patch is cropped, then centre-cropped to 88×88 and normalised (mean 0.421, std 0.165). A failed crop gives `occlusion` and stops.
5. **Gate: mouth motion.** This is the mean absolute frame-to-frame change of the mouth crop. Under **2.2** gives `low_mouth_motion`. This gate matters: on a still face, Auto-AVSR invents fluent sentences ("THAT'S WHAT I'M GOING TO DO"). The gate flags 8 of 8 still-face clips, and their output is never presented as a reading.
6. **Encode** once with the Conformer encoder, giving features of shape `(frames, 768)`.
7. **Run the modes** on those features, stopping at the first that gives readings (`quality.mode` says which):
   - **Enrollment** (if any phrase is recorded): the 3 closest phrases by DTW distance, keeping only those within `UNSPOKEN_ENROLL_MAX_DIST` (0.30) (§5.2).
   - **Phrase list** (if `UNSPOKEN_VSR_PHRASES` is set): top 3 phrases by log-likelihood (§5.3).
   - **Free text**: beam search with beam 12, CTC weight 0.1, LM weight 0.3, keeping the 5 best unique texts.
   The modes' scores are on different scales and the policy ranks candidates by score, so they are never mixed in one answer. A clip that matches a recording also skips beam search, the slowest step.
8. **Status.** `proposed`, or `ambiguous` if the top-2 score gap is under 1.5, or `unknown` if `no_face` or `low_mouth_motion` was raised. The coordinator offers the top reading alone when the gap is over 3, otherwise the top 3.

The response is an `Evidence` object (`shared/schemas.py`): `hypotheses[]` (text, score, score_kind), `quality_flags[]`, `quality{}` (duration, face ratio, mouth motion, timings, 8 mouth thumbnails), `provenance` (checkpoint, device, latency) and `status`.

---

## 4. How it is called

### From the app
- **The room** (`/`, or `/?lips` for the lips-only room): mouth-motion detection cuts a clip when the mouth starts moving and ends it after 0.9 s of stillness, then calls `POST /api/{session}/capture/lip`.
- **Recording phrases:** the temporary recording page was removed once the 20 phrases were recorded. To add takes, send a clip to the API: `curl -F video=@clip.webm -F "phrase=I need water" http://127.0.0.1:8000/api/enroll` (negatives: `phrase=[unknown] I don't need water`).
- **Lips-only mode for development:** `scripts/run_all.sh --lips` (Mac) or `bash scripts/brev/start_app.sh lips` (Brev) starts only the gateway and the lip agent, and `/?lips` turns off voice, blink, head and the eye-calibration prompt. The gateway's deterministic policy stands in for the coordinator.

### HTTP API

| Method | Path (gateway) | Body | Returns |
|---|---|---|---|
| POST | `/api/{sid}/capture/lip` | multipart `video` | `{event: Evidence, state: Session}` |
| GET | `/api/enroll` | — | `{phrases: [{phrase, count}], suggestions: [str]}` |
| POST | `/api/enroll` | multipart `video`, `phrase` | `{phrase, count, quality, phrases}`, or **422** "recording not saved: no face / too short / low mouth motion / occlusion" |
| DELETE | `/api/enroll?phrase=…` | — | `{phrases}` |

The VSR service exposes the same operations directly on port 8101: `/infer`, `/enroll`, and `/health`. `/health` reports the device, parameter count, phrase-list size and number of enrolled phrases.

```bash
curl -F video=@clip.webm http://127.0.0.1:8000/api/demo/capture/lip
curl -F video=@clip.webm -F "phrase=I need water" http://127.0.0.1:8000/api/enroll
```

### Configuration

| Variable | Default | Effect |
|---|---|---|
| `UNSPOKEN_VSR_DEVICE` | `mps`, else `cpu` | `cuda` on NVIDIA GPUs |
| `UNSPOKEN_VSR_BEAM` | `12` | Free-text beam width (smaller is faster) |
| `UNSPOKEN_VSR_MOTION_THRESHOLD` | `2.2` | Still-mouth gate |
| `UNSPOKEN_VSR_PHRASES` | unset | Path to a phrase list, e.g. `services/vsr/patient_phrases.txt` |
| `UNSPOKEN_VSR_DECODER` | unset | Fine-tuned decoder + CTC checkpoint (the GRID one only knows GRID words, so use it for demos only) |
| `UNSPOKEN_ENROLL_DIR` | `data/enroll` | Where enrollment features are stored |
| `UNSPOKEN_ENROLL_MAX_DIST` | `0.30` | A clip farther than this from every recording is not one of the person's phrases |
| `UNSPOKEN_VSR_FREE_TEXT` | `1` | `0`: with phrases recorded, an unmatched clip gives no reading instead of a free-text guess |

---

## 5. The three modes in detail

### 5.1 Free text
Standard Auto-AVSR decoding. It can produce any sentence, but on speakers, words and cameras it never saw, it is mostly wrong (§7). It is kept because it is the only mode that can catch something not on any list.

### 5.2 Personal enrollment
- **Recording:** each clip goes through steps 1–6 above. Clips that fail a gate are rejected (422) instead of saved. The encoder features are L2-normalised per frame and saved as `data/enroll/<hash>/<n>.npy` (float16), next to a `phrase.txt`. **No video is stored.**
- **Matching:** the new clip's features are compared with every recording using **dynamic time warping** (DTW) over cosine distance, normalised by path length. DTW lets the same phrase be said faster or slower. A phrase's distance is its **closest** recording. The 3 closest phrases within 0.30 become candidates, with score = −100 × distance (0 means identical).
- **Thresholds, from the first 10 webcam recordings (3 phrases, one person):** leave-one-out, every take was matched to its own phrase (10/10). A take was 0.13–0.28 from the other takes of its phrase, ≥ 0.22 from other phrases, and an unrelated GRID speaker's clip was ≈ 0.33 from all of them. Held-out takes beat the next phrase by ≥ 0.048, so a gap over 0.03 (score 3) offers that phrase alone. **Limit:** the same person mouthing a phrase they never recorded can still land within 0.30 of a recorded one (a GRID test: 0.18), so recording every phrase the person needs matters, and "No" (none of these) stays one look away.
- **Negatives ("must become unknown"):** the near-miss sentences in `services/vsr/negative_phrases.txt` (the voice agent's hard negatives, e.g. "I don't need water") can be recorded too; they are stored as `[unknown] <sentence>`. When a clip's closest recording is a negative, the lip agent returns no reading (`quality.mode = "negative"`) and runs no free text. Suggestions (`patient_phrases.txt`) are the voice agent's 20 phrases in the same wording, so a lip and a voice reading of the same phrase fuse into one candidate.
- **The recordings on this branch:** one person, webcam, the 20 phrases with 3–5 takes each (83 takes), plus two older phrases ("I am thirsty", "I am hungry") from the first recorder. No negatives recorded yet. Leave-one-out over the 83 takes: **top-1 73% (61/83), top-3 83% (69/83)**; 79/83 takes are within the 0.30 cut-off of their own phrase. Most confusions are short phrases that look alike: "I'm tired" / "I'm scared" / "Help me", "No" / "I'm cold" / "Yes", "Fix my pillow" / "Where is my family". With the clip's own take removed, each phrase has only 2–4 takes left, so live use with all takes should do a little better.
- **Privacy:** features stay on the machine running the lip agent; on this branch they are committed, at the recorded person's request. They are used only for this matching, are never uploaded, and train no model. The user can delete any phrase.
- **Why it works:** it compares a person with *themselves*, which removes the two largest error sources: an unfamiliar face, and the gap between silent mouthing and speaking aloud.

### 5.3 Phrase list
For each phrase in the list, the decoder computes log p(phrase | video) by teacher forcing (0.7 × attention decoder + 0.3 × CTC), all phrases in one batched pass. The 3 most likely phrases become candidates. No training or recordings are needed, and the list can be edited freely. The tokenizer greedily matches the model's SentencePiece units, and every word in `patient_phrases.txt` and GRID is a single unit.

---

## 6. Training (adapting to a fixed vocabulary)

`scripts/finetune_vsr_grid.py` adapts Auto-AVSR to a fixed sentence set without new parameters:

| Stage | What happens |
|---|---|
| `extract` | Each clip goes through steps 1–6 with the **frozen** original encoder. Features are cached as `data/cache/vsr_ft/<speaker>/<clip>.npy` (float16). This is the slow, one-time part |
| `train` | **Encoder frozen.** Fine-tune the decoder + CTC head (68.4M trainable parameters) on cached features. Loss = 0.7 × label-smoothed attention CE + 0.3 × CTC. AdamW, lr 1e-4, weight decay 0.01, grad-clip 5, batch 16 (Mac) or 32 (Brev), 4 epochs. 5% of clips are held out for validation, and the best validation checkpoint is kept |
| `eval` | Free-text WER on held-out speakers, comparing the original decoder with the fine-tuned one (with and without LM) |
| `phrases` | Closed-list accuracy (top-1 / top-3) for N = 10/50/100 candidates. `--max-diff 1` makes every distractor differ from the truth in only one word |

**Data: GRID corpus** (CC BY 4.0). 33 speakers each say 1,000 sentences from a fixed six-slot grammar ("bin blue at f two now"), 51 words in total. **s1 and s2 are always held out for testing.** Training used s3–s8 on the Mac, and s3–s34 minus s21 on Brev (s21 has no video).

**Data pitfalls found and fixed:**
- Zenodo's GRID alignment folders do not match the speaker videos for 17 of 31 training speakers (for example, s10's videos match the "s13" transcripts).
- `setup_models.sh` therefore checks each speaker and, on a mismatch, fetches that speaker's alignments from the Sheffield GRID site.
- Videos come from Sheffield as well, which is about 30× faster than Zenodo (8.4 vs 0.3 MB/s on Brev).

**Limit:** the fine-tuned decoder learns GRID's 51-word grammar, so it is a **proof of the recipe**, not the deployed model. Applying the recipe to patient phrases requires clips of those phrases.

```bash
export PYTHONPATH=$PWD
.venv/bin/python scripts/finetune_vsr_grid.py extract --speakers s1 s2 s3 s4 s5 s6 s7 s8 --per-speaker 500
.venv/bin/python scripts/finetune_vsr_grid.py train --train s3 s4 s5 s6 s7 s8
.venv/bin/python scripts/finetune_vsr_grid.py eval --test s1 s2
.venv/bin/python scripts/finetune_vsr_grid.py phrases --test s1 s2 [--max-diff 1]
.venv/bin/python scripts/eval_enrollment_miracl.py extract && .venv/bin/python scripts/eval_enrollment_miracl.py eval
./scripts/brev_bench.sh setup && ./scripts/brev_bench.sh vsr      # on a Brev GPU
```

---

## 7. Accuracy

All test speakers are unseen by every model being tested.

### 7.1 Free text (GRID s1+s2)
| Model | Clips | WER | Notes |
|---|---:|---:|---|
| Original (deployed) | 40 | **79.6%** | best-of-5 71.3%, CER 55.0% (`vsr.json`) |
| Original (deployed) | 200 | **80.7%** | same model, larger test set |
| Fine-tuned decoder, 6 speakers (Mac), no LM | 200 | **4.6%** | + LRS3 LM: 6.4% |
| Fine-tuned decoder, 31 speakers (Brev), no LM | 200 | **5.8%** | + LRS3 LM: 6.9%. More speakers did not help |

Example for the truth "bin blue at s three again": the original model gives *"people who ask for the day"*. The fine-tuned model gives *"bin blue at three again"* (one letter missed). Remaining errors are mostly single letters, which are nearly invisible on the lips.

### 7.2 Phrase list (GRID s1+s2, 200 clips, top-1 / top-3)
| Distractors | N | Original | Fine-tuned (6 spk) | Fine-tuned (31 spk) |
|---|---:|---:|---:|---:|
| random GRID sentences | 10 | 100 / 100% | 100 / 100% | 100 / 100% |
| random | 50 | 99.5 / 100% | 100 / 100% | 100 / 100% |
| random | 100 | 100 / 100% | 99.5 / 100% | 99.5 / 100% |
| differ in **one word** | 10 | 90.5 / 99.0% | 94.0 / 100% | 90.0 / 99.5% |
| differ in **one word** | 50 | 76.5 / 98.0% | 81.5 / 99.0% | 78.0 / 98.5% |

### 7.3 Personal enrollment (MIRACL-VC1, 15 speakers, 2 random splits)
For each speaker, k repetitions per class are enrolled and the remaining repetitions are tested.

| Classes | k | Test clips | Phrase list | **Enrollment (DTW)** | Combined |
|---|---:|---:|---:|---:|---:|
| 10 phrases | 1 | 2,700 | 72.0 / 89.7% | **96.1 / 99.2%** | 95.2 / 99.0% |
| 10 phrases | 3 | 2,100 | 71.7 / 89.6% | **99.1 / 99.9%** | 96.6 / 99.6% |
| 10 words + 10 phrases | 1 | 5,400 | 66.2 / 80.9% | **94.2 / 98.9%** | 93.4 / 98.1% |
| 10 words + 10 phrases | 3 | 4,200 | 66.1 / 80.8% | **97.9 / 99.6%** | 95.9 / 98.9% |

(top-1 / top-3. Chance is 10% for 10 classes and 5% for 20.) The weakest speaker improves from 53% (phrase list) to 94–99% (enrollment). Adding the phrase-list score does not beat enrollment alone.

### 7.4 Still face
8 of 8 frozen-frame clips were flagged `low_mouth_motion`, and 0% produced an unflagged fluent phrase. Median mouth motion is 0.09 for still clips and 2.62 for speech.

---

## 8. Latency

| Measurement | Apple M2, 8 GB (MPS) | RTX PRO 6000 Blackwell (CUDA, Brev) |
|---|---:|---:|
| Full `/infer`, original model, p50 | **≈3.6 s** (6 webcam-length GRID clips, 51 enrolled recordings) · 7.9 s in the original baseline run, measured while the coordinator benchmark shared the GPU | — |
| Beam search (beam 12, LM 0.3), p50 | 2.0–2.9 s | **0.29 s** |
| Beam search, fine-tuned decoder, no LM | 1.1 s | **0.14 s** |
| Phrase-list scoring, 44 phrases | **0.09 s** | — |
| Enrollment matching, 2 recordings | **0.04 s** | — |
| Enrollment matching, 51 recordings | **0.3–0.5 s** (≈7–10 ms per recording, grows linearly) | — |
| Feature extraction (decode + face + crop + encode), per clip | 0.7 s | 0.24–0.33 s (8 in parallel) |
| Fine-tuning, 4 epochs | ≈6 min (2,850 clips) | ≈6 min (6,650 clips; 14 s/epoch once cached in memory) |
| Model load + warm-up at start-up | 1–3 min | — |

**Where the Mac time goes:** beam search is most of `/infer`. Lowering `UNSPOKEN_VSR_BEAM` to 4–5 trades a little free-text accuracy for speed. Enrollment and the phrase list add little on top of the encoder pass.

---

## 9. Limits and open work

- **Webcam accuracy is not yet measured.** The GRID and MIRACL numbers come from studio or Kinect video of healthy people. MIRACL enrollment and test clips also come from the same session. Expect lower numbers across days, lighting changes and for patients.
- **Single user.** Enrollment stores one person's phrases per machine.
- **Scaling enrollment.** Matching cost grows with the number of recordings. Beyond a few hundred it will need pruning or an index.
- **MIRACL-VC1 is licensed for research only** (Rekik, Ben-Hamadou & Mahdi, 2016), so use it for evaluation, not in a product. GRID is CC BY 4.0. Auto-AVSR code is Apache-2.0, and its weights follow the LRS3 terms.
- **Possible next steps:**
  - Measure on consenting webcam users (selections per message, top-3 hit rate).
  - Unfreeze the encoder when fine-tuning on a GPU.
  - Narrow the phrase list using conversation context.
  - Use audio and video together when some voice remains.

Raw results are in `evaluations/results/vsr.json`, `vsr_finetune.json`, `vsr_phrases*.json` (Mac) and `evaluations/results/brev/` (Brev, including `vsr_enrollment_miracl.json`).
