"""Phase 3: ASR models behind one interface.

    load()                         -> None
    info()                         -> {"checkpoint", "revision", "params", ...}
    transcribe(wav)                -> {"text": str, "nbest": [str], ...extra signals}
    score_phrases(wav, texts)      -> [(log P(text | audio), n_tokens)] or None if unsupported
    unload()

score_phrases reuses the encoder pass of the preceding transcribe() on the same
array, so "one encoder pass" holds for transcript + all phrase scores.
"""

from __future__ import annotations

import gc

import numpy as np

SR = 16000
IMPOSSIBLE = -1e6  # log P for an alignment that cannot exist (target longer than the audio allows)


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class Model:
    supports_scoring = False

    def __init__(self, spec: dict, cfg):
        self.spec, self.cfg = spec, cfg
        self.name, self.checkpoint = spec["name"], spec["checkpoint"]

    def info(self) -> dict:
        return {"checkpoint": self.checkpoint}

    def unload(self):
        import torch

        for attr in ("model", "processor"):
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def resolve_revision(repo: str, revision: str | None) -> str:
    from huggingface_hub import HfApi

    return HfApi().model_info(repo, revision=revision).sha


# ---- Parakeet CTC (Hugging Face transformers format) -------------------------------------------

class HFCTC(Model):
    """nvidia/parakeet-ctc-0.6b and fine-tunes shipped in transformers format.

    Forced scoring: log P(phrase | audio) = -ctc_loss over all alignments, computed for
    every phrase from the same frame log-probabilities. Infeasible targets keep +inf loss
    (zero_infinity would turn them into a perfect score) and are mapped to IMPOSSIBLE.
    """

    supports_scoring = True

    def load(self):
        import torch
        from transformers import AutoProcessor, ParakeetForCTC

        self.dev = device()
        self.revision = resolve_revision(self.checkpoint, self.spec.get("revision"))
        self.dtype = getattr(torch, self.cfg["latency"]["dtype_cuda"]) if self.dev == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(self.checkpoint, revision=self.revision)
        self.model = ParakeetForCTC.from_pretrained(self.checkpoint, revision=self.revision, dtype=self.dtype).to(self.dev).eval()
        self.blank = self.model.config.pad_token_id
        self._cache = None

    def info(self) -> dict:
        return {"checkpoint": self.checkpoint, "revision": self.revision, "params": sum(p.numel() for p in self.model.parameters()),
                "device": self.dev, "dtype": str(self.dtype).replace("torch.", ""), "blank_id": self.blank}

    def _logprobs(self, wav):
        import torch

        if self._cache is not None and self._cache[0] is wav:  # holds the array, so identity cannot be reused
            return self._cache[1]
        inputs = self.processor(wav, sampling_rate=SR, return_tensors="pt")
        feats = inputs["input_features"].to(self.dev, self.dtype)
        mask = inputs.get("attention_mask")
        with torch.inference_mode():
            logits = self.model(input_features=feats, attention_mask=mask.to(self.dev) if mask is not None else None).logits
        lp = torch.log_softmax(logits.float(), dim=-1)[0]
        self._cache = (wav, lp)
        return lp

    def transcribe(self, wav):
        lp = self._logprobs(wav)
        best = lp.argmax(-1).tolist()
        ids, prev = [], None
        for t in best:
            if t != prev and t != self.blank:
                ids.append(t)
            prev = t
        text = self.processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
        return {"text": text, "nbest": [text], "best_path_logp": round(float(lp.max(-1).values.sum()), 3)}

    def score_phrases(self, wav, texts):
        import torch
        import torch.nn.functional as F

        lp = self._logprobs(wav)
        tok = self.processor.tokenizer
        targets = [tok(t, add_special_tokens=False)["input_ids"] if t.strip() else [] for t in texts]
        b, s = len(targets), max(1, max(len(t) for t in targets))
        padded = torch.full((b, s), self.blank, dtype=torch.long)
        for i, t in enumerate(targets):
            if t:
                padded[i, : len(t)] = torch.tensor(t)
        lens = torch.tensor([len(t) for t in targets], dtype=torch.long)
        steps = torch.full((b,), lp.shape[0], dtype=torch.long)
        with torch.inference_mode():
            nll = F.ctc_loss(lp.unsqueeze(1).expand(-1, b, -1), padded.to(lp.device), steps, lens, blank=self.blank,
                             reduction="none", zero_infinity=False)
        return [(IMPOSSIBLE if not np.isfinite(v) else -float(v), len(t)) for v, t in zip(nll.cpu().tolist(), targets)]


# ---- NeMo transducers: Parakeet-TDT, Nemotron Speech Streaming ------------------------------------

class NemoTransducer(Model):
    """NeMo RNNT/TDT model: greedy transcript plus forced phrase scoring with the model's own transducer loss.

    score_phrases: log P(phrase | audio) = -loss over all alignments, from the same encoder pass as the
    transcript (prediction network + joint batched over every phrase form). The loss comes from the model's
    own config (TDT for Parakeet-TDT, RNNT for Nemotron) with FastEmit off: that is a training regulariser and
    would bias scores. If the GPU loss kernel is unavailable, the joint output is scored on the CPU.

    `att_context` (e.g. [70, 1]) sets a cache-aware model's right context. Offline decoding at that context
    reproduces its streaming output at that lookahead (80 ms frames: [70,1] ~ 160 ms, [70,6] ~ 560 ms).
    """

    supports_scoring = True

    def load(self):
        import nemo.collections.asr as nemo_asr
        import torch
        from nemo.collections.asr.losses.rnnt import RNNTLoss

        self.dev = device()
        self.revision = resolve_revision(self.checkpoint, self.spec.get("revision"))
        self.notes: list[str] = []
        m = nemo_asr.models.ASRModel.from_pretrained(self.checkpoint, map_location=torch.device(self.dev))
        m.eval()
        for attr, val in (("dither", 0.0), ("pad_to", 0)):  # deterministic features
            try:
                setattr(m.preprocessor.featurizer, attr, val)
            except Exception:
                pass
        self.att_all = getattr(m.encoder, "att_context_size_all", None)
        if self.spec.get("att_context"):
            m.encoder.set_default_att_context_size(self.spec["att_context"])
            self.notes.append(f"att_context_size={self.spec['att_context']} (available: {self.att_all})")
        self.model = m
        self.direct = True
        self._cache = None
        loss_name, loss_kwargs = m.extract_rnnt_loss_cfg(m.cfg.get("loss", None))
        loss_kwargs = dict(loss_kwargs or {})
        if "fastemit_lambda" in loss_kwargs:
            loss_kwargs["fastemit_lambda"] = 0.0
        n = m.joint.num_classes_with_blank - 1
        if loss_name == "tdt":
            n -= getattr(m.joint, "num_extra_outputs", 0)
        self.loss = RNNTLoss(num_classes=n, loss_name=loss_name, loss_kwargs=loss_kwargs, reduction=None)
        self.notes.append(f"forced scoring loss={loss_name}")
        self.loss_on_cpu = False

    def info(self) -> dict:
        return {"checkpoint": self.checkpoint, "revision": self.revision, "params": sum(p.numel() for p in self.model.parameters()),
                "device": self.dev, "dtype": "bfloat16 autocast" if self.dev == "cuda" else "float32",
                "class": type(self.model).__name__, "att_context_available": str(self.att_all), "notes": self.notes}

    def _autocast(self):
        import torch

        return torch.autocast("cuda", dtype=torch.bfloat16) if self.dev == "cuda" else torch.autocast("cpu", enabled=False)

    def _encode(self, wav):
        import torch

        if self._cache is not None and self._cache[0] is wav:  # holds the array, so identity cannot be reused
            return self._cache[1], self._cache[2]
        sig = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))[None].to(self.dev)
        ln = torch.tensor([len(wav)], device=self.dev)
        with torch.inference_mode(), self._autocast():
            enc, enc_len = self.model.forward(input_signal=sig, input_signal_length=ln)
        self._cache = (wav, enc, enc_len)
        return enc, enc_len

    def transcribe(self, wav):
        import torch

        if self.direct:
            try:
                enc, enc_len = self._encode(wav)
                with torch.inference_mode(), self._autocast():
                    hyps = self.model.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc, encoded_lengths=enc_len,
                                                                                return_hypotheses=True)
                if isinstance(hyps, tuple):  # older API: (best, all)
                    hyps = hyps[0]
                h = hyps[0]
                text = h.text if isinstance(h.text, str) else self.model.tokenizer.ids_to_text([int(t) for t in h.y_sequence])
                return {"text": text.strip(), "nbest": [text.strip()], "path": "direct"}
            except Exception as exc:  # fall back to the library's own transcribe() and say so
                self.direct = False
                self.notes.append(f"direct decode failed ({type(exc).__name__}: {str(exc)[:120]}); using model.transcribe()")
        out = self.model.transcribe([np.ascontiguousarray(wav, dtype=np.float32)], batch_size=1, verbose=False)
        out = out[0] if isinstance(out, tuple) else out
        h = out[0]
        text = h.text if hasattr(h, "text") else str(h)
        return {"text": text.strip(), "nbest": [text.strip()], "path": "transcribe"}

    def score_phrases(self, wav, texts):
        import torch

        enc, enc_len = self._encode(wav)
        tok = self.model.tokenizer
        ids = [tok.text_to_ids(t) if t.strip() else [] for t in texts]
        b, u = len(ids), max(1, max(len(i) for i in ids))
        targets = torch.zeros(b, u, dtype=torch.long)
        for k, seq in enumerate(ids):
            if seq:
                targets[k, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        tlen = torch.tensor([len(i) for i in ids], dtype=torch.long)
        targets, tlen = targets.to(self.dev), tlen.to(self.dev)
        with torch.inference_mode():
            with self._autocast():
                g, _, _ = self.model.decoder(targets=targets, target_length=tlen)
                f = enc.expand(b, -1, -1)
                logits = self.model.joint.joint(f.transpose(1, 2), g.transpose(1, 2))
            # the CUDA loss kernels need contiguous length tensors (expand() gives a stride-0 view)
            nll = self._nll(logits.float().contiguous(), targets.contiguous(), enc_len.expand(b).long().contiguous(), tlen.contiguous())
        out = []
        for k in range(b):
            v = float(nll[k])
            out.append((IMPOSSIBLE if not np.isfinite(v) else -v, len(ids[k]) + 1))
        return out

    def _nll(self, logits, targets, enc_len, tlen):
        if not self.loss_on_cpu:
            try:
                return self.loss(log_probs=logits, targets=targets, input_lengths=enc_len, target_lengths=tlen).reshape(-1)
            except Exception as exc:
                self.loss_on_cpu = True
                self.notes.append(f"GPU transducer loss failed ({type(exc).__name__}); scoring on CPU")
        return self.loss(log_probs=logits.cpu(), targets=targets.cpu(), input_lengths=enc_len.cpu(), target_lengths=tlen.cpu()).reshape(-1)


# ---- faster-whisper (CTranslate2) -------------------------------------------------------------------

def _preload_cudnn():
    """CTranslate2 needs cuDNN 9; torch's pip wheel ships it under site-packages/nvidia/cudnn/lib."""
    import ctypes
    import glob
    import site

    for sp in site.getsitepackages():
        for lib in sorted(glob.glob(f"{sp}/nvidia/cudnn/lib/libcudnn*.so.9")) + sorted(glob.glob(f"{sp}/nvidia/cublas/lib/libcublas*.so.12")):
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


class FasterWhisper(Model):
    """Whisper via faster-whisper: fp16, beam 3, temperature 0, no previous-text conditioning, no built-in VAD.

    no_speech_threshold=None keeps every segment (Whisper's own skip rule would hide text and its
    signals); no_speech_prob and avg_logprob are recorded per clip instead. Text method only.
    """

    def load(self):
        import torch

        if torch.cuda.is_available():
            _preload_cudnn()
        from faster_whisper import WhisperModel
        from faster_whisper.utils import _MODELS

        self.dev = device()
        self.repo = _MODELS.get(self.checkpoint, self.checkpoint)
        self.revision = resolve_revision(self.repo, self.spec.get("revision"))
        self.compute_type = "float16" if self.dev == "cuda" else "int8"
        self.model = WhisperModel(self.repo, device=self.dev, compute_type=self.compute_type, revision=self.revision)

    def info(self) -> dict:
        params, source = None, None
        if self.spec.get("params_from"):
            try:
                from huggingface_hub import HfApi

                params = sum(HfApi().get_safetensors_metadata(self.spec["params_from"]).parameter_count.values())
                source = f"{self.spec['params_from']} safetensors headers"
            except Exception as exc:
                source = f"unavailable ({type(exc).__name__})"
        return {"checkpoint": self.repo, "revision": self.revision, "params": params, "params_source": source,
                "device": self.dev, "dtype": self.compute_type, "runtime": "ctranslate2"}

    def transcribe(self, wav):
        segs, info = self.model.transcribe(np.ascontiguousarray(wav, dtype=np.float32), language="en", beam_size=3, temperature=0.0,
                                           condition_on_previous_text=False, vad_filter=False, no_speech_threshold=None)
        segs = list(segs)
        text = " ".join(s.text.strip() for s in segs).strip()
        out = {"text": text, "nbest": [text]}
        if segs:
            out["avg_logprob"] = round(float(np.mean([s.avg_logprob for s in segs])), 4)
            out["no_speech_prob"] = round(float(max(s.no_speech_prob for s in segs)), 4)
        return out


KINDS = {"hf_ctc": HFCTC, "nemo_transducer": NemoTransducer, "faster_whisper": FasterWhisper}


def make(spec: dict, cfg) -> Model:
    if spec["kind"] not in KINDS:
        raise NotImplementedError(f"model kind {spec['kind']!r} not implemented yet")
    return KINDS[spec["kind"]](spec, cfg)
