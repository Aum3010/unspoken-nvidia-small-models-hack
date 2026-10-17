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


KINDS = {"hf_ctc": HFCTC}


def make(spec: dict, cfg) -> Model:
    if spec["kind"] not in KINDS:
        raise NotImplementedError(f"model kind {spec['kind']!r} not implemented yet")
    return KINDS[spec["kind"]](spec, cfg)
