"""Closed phrase-list recognition on top of Auto-AVSR.

Instead of free decoding, every phrase in a fixed list is scored against the
video (teacher-forced decoder + CTC log-likelihood, one batched pass) and the
best matches become candidates. The model still only ranks phrases a person
wrote down in advance; it never generates or edits words.
"""

from __future__ import annotations

import numpy as np
import torch

# log P for a phrase the clip cannot contain (more tokens than CTC can align to its frames); the same value and
# meaning as asr_eval/harness/models.py IMPOSSIBLE. Finite, so evidence JSON stays valid; it always ranks last.
IMPOSSIBLE = -1e6


def tokenize(text: str, piece_to_id: dict[str, int]) -> list[int]:
    """Greedy longest-match over the model's unigram5000 pieces."""
    ids = []
    for word in text.upper().split():
        s = "▁" + word
        while s:
            for end in range(len(s), 0, -1):
                if s[:end] in piece_to_id:
                    ids.append(piece_to_id[s[:end]])
                    s = s[end:]
                    break
            else:
                ids.append(piece_to_id["<unk>"])
                s = s[1:]
    return ids


def score_phrases(model, enc: torch.Tensor, phrase_tokens: list[list[int]], ctc_weight: float = 0.3) -> torch.Tensor:
    """log p(phrase | video) for every phrase in the list."""
    from espnet.nets.pytorch_backend.nets_utils import pad_list
    from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
    from espnet.nets.pytorch_backend.transformer.mask import target_mask

    n = len(phrase_tokens)
    ys = pad_list([torch.tensor(t) for t in phrase_tokens], model.ignore_id).to(enc.device)
    ys_in, ys_out = add_sos_eos(ys, model.sos, model.eos, model.ignore_id)
    hs = enc.unsqueeze(0).expand(n, -1, -1)
    logp = torch.log_softmax(model.decoder(ys_in, target_mask(ys_in, model.ignore_id), hs, None)[0], dim=-1)
    valid = ys_out != model.ignore_id
    att = (logp.gather(-1, ys_out.clamp(min=0).unsqueeze(-1)).squeeze(-1) * valid).sum(-1)

    ctc_lp = torch.log_softmax(model.ctc.ctc_lo(enc), dim=-1).cpu()  # CTC loss runs on CPU
    ctc = ctc_logp(ctc_lp, phrase_tokens)
    out = (1 - ctc_weight) * att.cpu() + ctc_weight * ctc
    return torch.where(ctc <= IMPOSSIBLE, torch.full_like(out, IMPOSSIBLE), out)


def ctc_logp(ctc_lp: torch.Tensor, phrase_tokens: list[list[int]]) -> torch.Tensor:
    """CTC log P(phrase | frames) for every phrase, from one clip's (frames, vocab) log-probabilities.

    An alignment that cannot exist gets IMPOSSIBLE. (zero_infinity=True used to turn its infinite loss into 0, a
    perfect score, so an impossible phrase could rank first.)"""
    n = len(phrase_tokens)
    targets = torch.cat([torch.tensor(t) for t in phrase_tokens])
    nll = torch.nn.functional.ctc_loss(
        ctc_lp.unsqueeze(1).expand(-1, n, -1), targets, torch.full((n,), len(ctc_lp)),
        torch.tensor([len(t) for t in phrase_tokens]), reduction="none", zero_infinity=False)
    return torch.where(torch.isfinite(nll), -nll, torch.full_like(nll, IMPOSSIBLE))


def dtw(a: np.ndarray, b: np.ndarray) -> float:
    """Length-normalised DTW over cosine distance (anti-diagonal vectorised)."""
    d = 1.0 - a @ b.T
    n, m = d.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for s in range(2, n + m + 1):
        i = np.arange(max(1, s - m), min(n, s - 1) + 1)
        j = s - i
        acc[i, j] = d[i - 1, j - 1] + np.minimum(np.minimum(acc[i - 1, j], acc[i, j - 1]), acc[i - 1, j - 1])
    return float(acc[n, m] / (n + m))


def dtw_batch(a: np.ndarray, bs: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """`dtw(a, bs[k, :lens[k]])` for every k in one pass: the templates are zero-padded to one length and their
    cost matrices filled together, one anti-diagonal at a time. Padding only adds cells past each template's end,
    which its own result never reads, so every distance is the one `dtw` returns."""
    d = 1.0 - a @ bs.transpose(0, 2, 1)
    k, n, m = d.shape
    acc = np.full((k, n + 1, m + 1), np.inf)
    acc[:, 0, 0] = 0.0
    for s in range(2, n + m + 1):
        i = np.arange(max(1, s - m), min(n, s - 1) + 1)
        j = s - i
        acc[:, i, j] = d[:, i - 1, j - 1] + np.minimum(np.minimum(acc[:, i - 1, j], acc[:, i, j - 1]), acc[:, i - 1, j - 1])
    return acc[np.arange(k), n, lens] / (n + lens)
