"""Closed phrase-list scoring (services/vsr/phrases.py, only used with UNSPOKEN_VSR_PHRASES set): a phrase the clip
cannot contain must never outrank one it can.

Run: .venv/bin/python -m pytest evaluations/safety_tests -q
"""

import torch

from services.vsr.phrases import IMPOSSIBLE, ctc_logp


def frames(n_frames: int, vocab: int = 6) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.log_softmax(torch.randn(n_frames, vocab), dim=-1)


def test_impossible_alignment_scores_impossible():
    lp = frames(3)
    scores = ctc_logp(lp, [[1], [1, 2, 3, 4, 5]])   # 5 tokens cannot align to 3 frames
    assert scores[1].item() == IMPOSSIBLE
    assert IMPOSSIBLE < scores[0].item() < 0


def test_impossible_alignment_never_ranks_first():
    lp = frames(4)
    phrases = [[1, 2], [1, 1, 1, 1, 1, 1], [3], [2, 3, 4, 5, 1, 2, 3]]
    scores = ctc_logp(lp, phrases)
    assert scores.argmax().item() in (0, 2)
    assert scores.argsort(descending=True)[:2].tolist() == sorted([0, 2], key=lambda i: -scores[i].item())


def test_feasible_scores_unchanged():
    lp = frames(8)
    phrases = [[1, 2], [3, 4, 5]]
    expected = -torch.nn.functional.ctc_loss(lp.unsqueeze(1).expand(-1, 2, -1), torch.tensor([1, 2, 3, 4, 5]),
                                             torch.full((2,), 8), torch.tensor([2, 3]), reduction="none")
    assert torch.allclose(ctc_logp(lp, phrases), expected)
