"""Stateful EEG replay for the `neuro_comm` environment (pure Python; no NeMo Gym import).

One episode = one recorded Study F character selection, replayed from the frozen evidence artifacts
(`data/evidence/study_f/`, built by scripts/benchmark/build_study_f_evidence.py). Nothing is simulated.

Visibility contract
  agent-visible : `EpisodeEvidence` (per-repetition character log-evidence increments, recorded timestamps, grid labels)
  verifier-only : the intended target, held in a separate `GroundTruth` mapping that `ReplaySession` never receives.
Every tool response is built from `EpisodeEvidence` alone, so a policy cannot read the target through the tools.

Replay constraint: only repetitions that were recorded can be revealed. `request_more_eeg` past the last recorded
repetition is rejected (counted as an invalid call). No audio, gaze or speech evidence exists for these episodes, so
`request_other_modality` always answers "unavailable"; it never fabricates evidence.

Posterior shown to the agent: softmax(L(r)) with alpha = 1 (plain Bayesian rule, the same posterior the standalone
benchmark's `bayes_tau*` policies use). Time fields are recorded stimulus timestamps, never wall-clock replay time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from services.eeg.study_f import stopping as st

SOA_S, EPOCH_END_S = 0.25, 0.7
MODALITIES_KNOWN = ("lips", "voice", "gaze", "speech")
TOOLS = ("get_state", "request_more_eeg", "commit_character", "abstain", "request_other_modality")


@dataclass
class RewardConfig:
    """Reward = outcome - cost terms. All coefficients are configurable (see docs/NEUROCOMM_GYM_EXPERIMENT.md)."""

    r_correct: float = 1.0        # + correct commit
    p_wrong: float = 1.0          # - incorrect commit
    p_abstain: float = 0.2        # - abstention (a safe but non-communicating outcome)
    p_timeout: float = 0.5        # - episode ended without commit/abstain (step budget, or recording ran out untouched)
    c_sec: float = 0.01           # - per recorded second of EEG acquisition consumed (first flash onset .. last used flash + SOA)
    c_turn: float = 0.005         # - per tool call (one model turn each): penalises unnecessary model/tool calls
    p_invalid: float = 0.1        # - per rejected call (bad character, beyond recording, unknown tool, after the end)
    p_modality: float = 0.05      # - per request for an unavailable modality
    max_calls: int = 40           # hard step budget

    @classmethod
    def from_dict(cls, d: dict | None) -> "RewardConfig":
        d = d or {}
        return cls(**{k: float(v) if k != "max_calls" else int(v) for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class EpisodeEvidence:
    """Agent-visible evidence of one selection. No target field exists on this type."""

    episode_id: str
    participant_id: str
    role: str
    L: np.ndarray            # (n_avail, 36) cumulative character log-evidence after each recorded repetition
    first_onset_s: float
    last_onset_s: np.ndarray  # (14,) recorded onset of the last flash of each repetition (NaN if not recorded)
    n_flashes: np.ndarray     # (14,)
    chars: list[str]          # grid labels in channel order

    @property
    def n_avail(self) -> int:
        return len(self.L)


def load_char_labels(evd: Path) -> list[str]:
    p = Path(evd) / "char_labels.json"
    if not p.exists():
        p = Path(__file__).resolve().parents[3] / "evaluations" / "dataset_manifests" / "study_f_char_labels.json"
    return json.loads(p.read_text())


def load_episodes(evd: Path, decoder: str = "xdawn_riemann_lr", roles: tuple[str, ...] = ("evaluation",)) -> dict[str, EpisodeEvidence]:
    import pandas as pd

    evd = Path(evd)
    z = np.load(evd / f"rep_evidence_{decoder}.npz", allow_pickle=False)
    rl = pd.read_csv(evd / "selection_roles.csv").set_index("selection_id")
    chars = load_char_labels(evd)
    out = {}
    for i, sid in enumerate(z["selection_id"]):
        sid = str(sid)
        if rl.loc[sid, "role"] not in roles:
            continue
        d = z["delta"][i]
        ok = ~np.isnan(d).any(1)
        n = int(ok.argmin()) if not ok.all() else len(ok)
        if n == 0:
            continue
        out[sid] = EpisodeEvidence(sid, str(rl.loc[sid, "participant_id"]), str(rl.loc[sid, "role"]), np.cumsum(d[:n], 0),
                                   float(z["first_onset_s"][i]), z["last_onset_s"][i].copy(), z["n_flashes"][i].copy(), chars)
    return out


def load_ground_truth(evd: Path) -> dict[str, int]:
    """Verifier-only: selection_id -> 0-based target character index."""
    import pandas as pd

    gt = pd.read_csv(Path(evd) / "ground_truth.csv").set_index("selection_id")
    return {str(k): int(v) - 1 for k, v in gt.target_index.items()}


class ReplaySession:
    """State of one episode. Holds agent-visible evidence only; the verifier supplies the target to `outcome()`."""

    def __init__(self, ev: EpisodeEvidence, cfg: RewardConfig | None = None):
        self.ev, self.cfg = ev, cfg or RewardConfig()
        self.r = 1                       # repetitions revealed so far (episode starts with the first recorded repetition)
        self.done = False
        self.decision: dict | None = None  # {"type": "commit"|"abstain", "char": idx|None, "reps": r}
        self.n_calls = self.n_invalid = self.n_modality = self.n_more = 0
        self.events: list[dict] = []

    # ---- observation (built from agent-visible evidence only) -------------------------------------------------------
    def _timing(self, r: int) -> tuple[float, float, int]:
        last = float(self.ev.last_onset_s[r - 1])
        return last - self.ev.first_onset_s + SOA_S, last - self.ev.first_onset_s + EPOCH_END_S, int(self.ev.n_flashes[:r].sum())

    def observation(self) -> dict:
        r, n = self.r, self.ev.n_avail
        p = st.softmax(self.ev.L[r - 1])
        order = np.argsort(-p)
        acq, lat, flashes = self._timing(r)
        top = [{"character": self.ev.chars[i], "posterior": round(float(p[i]), 6)} for i in order[:5]]
        ent = float(-(p * np.log2(np.clip(p, 1e-12, None))).sum())
        return {
            "episode_id": self.ev.episode_id, "repetition": r, "repetitions_remaining": n - r, "evidence_exhausted": r >= n,
            "top_candidate": top[0]["character"], "top_posterior": top[0]["posterior"],
            "second_posterior": top[1]["posterior"], "top_candidates": top, "entropy_bits": round(ent, 4),
            "acquisition_s_so_far": round(acq, 3), "decision_latency_s_so_far": round(lat, 3), "flashes_so_far": flashes,
            "previous_characters": "unavailable: this episode is a single selection; no emitted text exists",
            "other_modalities": "unavailable: no recorded audio, gaze or speech for this episode",
            "grid": self.ev.chars, "done": self.done,
        }

    def _log(self, tool: str, args: dict, resp: dict, ok: bool) -> dict:
        self.events.append({"call": self.n_calls, "tool": tool, "args": args, "ok": ok, "repetition": self.r,
                            "done": self.done, "response_keys": sorted(resp)})
        return resp

    def _reject(self, tool: str, args: dict, why: str) -> dict:
        self.n_invalid += 1
        return self._log(tool, args, {"ok": False, "error": why, "observation": self.observation()}, False)

    def _call(self, tool: str, args: dict) -> dict | None:
        """Common gate. Returns a rejection response, or None when the call may proceed."""
        self.n_calls += 1
        if self.done:
            return self._reject(tool, args, "episode already finished")
        if self.n_calls > self.cfg.max_calls:
            self.done, self.decision = True, {"type": "timeout", "char": None, "reps": self.r}
            return self._reject(tool, args, "step budget exhausted; episode ended without a decision")
        return None

    # ---- tools -----------------------------------------------------------------------------------------------------
    def get_state(self) -> dict:
        return self._call("get_state", {}) or self._log("get_state", {}, {"ok": True, "observation": self.observation()}, True)

    def request_more_eeg(self) -> dict:
        rej = self._call("request_more_eeg", {})
        if rej:
            return rej
        if self.r >= self.ev.n_avail:
            return self._reject("request_more_eeg", {}, "evidence exhausted: no further repetitions were recorded for this selection")
        self.r += 1
        self.n_more += 1
        return self._log("request_more_eeg", {}, {"ok": True, "observation": self.observation()}, True)

    def commit_character(self, character: str) -> dict:
        args = {"character": character}
        rej = self._call("commit_character", args)
        if rej:
            return rej
        if character not in self.ev.chars:
            return self._reject("commit_character", args, f"unknown character; valid: {self.ev.chars}")
        self.done = True
        self.decision = {"type": "commit", "char": self.ev.chars.index(character), "reps": self.r}
        return self._log("commit_character", args, {"ok": True, "committed": character, "observation": self.observation()}, True)

    def abstain(self) -> dict:
        rej = self._call("abstain", {})
        if rej:
            return rej
        self.done = True
        self.decision = {"type": "abstain", "char": None, "reps": self.r}
        return self._log("abstain", {}, {"ok": True, "abstained": True, "observation": self.observation()}, True)

    def request_other_modality(self, modality: str) -> dict:
        args = {"modality": modality}
        rej = self._call("request_other_modality", args)
        if rej:
            return rej
        self.n_modality += 1
        return self._log("request_other_modality", args, {
            "ok": False, "error": f"modality '{modality}' unavailable: no recorded {modality} evidence exists for this episode; "
                                  "nothing was fabricated", "observation": self.observation()}, False)

    def dispatch(self, tool: str, args: dict | None = None) -> dict:
        """Single entry point used by policies and the HTTP server: unknown tools and bad arguments are rejected, not raised."""
        args = args or {}
        if tool == "get_state":
            return self.get_state()
        if tool == "request_more_eeg":
            return self.request_more_eeg()
        if tool == "abstain":
            return self.abstain()
        if tool == "commit_character" and isinstance(args.get("character"), str):
            return self.commit_character(args["character"])
        if tool == "request_other_modality" and isinstance(args.get("modality"), str):
            return self.request_other_modality(args["modality"])
        rej = self._call(tool, args)
        return rej or self._reject(tool, args, f"unknown tool or malformed arguments: {tool}({sorted(args)})")

    # ---- verifier (the only place the target is used) ----------------------------------------------------------------
    def outcome(self, target: int) -> dict:
        c = self.cfg
        dec = self.decision or {"type": "timeout", "char": None, "reps": self.r}
        committed = dec["type"] == "commit"
        correct = bool(committed and dec["char"] == target)
        r = min(dec["reps"], self.ev.n_avail)
        acq, lat, flashes = self._timing(r)
        comp = {
            "correct": c.r_correct if correct else 0.0,
            "incorrect_commit": -c.p_wrong if committed and not correct else 0.0,
            "abstain": -c.p_abstain if dec["type"] == "abstain" else 0.0,
            "timeout": -c.p_timeout if dec["type"] == "timeout" else 0.0,
            "acquisition": -c.c_sec * acq,
            "turns": -c.c_turn * self.n_calls,
            "invalid_calls": -c.p_invalid * self.n_invalid,
            "modality_requests": -c.p_modality * self.n_modality,
        }
        return {"reward": float(sum(comp.values())), "components": comp, "decision": dec["type"], "committed": committed,
                "correct": correct, "incorrect_commit": bool(committed and not correct), "abstained": dec["type"] == "abstain",
                "timed_out": dec["type"] == "timeout", "reps": r, "flashes": flashes, "acq_s": acq, "latency_s": lat,
                "exhausted_at_stop": r >= self.ev.n_avail, "n_calls": self.n_calls, "n_invalid": self.n_invalid,
                "n_modality_requests": self.n_modality, "n_more_eeg": self.n_more, "config": asdict(c)}
