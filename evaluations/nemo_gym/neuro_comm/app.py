"""NeMo Gym resources server: `neuro_comm`.

Stateful EEG replay for sequential communication decisions on real recorded ALS EEG (bigP3BCI Study F). One episode = one
recorded character selection, replayed from frozen evidence (see services/eeg/study_f/replay.py for the visibility
contract). The model sees only tool responses; the intended target is read only inside `verify`.

Tools (POST, one session per rollout via the session cookie): get_state, request_more_eeg, commit_character,
abstain, request_other_modality. Seed body: {"episode_id": "<selection id>"}.

Run from a NeMo Gym checkout with the Unspoken repo importable (PYTHONPATH=<unspoken> or UNSPOKEN_REPO=<unspoken>).
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from nemo_gym.base_resources_server import (
    BaseMultiRewardVerifyResponse,
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY

_repo = os.environ.get("UNSPOKEN_REPO") or str(Path(__file__).resolve().parents[3])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from services.eeg.study_f import replay  # noqa: E402


class NeuroCommConfig(BaseResourcesServerConfig):
    evidence_dir: str = "data/evidence/study_f"
    decoder: str = "xdawn_riemann_lr"
    roles: List[str] = Field(default_factory=lambda: ["calibration", "evaluation", "evaluation_secondary"])
    reward: Dict[str, float] = Field(default_factory=dict)   # overrides of replay.RewardConfig
    rollout_log: Optional[str] = None                       # append one JSON line per finished episode


class SeedRequest(BaseSeedSessionRequest):
    episode_id: str


class CommitRequest(BaseModel):
    character: str


class ModalityRequest(BaseModel):
    modality: str


class ToolResponse(BaseModel):
    ok: bool
    error: Optional[str] = None
    observation: Dict[str, Any] = Field(default_factory=dict)
    committed: Optional[str] = None
    abstained: Optional[bool] = None


class NeuroCommVerifyRequest(BaseVerifyRequest):
    episode_id: str


class NeuroCommVerifyResponse(BaseMultiRewardVerifyResponse):
    episode_id: str = ""
    decision: str = ""
    correct: float = 0.0
    incorrect_commit: float = 0.0
    abstained: float = 0.0
    reps: float = 0.0
    acq_s: float = 0.0
    n_calls: float = 0.0
    n_invalid: float = 0.0


class NeuroCommResourcesServer(SimpleResourcesServer):
    ray_enabled = False
    config: NeuroCommConfig
    sessions: Dict[str, Any] = Field(default_factory=dict)
    episodes: Dict[str, Any] = Field(default_factory=dict)
    ground_truth: Dict[str, int] = Field(default_factory=dict)   # verifier-only; never serialised into a tool response

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        evd = Path(os.environ.get("UNSPOKEN_EVIDENCE_DIR") or self.config.evidence_dir)   # env var wins: the Gym cwd is not the repo
        self.episodes = replay.load_episodes(evd, self.config.decoder, tuple(self.config.roles))
        self.ground_truth = replay.load_ground_truth(evd)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        for name in ("get_state", "request_more_eeg", "abstain"):
            app.post(f"/{name}")(getattr(self, f"_{name}"))
        app.post("/commit_character")(self._commit_character)
        app.post("/request_other_modality")(self._request_other_modality)
        return app

    # ---- session ------------------------------------------------------------------------------------------------------
    async def seed_session(self, request: Request, body: SeedRequest) -> BaseSeedSessionResponse:
        sid = request.session[SESSION_ID_KEY]
        ev = self.episodes.get(body.episode_id)
        if ev is None:
            raise HTTPException(status_code=400, detail=f"unknown episode {body.episode_id!r} (not in roles {self.config.roles})")
        self.sessions[sid] = replay.ReplaySession(ev, replay.RewardConfig.from_dict(self.config.reward))
        return BaseSeedSessionResponse()

    def _session(self, request: Request):
        return self.sessions.get(request.session[SESSION_ID_KEY])

    def _tool(self, request: Request, tool: str, args: dict) -> ToolResponse:
        s = self._session(request)
        if s is None:
            return ToolResponse(ok=False, error="no episode seeded for this session")
        return ToolResponse(**s.dispatch(tool, args))

    async def _get_state(self, request: Request) -> ToolResponse:
        return self._tool(request, "get_state", {})

    async def _request_more_eeg(self, request: Request) -> ToolResponse:
        return self._tool(request, "request_more_eeg", {})

    async def _abstain(self, request: Request) -> ToolResponse:
        return self._tool(request, "abstain", {})

    async def _commit_character(self, request: Request, body: CommitRequest) -> ToolResponse:
        return self._tool(request, "commit_character", {"character": body.character})

    async def _request_other_modality(self, request: Request, body: ModalityRequest) -> ToolResponse:
        return self._tool(request, "request_other_modality", {"modality": body.modality})

    # ---- verifier -----------------------------------------------------------------------------------------------------
    async def verify(self, request: Request, body: NeuroCommVerifyRequest) -> NeuroCommVerifyResponse:
        sid = request.session[SESSION_ID_KEY]
        s = self.sessions.get(sid)
        if s is None or body.episode_id != s.ev.episode_id:
            comps = {k: 0.0 for k in ("correct", "incorrect_commit", "abstain", "timeout", "acquisition", "turns", "invalid_calls", "modality_requests")}
            return NeuroCommVerifyResponse(**body.model_dump(), reward=0.0, reward_components=comps, decision="no_session",
                                           mask_sample=True, failure_kind=None, failure_reason="no seeded session for this episode")
        out = s.outcome(self.ground_truth[body.episode_id])
        log = os.environ.get("UNSPOKEN_ROLLOUT_LOG") or self.config.rollout_log
        if log:                                                  # logging must never change or fail a verification
            try:
                Path(log).parent.mkdir(parents=True, exist_ok=True)
                with open(log, "a") as f:
                    f.write(json.dumps({"ts": time.time(), "episode_id": body.episode_id, "outcome": out, "events": s.events}) + "\n")
            except OSError:
                pass
        self.sessions.pop(sid, None)
        return NeuroCommVerifyResponse(
            **body.model_dump(), reward=out["reward"], reward_components=out["components"], decision=out["decision"],
            correct=float(out["correct"]), incorrect_commit=float(out["incorrect_commit"]), abstained=float(out["abstained"]),
            reps=float(out["reps"]), acq_s=out["acq_s"], n_calls=float(out["n_calls"]), n_invalid=float(out["n_invalid"]))


if __name__ == "__main__":
    NeuroCommResourcesServer.run_webserver()
