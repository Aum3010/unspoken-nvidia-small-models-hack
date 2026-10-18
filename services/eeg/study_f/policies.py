"""Policies that drive a `neuro_comm` episode through the replay tools: Bayesian fast path, Nemotron tool-calling agent,
and the hybrid (Bayesian fast path + deterministic escalation to Nemotron).

All policies talk to an env object with `call(tool, args) -> dict` (in-process `LocalEnv` or `HttpEnv` against the
Gym resources server), so the same code path is exercised for every policy. Policies see only tool responses.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import httpx

from services.eeg.study_f.replay import EpisodeEvidence, ReplaySession, RewardConfig

TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "get_state", "description": "Read the current EEG evidence state (no new evidence).",
                                      "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "request_more_eeg", "description": (
        "Reveal the next recorded repetition of EEG evidence for this selection. Costs recorded acquisition time (about 2.5 s). "
        "Rejected when evidence_exhausted is true."), "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "commit_character", "description": "Commit the selected character. Final; ends the episode.",
                                      "parameters": {"type": "object", "properties": {"character": {"type": "string", "description": "A grid label from `grid`."}},
                                                     "required": ["character"]}}},
    {"type": "function", "function": {"name": "abstain", "description": "Decline to select a character (ask the user to retry another way). Final; ends the episode.",
                                      "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "request_other_modality", "description": (
        "Ask for lips, voice, gaze or speech evidence. In this replay none is recorded, so this always fails and is penalised."),
        "parameters": {"type": "object", "properties": {"modality": {"type": "string"}}, "required": ["modality"]}}},
]


def system_prompt(cfg: RewardConfig) -> str:
    return (
        "You control a P300 speller for a person with ALS. One episode = one character selection from a 36-character grid. "
        "EEG evidence arrives in repetitions; after each, the tools return the posterior over characters, computed by a fixed "
        "decoder. You cannot see the intended character. Decide, using only tool results, when to stop and what to do.\n"
        "Tools: get_state, request_more_eeg, commit_character(character), abstain, request_other_modality(modality). "
        "Call exactly one tool per turn and finish with commit_character or abstain.\n"
        "Scoring: correct commit +%.2f; wrong commit -%.2f; abstain -%.2f; no decision -%.2f; each recorded second of EEG acquired -%.3f "
        "(one repetition is about 2.5 s); each tool call -%.3f; each rejected call -%.2f; each unavailable-modality request -%.2f. "
        "request_more_eeg is rejected once evidence_exhausted is true. Other modalities are never available. "
        "Never invent evidence or characters; commit only grid labels." % (cfg.r_correct, cfg.p_wrong, cfg.p_abstain, cfg.p_timeout,
                                                                           cfg.c_sec, cfg.c_turn, cfg.p_invalid, cfg.p_modality))


# ---- environments ------------------------------------------------------------------------------------------------------

class LocalEnv:
    """In-process env around a ReplaySession (the same class the Gym resources server uses)."""

    def __init__(self, ev: EpisodeEvidence, cfg: RewardConfig | None = None):
        self.session = ReplaySession(ev, cfg)

    def call(self, tool: str, args: dict | None = None) -> dict:
        return self.session.dispatch(tool, args or {})

    @property
    def done(self) -> bool:
        return self.session.done


class HttpEnv:
    """Same tool surface over HTTP against the Gym resources server (real server code, own session cookie)."""

    def __init__(self, client, episode_id: str):
        self.client, self.episode_id, self.cookies, self._done = client, episode_id, None, False
        r = client.post("/seed_session", json={"episode_id": episode_id})
        r.raise_for_status()
        self.cookies = r.cookies

    def call(self, tool: str, args: dict | None = None) -> dict:
        r = self.client.post(f"/{tool}", json=args or {}, cookies=self.cookies)
        r.raise_for_status()
        out = r.json()
        self._done = bool(out.get("observation", {}).get("done", False)) or self._done
        return out

    @property
    def done(self) -> bool:
        return self._done


# ---- Bayesian fast path ------------------------------------------------------------------------------------------------

@dataclass
class BayesPolicy:
    """Deterministic fast path: stop at the first repetition with top posterior >= tau, else commit the argmax when the
    recording is exhausted (the standalone benchmark's `bayes_tau<t>`)."""

    tau: float = 0.99
    name: str = "bayes"

    def run(self, env) -> dict:
        obs = env.call("get_state")["observation"]
        while not env.done:
            if obs["top_posterior"] >= self.tau or obs["evidence_exhausted"]:
                obs = env.call("commit_character", {"character": obs["top_candidate"]})["observation"]
            else:
                obs = env.call("request_more_eeg")["observation"]
        return {"policy": self.name, "model_calls": 0, "model_latency_s": 0.0, "escalated": False}


@dataclass
class BayesAbstainPolicy:
    """Bayesian stopping that abstains instead of committing when the recording ends below tau (the standalone benchmark's
    `bayes_abstain_tau<t>`). This is the strong classical baseline a model has to beat on reward, since abstaining is its
    most obvious gain."""

    tau: float = 0.99
    name: str = "bayes_abstain"

    def run(self, env) -> dict:
        obs = env.call("get_state")["observation"]
        while not env.done:
            if obs["top_posterior"] >= self.tau:
                obs = env.call("commit_character", {"character": obs["top_candidate"]})["observation"]
            elif obs["evidence_exhausted"]:
                obs = env.call("abstain")["observation"]
            else:
                obs = env.call("request_more_eeg")["observation"]
        return {"policy": self.name, "model_calls": 0, "model_latency_s": 0.0, "escalated": False}


# ---- Nemotron (OpenAI-compatible chat/completions with tools: vLLM or Ollama) ----------------------------------------

class OpenAICompatClient:
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", timeout: float = 120.0, temperature: float = 0.0,
                 extra_body: dict | None = None):
        self.url = base_url.rstrip("/") + "/v1/chat/completions" if not base_url.rstrip("/").endswith("/v1") else base_url.rstrip("/") + "/chat/completions"
        self.model, self.temperature, self.extra = model, temperature, extra_body or {}
        self.http = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {api_key}"})

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        body = {"model": self.model, "messages": messages, "tools": tools, "tool_choice": "auto", "temperature": self.temperature,
                **self.extra}
        r = self.http.post(self.url, json=body)
        r.raise_for_status()
        return r.json()


def _parse_calls(msg: dict) -> list[tuple[str, dict, str]]:
    out = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        raw = fn.get("arguments", "{}")
        try:
            args = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
        except (json.JSONDecodeError, TypeError, ValueError):
            args = {"__malformed__": str(raw)}
        out.append((fn.get("name", ""), args, tc.get("id", "")))
    return out


@dataclass
class NemotronAgent:
    """Tool-calling LLM loop. Every model call, tool call, response and latency is recorded in `trace`."""

    llm: object
    cfg: RewardConfig = field(default_factory=RewardConfig)
    max_turns: int = 30
    name: str = "nemotron"
    trace: list = field(default_factory=list)

    def run(self, env, initial_observation: dict | None = None, preamble: str = "") -> dict:
        msgs = [{"role": "system", "content": system_prompt(self.cfg)}]
        if initial_observation is None:
            msgs.append({"role": "user", "content": preamble + "Decide for this selection. Start by calling get_state."})
        else:
            msgs.append({"role": "user", "content": preamble + "Current state (from the deterministic fast path):\n"
                         + json.dumps(initial_observation) + "\nDecide: request_more_eeg, commit_character, or abstain."})
        model_calls, model_latency, invalid_calls = 0, 0.0, 0
        for _ in range(self.max_turns):
            if env.done:
                break
            t0 = time.perf_counter()
            try:
                resp = self.llm.chat(msgs, TOOL_SCHEMAS)
            except Exception as e:  # endpoint failure: recorded, episode left to the caller's fallback
                self.trace.append({"event": "llm_error", "error": f"{type(e).__name__}: {e}"})
                break
            dt = time.perf_counter() - t0
            model_calls += 1
            model_latency += dt
            msg = resp["choices"][0]["message"]
            calls = _parse_calls(msg)
            self.trace.append({"event": "model", "latency_s": round(dt, 4), "content": (msg.get("content") or "")[:500],
                               "tool_calls": [{"name": n, "arguments": a} for n, a, _ in calls],
                               "usage": resp.get("usage")})
            msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": msg.get("tool_calls") or []})
            if not calls:
                msgs.append({"role": "user", "content": "Call exactly one tool."})
                invalid_calls += 1
                continue
            name, args, cid = calls[0]                       # one tool per turn; extras ignored and recorded
            result = env.call(name, args)
            if not result.get("ok", False):
                invalid_calls += 1
            self.trace.append({"event": "tool", "name": name, "arguments": args, "ok": result.get("ok"), "error": result.get("error")})
            msgs.append({"role": "tool", "tool_call_id": cid or "0", "content": json.dumps(result)})
            if len(calls) > 1:
                self.trace.append({"event": "extra_tool_calls_ignored", "count": len(calls) - 1})
        return {"policy": self.name, "model_calls": model_calls, "model_latency_s": model_latency, "escalated": True,
                "invalid_model_calls": invalid_calls}


# ---- Hybrid -------------------------------------------------------------------------------------------------------------

@dataclass
class HybridPolicy:
    """Bayesian fast path with deterministic escalation to Nemotron.

    Per repetition, using only the tool observation:
      commit (no model call)   if top_posterior >= tau_commit
      escalate  T1 (exhausted) if evidence_exhausted and top_posterior < tau_commit   -> Nemotron: commit or abstain
      escalate  T2 (stalled)   if repetition >= r_stall and top_posterior < tau_stall -> Nemotron: more / commit / abstain
      otherwise request_more_eeg
    Escalation is final for the episode. If Nemotron fails to reach a terminal action, fallback = commit the current argmax.
    """

    agent: NemotronAgent
    tau_commit: float = 0.99
    r_stall: int = 7
    tau_stall: float = 0.9
    use_stall_trigger: bool = True
    name: str = "bayes+nemotron"

    def trigger(self, obs: dict) -> str | None:
        if obs["top_posterior"] >= self.tau_commit:
            return None
        if obs["evidence_exhausted"]:
            return "T1_exhausted_below_threshold"
        if self.use_stall_trigger and obs["repetition"] >= self.r_stall and obs["top_posterior"] < self.tau_stall:
            return "T2_stalled"
        return None

    def run(self, env) -> dict:
        obs = env.call("get_state")["observation"]
        while not env.done:
            if obs["top_posterior"] >= self.tau_commit:
                obs = env.call("commit_character", {"character": obs["top_candidate"]})["observation"]
                continue
            trig = self.trigger(obs)
            if trig is None:
                obs = env.call("request_more_eeg")["observation"]
                continue
            info = self.agent.run(env, initial_observation=obs)
            info.update(policy=self.name, trigger=trig, escalated=True, fallback_used=False)
            if not env.done:                                    # model failed to decide: deterministic fallback
                s = getattr(env, "session", None)
                cur = env.call("get_state")["observation"]
                env.call("commit_character", {"character": cur["top_candidate"]})
                info["fallback_used"] = True
            return info
        return {"policy": self.name, "model_calls": 0, "model_latency_s": 0.0, "escalated": False, "trigger": None, "fallback_used": False}
