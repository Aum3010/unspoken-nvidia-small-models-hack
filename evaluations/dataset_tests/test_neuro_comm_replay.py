"""neuro_comm replay, reward, leakage, policy-plumbing and Bayesian-parity tests.

The scripted LLM below is a *test double* for the chat endpoint: it exercises the tool-calling plumbing only. It is never
used to produce reported results."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.eeg.study_f import policies as pol  # noqa: E402
from services.eeg.study_f import replay  # noqa: E402
from services.eeg.study_f import stopping as st  # noqa: E402

EVD = ROOT / "data" / "evidence" / "study_f"
HAVE_EVD = (EVD / "manifest.json").exists()
CHARS = [f"c{i}" for i in range(36)]


def synth(n_reps=6, winner=3, gain=1.0, pid="P", sid="e0", noise=0.0, seed=0) -> replay.EpisodeEvidence:
    rng = np.random.default_rng(seed)
    d = rng.normal(0, noise, (n_reps, 36))
    d[:, winner] += gain
    last = np.full(14, np.nan); last[:n_reps] = 2.0 + 2.25 * np.arange(n_reps)
    nf = np.zeros(14, int); nf[:n_reps] = 9
    return replay.EpisodeEvidence(sid, pid, "evaluation", np.cumsum(d, 0), 0.0, last, nf, CHARS)


def env_for(ev=None, **cfg):
    return pol.LocalEnv(ev or synth(), replay.RewardConfig(**cfg))


# ---- replay semantics --------------------------------------------------------------------------------------------------

def test_episode_starts_at_first_recorded_repetition_and_reveals_in_order():
    e = env_for()
    o = e.call("get_state")["observation"]
    assert o["repetition"] == 1 and o["repetitions_remaining"] == 5 and not o["evidence_exhausted"]
    assert e.call("request_more_eeg")["observation"]["repetition"] == 2


def test_request_beyond_recording_is_rejected_and_counted():
    e = env_for(synth(n_reps=3))
    for _ in range(2):
        assert e.call("request_more_eeg")["ok"]
    out = e.call("request_more_eeg")
    assert not out["ok"] and "exhausted" in out["error"] and out["observation"]["repetition"] == 3
    assert e.session.n_invalid == 1


def test_evidence_exhausted_flag_and_timing_use_recorded_timestamps():
    e = env_for(synth(n_reps=2))
    e.call("request_more_eeg")
    o = e.call("get_state")["observation"]
    assert o["evidence_exhausted"] and o["flashes_so_far"] == 18
    assert o["acquisition_s_so_far"] == pytest.approx(4.25 - 0.0 + replay.SOA_S)        # last onset 4.25 s, first 0 s
    assert o["decision_latency_s_so_far"] == pytest.approx(4.25 + replay.EPOCH_END_S)


def test_other_modalities_never_fabricated():
    e = env_for()
    for m in ("gaze", "lips", "voice", "speech", "telepathy"):
        out = e.call("request_other_modality", {"modality": m})
        assert out["ok"] is False and "unavailable" in out["error"] and out["observation"]["repetition"] == 1
    assert e.session.n_modality == 5 and not e.done


def test_previous_characters_reported_unavailable_not_invented():
    o = env_for().call("get_state")["observation"]
    assert o["previous_characters"].startswith("unavailable")


def test_invalid_inputs_are_rejected_not_raised():
    e = env_for()
    assert not e.call("commit_character", {"character": "not-a-char"})["ok"]
    assert not e.call("commit_character", {})["ok"]
    assert not e.call("launch_missiles", {"x": 1})["ok"]
    assert not e.call("request_other_modality", {"modality": 5})["ok"]
    assert e.session.n_invalid == 4 and not e.done


def test_calls_after_the_end_and_step_budget():
    e = env_for(max_calls=3)
    e.call("abstain")
    assert not e.call("request_more_eeg")["ok"]
    e = env_for(max_calls=3)
    for _ in range(4):
        e.call("get_state")
    assert e.done and e.session.decision["type"] == "timeout"


# ---- verifier / reward ------------------------------------------------------------------------------------------------

def finish(commit_after=0, char=3, **cfg):
    e = env_for(synth(), **cfg)
    for _ in range(commit_after):
        e.call("request_more_eeg")
    e.call("commit_character", {"character": CHARS[char]})
    return e.session.outcome(3)


def test_reward_components_and_sign_conventions():
    ok, bad = finish(char=3), finish(char=4)
    assert ok["correct"] and ok["reward"] > 0.9 and ok["components"]["incorrect_commit"] == 0
    assert bad["incorrect_commit"] and bad["components"]["incorrect_commit"] == -1.0 and bad["reward"] < -1.0
    e = env_for(); e.call("abstain")
    ab = e.session.outcome(3)
    assert ab["abstained"] and ab["components"]["abstain"] == -0.2
    e = env_for(); e.call("get_state")
    to = e.session.outcome(3)
    assert to["timed_out"] and to["components"]["timeout"] == -0.5


def test_reward_coefficients_change_rewards_in_the_documented_direction():
    base = finish(commit_after=3)["reward"]
    assert finish(commit_after=3, c_sec=0.05)["reward"] < base               # acquisition cost
    assert finish(commit_after=3, c_turn=0.05)["reward"] < base              # per-call cost
    assert finish(commit_after=3, r_correct=2.0)["reward"] > base
    assert finish(char=4, p_wrong=3.0)["reward"] < finish(char=4)["reward"]
    # more evidence costs reward at fixed outcome
    assert finish(commit_after=5)["reward"] < finish(commit_after=0)["reward"]
    # unnecessary calls cost reward
    e = env_for(); e.call("request_other_modality", {"modality": "gaze"}); e.call("get_state")
    e.call("commit_character", {"character": CHARS[3]})
    assert e.session.outcome(3)["reward"] < finish()["reward"]


def test_penalties_make_abstain_optimal_only_when_commit_is_too_risky():
    """With p_wrong high enough and a coin-flip posterior, expected commit reward < abstain reward (and not otherwise)."""
    def ev_commit(p_right, p_wrong, p_abs=0.2):
        return p_right * 1.0 - (1 - p_right) * p_wrong, -p_abs
    for pw, expect_abstain in ((0.2, False), (1.0, True), (3.0, True)):
        c, a = ev_commit(0.3, pw)   # abstain wins iff p < (p_wrong - p_abstain) / (1 + p_wrong)
        assert (a > c) == expect_abstain
    assert ev_commit(0.95, 1.0)[0] > ev_commit(0.95, 1.0)[1]


# ---- leakage / safety -------------------------------------------------------------------------------------------------

def test_agent_visible_types_have_no_target_field():
    assert not any("target" in f.lower() or "truth" in f.lower() for f in replay.EpisodeEvidence.__dataclass_fields__)
    s = env_for().session
    assert not any("target" in k.lower() or "truth" in k.lower() for k in vars(s))


def test_tool_trace_is_independent_of_ground_truth():
    """Same evidence, different (hidden) targets -> byte-identical tool responses for an identical call sequence."""
    def trace(truth):
        e = env_for(synth(noise=0.5, seed=4))
        out = [e.call("get_state"), e.call("request_more_eeg"), e.call("request_other_modality", {"modality": "gaze"}),
               e.call("commit_character", {"character": CHARS[5]})]
        e.session.outcome(truth)
        return json.dumps(out, sort_keys=True)
    assert trace(0) == trace(17)


def test_observation_values_come_from_posterior_not_target():
    e = env_for(synth(winner=9, gain=2.0))
    o = e.call("get_state")["observation"]
    assert o["top_candidate"] == CHARS[9]       # the *evidence's* top candidate (here also what the decoder favours)
    assert o["top_posterior"] == pytest.approx(float(st.softmax(synth(winner=9, gain=2.0).L[0]).max()), abs=1e-5)


@pytest.mark.skipif(not HAVE_EVD, reason="frozen evidence not built")
def test_task_files_hold_no_targets_and_train_eval_split_is_clean():
    d = ROOT / "evaluations/nemo_gym/neuro_comm/data"
    rl = __import__("pandas").read_csv(EVD / "selection_roles.csv").set_index("selection_id")
    gt = __import__("pandas").read_csv(EVD / "ground_truth.csv")
    seen = {}
    for name in ("train", "eval", "example"):
        rows = [json.loads(l) for l in open(d / f"{name}.jsonl")]
        seen[name] = {r["episode_id"] for r in rows}
        text = "\n".join(open(d / f"{name}.jsonl"))
        assert "target_index" not in text and "target_char" not in text
        assert set(rl.loc[list(seen[name]), "role"]) == ({"calibration"} if name != "eval" else {"evaluation"})
    assert not seen["train"] & seen["eval"] and set(rl.loc[list(seen["train"]), "session"]) <= {1, 2}
    assert set(rl.loc[list(seen["eval"]), "session"]) == {3} and len(seen["eval"]) == 120 and seen["example"] <= seen["train"]


# ---- policies ---------------------------------------------------------------------------------------------------------

def test_bayes_policy_stops_at_threshold_or_commits_at_exhaustion():
    e = env_for(synth(n_reps=6, gain=1.5)); pol.BayesPolicy(0.9).run(e)
    o = e.session.outcome(3)
    assert o["committed"] and o["correct"] and o["reps"] < 6
    e = env_for(synth(n_reps=3, gain=0.01)); pol.BayesPolicy(0.99).run(e)
    o = e.session.outcome(3)
    assert o["committed"] and o["reps"] == 3 and o["exhausted_at_stop"]


def test_bayes_abstain_policy_abstains_only_when_exhausted_below_threshold():
    e = env_for(synth(n_reps=3, gain=0.01)); pol.BayesAbstainPolicy(0.99).run(e)
    o = e.session.outcome(3)
    assert o["abstained"] and o["reps"] == 3 and o["incorrect_commit"] is False
    e = env_for(synth(n_reps=6, gain=1.5)); pol.BayesAbstainPolicy(0.9).run(e)
    assert e.session.outcome(3)["correct"]


class ScriptedLLM:
    """Test double: returns pre-written assistant messages in OpenAI chat format."""

    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def chat(self, messages, tools):
        self.calls += 1
        if not self.script:
            raise RuntimeError("endpoint down")
        item = self.script.pop(0)
        calls = [{"id": f"c{self.calls}_{i}", "type": "function", "function": {"name": n, "arguments": a if isinstance(a, str) else json.dumps(a)}}
                 for i, (n, a) in enumerate(item)]
        return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": calls}}], "usage": {"total_tokens": 10}}


def test_nemotron_agent_executes_real_tool_calls_and_records_everything():
    llm = ScriptedLLM([[("get_state", {})], [("request_more_eeg", {})], [("commit_character", {"character": CHARS[3]})]])
    ag = pol.NemotronAgent(llm); e = env_for()
    info = ag.run(e)
    o = e.session.outcome(3)
    assert o["correct"] and o["reps"] == 2 and info["model_calls"] == 3 and info["model_latency_s"] >= 0
    assert [t["name"] for t in ag.trace if t["event"] == "tool"] == ["get_state", "request_more_eeg", "commit_character"]
    assert all("latency_s" in t for t in ag.trace if t["event"] == "model")


def test_nemotron_invalid_calls_are_counted_and_do_not_end_the_episode():
    llm = ScriptedLLM([[("fly", {})], [("commit_character", "{not json")], [("request_other_modality", {"modality": "voice"})],
                       [("request_more_eeg", {}), ("abstain", {})], [("abstain", {})]])
    ag = pol.NemotronAgent(llm); e = env_for()
    info = ag.run(e)
    o = e.session.outcome(3)
    assert o["abstained"] and o["n_invalid"] == 2 and o["n_modality_requests"] == 1
    assert any(t["event"] == "extra_tool_calls_ignored" for t in ag.trace)


def test_nemotron_respects_evidence_exhaustion():
    llm = ScriptedLLM([[("request_more_eeg", {})]] * 4 + [[("abstain", {})]])
    ag = pol.NemotronAgent(llm); e = env_for(synth(n_reps=2))
    ag.run(e)
    o = e.session.outcome(3)
    assert o["reps"] == 2 and o["n_invalid"] == 3 and o["abstained"]


def test_hybrid_escalation_trigger_is_deterministic():
    h = pol.HybridPolicy(pol.NemotronAgent(ScriptedLLM([])), tau_commit=0.99, r_stall=7, tau_stall=0.9)
    base = {"top_posterior": 0.5, "evidence_exhausted": False, "repetition": 3}
    assert h.trigger({**base}) is None
    assert h.trigger({**base, "top_posterior": 0.995, "evidence_exhausted": True}) is None
    assert h.trigger({**base, "evidence_exhausted": True}).startswith("T1")
    assert h.trigger({**base, "repetition": 7}).startswith("T2")
    assert h.trigger({**base, "repetition": 7, "top_posterior": 0.95}) is None
    assert pol.HybridPolicy(pol.NemotronAgent(ScriptedLLM([])), use_stall_trigger=False).trigger({**base, "repetition": 9}) is None


def test_hybrid_is_identical_to_bayes_when_nothing_escalates_and_makes_no_model_calls():
    ev = synth(n_reps=6, gain=2.0)
    llm = ScriptedLLM([])
    e1, e2 = env_for(ev), env_for(ev)
    pol.BayesPolicy(0.99).run(e1)
    info = pol.HybridPolicy(pol.NemotronAgent(llm), tau_commit=0.99).run(e2)
    assert llm.calls == 0 and not info["escalated"] and e1.session.outcome(3)["reps"] == e2.session.outcome(3)["reps"]


def test_hybrid_escalates_on_exhaustion_and_nemotron_can_abstain():
    ev = synth(n_reps=3, gain=0.02, noise=0.3)
    llm = ScriptedLLM([[("abstain", {})]])
    e = env_for(ev)
    info = pol.HybridPolicy(pol.NemotronAgent(llm), tau_commit=0.99).run(e)
    assert info["escalated"] and info["trigger"].startswith("T1") and e.session.outcome(3)["abstained"] and not info["fallback_used"]


def test_hybrid_falls_back_to_bayes_commit_when_model_endpoint_fails():
    ev = synth(n_reps=3, gain=0.02, noise=0.3)
    e = env_for(ev)
    info = pol.HybridPolicy(pol.NemotronAgent(ScriptedLLM([])), tau_commit=0.99).run(e)   # endpoint raises immediately
    o = e.session.outcome(3)
    assert info["fallback_used"] and o["committed"] and o["reps"] == 3


# ---- Bayesian parity with the standalone benchmark ---------------------------------------------------------------------

@pytest.mark.skipif(not HAVE_EVD, reason="frozen evidence not built")
def test_bayes_through_replay_tools_matches_standalone_benchmark_exactly():
    spec = importlib.util.spec_from_file_location("run_neuro_comm_eval", ROOT / "scripts/benchmark/run_neuro_comm_eval.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    eps = replay.load_episodes(EVD, "xdawn_riemann_lr", ("evaluation",))
    truth = replay.load_ground_truth(EVD)
    ids = sorted(eps)[::6]                                    # 20 episodes across participants keeps the test quick
    d = m.parity(EVD, {i: eps[i] for i in ids}, truth, taus=(0.9, 0.99))
    assert (d.gym_committed == d.std_committed).all() and (d.gym_correct == d.std_correct).all()
    assert (d.gym_reps == d.std_reps).all() and (d.gym_flashes == d.std_flashes).all()
    assert np.allclose(d.gym_acq_s, d.std_acq_s) and np.allclose(d.gym_latency_s, d.std_latency_s)


@pytest.mark.skipif(not HAVE_EVD, reason="frozen evidence not built")
def test_replayed_evidence_matches_benchmark_loader_arrays():
    spec = importlib.util.spec_from_file_location("run_study_f_baselines", ROOT / "scripts/benchmark/run_study_f_baselines.py")
    b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
    items = {i["sid"]: i for i in b.load(EVD, "xdawn_riemann_lr") if i["role"] == "evaluation"}
    eps = replay.load_episodes(EVD, "xdawn_riemann_lr", ("evaluation",))
    assert set(items) == set(eps)
    for sid, ev in eps.items():
        assert np.allclose(items[sid]["L"], ev.L) and items[sid]["n_avail"] == ev.n_avail
