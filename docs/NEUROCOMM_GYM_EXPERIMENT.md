# neuro_comm: NeMo Gym EEG agent experiment

**Question:** can Nemotron improve sequential communication decisions (when to stop, what to commit, when to abstain) over
classical Bayesian stopping on real recorded ALS EEG?

**Status (honest summary):** the environment, the replay tools, the Bayesian baseline *inside* Gym with exact parity, the
Nemotron tool-calling agent and the hybrid coordinator are implemented and tested. **No Nemotron results exist yet**: this
build environment has no reachable Nemotron endpoint and no GPU (details in section 7), so the Nemotron-only and hybrid
policies have been exercised only against a scripted test double, never reported as results. The comparison is one command
away on a machine that serves Nemotron (section 6). No training was run; no model-improvement claim is made.

## 1. What was built

| Piece | Where |
|---|---|
| Replay core (stateful episode, tools, reward, verifier) | `services/eeg/study_f/replay.py` |
| Policies: Bayesian, Bayesian+abstain, Nemotron tool-calling agent, hybrid | `services/eeg/study_f/policies.py` |
| NeMo Gym resources server `neuro_comm` (stateful tools via session cookie, `/verify`, rollout log) | `evaluations/nemo_gym/neuro_comm/app.py` |
| Gym config, task files, README | `evaluations/nemo_gym/neuro_comm/{configs,data,README.md}` |
| Held-out evaluation + Bayesian parity | `scripts/benchmark/run_neuro_comm_eval.py` |
| Live-server check, reward sweep | `scripts/benchmark/run_neuro_comm_live_server.py`, `run_neuro_comm_reward_sweep.py` |
| Tests | `evaluations/dataset_tests/test_neuro_comm_replay.py` (25), `evaluations/nemo_gym/neuro_comm/tests/test_neuro_comm.py` (11) |

Built against **NeMo Gym 0.7.0** (installed with `pip install nemo-gym`; the repo's older `unspoken_communication` config declares
a dataset of type `eval`, which 0.7.0 rejects, and is left untouched). Following the Gym stateful-environment pattern
(`SESSION_ID_KEY`, per-session state, `seed_session`, `verify`), `neuro_comm` was started with the real
`gym env start --resources-server neuro_comm` (all 4 servers reported ready) and driven over HTTP (section 4).

## 2. Environment

One episode = one recorded Study F character selection, replayed from the frozen evidence artifacts. The same decoder and
the same evidence are used for every policy.

**Agent-visible** (every tool response): repetition index, repetitions remaining, `evidence_exhausted`, top candidate and top-5
posteriors (softmax of the per-repetition log-evidence sums, alpha = 1, exactly the benchmark's Bayesian rule), entropy, recorded
acquisition time and decision latency so far (from stimulus timestamps), flashes so far, the grid labels. `previous_characters`
and `other_modalities` are returned as explicit "unavailable" strings: the episode is a single selection (no emitted text exists,
and the true intended prefix is never exposed) and no audio/gaze/speech was recorded.

**Verifier-only:** the intended target. It lives in a separate mapping loaded by the server; `ReplaySession` never receives it and
`outcome(target)` is called by `verify` only. `EpisodeEvidence` has no target field (tested).

| Tool | Behaviour |
|---|---|
| `get_state` | read-only observation |
| `request_more_eeg` | reveals the next recorded repetition; **rejected** (counted invalid) when none remain |
| `commit_character(character)` | grid label required; unknown labels rejected; final |
| `abstain` | final |
| `request_other_modality(modality)` | always rejected, nothing fabricated; penalised |

Anything else (unknown tool, malformed arguments, calls after the end, more than `max_calls`=40) is rejected, not raised; the step
budget ends the episode as a "no decision".

### Reward (configurable: `configs/neuro_comm.yaml` or `RewardConfig`)

`reward = outcome - cost terms`

| Term | Default | Meaning |
|---|---:|---|
| `r_correct` | +1.0 | correct commit |
| `p_wrong` | -1.0 | incorrect commit |
| `p_abstain` | -0.2 | abstention (safe but not communicating) |
| `p_timeout` | -0.5 | episode ended with no decision |
| `c_sec` | -0.01 / s | recorded EEG acquisition (first flash to last used flash + 0.25 s); one repetition is about 2.5 s |
| `c_turn` | -0.005 / call | every tool call = one model turn (unnecessary model/tool calls) |
| `p_invalid` | -0.1 / call | rejected call (bad character, beyond recording, unknown tool, after end) |
| `p_modality` | -0.05 / call | request for an unavailable modality |

Rationale: with these defaults abstaining beats committing when the commit success probability p satisfies
`p < (p_wrong - p_abstain) / (1 + p_wrong)` = 0.4, i.e. the agent is rewarded for abstaining on near-chance decisions but not for
abstaining on a 70% one; a full 14-repetition acquisition costs about 0.32, so waiting is not free. Effects are tested
(`test_reward_coefficients_change_rewards_in_the_documented_direction`, `test_penalties_make_abstain_optimal_...`) and measured
for the Bayesian policy in `evaluations/results/neuro_comm/reward_sweep_bayes.csv` (threshold chosen on calibration episodes,
scored on session 3):

| Config | Best tau on calibration | Session-3 reward at that tau |
|---|---:|---:|
| default | 0.95 | 0.254 |
| wrong x0.25 | 0.95 | 0.442 |
| wrong x3 | 0.95 | -0.246 |
| time x0.1 | 0.95 | 0.432 |
| time x5 | 0.70 | -0.529 |

Only a high time cost moves the Bayesian threshold; the plain Bayesian rule cannot react to a high error cost because it never
abstains (that is exactly where an agent could add value).

## 3. Data and leakage controls

- Tasks: `train.jsonl` = 600 calibration-role episodes (sessions 1-2; labels may legitimately be used for training; their evidence is
  out-of-fold); `eval.jsonl` = the 120 held-out session-3 Static Test episodes; `example.jsonl` = 5 train episodes. **No task row contains a
  target** (tested) and train and eval are disjoint (tested).
- Tool traces are byte-identical when the hidden target is changed (`test_tool_trace_is_independent_of_ground_truth`).
- The server returns a controlled error for unseeded sessions and unknown episodes, isolates concurrent sessions by cookie, and flags
  a verify without a matching session as `mask_sample` rather than scoring it.
- Nothing is trained on session 3. The frozen decoder, calibration `alpha` and thresholds were fitted on sessions 1-2 only (benchmark).

## 4. Bayesian parity (gate passed)

The Bayesian policy was run through exactly the tool interface a model uses (`get_state` -> `request_more_eeg` / `commit_character`) and
compared episode by episode with the standalone benchmark policy on the same frozen evidence, for 120 held-out episodes:
`evaluations/results/neuro_comm/bayes_parity_{per_episode,summary}.csv`.

| Policy | tau | Not-correct (Gym / standalone) | Abstentions | Mean repetitions | Mismatched episodes |
|---|---:|---|---|---|---:|
| Bayes | 0.9 | 34 / 34 | 0 / 0 | 8.200 / 8.200 | **0** |
| Bayes | 0.99 | 29 / 29 | 0 / 0 | 9.375 / 9.375 | **0** |
| Bayes | 0.9999 | 29 / 29 | 0 / 0 | 10.550 / 10.550 | **0** |
| Bayes + abstain | 0.9 / 0.99 / 0.9999 | 48 / 52 / 60 (both) | 43 / 52 / 60 (both) | same | **0** |

Predictions, repetitions, flashes, commit/abstain and recorded acquisition and latency times are identical in all 720 episode comparisons.
(`bayes_tau0.99`: accuracy 0.758, 29 incorrect, 9.375 repetitions, 21.6 s recorded latency, 3.22 correct characters/min: the same
row as `docs/EEG_STOPPING_BENCHMARK.md`.) No discrepancy was found. Parity is also a test (`test_bayes_through_replay_tools_matches_...`),
and over HTTP against the running Gym server the Bayesian policy and `/verify` reproduce the in-process rewards exactly
(`test_bayes_policy_over_the_gym_server_...`; live server: section 4b).

### 4b. Live Gym server

The real Gym launcher (`gym env start`, resources server `neuro_comm`, NeMo Gym 0.7.0, workspace containing `resources_servers/neuro_comm`) brought all
four servers up. `run_neuro_comm_live_server.py` then ran the Bayesian policy (tau 0.99) against the live `neuro_comm` server over HTTP for all 120
held-out episodes, each with its own session cookie, seed, tool calls and `/verify`: **mean reward 0.2534, accuracy 0.758, identical reward, correctness
and repetitions to the in-process replay in every episode**. The server wrote its own rollout log (120 lines with per-episode tool-call events:
`evaluations/results/neuro_comm/gym_rollouts.jsonl`). The policy-model server pointed at an unreachable dummy URL, so an agent-driven `gym eval run`
with a model was not exercised.

## 5. Policies

**Bayesian fast path** (`BayesPolicy`): stop at the first repetition with top posterior >= tau (default 0.99), commit the argmax; if
the recording ends first, commit the argmax. **Bayes + abstain** (`BayesAbstainPolicy`): same, but abstain when the recording ends
below tau. This is the strong classical baseline for reward, since abstention is the obvious thing a model could add.

**Nemotron-only** (`NemotronAgent`): tool-calling loop over an OpenAI-compatible chat endpoint (vLLM or Ollama `/v1`), one tool per
turn, system prompt = task, tools and the reward coefficients (no target, no data). Every model call (latency, token usage, content),
tool call, tool response and error is recorded; malformed or unknown calls are rejected by the environment and counted; extra tool calls in a
turn are ignored and logged; an endpoint failure ends the loop.

**Hybrid** (`HybridPolicy`, deterministic escalation, no model call unless triggered):
1. commit with no model call when top posterior >= tau_commit (0.99);
2. **T1** escalate when the recording is exhausted below tau_commit (Nemotron: commit or abstain);
3. **T2** escalate when repetition >= 7 (the original protocol's static default) and top posterior < 0.9 (Nemotron: more / commit / abstain);
4. otherwise `request_more_eeg`.
Escalation is final for the episode. If Nemotron fails to reach a terminal action (endpoint down, budget), the deterministic fallback commits the
current argmax and `fallback_used` is recorded.

Latency accounting: `end_to_end_s = recorded EEG decision latency + measured model inference seconds` (sequential, no overlap assumed).
Offline replay speed is never reported as patient speed.

## 6. Running the Nemotron comparison (not yet run)

Serve Nemotron with the existing infrastructure (`scripts/serve_vllm_brev.sh`, vLLM with `--enable-auto-tool-choice`; or local Ollama), then:

```bash
python scripts/benchmark/run_neuro_comm_eval.py --policies bayes bayes_abstain nemotron hybrid --tau 0.99 \
  --base-url http://<host>:8001 --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 --workers 4 --out evaluations/results/neuro_comm
# or against the Gym server's own agent path:  gym eval run --agent neuro_comm_simple_agent --input .../data/eval.jsonl (see the env README)
```

It writes per-participant and aggregate tables (accuracy, incorrect commits, abstentions, mean repetitions, recorded acquisition and
EEG latency, Nemotron latency, end-to-end latency, correct characters/min both EEG-only and end-to-end, invalid tool calls, escalation
frequency, mean reward) with participant-bootstrap intervals, plus full rollouts (`rollouts.jsonl`, with the tool-call trace). The
script refuses to report a Nemotron row without `--base-url/--model`; nothing is substituted.

## 7. Blockers and what was checked

- **No Nemotron endpoint reachable from the build container**: no `NVIDIA_API_KEY`, `integrate.api.nvidia.com`, `ollama.com` and `openrouter.ai`
  are blocked by the egress proxy, no GPU (`nvidia-smi` absent; 4 CPU, 15 GB RAM), Hugging Face blocked (cannot fetch weights), and the
  Mac's Ollama is not reachable from the cloud container. Therefore Section 5's Nemotron-only and hybrid policies have **no
  measured results**. The scripted LLM in the tests is a test double for the plumbing only.
- **Model-improvement (NeMo Gym training) was not attempted.** Gym 0.7.0 supplies environments, rollout collection and verification; the weight
  updates happen in an external trainer on GPUs (none here), and no Nemotron checkpoint can be fetched. Training episodes exist
  (`train.jsonl`, 600 calibration-role episodes) and the held-out set is separate, so the experiment is ready when compute is. **This
  work does not support a NeMo Gym model-improvement bounty claim.**
- With the decoder as is, three participants (F_06, F_07, F_25) decode near chance. No policy can create evidence that was not recorded, so the
  realistic headroom is in abstaining / escalating on them, not in the other seven.
- Latency of the 30B model on the Brev/Mac setup is unmeasured here.

## 8. Reproduction

```bash
.venv/bin/python scripts/benchmark/build_study_f_evidence.py          # frozen evidence (see EEG_STOPPING_BENCHMARK.md)
.venv/bin/python evaluations/nemo_gym/neuro_comm/create_examples.py   # task files
.venv/bin/python scripts/benchmark/run_neuro_comm_eval.py --parity --policies bayes bayes_abstain
.venv/bin/python scripts/benchmark/run_neuro_comm_reward_sweep.py
.venv/bin/python -m pytest -q                                          # replay, leakage, reward, policy, parity tests
# Gym server tests and live server need NeMo Gym 0.7.0:  pip install nemo-gym  (separate venv)
python -m pytest evaluations/nemo_gym/neuro_comm/tests -q
# live server: in a workspace containing resources_servers/neuro_comm
UNSPOKEN_REPO=<repo> UNSPOKEN_EVIDENCE_DIR=<repo>/data/evidence/study_f gym env start --resources-server neuro_comm --model-type openai_model --model <m> --model-url <url> --model-api-key x
.venv/bin/python scripts/benchmark/run_neuro_comm_live_server.py --url http://127.0.0.1:<neuro_comm port>
```
