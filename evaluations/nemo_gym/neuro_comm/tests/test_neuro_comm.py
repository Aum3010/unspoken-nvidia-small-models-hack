"""Server tests for the `neuro_comm` resources server (needs nemo_gym and the frozen evidence; skipped otherwise)."""

import json
import sys
from http.cookiejar import CookieJar
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[4]
EVD = ROOT / "data" / "evidence" / "study_f"
pytest.importorskip("nemo_gym")
pytestmark = pytest.mark.skipif(not (EVD / "manifest.json").exists(), reason="frozen evidence not built")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming  # noqa: E402
from nemo_gym.server_utils import ServerClient  # noqa: E402

from app import NeuroCommConfig, NeuroCommResourcesServer  # noqa: E402
from services.eeg.study_f import replay  # noqa: E402


class _NoStickyCookies(CookieJar):
    def extract_cookies(self, response, request):  # sessions come only from explicit cookies
        pass


@pytest.fixture(scope="module")
def server():
    cfg = NeuroCommConfig(host="0.0.0.0", port=8080, entrypoint="", name="neuro_comm", evidence_dir=str(EVD), roles=["evaluation"],
                          reward={"c_sec": 0.01})
    return NeuroCommResourcesServer(config=cfg, server_client=MagicMock(spec=ServerClient))


@pytest.fixture()
def client(server):
    return TestClient(server.setup_webserver(), cookies=_NoStickyCookies())


def seed(client, eid):
    r = client.post("/seed_session", json={"episode_id": eid})
    assert r.status_code == 200, r.text
    return r.cookies


def verify(client, cookies, eid):
    resp = NeMoGymResponse(id="r", created_at=0, model="m", object="response", output=[], parallel_tool_calls=False, tool_choice="auto", tools=[])
    body = {"responses_create_params": NeMoGymResponseCreateParamsNonStreaming(input="x").model_dump(exclude_unset=True),
            "response": resp.model_dump(), "episode_id": eid}
    r = client.post("/verify", json=body, cookies=cookies)
    assert r.status_code == 200, r.text
    return r.json()


EPS = replay.load_episodes(EVD, roles=("evaluation",))
TRUTH = replay.load_ground_truth(EVD)
EID = sorted(EPS)[0]
CHARS = EPS[EID].chars


def test_tool_responses_never_contain_the_target(client):
    ck = seed(client, EID)
    target_char = CHARS[TRUTH[EID]]
    blob = json.dumps(client.post("/get_state", cookies=ck).json())
    for _ in range(EPS[EID].n_avail - 1):
        blob += json.dumps(client.post("/request_more_eeg", cookies=ck).json())
    assert "target" not in blob.lower().replace("top_candidate", "")      # no target field anywhere
    assert "ground_truth" not in blob and "intended" not in blob.lower()


def test_unseeded_session_is_rejected_not_crashing(client):
    r = client.post("/get_state")
    assert r.status_code == 200 and r.json()["ok"] is False


def test_sessions_are_isolated(client):
    a, b = seed(client, EID), seed(client, sorted(EPS)[1])
    client.post("/request_more_eeg", cookies=a)
    assert client.post("/get_state", cookies=a).json()["observation"]["repetition"] == 2
    assert client.post("/get_state", cookies=b).json()["observation"]["repetition"] == 1


def test_beyond_recording_is_rejected(client):
    ck = seed(client, EID)
    n = EPS[EID].n_avail
    for _ in range(n - 1):
        assert client.post("/request_more_eeg", cookies=ck).json()["ok"]
    out = client.post("/request_more_eeg", cookies=ck).json()
    assert out["ok"] is False and "exhausted" in out["error"] and out["observation"]["repetition"] == n


def test_other_modality_never_fabricates(client):
    ck = seed(client, EID)
    out = client.post("/request_other_modality", json={"modality": "gaze"}, cookies=ck).json()
    assert out["ok"] is False and "unavailable" in out["error"] and out["observation"]["repetition"] == 1


def test_verify_scores_commit_with_cost_terms(client):
    ck = seed(client, EID)
    right = CHARS[TRUTH[EID]]
    assert client.post("/commit_character", json={"character": right}, cookies=ck).json()["ok"]
    v = verify(client, ck, EID)
    assert v["correct"] == 1.0 and v["decision"] == "commit"
    c = v["reward_components"]
    assert c["correct"] == 1.0 and c["incorrect_commit"] == 0.0 and c["acquisition"] < 0 and c["turns"] == pytest.approx(-0.005)
    assert v["reward"] == pytest.approx(sum(c.values()))


def test_verify_wrong_commit_and_abstain(client):
    ck = seed(client, EID)
    wrong = CHARS[(TRUTH[EID] + 1) % 36]
    client.post("/commit_character", json={"character": wrong}, cookies=ck)
    v = verify(client, ck, EID)
    assert v["incorrect_commit"] == 1.0 and v["reward_components"]["incorrect_commit"] == -1.0
    ck = seed(client, EID)
    client.post("/abstain", cookies=ck)
    v = verify(client, ck, EID)
    assert v["abstained"] == 1.0 and v["reward_components"]["abstain"] == -0.2 and v["correct"] == 0.0


def test_verify_without_decision_is_timeout(client):
    ck = seed(client, EID)
    client.post("/request_more_eeg", cookies=ck)
    v = verify(client, ck, EID)
    assert v["decision"] == "timeout" and v["reward_components"]["timeout"] == -0.5


def test_verify_requires_matching_seeded_episode_and_is_masked_otherwise(client):
    v = verify(client, None, EID)
    assert v["mask_sample"] is True and v["reward"] == 0.0


def test_calibration_role_not_served_when_not_configured(client):
    r = client.post("/seed_session", json={"episode_id": "bigp3bci/F_03/SE001/Test/Static01#t00"})
    assert r.status_code == 400                               # unknown episode must not create a session
    assert client.post("/get_state", cookies=r.cookies).json()["ok"] is False


def test_bayes_policy_over_the_gym_server_matches_in_process_replay_and_rewards(client):
    """The same BayesPolicy through the real resources server (HTTP tools + verify) and in process agree exactly."""
    from services.eeg.study_f import policies as pol

    for eid in sorted(EPS)[::10]:
        http = pol.HttpEnv(client, eid)
        pol.BayesPolicy(0.99).run(http)
        v = verify(client, http.cookies, eid)
        local = pol.LocalEnv(EPS[eid], replay.RewardConfig(c_sec=0.01))
        pol.BayesPolicy(0.99).run(local)
        o = local.session.outcome(TRUTH[eid])
        assert v["reward"] == pytest.approx(o["reward"]) and v["reps"] == o["reps"] and bool(v["correct"]) == o["correct"]
