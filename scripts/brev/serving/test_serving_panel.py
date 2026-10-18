"""Serving card (UNSPOKEN_SERVING_PANEL, off by default): absent when off (404); when vLLM is down it answers
"unavailable" and the rest of the gateway is unaffected; /metrics text is read into the card's numbers.

Run: PYTHONPATH=. python -m pytest scripts/brev/serving/test_serving_panel.py -q
"""

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import services.gateway.app as gateway

METRICS = """# HELP vllm:num_requests_running Number of requests in model execution batches.
vllm:num_requests_running{engine="0",model_name="m"} 3.0
vllm:num_requests_waiting{engine="0",model_name="m"} 1.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.125
vllm:prefix_cache_queries_total{engine="0",model_name="m"} 1000.0
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 250.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.1",model_name="m"} 2.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.2",model_name="m"} 6.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf",model_name="m"} 8.0
vllm:time_to_first_token_seconds_count{engine="0",model_name="m"} 8.0
vllm:time_to_first_token_seconds_created{engine="0",model_name="m"} 1.79e+09
"""


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(gateway, "_serving_cache", {"ts": 0.0, "data": None})
    monkeypatch.setattr(gateway, "_ttft_snapshots", [])


def panel_app() -> TestClient:
    app = FastAPI()
    app.add_api_route("/api/serving_metrics", gateway.serving_metrics, methods=["GET"])
    return TestClient(app)


def test_flag_off_by_default_means_404():
    assert gateway.SERVING_PANEL is False
    r = TestClient(gateway.app).get("/api/serving_metrics")
    assert r.status_code == 404


def test_vllm_down_is_unavailable_and_health_still_answers(monkeypatch):
    monkeypatch.setattr(gateway, "VLLM_METRICS_URL", "http://127.0.0.1:9/metrics")   # nothing listens on port 9
    r = panel_app().get("/api/serving_metrics")
    assert r.status_code == 200 and r.json()["state"] == "unavailable"
    assert TestClient(gateway.app).get("/api/health").status_code == 200


def test_metrics_are_read(monkeypatch):
    real = httpx.AsyncClient
    monkeypatch.setattr(gateway.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(lambda req: httpx.Response(200, text=METRICS)), **kw))
    m = panel_app().get("/api/serving_metrics").json()
    assert m["state"] == "ready"
    assert (m["running"], m["waiting"], m["kv_cache_usage_pct"], m["prefix_cache_hit_rate"]) == (3, 1, 12.5, 0.25)
    assert m["ttft_p50_ms"] == pytest.approx(150.0)   # the 4th of 8 requests: halfway through the 0.1-0.2 s bucket
    assert m["spec_decode_acceptance"] is None        # no MTP counters, no acceptance


def test_slow_vllm_times_out_fast(monkeypatch):
    async def slow(request):
        raise httpx.ReadTimeout("slow", request=request)
    real = httpx.AsyncClient
    monkeypatch.setattr(gateway.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(slow), **kw))
    assert panel_app().get("/api/serving_metrics").json() == {"state": "unavailable", "error": "ReadTimeout"}
