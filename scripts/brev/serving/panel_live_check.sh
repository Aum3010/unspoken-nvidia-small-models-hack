#!/usr/bin/env bash
# Experiment B, live: the serving card's route on test gateways next to the running app (the gateway on :8000 is
# not touched). Shares the running agents and vLLM.
#   :8010  UNSPOKEN_SERVING_PANEL=1           -> 200 with vLLM's numbers; smoke test through this gateway
#   :8011  flag unset (the default)           -> 404
#   :8012  flag on, vLLM URL pointed at a closed port -> {"state":"unavailable"}, /api/health still 200
# Then the safety tests and the card's unit tests. Writes evaluations/results/serving/panel_check.{json,log}.
#   nohup bash scripts/brev/serving/panel_live_check.sh > .run/logs/serving_B.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT=evaluations/results/serving
PY=/workspace/venv-app/bin/python
mkdir -p "$OUT" .run/logs
export PYTHONPATH=$PWD TOKENIZERS_PARALLELISM=false UNSPOKEN_HOST_LABEL="NVIDIA Brev"
export UNSPOKEN_LLM_BACKEND=openai UNSPOKEN_LLM_BASE_URL=http://127.0.0.1:8001 UNSPOKEN_LLM_MODEL=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done; }
trap cleanup EXIT INT TERM

UNSPOKEN_SERVING_PANEL=1 nohup $PY -m uvicorn services.gateway.app:app --host 127.0.0.1 --port 8010 --log-level warning > .run/logs/serving_B_gw8010.log 2>&1 < /dev/null & pids+=($!)
nohup $PY -m uvicorn services.gateway.app:app --host 127.0.0.1 --port 8011 --log-level warning > .run/logs/serving_B_gw8011.log 2>&1 < /dev/null & pids+=($!)
UNSPOKEN_SERVING_PANEL=1 UNSPOKEN_LLM_BASE_URL=http://127.0.0.1:9 nohup $PY -m uvicorn services.gateway.app:app --host 127.0.0.1 --port 8012 --log-level warning > .run/logs/serving_B_gw8012.log 2>&1 < /dev/null & pids+=($!)
for port in 8010 8011 8012; do
  for _ in $(seq 1 60); do curl -s -m 2 -o /dev/null "http://127.0.0.1:$port/api/health" && break; sleep 1; done
done

code() { curl -s -m 5 -o /dev/null -w '%{http_code}' "$1"; }
on_code=$(code http://127.0.0.1:8010/api/serving_metrics); on_body=$(curl -s -m 5 http://127.0.0.1:8010/api/serving_metrics)
off_code=$(code http://127.0.0.1:8011/api/serving_metrics)
down_code=$(code http://127.0.0.1:8012/api/serving_metrics); down_body=$(curl -s -m 5 http://127.0.0.1:8012/api/serving_metrics)
down_health=$(code http://127.0.0.1:8012/api/health)
t0=$(date +%s%N); curl -s -m 5 -o /dev/null http://127.0.0.1:8012/api/serving_metrics; down_ms=$(( ($(date +%s%N) - t0) / 1000000 ))
echo "flag on:  $on_code $on_body"
echo "flag off: $off_code"
echo "vLLM down: $down_code $down_body ($down_ms ms, cache may answer) · /api/health $down_health"

$PY scripts/brev/serving/smoke_box.py --gateway http://127.0.0.1:8010 --checks health accepted ambiguous_no yes --out /tmp/serving_B_smoke.json
smoke=$?
$PY -m pytest -q evaluations/safety_tests 2>&1 | tail -n 3 | tee /tmp/serving_B_safety.txt
safety=${PIPESTATUS[0]}
$PY -m pytest -q scripts/brev/serving/test_serving_panel.py 2>&1 | tail -n 3 | tee /tmp/serving_B_unit.txt
unit=${PIPESTATUS[0]}

$PY - "$on_code" "$on_body" "$off_code" "$down_code" "$down_body" "$down_health" "$smoke" "$safety" "$unit" <<'PY'
import json, sys, time
on_code, on_body, off_code, down_code, down_body, down_health, smoke, safety, unit = sys.argv[1:]
def j(s):
    try: return json.loads(s)
    except Exception: return s
def d(s):
    return j(s) if isinstance(j(s), dict) else {}
res = {"date": time.strftime("%Y-%m-%d %H:%M:%S"),
       "flag_on": {"http": int(on_code), "body": j(on_body), "ok": on_code == "200" and d(on_body).get("state") == "ready"},
       "flag_off": {"http": int(off_code), "ok": off_code == "404"},
       "vllm_down": {"http": int(down_code), "body": j(down_body), "health_http": int(down_health),
                     "ok": down_code == "200" and d(down_body).get("state") == "unavailable" and down_health == "200"},
       "smoke_via_flag_on_gateway": json.load(open("/tmp/serving_B_smoke.json")) if smoke in ("0", "1") else None,
       "safety_tests": open("/tmp/serving_B_safety.txt").read().strip().splitlines()[-1],
       "panel_unit_tests": open("/tmp/serving_B_unit.txt").read().strip().splitlines()[-1]}
res["smoke_ok"] = smoke == "0"
res["safety_ok"] = safety == "0"
res["unit_ok"] = unit == "0"
res["works"] = all([res["flag_on"]["ok"], res["flag_off"]["ok"], res["vllm_down"]["ok"], res["smoke_ok"], res["safety_ok"], res["unit_ok"]])
open("evaluations/results/serving/panel_check.json", "w").write(json.dumps(res, indent=2))
print("B WORKS" if res["works"] else "B FAILED", json.dumps({k: v for k, v in res.items() if k.endswith("ok") or k == "works"}))
PY
