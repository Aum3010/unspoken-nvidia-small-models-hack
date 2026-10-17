#!/usr/bin/env bash
# Health of every service + one board -> confirm -> speak round trip.
set -euo pipefail
GW=${GW:-http://127.0.0.1:8000}
curl -sf $GW/api/health | python3 -c "
import json,sys; d=json.load(sys.stdin)
for k,v in d.items(): print(f'  {k:<12} {v.get(\"state\")}')
assert all(v.get('state')=='ready' for v in d.values()), 'a service is not ready'"
SID=smoke-$RANDOM
curl -sf -XPOST $GW/api/$SID/board -H 'content-type: application/json' -d '{"text":"I need water","origin":"board"}' >/dev/null
OUT=$(curl -sf -XPOST $GW/api/$SID/confirm -H 'content-type: application/json' -d '{"method":"keyboard_confirmed"}')
echo "$OUT" | python3 -c "import json,sys; d=json.load(sys.stdin); assert d['text']=='I need water' and d['audio_url']; print('  spoke:', d['text'])"
curl -sf -XPOST $GW/api/$SID/safety/attempt_unconfirmed_speech | grep -q blocked && echo "  policy gate: blocked unconfirmed speech"
echo "smoke test OK"
