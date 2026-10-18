"""Smoke test on the Brev box against any gateway (the live one on :8000, or a test gateway on another port).

    PYTHONPATH=. /workspace/venv-app/bin/python scripts/brev/serving/smoke_box.py --gateway http://127.0.0.1:8010 \
        --clips ~ --checks health accepted ambiguous_no yes --out /tmp/smoke.json

Checks (scripts/checks/smoke.py does the same through the tunnel, from the laptop):
  health        voice, lips, TTS (and, with --all-health, every service) report ready
  accepted      an accepted voice phrase (need_water.wav) is auto-spoken
  ambiguous_no  an ambiguous phrase (dont_need_water.wav) waits for Yes/No; No rejects it, nothing spoken
  yes           an ambiguous phrase (water.wav): Yes (select + confirm by button) speaks it, by the person
  rules_question an ambiguous phrase (water.wav) still gets a question at once, nothing spoken (used with vLLM down)
Sessions are named serving-exp-*, so they are told apart from real use. Exit code 0 only if every check passed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

import httpx

results: dict[str, dict] = {}


def check(name: str, ok: bool, detail: str = ""):
    results[name] = {"ok": bool(ok), "detail": detail}
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default="http://127.0.0.1:8000")
    ap.add_argument("--clips", default=str(Path.home()))
    ap.add_argument("--checks", nargs="+", default=["health", "accepted", "ambiguous_no", "yes"])
    ap.add_argument("--all-health", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    G, clips = args.gateway.rstrip("/"), Path(args.clips).expanduser()

    async with httpx.AsyncClient(timeout=30) as c:
        async def state(sid):
            return (await c.get(f"{G}/api/{sid}/state")).json()

        async def settle(sid, timeout=8.0):
            t0 = time.time()
            while time.time() - t0 < timeout:
                s = await state(sid)
                if s["spoken"] or (s.get("plan") and not any(t["level"] == "pending" for t in s["timeline"])):
                    return s
                await asyncio.sleep(0.05)
            return await state(sid)

        async def voice(sid, wav):
            r = await c.post(f"{G}/api/{sid}/capture/audio", files={"audio": ("clip.wav", (clips / wav).read_bytes(), "audio/wav")})
            r.raise_for_status()
            return r.json()["event"]["status"]

        def sid():
            return f"serving-exp-{uuid.uuid4().hex[:6]}"

        for name in args.checks:
            try:
                if name == "health":
                    h = (await c.get(f"{G}/api/health")).json()
                    want = [k for k in h if k != "gateway"] if args.all_health else ["audio", "vsr", "tts"]
                    bad = [k for k in want if (h.get(k) or {}).get("state") != "ready"]
                    check("health", not bad, f"not ready={bad}")
                elif name == "accepted":
                    s_id, t0 = sid(), time.perf_counter()
                    st = await voice(s_id, "need_water.wav")
                    s = await settle(s_id)
                    for _ in range(60):
                        if s["spoken"]:
                            break
                        await asyncio.sleep(0.05)
                        s = await state(s_id)
                    sp = s["spoken"][-1] if s["spoken"] else {}
                    check("accepted", st == "proposed" and sp.get("method") == "demo_auto_spoken",
                          f"status={st} spoken={sp.get('text')!r} by={sp.get('method')} {int((time.perf_counter() - t0) * 1000)} ms")
                elif name == "ambiguous_no":
                    s_id = sid()
                    st = await voice(s_id, "dont_need_water.wav")
                    await settle(s_id)
                    await asyncio.sleep(1.5)
                    s = await state(s_id)
                    waits = st == "ambiguous" and not s["spoken"] and s.get("auto_speak_candidate_id") is None
                    r = await c.post(f"{G}/api/{s_id}/reject")
                    s = await state(s_id)
                    check("ambiguous_no", waits and r.status_code == 200 and s["message"]["status"] == "rejected" and not s["spoken"],
                          f"status={st} waited={waits} after No: message={s['message']['status']} spoken={len(s['spoken'])} "
                          f"plan_by={(s.get('plan') or {}).get('by')}")
                elif name == "yes":
                    s_id = sid()
                    st = await voice(s_id, "water.wav")
                    s = await settle(s_id)
                    shown = s["shown_candidate_ids"] or [x["id"] for x in s["message"]["candidates"]]
                    await c.post(f"{G}/api/{s_id}/select", json={"candidate_id": shown[0], "method": "accessible_button_confirmed"})
                    r = await c.post(f"{G}/api/{s_id}/confirm", json={"method": "accessible_button_confirmed"})
                    body = r.json() if r.status_code == 200 else {}
                    audio = await c.get(f"{G}{body['audio_url']}") if body.get("audio_url") else None
                    s = await state(s_id)
                    sp = s["spoken"][-1] if s["spoken"] else {}
                    check("yes", r.status_code == 200 and sp.get("method") == "accessible_button_confirmed"
                          and audio is not None and audio.status_code == 200,
                          f"status={st} spoken={sp.get('text')!r} by={sp.get('method')} audio={len(audio.content) if audio else 0} B")
                elif name == "rules_question":
                    s_id, t0 = sid(), time.perf_counter()
                    st = await voice(s_id, "water.wav")
                    s = await state(s_id)
                    shown = s["shown_candidate_ids"] or []
                    first_ms = int((time.perf_counter() - t0) * 1000)
                    s2 = await settle(s_id, timeout=25)
                    check("rules_question", st == "ambiguous" and bool(s.get("plan")) and bool(shown) and not s["spoken"] and not s2["spoken"],
                          f"status={st} question on screen after {first_ms} ms: {s.get('plan', {}).get('name')} {shown}; "
                          f"final plan by={(s2.get('plan') or {}).get('by')} spoken={len(s2['spoken'])}")
                else:
                    check(name, False, "unknown check")
            except Exception as exc:
                check(name, False, f"{type(exc).__name__}: {exc}"[:200])

    ok = all(v["ok"] for v in results.values())
    if args.out:
        Path(args.out).write_text(json.dumps({"gateway": G, "ok": ok, "checks": results}, indent=2))
    print("SMOKE", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
