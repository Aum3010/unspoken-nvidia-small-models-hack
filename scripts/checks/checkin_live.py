"""Live check of the vision-preview check-in through the tunnel.
  off: python checkin_live.py off   -> the routes return 404
  on:  python checkin_live.py on    -> nurse alert before the question; Yes answers it; nothing spoken, no message;
                                       queued while an ambiguous message is pending, opens after No"""
import asyncio, sys, uuid

import httpx

G = "http://127.0.0.1:8000"
SP = sys.argv[2] if len(sys.argv) > 2 else "."


async def main(mode):
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")

    async with httpx.AsyncClient(timeout=60) as c:
        h = (await c.get(f"{G}/api/health")).json()["gateway"]
        sid = f"vis-{uuid.uuid4().hex[:6]}"
        if mode == "off":
            r1 = await c.post(f"{G}/api/{sid}/vision/checkin")
            r2 = await c.get(f"{G}/api/vision/data")
            check("flag off: routes absent", r1.status_code == 404 and r2.status_code == 404 and "checkin" not in h,
                  f"checkin={r1.status_code} data={r2.status_code} health.gateway={h}")
        else:
            d = (await c.get(f"{G}/api/vision/data")).json()
            check("data served", d.get("event", {}).get("glucose", {}).get("value") is not None and d.get("profile", {}).get("synthetic"),
                  f"glucose={d['event']['glucose']['value']} {d['event']['glucose']['unit']} name={d['profile']['preferred_name']} label={d['label']!r}")
            r = (await c.post(f"{G}/api/{sid}/vision/checkin")).json()
            titles = [t["title"] for t in r["state"]["timeline"]]
            check("nurse alert before the question", titles.index("🔔 Nurse alerted (vision preview)") < titles.index("Check-in question shown")
                  and r["checkin"]["status"] == "open", f"timeline={titles[-2:]}")
            r = (await c.post(f"{G}/api/{sid}/vision/checkin/answer", json={"answer": "yes", "method": "deliberate_blink_confirmed"})).json()
            s = r["state"]
            audit = (await c.get(f"{G}/api/{sid}/audit")).json()
            check("Yes answers it: logged, nothing spoken, no message",
                  r["checkin"]["answer"] == "yes" and not s["spoken"] and not s["message"]["candidates"]
                  and any(e["kind"] == "checkin_response" for e in audit) and not any(e["kind"] == "speech_authorized" for e in audit),
                  f"last={s['timeline'][-1]['title']!r} spoken={len(s['spoken'])} candidates={len(s['message']['candidates'])}")

            sid = f"vis-{uuid.uuid4().hex[:6]}"
            wav = open(f"{SP}/dont_need_water.wav", "rb").read()
            await c.post(f"{G}/api/{sid}/capture/audio", files={"audio": ("clip.wav", wav, "audio/wav")})
            await asyncio.sleep(0.5)
            r = (await c.post(f"{G}/api/{sid}/vision/checkin")).json()
            queued = r["checkin"]["status"]
            await c.post(f"{G}/api/{sid}/reject")
            await asyncio.sleep(1.0)
            s = (await c.get(f"{G}/api/{sid}/state")).json()
            check("queued while a message is pending, opens after No", queued == "queued" and s["checkin"]["status"] == "open",
                  f"{queued} -> {s['checkin']['status']}")
    print("CHECKIN", "PASS" if ok else "FAIL")


asyncio.run(main(sys.argv[1]))
