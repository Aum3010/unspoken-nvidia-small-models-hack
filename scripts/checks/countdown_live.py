"""Live check of UNSPOKEN_AUTO_SPEAK_DELAY_MS through the tunnel (gateway started with the flag):
  1. accepted voice phrase -> countdown shown (auto_speak_in_ms > 0) -> No inside the window -> nothing spoken
  2. accepted voice phrase -> no answer -> spoken exactly once, after the delay
Usage: python countdown_live.py <scratchpad>"""
import asyncio, sys, time, uuid
from pathlib import Path

import httpx

SP = Path(sys.argv[1])
G = "http://127.0.0.1:8000"


async def main():
    ok = True
    async with httpx.AsyncClient(timeout=60) as c:
        wav = (SP / "need_water.wav").read_bytes()

        sid = f"cd-{uuid.uuid4().hex[:6]}"
        await c.post(f"{G}/api/{sid}/capture/audio", files={"audio": ("clip.wav", wav, "audio/wav")})
        s = (await c.get(f"{G}/api/{sid}/state")).json()
        left = s.get("auto_speak_in_ms")
        await c.post(f"{G}/api/{sid}/reject")
        await asyncio.sleep(3.5)
        s = (await c.get(f"{G}/api/{sid}/state")).json()
        r1 = left is not None and left > 0 and not s["spoken"] and s["message"]["status"] == "rejected"
        print(f"{'PASS' if r1 else 'FAIL'}  countdown {left} ms shown, No inside the window -> spoken={len(s['spoken'])} status={s['message']['status']}")

        sid = f"cd-{uuid.uuid4().hex[:6]}"
        t0 = time.perf_counter()
        await c.post(f"{G}/api/{sid}/capture/audio", files={"audio": ("clip.wav", wav, "audio/wav")})
        t_spoken = None
        while time.perf_counter() - t0 < 6:
            s = (await c.get(f"{G}/api/{sid}/state")).json()
            if s["spoken"] and t_spoken is None:
                t_spoken = time.perf_counter() - t0
            await asyncio.sleep(0.1)
        r2 = len(s["spoken"]) == 1 and s["spoken"][0]["method"] == "demo_auto_spoken" and t_spoken is not None and t_spoken > 2.0
        print(f"{'PASS' if r2 else 'FAIL'}  no answer -> spoken {len(s['spoken'])}x by {s['spoken'][0]['method'] if s['spoken'] else '-'} after {t_spoken and round(t_spoken, 2)} s")
        ok = r1 and r2
    print("COUNTDOWN", "PASS" if ok else "FAIL")


asyncio.run(main())
