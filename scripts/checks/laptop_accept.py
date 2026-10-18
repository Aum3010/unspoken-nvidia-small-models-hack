"""From the laptop through the tunnel: accepted voice clip sent -> the gateway reports it spoken (auto-speak).
Usage: python laptop_accept.py <scratchpad> [n]"""
import asyncio, statistics, sys, time, uuid
from pathlib import Path

import httpx

SP = Path(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
G = "http://127.0.0.1:8000"


async def main():
    ms = []
    async with httpx.AsyncClient(timeout=60) as c:
        for i in range(N):
            wav = (SP / ("need_water.wav" if i % 2 == 0 else "pain.wav")).read_bytes()
            sid = f"lapacc-{uuid.uuid4().hex[:6]}"
            t0 = time.perf_counter()
            r = await c.post(f"{G}/api/{sid}/capture/audio", files={"audio": ("clip.wav", wav, "audio/wav")})
            if r.json()["event"]["status"] != "proposed":
                continue
            for _ in range(200):
                s = (await c.get(f"{G}/api/{sid}/state")).json()
                if s["spoken"]:
                    ms.append((time.perf_counter() - t0) * 1000)
                    break
                await asyncio.sleep(0.01)
    ms.sort()
    print(f"accepted voice, laptop: clip sent -> spoken  n={len(ms)}  p50 {statistics.median(ms):.0f} ms  p95 {ms[int(round(0.95 * (len(ms) - 1)))]:.0f} ms")


asyncio.run(main())
