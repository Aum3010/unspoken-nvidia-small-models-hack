"""Smoke test of the live system through the tunnel (laptop -> gateway on Brev):
  1. accepted voice phrase -> auto-spoken
  2. ambiguous voice phrase -> waits (nothing spoken), then No -> rejected, nothing spoken
  3. ambiguous voice phrase -> Yes (select + confirm by button) -> spoken by the person's method
  4. streamed lip clip (pieces over the WebSocket, then finish) -> reading returned
Usage: python smoke.py <repo> <scratchpad> [out.json]"""
import asyncio, json, sys, time, uuid
from pathlib import Path

import httpx, websockets

REPO, SP = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(Path(__file__).resolve().parent))  # laptop_lip.py sits next to this file
from laptop_lip import live_webm, frame  # noqa: E402

G, WS = "http://127.0.0.1:8000", "ws://127.0.0.1:8000"
results = {}


def check(name, ok, detail=""):
    results[name] = {"ok": bool(ok), "detail": detail}
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


async def state(c, sid):
    return (await c.get(f"{G}/api/{sid}/state")).json()


async def settle(c, sid, timeout=8.0):
    """Wait until a plan is on screen and no step is pending (or something was spoken)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = await state(c, sid)
        if s["spoken"] or (s.get("plan") and not any(t["level"] == "pending" for t in s["timeline"])):
            return s
        await asyncio.sleep(0.05)
    return await state(c, sid)


async def voice(c, sid, wav):
    r = await c.post(f"{G}/api/{sid}/capture/audio", files={"audio": ("clip.wav", (SP / wav).read_bytes(), "audio/wav")})
    return r.json()["event"]["status"]


async def main():
    async with httpx.AsyncClient(timeout=60) as c:
        h = (await c.get(f"{G}/api/health")).json()
        bad = [k for k, v in h.items() if isinstance(v, dict) and v.get("state") != "ready"]
        check("health: every service ready", not bad, f"host={h['gateway'].get('host')} not ready={bad}")

        sid = f"smoke-{uuid.uuid4().hex[:6]}"
        t0 = time.perf_counter()
        st = await voice(c, sid, "need_water.wav")
        s = await settle(c, sid)
        for _ in range(60):
            if s["spoken"]:
                break
            await asyncio.sleep(0.05); s = await state(c, sid)
        sp = s["spoken"][-1] if s["spoken"] else {}
        check("accepted voice phrase is auto-spoken", st == "proposed" and sp.get("method") == "demo_auto_spoken",
              f"status={st} spoken={sp.get('text')!r} by={sp.get('method')} {int((time.perf_counter() - t0) * 1000)} ms")

        sid = f"smoke-{uuid.uuid4().hex[:6]}"
        st = await voice(c, sid, "dont_need_water.wav")
        s = await settle(c, sid)
        await asyncio.sleep(1.5)
        s = await state(c, sid)
        check("ambiguous voice phrase waits for Yes/No", st == "ambiguous" and not s["spoken"] and s.get("auto_speak_candidate_id") is None,
              f"status={st} message={s['message']['status']} spoken={len(s['spoken'])}")
        r = await c.post(f"{G}/api/{sid}/reject")
        s = await state(c, sid)
        check("No rejects it; nothing spoken", r.status_code == 200 and s["message"]["status"] == "rejected" and not s["spoken"],
              f"message={s['message']['status']} spoken={len(s['spoken'])}")

        sid = f"smoke-{uuid.uuid4().hex[:6]}"
        st = await voice(c, sid, "water.wav")
        s = await settle(c, sid)
        shown = s["shown_candidate_ids"] or [x["id"] for x in s["message"]["candidates"]]
        await c.post(f"{G}/api/{sid}/select", json={"candidate_id": shown[0], "method": "accessible_button_confirmed"})
        r = await c.post(f"{G}/api/{sid}/confirm", json={"method": "accessible_button_confirmed"})
        body = r.json() if r.status_code == 200 else {}
        audio = await c.get(f"{G}{body['audio_url']}") if body.get("audio_url") else None
        s = await state(c, sid)
        sp = s["spoken"][-1] if s["spoken"] else {}
        check("Yes speaks the chosen reading, by the person", r.status_code == 200 and sp.get("method") == "accessible_button_confirmed"
              and audio is not None and audio.status_code == 200,
              f"status={st} spoken={sp.get('text')!r} by={sp.get('method')} audio={len(audio.content) if audio else 0} B phrase_audio={body.get('phrase_audio_id') is not None}")

        sid = f"smoke-{uuid.uuid4().hex[:6]}"
        clip = sorted((SP / "grid").glob("s*/*.mpg"))[0]
        data, pieces = live_webm(clip)
        async with websockets.connect(f"{WS}/ws/{sid}", max_size=None) as ws:
            await ws.recv()
            sp_id = "lip" + uuid.uuid4().hex[:6]
            for i, p in enumerate(pieces):
                await ws.send(frame({"id": sp_id, "seq": i, "mime": "video/webm;codecs=vp9"}, p))
                if i < len(pieces) - 1:
                    await asyncio.sleep(0.25)
            t0 = time.perf_counter()
            r = await c.post(f"{G}/api/{sid}/capture/lip/stream", json={"stream_id": sp_id, "chunks": len(pieces)})
            ms = int((time.perf_counter() - t0) * 1000)
        ev = r.json().get("event", {}) if r.status_code == 200 else {}
        q = ev.get("quality", {})
        check("streamed lip clip is read", r.status_code == 200 and q.get("streamed") is True,
              f"http={r.status_code} {len(pieces)} pieces, frames={q.get('frames')} face={q.get('face_detected_ratio')} "
              f"ignored={r.json().get('ignored') if r.status_code == 200 else '-'} {ms} ms after the end")

    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(json.dumps(results, indent=2))
    print("\nSMOKE", "PASS" if all(v["ok"] for v in results.values()) else "FAIL")


asyncio.run(main())
