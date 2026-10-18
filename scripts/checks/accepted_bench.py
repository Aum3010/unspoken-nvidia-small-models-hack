"""Voice-only messages through the gateway: time from sending the clip until it is spoken (auto-speak) or the
question is on screen (confirm mode), split by whether the voice agent accepted the reading. Runs on Brev."""
import asyncio
import statistics
import sys
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path.home() / "unspoken"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "asr_eval"))
from harness.config import load_config  # noqa: E402
from shared.confirmation import sign  # noqa: E402

G, TTS = "http://127.0.0.1:8000", "http://127.0.0.1:8104"


async def main(n_rounds: int = 2):
    phrases = list(load_config(ROOT / "asr_eval" / "configs" / "eval.yaml").phrases.values())
    rows = []
    async with httpx.AsyncClient(timeout=60) as c:
        wavs = []
        for p in phrases:
            mid = f"acc-{uuid.uuid4().hex[:6]}"
            wavs.append((await c.post(f"{TTS}/speak", json={"message_id": mid, "text": p, "token": sign(mid, p), "voice": "af_heart"})).content)
        for _ in range(n_rounds):
            for wav in wavs:
                sid = f"acc-{uuid.uuid4().hex[:8]}"
                t0 = time.perf_counter()
                r = (await c.post(f"{G}/api/{sid}/capture/audio", files={"audio": ("clip.wav", wav, "audio/wav")})).json()
                status = r["event"]["status"]
                t_ready, by = None, None
                for _ in range(400):
                    s = (await c.get(f"{G}/api/{sid}/state")).json()
                    plan = s.get("plan") or {}
                    shown = plan and not any(t["level"] == "pending" for t in s["timeline"]) and s["message"]["status"] in ("clarifying", "selected", "confirmed", "spoken")
                    waits = shown and s.get("auto_speak_candidate_id") is None
                    if s["spoken"] or ((not s["settings"].get("auto_speak") or waits) and shown):
                        t_ready, by = time.perf_counter(), plan.get("by") + (" spoken" if s["spoken"] else " asked")
                        break
                    if r.get("ignored"):
                        break
                    await asyncio.sleep(0.01)
                if t_ready:
                    rows.append((status, by, (t_ready - t0) * 1000))
                await asyncio.sleep(0.3)
    for status in ("proposed", "ambiguous"):
        v = sorted(ms for st, _, ms in rows if st == status)
        bys = sorted({b for st, b, _ in rows if st == status})
        if v:
            print(f"voice {status:<9} (accepted={status == 'proposed'}) n={len(v):>2}  clip -> spoken/asked  p50 {statistics.median(v):5.0f} ms  p95 {v[int(round(0.95 * (len(v) - 1)))]:5.0f} ms  plan by {bys}")


asyncio.run(main())
