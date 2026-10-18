"""Latency of every agent alone, then with all of them running at once, on the Brev GPU (warm, batch 1).

    cd ~/unspoken && PYTHONPATH=. /workspace/venv-app/bin/python scripts/brev/bench_latency.py --n 20

Inputs: Kokoro speaks the 20 voice phrases (that is also the TTS benchmark) and those clips feed the voice
agent; GRID videos feed the lip agent; a BNCI subject feeds the EEG agent; a fixed two-candidate question
feeds the coordinator. Times are client round trips to each service on 127.0.0.1 (no network to the laptop).

  1. alone       each agent by itself, one request at a time
  2. together    every agent at once: one closed loop per agent, all running concurrently (GPU sampled)
  3. end to end  through the gateway, like one real message: voice + lip clip sent together -> evidence ->
                 plan -> spoken. With demo auto-speak an accepted reading is spoken by the gateway; anything
                 else is selected and confirmed by the bench (counted in `_notes`)
  4. voice wait  from the end of an utterance (700 ms of silence) until its evidence is in the session:
                 sent at that moment (as before) vs early decode (sent at 350 ms of silence, committed at 700 ms)

Writes evaluations/results/latency_brev.json and prints a markdown table.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.confirmation import sign  # noqa: E402  (same secret file as the TTS service on this box)

URL = {"gateway": "http://127.0.0.1:8000", "vsr": "http://127.0.0.1:8101", "audio": "http://127.0.0.1:8102",
       "eeg": "http://127.0.0.1:8103", "tts": "http://127.0.0.1:8104", "coordinator": "http://127.0.0.1:8105"}
PLAN = {"message_id": "bench", "available_inputs": ["gaze", "blink", "button", "keyboard"], "gaze_calibrated": True, "history": [],
        "candidates": [{"id": "c1", "text": "I need water", "modality": "audio", "score": -0.4, "supporting_modalities": ["audio"]},
                       {"id": "c2", "text": "I need the bathroom", "modality": "lip", "score": -2.1, "supporting_modalities": ["lip"]}],
        "evidence": [{"modality": "audio", "status": "ambiguous", "quality_flags": [], "top2_margin": 0.3, "n_hypotheses": 3},
                     {"modality": "lip", "status": "proposed", "quality_flags": [], "top2_margin": 1.2, "n_hypotheses": 5}]}


def stats(ms: list[float]) -> dict:
    if not ms:
        return {"n": 0}
    s = sorted(ms)
    return {"n": len(s), "p50": round(statistics.median(s)), "p95": round(s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]),
            "max": round(s[-1]), "mean": round(statistics.fmean(s))}


class Agents:
    def __init__(self, client: httpx.AsyncClient, phrases: list[str], videos: list[Path], subject: str):
        self.c, self.phrases, self.videos, self.subject = client, phrases, [v.read_bytes() for v in videos], subject
        self.wavs: list[bytes] = []
        self.i = 0
        self.detail: dict[str, list[dict]] = {"audio": [], "vsr": [], "coordinator": []}

    async def tts(self, text: str) -> bytes:
        mid = f"bench-{uuid.uuid4().hex[:8]}"
        r = await self.c.post(f"{URL['tts']}/speak", json={"message_id": mid, "text": text, "token": sign(mid, text), "voice": "af_heart"})
        r.raise_for_status()
        return r.content

    async def call(self, agent: str):
        self.i += 1
        if agent == "tts":
            await self.tts(self.phrases[self.i % len(self.phrases)])
        elif agent == "audio":
            r = await self.c.post(f"{URL['audio']}/infer", files={"audio": ("clip.wav", self.wavs[self.i % len(self.wavs)], "audio/wav")})
            r.raise_for_status()
            self.detail["audio"].append(r.json()["quality"].get("timings_ms") or {})
        elif agent == "vsr":
            r = await self.c.post(f"{URL['vsr']}/infer", files={"video": ("clip.mpg", self.videos[self.i % len(self.videos)], "video/mpeg")})
            r.raise_for_status()
            q = r.json()["quality"]
            self.detail["vsr"].append({k: q.get(k) for k in ("mode", "enroll_ms", "phrase_ms", "decode_ms", "enroll_best_dist")} | {"n_hyp": len(r.json()["hypotheses"]), "service_ms": r.json()["provenance"]["latency_ms"]})
        elif agent == "coordinator":
            r = await self.c.post(f"{URL['coordinator']}/plan", json={**PLAN, "message_id": f"bench-{self.i}"})
            r.raise_for_status()
            if r.json().get("by") != "llm":  # a fallback answer would time the deterministic policy, not Nemotron
                raise RuntimeError(f"coordinator fell back: {r.json().get('raw')}")
            self.detail["coordinator"].append({"tool": r.json()["name"], **((r.json().get("raw") or {}).get("usage") or {})})
        elif agent == "eeg":
            r = await self.c.post(f"{URL['eeg']}/evidence", json={"subject": self.subject, "chars": [self.i % 35]})
            r.raise_for_status()

    async def timed(self, agent: str) -> float:
        t = time.perf_counter()
        await self.call(agent)
        return (time.perf_counter() - t) * 1000


AGENTS = ["audio", "vsr", "coordinator", "tts", "eeg"]


async def phase_alone(a: Agents, n: int) -> dict:
    out = {}
    for agent in AGENTS:
        for _ in range(3):  # warm-up
            await a.call(agent)
        out[agent] = stats([await a.timed(agent) for _ in range(n)])
        print(f"  alone    {agent:<12} {out[agent]}", flush=True)
    return out


def gpu_sampler(stop: asyncio.Event, samples: list):
    async def run():
        while not stop.is_set():
            q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True).stdout.strip().split(",")
            if len(q) == 2:
                samples.append((int(q[0]), int(q[1])))
            await asyncio.sleep(0.5)
    return run()


async def phase_together(a: Agents, n: int) -> dict:
    stop, gpu = asyncio.Event(), []
    sampler = asyncio.create_task(gpu_sampler(stop, gpu))

    async def loop(agent):
        return [await a.timed(agent) for _ in range(n)]

    t0 = time.perf_counter()
    res = await asyncio.gather(*[loop(x) for x in AGENTS])
    wall = time.perf_counter() - t0
    stop.set()
    await sampler
    out = {agent: stats(ms) for agent, ms in zip(AGENTS, res)}
    for agent in AGENTS:
        print(f"  together {agent:<12} {out[agent]}", flush=True)
    out["_gpu"] = {"util_mean_pct": round(statistics.fmean(u for u, _ in gpu)) if gpu else None,
                   "util_max_pct": max((u for u, _ in gpu), default=None), "mem_max_mib": max((m for _, m in gpu), default=None),
                   "wall_s": round(wall, 1)}
    print(f"  together GPU {out['_gpu']}", flush=True)
    return out


async def phase_end_to_end(a: Agents, n: int) -> dict:
    """One message through the gateway as the ui branch runs it: auto-speak (if on) speaks without a confirm."""
    parts: dict[str, list[float]] = {"evidence": [], "llm_plan": [], "spoken": [], "total": []}
    notes = {"auto_spoken": 0, "confirmed_by_bench": 0, "lip_ignored": 0, "no_reading": 0}
    g = URL["gateway"]
    for k in range(n):
        sid = f"bench-{uuid.uuid4().hex[:8]}"
        t0 = time.perf_counter()
        wav, vid = a.wavs[k % len(a.wavs)], a.videos[k % len(a.videos)]
        await asyncio.gather(
            a.c.post(f"{g}/api/{sid}/capture/audio", files={"audio": ("clip.wav", wav, "audio/wav")}, data={"enhance": "true"}),
            a.c.post(f"{g}/api/{sid}/capture/lip", files={"video": ("clip.mpg", vid, "video/mpeg")}))
        t_ev = time.perf_counter()
        t_plan = t_spoken = None
        for _ in range(500):
            s = (await a.c.get(f"{g}/api/{sid}/state")).json()
            now = time.perf_counter()
            if t_plan is None and (s.get("plan") or {}).get("by") == "llm":
                t_plan = now
            if s["message"]["status"] == "spoken" or s["spoken"]:
                t_spoken = now
                break
            if t_plan is not None and s["settings"].get("auto_speak") is False:
                break
            # auto-speak covers accepted readings only: once a plan is on screen without it, the person must answer
            pending = any(t["level"] == "pending" for t in s["timeline"])
            if s.get("plan") and not pending and "auto_speak_candidate_id" in s and s["auto_speak_candidate_id"] is None \
                    and s["message"]["status"] in ("clarifying", "selected"):
                t_plan = t_plan or now
                break
            if t_plan is not None and now - t_plan > 3:
                break
            await asyncio.sleep(0.02)
        if any("Ignored" in t["title"] and t["agent"] == "policy" for t in s["timeline"]):
            notes["lip_ignored"] += 1
        if t_spoken is None:
            shown = s["shown_candidate_ids"] or [c["id"] for c in s["message"]["candidates"]]
            if not shown:
                notes["no_reading"] += 1
                continue
            await a.c.post(f"{g}/api/{sid}/select", json={"candidate_id": shown[0], "method": "accessible_button_confirmed"})
            r = await a.c.post(f"{g}/api/{sid}/confirm", json={"method": "accessible_button_confirmed"})
            r.raise_for_status()
            t_spoken = time.perf_counter()
            notes["confirmed_by_bench"] += 1
        else:
            notes["auto_spoken"] += 1
        by = (s.get("plan") or {}).get("by")
        notes[f"plan_by_{by}"] = notes.get(f"plan_by_{by}", 0) + 1
        sp = s["spoken"][-1] if s["spoken"] else None
        if sp:
            (await a.c.get(f"{g}/api/{sid}/audio/{sp['message_id']}")).raise_for_status()
        t_end = time.perf_counter()
        parts["evidence"].append((t_ev - t0) * 1000)
        if t_plan:
            parts["llm_plan"].append((t_plan - t_ev) * 1000)
        parts["spoken"].append((t_spoken - t_ev) * 1000)
        parts["total"].append((t_end - t0) * 1000)
    out = {k: stats(v) for k, v in parts.items()}
    out["_notes"] = notes
    for k, v in out.items():
        print(f"  e2e      {k:<12} {v}", flush=True)
    return out


async def phase_voice_wait(a: Agents, n: int) -> dict:
    g, normal, early = URL["gateway"], [], []
    for k in range(n):
        wav = a.wavs[k % len(a.wavs)]
        t = time.perf_counter()  # the 700 ms of silence has just passed
        (await a.c.post(f"{g}/api/bench-n{k}-{uuid.uuid4().hex[:6]}/capture/audio", files={"audio": ("clip.wav", wav, "audio/wav")})).raise_for_status()
        normal.append((time.perf_counter() - t) * 1000)
        sid = f"bench-e{k}-{uuid.uuid4().hex[:6]}"
        t_early = time.perf_counter()  # 350 ms of silence: sent early
        r = await a.c.post(f"{g}/api/{sid}/capture/audio/early", files={"audio": ("clip.wav", wav, "audio/wav")})
        r.raise_for_status()
        await asyncio.sleep(max(0.0, 0.35 - (time.perf_counter() - t_early)))
        t = time.perf_counter()  # 700 ms of silence: commit
        (await a.c.post(f"{g}/api/{sid}/capture/audio/commit", json={"early_id": r.json()["early_id"]})).raise_for_status()
        early.append((time.perf_counter() - t) * 1000)
    out = {"sent_at_700ms": stats(normal), "early_decode": stats(early)}
    for k, v in out.items():
        print(f"  voice    {k:<14} {v}", flush=True)
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "evaluations" / "results" / "latency_brev.json"))
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT / "asr_eval"))
    from harness.config import load_config

    phrases = list(load_config(ROOT / "asr_eval" / "configs" / "eval.yaml").phrases.values())
    random.seed(7)
    videos = random.sample(sorted((ROOT / "data" / "grid").glob("s*/*.mpg")), 12)
    async with httpx.AsyncClient(timeout=120) as c:
        subject = (await c.get(f"{URL['eeg']}/subjects")).json()["subjects"][0]["subject"]
        a = Agents(c, phrases, videos, subject)
        print("preparing voice clips with Kokoro...", flush=True)
        a.wavs = [await a.tts(p) for p in phrases]
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        print(f"GPU: {gpu}\n1) each agent alone", flush=True)
        alone = await phase_alone(a, args.n)
        print("2) all agents at once", flush=True)
        together = await phase_together(a, args.n)
        print("3) one message end to end, through the gateway", flush=True)
        e2e = await phase_end_to_end(a, args.n)
        print("4) voice: end of utterance -> evidence in the session", flush=True)
        voice_wait = await phase_voice_wait(a, args.n)
    res = {"gpu": gpu, "n": args.n, "date": time.strftime("%Y-%m-%d"), "alone": alone, "together": together, "end_to_end": e2e,
           "detail": a.detail, "voice_wait_after_utterance": voice_wait,
           "inputs": {"voice": "Kokoro af_heart speaking the 20 phrases", "lips": "12 GRID videos (3 s, 25 fps)", "eeg": f"BNCI {subject}, one letter",
                      "coordinator": "2 candidates, ambiguous audio + proposed lip"}}
    Path(args.out).write_text(json.dumps(res, indent=2))
    print("\n| Agent | alone p50 / p95 ms | all together p50 / p95 ms |\n|---|---:|---:|")
    for x in AGENTS:
        print(f"| {x} | {alone[x]['p50']} / {alone[x]['p95']} | {together[x]['p50']} / {together[x]['p95']} |")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
