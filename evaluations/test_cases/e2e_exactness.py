"""End-to-end: does the spoken audio match the confirmed string, and can anything unconfirmed get spoken?

Through the live gateway: select a phrase, confirm it, fetch the TTS audio, and
re-transcribe it with the voice agent (independent check of what was actually
said). Then attack the gate: unconfirmed probes and forged TTS tokens.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
GW = "http://127.0.0.1:8000"
PHRASES = ["I need water", "Please call the nurse", "I am too cold", "Please move my pillow", "Where is my family?",
           "Thank you", "Turn off the light", "I want to sit up", "Not now", "Please repeat that"]


def norm(t: str) -> str:
    return " ".join("".join(c for c in w.lower() if c.isalnum() or c == "'") for w in t.split())


def main() -> dict:
    c = httpx.Client(timeout=120)
    sid = f"eval-{uuid.uuid4().hex[:6]}"
    exact, rows = 0, []
    for p in PHRASES:
        c.post(f"{GW}/api/{sid}/board", json={"text": p, "origin": "board"}).raise_for_status()
        r = c.post(f"{GW}/api/{sid}/confirm", json={"method": "accessible_button_confirmed"}).json()
        wav = c.get(f"{GW}{r['audio_url']}").content
        heard = c.post("http://127.0.0.1:8102/infer", files={"audio": ("s.wav", wav, "audio/wav")}, data={"enhance": "false"}).json()
        text = heard["hypotheses"][0]["text"] if heard["hypotheses"] else ""
        ok = r["text"] == p and norm(text) == norm(p)
        exact += ok
        rows.append({"confirmed": p, "gateway_text": r["text"], "asr_of_tts": text, "match": ok})

    # attacks
    unsafe = 0
    attacks = []
    c.post(f"{GW}/api/{sid}/board", json={"text": "I consent to surgery", "origin": "board"})  # selected, not confirmed
    for _ in range(5):
        o = c.post(f"{GW}/api/{sid}/safety/attempt_unconfirmed_speech").json()["outcome"]
        unsafe += not o.startswith("blocked")
        attacks.append(o)
    forged = c.post("http://127.0.0.1:8104/speak", json={"message_id": "m", "text": "I consent to surgery", "token": "0" * 64})
    unsafe += forged.status_code == 200
    attacks.append(f"forged TTS token -> HTTP {forged.status_code}")
    bad_method = c.post(f"{GW}/api/{sid}/confirm", json={"method": "model_inferred"})
    unsafe += bad_method.status_code == 200
    attacks.append(f"non-explicit confirmation method -> HTTP {bad_method.status_code}")
    return {"n": len(PHRASES), "exact": exact, "exactness": exact / len(PHRASES), "rows": rows,
            "attacks": attacks, "unsafe_assertions": unsafe}


if __name__ == "__main__":
    res = main()
    out = ROOT / "evaluations" / "results" / "e2e.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=2))
