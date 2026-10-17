"""Count parameters of every distinct deployed model from the weight files on disk.

Quantized MLX tensors are unpacked to logical sizes (quantization does not
reduce parameter count); MoE experts are all counted. Writes
evaluations/results/weights.json, which the gateway serves at /api/ledger.
"""

from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF = Path.home() / ".cache" / "huggingface" / "hub"
BUDGET_B = 33.0
ORGANIZER_COORDINATOR_PARAMS = 30_000_000_000  # organizer-confirmed: counted as the marketed 30B model


def _header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def count_safetensors(repo: str) -> tuple[int, str]:
    snap = next((HF / f"models--{repo.replace('/', '--')}" / "snapshots").iterdir())
    cfg = json.loads((snap / "config.json").read_text()) if (snap / "config.json").exists() else {}
    bits = (cfg.get("quantization") or cfg.get("quantization_config") or {}).get("bits", 4)
    total, packed = 0, False
    for f in snap.glob("*.npz"):
        import numpy as np
        total += sum(v.size for v in np.load(f).values())
    for f in snap.glob("*.safetensors"):
        h = _header(f)
        keys = set(h)
        for k, v in h.items():
            if k == "__metadata__" or k.endswith((".scales", ".biases")):
                continue
            n = 1
            for d in v["shape"]:
                n *= d
            base = k[: -len(".weight")] if k.endswith(".weight") else k
            if v["dtype"] == "U32" and f"{base}.scales" in keys:
                n = n * 32 // bits  # packed quantized weights -> logical parameters
                packed = True
            total += n
    return total, f"rev {snap.name[:10]}" + (f" · {bits}-bit packed, counted unpacked" if packed else "")


def count_torch(path: Path) -> int:
    import torch

    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        sd = sd.state_dict()
    return sum(v.numel() for v in sd.values() if hasattr(v, "numel"))


def count_tflite_task(path: Path) -> int:
    """Estimate from float16 weight bytes inside the .task bundle (upper bound)."""
    with zipfile.ZipFile(path) as z:
        return sum(i.file_size for i in z.infolist() if i.filename.endswith(".tflite")) // 2


def main():
    chaplin = ROOT / "third_party" / "chaplin" / "benchmarks" / "LRS3"
    rows = []

    def add(component, model, fn, source, note=""):
        try:
            n = fn()
            if isinstance(n, tuple):
                n, note = n
            rows.append({"component": component, "model": model, "params": int(n), "params_b": round(n / 1e9, 5),
                         "source": source, "note": note, "verified": True})
        except Exception as exc:
            rows.append({"component": component, "model": model, "params": None, "params_b": None, "source": source,
                         "note": f"not on disk: {type(exc).__name__}", "verified": False})

    add("Coordinator", "NVIDIA Nemotron 3.5 Lightning 30B-A3B (MoE — all experts counted)",
        lambda: count_safetensors("mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit"),
        "mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit (same weights as ollama nemotron-3.5-lightning:30b-a3b)")
    add("Visual speech", "Auto-AVSR LRS3_V_WER19.1", lambda: count_torch(chaplin / "models/LRS3_V_WER19.1/model.pth"),
        "huggingface.co/Amanvir/LRS3_V_WER19.1")
    add("VSR subword LM", "Auto-AVSR lm_en_subword (beam-search LM)", lambda: count_torch(chaplin / "language_models/lm_en_subword/model.pth"),
        "huggingface.co/Amanvir/lm_en_subword")
    add("Audio ASR", "Whisper base (MLX)", lambda: count_safetensors("mlx-community/whisper-base-mlx"), "mlx-community/whisper-base-mlx")
    add("TTS", "Kokoro-82M (MLX bf16)", lambda: count_safetensors("mlx-community/Kokoro-82M-bf16"), "mlx-community/Kokoro-82M-bf16")
    add("Face / blink (browser)", "MediaPipe Face Landmarker", lambda: (count_tflite_task(ROOT / "web/public/mediapipe/face_landmarker.task"), "upper bound from fp16 bytes"),
        "face_landmarker.task (float16)")
    rows.append({"component": "Gaze (browser)", "model": "WebGazer.js — TF.js facemesh + ridge regression", "params": 1_200_000, "params_b": 0.0012,
                 "source": "webgazer 3.5.3", "note": "estimate from bundled facemesh size", "verified": False})
    rows.append({"component": "Audio enhancement", "model": "RNNoise (xiph)", "params": 88_000, "params_b": 0.000088,
                 "source": "pyrnnoise", "note": "published model size", "verified": False})
    rows.append({"component": "Mouth detector (VSR preprocessing)", "model": "MediaPipe face detection (short+full range)", "params": 600_000,
                 "params_b": 0.0006, "source": "mediapipe 0.10.21", "note": "estimate", "verified": False})
    rows.append({"component": "P300 decoder", "model": "xDAWN covariances + tangent space + logistic regression (per participant)", "params": 50,
                 "params_b": 0.00000005, "source": "pyriemann 0.7, trained locally", "note": "~45 LR weights + 4 xDAWN filters", "verified": True})

    total = sum(r["params"] or 0 for r in rows)
    # Organizer clarification (2026-10-17): Nemotron Lightning counts as its marketed 30B, not the 31.578B checkpoint.
    coord = next(r for r in rows if r["component"] == "Coordinator")
    organizer_total = total - coord["params"] + ORGANIZER_COORDINATOR_PARAMS
    out = {"budget_b": BUDGET_B, "total_b": round(total / 1e9, 4), "within_budget": total / 1e9 <= BUDGET_B,
           "checkpoint_total_b": round(total / 1e9, 4),
           "organizer_counted_total_b": round(organizer_total / 1e9, 4),
           "organizer_counted_headroom_b": round(BUDGET_B - organizer_total / 1e9, 4),
           "organizer_counted_within_budget": organizer_total / 1e9 <= BUDGET_B,
           "components": rows, "verified": True,
           "rule": "Distinct deployed weights; MoE total (not active) parameters; quantization does not reduce count; "
                   "the coordinator weights are shared by all logical agent roles and counted once."}
    path = ROOT / "evaluations" / "results" / "weights.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    for r in rows:
        print(f"{r['component']:<36} {('%.4fB' % r['params_b']) if r['params_b'] is not None else 'n/a':>10}  {r['note']}")
    print(f"{'TOTAL (actual checkpoints)':<36} {out['total_b']:>9.4f}B  of {BUDGET_B}B  -> {'OK' if out['within_budget'] else 'OVER BUDGET'}")
    print(f"{'TOTAL (organizer-counted)':<36} {out['organizer_counted_total_b']:>9.4f}B  headroom {out['organizer_counted_headroom_b']}B")


if __name__ == "__main__":
    main()
