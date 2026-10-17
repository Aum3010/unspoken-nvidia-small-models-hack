"""Model/event/source metadata and the distinct-weights parameter ledger."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SERVICE_PORTS = {
    "gateway": int(os.environ.get("UNSPOKEN_GATEWAY_PORT", 8000)),
    "vsr": int(os.environ.get("UNSPOKEN_VSR_PORT", 8101)),
    "audio": int(os.environ.get("UNSPOKEN_AUDIO_PORT", 8102)),
    "eeg": int(os.environ.get("UNSPOKEN_EEG_PORT", 8103)),
    "tts": int(os.environ.get("UNSPOKEN_TTS_PORT", 8104)),
    "coordinator": int(os.environ.get("UNSPOKEN_COORDINATOR_PORT", 8105)),
}


def service_url(name: str) -> str:
    host = os.environ.get(f"UNSPOKEN_{name.upper()}_HOST", "127.0.0.1")
    return f"http://{host}:{SERVICE_PORTS[name]}"


# Distinct deployed weights. Counts are filled by scripts/verify_weights.py
# where the weights are on disk; the rest are documented upstream values.
MODEL_LEDGER = [
    {
        "component": "Coordinator",
        "model": "NVIDIA Nemotron 3.5 Lightning 30B-A3B (MoE, total params counted)",
        "params_b": None,
        "source": "ollama nemotron-3.5-lightning:30b-a3b / mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit",
    },
    {
        "component": "Visual speech",
        "model": "Auto-AVSR LRS3_V_WER19.1 (via Chaplin)",
        "params_b": 0.2504,
        "source": "huggingface.co/Amanvir/LRS3_V_WER19.1",
    },
    {
        "component": "VSR subword LM",
        "model": "Auto-AVSR lm_en_subword",
        "params_b": None,
        "source": "huggingface.co/Amanvir/lm_en_subword",
    },
    {"component": "Audio ASR", "model": "Whisper base (MLX)", "params_b": 0.074, "source": "mlx-community/whisper-base-mlx"},
    {"component": "Audio enhancement", "model": "RNNoise (xiph)", "params_b": 0.00009, "source": "pyrnnoise"},
    {"component": "TTS", "model": "Kokoro-82M (MLX bf16)", "params_b": 0.082, "source": "mlx-community/Kokoro-82M-bf16"},
    {"component": "Face / blink (browser)", "model": "MediaPipe Face Landmarker", "params_b": 0.0013, "source": "face_landmarker.task float16"},
    {"component": "Gaze (browser)", "model": "MobileGaze MobileOne-S0 (eye-image gaze) + iris ridge calibration", "params_b": 0.00124, "source": "yakhyo/gaze-estimation mobileone_s0_gaze.onnx (MIT)"},
    {"component": "P300 decoder", "model": "xDAWN covariances + tangent space + logistic regression", "params_b": 0.0000002, "source": "pyriemann, trained per participant"},
]
