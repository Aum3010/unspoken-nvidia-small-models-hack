"""Lip path stage timings on Brev, on browser-like clips: GRID video placed in a 1280x720 frame, VP8 webm at
2.5 Mbps (what MediaRecorder sends). Also: the size of a face-box crop of the same clip (item 8)."""
import io, sys, time, random, statistics
from pathlib import Path
import numpy as np, av, torch
ROOT = Path.home() / "unspoken"; sys.path.insert(0, str(ROOT))
import services.vsr.app as vsr
from shared.service import decode_video_rgb

def to_webm(frames, fps=30, rate=2_500_000):
    buf = io.BytesIO()
    with av.open(buf, "w", format="webm") as c:
        s = c.add_stream("libvpx", rate=fps); s.width, s.height = frames.shape[2], frames.shape[1]
        s.pix_fmt = "yuv420p"; s.bit_rate = rate
        for f in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")): c.mux(p)
        for p in s.encode(): c.mux(p)
    return buf.getvalue()

m = vsr._load()
random.seed(0)
clips = random.sample(sorted((ROOT / "data" / "grid").glob("s*/*.mpg")), 8)
rows = []
for path in clips:
    g, _ = decode_video_rgb(path.read_bytes(), 30)
    big = np.zeros((len(g), 720, 1280, 3), np.uint8)
    h, w = g.shape[1] * 2, g.shape[2] * 2
    up = g.repeat(2, axis=1).repeat(2, axis=2)
    y0, x0 = (720 - h) // 2, (1280 - w) // 2
    big[:, y0:y0 + h, x0:x0 + w] = up
    data = to_webm(big)
    crop = to_webm(np.ascontiguousarray(up[:, : h // 2 * 2, : w // 2 * 2]))
    t = {}
    t0 = time.perf_counter(); frames, dur = decode_video_rgb(data, 25); t["decode"] = time.perf_counter() - t0
    t0 = time.perf_counter(); lm = m["detector"].detect(frames, m["detector"].full_range_detector); t["detect"] = time.perf_counter() - t0
    t0 = time.perf_counter(); crops = m["process"](frames, lm); t["crop"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    x = m["transform"](torch.tensor(crops))
    with torch.no_grad():
        enc = m["avsr"].model.encode(x.to(m["avsr"].device)); torch.cuda.synchronize()
    t["encode"] = time.perf_counter() - t0
    t0 = time.perf_counter(); vsr.enrolled.match(enc.cpu().numpy(), 30); t["dtw"] = time.perf_counter() - t0
    t["frames"] = len(frames); t["dur"] = dur; t["webm_kb"] = len(data) / 1024; t["crop_kb"] = len(crop) / 1024
    t["crop_px"] = f"{up.shape[2]}x{up.shape[1]}"
    rows.append(t)
for k in ("decode", "detect", "crop", "encode", "dtw"):
    v = [r[k] * 1000 for r in rows]; print(f"{k:<7} p50 {statistics.median(v):6.1f} ms   max {max(v):6.1f} ms")
print(f"clip {statistics.median([r['dur'] for r in rows]):.1f} s, {statistics.median([r['frames'] for r in rows])} frames at 25 fps")
print(f"upload 1280x720 webm {statistics.median([r['webm_kb'] for r in rows]):.0f} KB vs face-box crop ({rows[0]['crop_px']}) {statistics.median([r['crop_kb'] for r in rows]):.0f} KB")
