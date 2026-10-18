"""Streamed mouth clips vs whole-clip upload, against the running lip agent on Brev.

GRID clips are placed in a 1280x720 frame and written as live webm (VP9 and VP8, to a non-seekable writer, so no
cues and unknown sizes, like MediaRecorder), then cut into 250 ms pieces. Each clip is read twice:
  whole:    POST /infer with all the bytes (the old path)
  streamed: pieces to /stream/<id>/chunk in real time, then /stream/<id>/finish
Same reading? (hypotheses, scores, face ratio, mouth motion, enrollment distance.) And how long after the end?"""
import io, random, statistics, sys, time, uuid
from pathlib import Path
import av, httpx, numpy as np

ROOT = Path.home() / "unspoken"
sys.path.insert(0, str(ROOT))
from shared.service import decode_video_rgb  # noqa: E402

VSR = "http://127.0.0.1:8101"


class Pipe:
    """Write-only, not seekable: the muxer writes a live stream."""
    def __init__(self):
        self.parts = []
    def write(self, b):
        self.parts.append(bytes(b)); return len(b)


def live_webm(path: Path, codec: str):
    g, _ = decode_video_rgb(path.read_bytes(), 30)
    out = Pipe()
    with av.open(out, "w", format="webm") as c:
        s = c.add_stream(codec, rate=30)
        s.width, s.height, s.pix_fmt, s.bit_rate = 1280, 720, "yuv420p", 2_500_000
        times = []
        for k, f in enumerate(g):
            nw = int(round(f.shape[1] * 720 / f.shape[0])) // 2 * 2
            fr = av.VideoFrame.from_ndarray(f, format="rgb24").reformat(width=nw, height=720).to_ndarray(format="rgb24")
            canvas = np.zeros((720, 1280, 3), np.uint8); x0 = (1280 - nw) // 2; canvas[:, x0:x0 + nw] = fr
            for p in s.encode(av.VideoFrame.from_ndarray(canvas, format="rgb24")):
                c.mux(p)
            times.append((k / 30, sum(map(len, out.parts))))
        for p in s.encode():
            c.mux(p)
    data = b"".join(out.parts)
    # 250 ms pieces: cut where the byte count stood at each 250 ms of media time
    cuts = sorted({n for t, n in times if abs((t * 4) - round(t * 4)) < 1e-6 and n > 0} | {len(data)})
    pieces, prev = [], 0
    for n in cuts:
        if n > prev:
            pieces.append(data[prev:n]); prev = n
    return data, pieces


def key(ev):
    q = ev["quality"]
    return ([(h["text"], h["score"]) for h in ev["hypotheses"]], ev["quality_flags"],
            q.get("face_detected_ratio"), q.get("mouth_motion"), q.get("enroll_best_dist"), q.get("frames"))


random.seed(1)
clips = random.sample(sorted((ROOT / "data" / "grid").glob("s*/*.mpg")), int(sys.argv[1]) if len(sys.argv) > 1 else 12)
same, whole_ms, tail_ms, n = 0, [], [], 0
with httpx.Client(timeout=120) as c:
    for i, path in enumerate(clips):
        codec = "libvpx-vp9" if i % 2 == 0 else "libvpx"
        data, pieces = live_webm(path, codec)
        t0 = time.perf_counter()
        a = c.post(f"{VSR}/infer", files={"video": ("clip.webm", data, "video/webm")}).json()
        whole_ms.append((time.perf_counter() - t0) * 1000)
        sid = f"eq-{uuid.uuid4().hex[:8]}"
        for p in pieces:  # real time: one piece per 250 ms
            c.post(f"{VSR}/stream/{sid}/chunk", content=p, params={"mime": "video/webm"})
            time.sleep(0.25)
        t0 = time.perf_counter()
        b = c.post(f"{VSR}/stream/{sid}/finish", data={"session_id": "eq"}).json()
        tail_ms.append((time.perf_counter() - t0) * 1000)
        ok = key(a) == key(b)
        same += ok; n += 1
        print(f"{path.parent.name}/{path.name} {codec:<10} {len(data)//1024:>5} KB {len(pieces):>2} pieces  "
              f"same={ok}  whole {whole_ms[-1]:5.0f} ms  streamed tail {tail_ms[-1]:5.0f} ms  "
              f"{'' if ok else str(key(a)[1:]) + ' vs ' + str(key(b)[1:])}", flush=True)
print(f"\nidentical readings {same}/{n}")
print(f"whole clip /infer   p50 {statistics.median(whole_ms):.0f} ms")
print(f"streamed, after end p50 {statistics.median(tail_ms):.0f} ms")
