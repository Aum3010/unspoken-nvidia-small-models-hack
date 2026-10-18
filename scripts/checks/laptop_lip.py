"""From the laptop, through the tunnel: a mouth clip the old way (whole clip POSTed after the pause) vs streamed
(pieces over the session WebSocket while mouthing, then the finish call). Time = end of clip -> reading back."""
import asyncio, json, statistics, sys, time, uuid
from pathlib import Path

import av, httpx, numpy as np, websockets

sys.path.insert(0, sys.argv[1])
from shared.service import decode_video_rgb  # noqa: E402

G, WS = "http://127.0.0.1:8000", "ws://127.0.0.1:8000"


class Pipe:
    def __init__(self):
        self.parts = []
    def write(self, b):
        self.parts.append(bytes(b)); return len(b)


def live_webm(path: Path):
    g, _ = decode_video_rgb(path.read_bytes(), 30)
    out, marks = Pipe(), []
    with av.open(out, "w", format="webm") as c:
        s = c.add_stream("libvpx-vp9", rate=30)
        s.width, s.height, s.pix_fmt, s.bit_rate = 1280, 720, "yuv420p", 2_500_000
        s.options = {"deadline": "realtime", "cpu-used": "8"}
        for k, f in enumerate(g):
            nw = int(round(f.shape[1] * 720 / f.shape[0])) // 2 * 2
            fr = av.VideoFrame.from_ndarray(f, format="rgb24").reformat(width=nw, height=720).to_ndarray(format="rgb24")
            canvas = np.zeros((720, 1280, 3), np.uint8); x0 = (1280 - nw) // 2; canvas[:, x0:x0 + nw] = fr
            for p in s.encode(av.VideoFrame.from_ndarray(canvas, format="rgb24")):
                c.mux(p)
            if k % 8 == 7:   # ~250 ms of video
                marks.append(sum(map(len, out.parts)))
        for p in s.encode():
            c.mux(p)
    data = b"".join(out.parts)
    cuts = sorted({m for m in marks if 0 < m < len(data)} | {len(data)})
    return data, [data[a:b] for a, b in zip([0] + cuts[:-1], cuts)]


def frame(header: dict, piece: bytes = b"") -> bytes:
    h = json.dumps(header).encode()
    return len(h).to_bytes(2, "big") + h + piece


async def main():
    clips = sorted(Path(sys.argv[2]).glob("s*/*.mpg"))
    old, new, same = [], [], 0
    async with httpx.AsyncClient(timeout=120) as c:
        for path in clips:
            data, pieces = live_webm(path)
            sid = f"lap-{uuid.uuid4().hex[:8]}"
            t0 = time.perf_counter()
            a = (await c.post(f"{G}/api/{sid}/capture/lip", files={"video": ("clip.webm", data, "video/webm")})).json()
            old.append((time.perf_counter() - t0) * 1000)
            sid = f"lap-{uuid.uuid4().hex[:8]}"
            async with websockets.connect(f"{WS}/ws/{sid}", max_size=None) as ws:
                await ws.recv()
                sp = "lip" + uuid.uuid4().hex[:6]
                for i, p in enumerate(pieces):
                    await ws.send(frame({"id": sp, "seq": i, "mime": "video/webm;codecs=vp9"}, p))
                    if i < len(pieces) - 1:
                        await asyncio.sleep(0.25)   # recorded in real time; the last piece comes at the stop
                t0 = time.perf_counter()
                b = (await c.post(f"{G}/api/{sid}/capture/lip/stream", json={"stream_id": sp, "chunks": len(pieces)})).json()
                new.append((time.perf_counter() - t0) * 1000)
            ka = (a["event"]["quality"].get("enroll_best_dist"), a["event"]["quality"].get("frames"), a["event"]["quality_flags"])
            kb = (b["event"]["quality"].get("enroll_best_dist"), b["event"]["quality"].get("frames"), b["event"]["quality_flags"])
            same += ka == kb
            print(f"{path.name} {len(data)//1024:>4} KB {len(pieces):>2} pieces  whole {old[-1]:5.0f} ms  streamed {new[-1]:5.0f} ms  same={ka == kb}", flush=True)
    print(f"\nsame reading {same}/{len(clips)} · end of clip -> reading, whole p50 {statistics.median(old):.0f} ms, "
          f"streamed p50 {statistics.median(new):.0f} ms")


if __name__ == "__main__":
    asyncio.run(main())
