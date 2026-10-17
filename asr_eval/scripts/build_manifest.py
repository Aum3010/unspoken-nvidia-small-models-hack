"""Phase 1: build data/manifest.csv from whatever data is present.

Columns (spec): path, source, speaker_id, condition, label_type, phrase_id, split, duration_s
Extra:          task (phrases20 | torgo_words), text (reference), origin (real | tts | generated),
                degraded (bool), derived_from, group (dysarthric | control | '')

Splits are by speaker within each source (TORGO stratified by dysarthric/control),
assigned before degradation so every augmented copy inherits its source's split.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from degrade import degrade_file  # noqa: E402
from harness import audio  # noqa: E402
from harness.config import load_config  # noqa: E402
from harness.text import normalize  # noqa: E402

MIN_SOURCE_S = 0.2
COLUMNS = ["path", "source", "speaker_id", "condition", "label_type", "phrase_id", "split", "duration_s",
           "task", "text", "origin", "degraded", "derived_from", "group"]


def row(**kw) -> dict:
    base = {"condition": "clean", "phrase_id": "", "task": "phrases20", "text": "", "degraded": False,
            "derived_from": "", "group": "", "split": ""}
    return {**base, **kw}


def rel(p: Path, data: Path) -> str:
    return p.resolve().relative_to(data).as_posix()


# ---- sources -------------------------------------------------------------------------------------

def own(cfg) -> list[dict]:
    rows, data = [], cfg.data
    negs = cfg["own"]["negatives"]
    for p in sorted((data / "own").glob("*.wav")):
        parts = p.stem.split("_")
        if len(parts) < 4:
            print(f"  skip (name): {p.name}")
            continue
        spk, cond, pid, _take = parts[0], parts[1], "_".join(parts[2:-1]), parts[-1]
        dur = sf.info(str(p)).duration
        if dur < MIN_SOURCE_S:  # released too early: no phrase fits
            print(f"  skip (too short, {dur:.2f} s): {p.name}")
            continue
        if pid in cfg.phrases:
            r = row(label_type="positive", phrase_id=pid, text=cfg.phrases[pid])
        elif pid.startswith("neg_"):
            name = pid[4:]
            if name in cfg.hard_negatives:
                r = row(label_type="hard_negative", text=cfg.hard_negatives[name])
            else:
                r = row(label_type=negs.get(name, "unknown_speech"))
        else:
            print(f"  skip (unknown phrase id): {p.name}")
            continue
        rows.append({**r, "path": rel(p, data), "source": "own", "speaker_id": f"own:{spk}", "condition": cond, "origin": "real"})
    return rows


def tts(cfg) -> list[dict]:
    rows, data = [], cfg.data
    for p in sorted((data / "tts").glob("*/*.wav")):
        tid, voice = p.stem, p.parent.name
        if tid in cfg.phrases:
            r = row(label_type="positive", phrase_id=tid, text=cfg.phrases[tid])
        elif tid in cfg.hard_negatives:
            r = row(label_type="hard_negative", text=cfg.hard_negatives[tid])
        else:
            continue
        rows.append({**r, "path": rel(p, data), "source": "tts", "speaker_id": f"tts:{voice}", "condition": "tts_clean", "origin": "tts"})
    return rows


def generated(cfg, rng) -> list[dict]:
    rows, data = [], cfg.data
    n = cfg["manifest"]["generated_nonspeech"]
    for i in range(n):
        p = data / "generated" / f"gen_{i:04d}.wav"
        dur = float(rng.uniform(1, 6))
        level = float(rng.uniform(-75, -45))
        if not p.exists():
            x = audio.digital_silence(dur) if i % 2 == 0 else audio.mic_noise(dur, level, rng)
            audio.save(p, x)
        kind = "digital_silence" if i % 2 == 0 else "mic_noise"
        rows.append(row(path=rel(p, data), source="generated", speaker_id=f"gen:{p.stem}", condition=kind,
                        label_type="silence", origin="generated"))
    return rows


def librispeech(cfg, rng) -> list[dict]:
    data = cfg.data
    root = data / "raw" / "librispeech"
    items = []
    for f in sorted(root.rglob("*.trans.txt")):
        for line in f.read_text().splitlines():
            uid, text = line.split(" ", 1)
            items.append((f.parent / f"{uid}.flac", text, uid.split("-")[0]))
    if not items:
        return []
    pick = rng.choice(len(items), size=min(cfg["manifest"]["librispeech_n"], len(items)), replace=False)
    return [row(path=rel(items[i][0], data), source="librispeech", speaker_id=f"ls:{items[i][2]}", label_type="unknown_speech",
                text=items[i][1].lower(), origin="real") for i in sorted(pick)]


def nonspeech_public(cfg, rng) -> tuple[list[dict], list[Path]]:
    """Cut MUSAN noise / Speech Commands background / ESC-50 categories into 1-6 s clips.

    MUSAN noise files are split by name hash: half are negatives, half feed the degradation grid,
    so no noise recording appears both as a test negative and inside a positive.
    """
    data, m = cfg.data, cfg["manifest"]
    raw = data / "raw"
    musan = sorted((raw / "musan").rglob("*.wav"))
    musan_neg = [p for p in musan if zlib.crc32(p.stem.encode()) % 2 == 0]
    musan_aug = [p for p in musan if zlib.crc32(p.stem.encode()) % 2 == 1]
    sc = sorted((raw / "speech_commands").rglob("_background_noise_/*.wav"))
    esc, esc_cat = [], {}
    meta = next((raw / "esc50").rglob("esc50.csv"), None)
    if meta:
        df = pd.read_csv(meta)
        esc_cat = {f: c for f, c in zip(df["filename"], df["category"]) if c in set(m["esc50_categories"])}
        esc = [meta.parent.parent / "audio" / f for f in sorted(esc_cat)]
    n = m["nonspeech_public_n"]
    # shares 50/15/35, renormalised over the sources actually present (e.g. MUSAN still downloading)
    present = [(src, files, w) for src, files, w in (("musan", musan_neg, 0.5), ("speech_commands", sc, 0.15), ("esc50", esc, 0.35)) if files]
    total_w = sum(w for _, _, w in present) or 1.0
    quotas = [(src, files, round(n * w / total_w)) for src, files, w in present]
    missing = {"musan", "speech_commands", "esc50"} - {q[0] for q in quotas}
    if missing:
        print(f"  non-speech sources missing (quota redistributed): {sorted(missing)}")
    rows = []
    for src, files, quota in quotas:
        if not files:
            continue
        for k in range(quota):
            f = files[k % len(files)]
            info = sf.info(str(f))
            dur = float(min(info.duration, rng.uniform(1, 6)))
            off = float(rng.uniform(0, max(0.0, info.duration - dur)))
            p = data / "nonspeech" / src / f"{f.stem}_{k:04d}.wav"
            if not p.exists():
                audio.save(p, audio.load(f, offset_s=off, dur_s=dur))
            cond = {"musan": f.parent.name, "esc50": esc_cat.get(f.name, ""), "speech_commands": f.stem}[src]
            rows.append(row(path=rel(p, data), source=src, speaker_id=f"{src}:{f.stem}", condition=cond,
                            label_type="non_speech", origin="real"))
    return rows, musan_aug


def torgo(cfg, rng) -> list[dict]:
    data = cfg.data
    root = data / "raw" / "torgo"
    if not root.exists():
        return []
    phrase_norms = {normalize(t) for t in {**cfg.phrases, **cfg.hard_negatives}.values()}
    singles, sentences, excluded = [], [], []
    for prompt in sorted(root.rglob("prompts/*.txt")):
        session = prompt.parent.parent
        spk = next((p for p in prompt.parts if re.fullmatch(r"[FM]C?\d\d", p)), None)
        wav = next((session / mic / f"{prompt.stem}.wav" for mic in ("wav_arrayMic", "wav_headMic")
                    if (session / mic / f"{prompt.stem}.wav").exists()), None)
        if not spk or wav is None:
            continue
        dur = sf.info(str(wav)).duration
        if dur < MIN_SOURCE_S:  # truncated recordings in the corpus (e.g. 9-75 ms); no word fits
            excluded.append({"path": rel(wav, data), "duration_s": round(dur, 3), "reason": f"shorter than {MIN_SOURCE_S} s"})
            continue
        text = prompt.read_text(errors="ignore").strip()
        if not text or any(c in text for c in "[]/") or "xxx" in text.lower() or ".jpg" in text.lower():
            continue
        words = re.findall(r"[a-z']+", text.lower())
        item = {"wav": wav, "spk": spk, "text": " ".join(words), "group": "control" if spk[1] == "C" else "dysarthric"}
        if len(words) == 1:
            singles.append(item)
        elif len(words) >= 3 and normalize(item["text"]) not in phrase_norms:
            sentences.append(item)
    by_word = defaultdict(set)
    counts = Counter()
    for s in singles:
        by_word[s["text"]].add(s["spk"])
        counts[s["text"]] += 1
    vocab = cfg["manifest"].get("torgo_vocab") or [w for w, _ in sorted(by_word.items(), key=lambda kv: (-len(kv[1]), -counts[kv[0]], kv[0]))
                                                   ][: cfg["manifest"]["torgo_vocab_size"]]
    (data / "torgo_vocab.json").write_text(json.dumps(
        {"vocab": vocab, "chosen_by": "config" if cfg["manifest"].get("torgo_vocab") else "most distinct speakers, then count",
         "stats": {w: {"speakers": len(by_word[w]), "clips": counts[w]} for w in vocab}}, indent=2))
    rows = []
    for s in singles:
        common = dict(path=rel(s["wav"], data), source="torgo", speaker_id=f"torgo:{s['spk']}", group=s["group"], text=s["text"], origin="real")
        if s["text"] in vocab:
            rows.append(row(**common, label_type="positive", phrase_id=s["text"], task="torgo_words"))
    # homophones of a vocabulary word ("know" for "no", "write" for "right") cannot be rejected acoustically,
    # so they are not valid negatives; identical CMU pronunciation decides
    from harness.decide import phones

    vocab_phones = {phones(w) for w in vocab} - {None}
    others, homophones = [], Counter()
    for s in singles:
        if s["text"] in vocab:
            continue
        if phones(s["text"]) in vocab_phones:
            homophones[s["text"]] += 1
        else:
            others.append(s)
    if homophones:
        print(f"  torgo: dropped homophone negatives {dict(homophones)}")
    for i in rng.permutation(len(others))[:300]:
        s = others[i]
        rows.append(row(path=rel(s["wav"], data), source="torgo", speaker_id=f"torgo:{s['spk']}", group=s["group"], text=s["text"],
                        origin="real", label_type="unknown_speech", task="torgo_words"))
    for i in rng.permutation(len(sentences))[:400]:
        s = sentences[i]
        rows.append(row(path=rel(s["wav"], data), source="torgo", speaker_id=f"torgo:{s['spk']}", group=s["group"], text=s["text"],
                        origin="real", label_type="unknown_speech", task="phrases20"))
    print(f"  torgo vocab ({len(vocab)}): {vocab}")
    if excluded:
        pd.DataFrame(excluded).to_csv(data / "excluded_clips.csv", index=False)
        print(f"  torgo: excluded {len(excluded)} truncated recordings -> excluded_clips.csv")
    return rows


# ---- split + degrade ------------------------------------------------------------------------------

def assign_splits(df: pd.DataFrame, cfg, seed: int) -> pd.DataFrame:
    frac = cfg["manifest"]["calib_fraction"]
    explicit = cfg["manifest"].get("calib_speakers") or {}
    split = {}
    for src, g in df.groupby("source"):
        if src in explicit:
            wanted = {f"{src}:{s}" for s in explicit[src]}
            for spk in g["speaker_id"].unique():
                split[spk] = "calib" if spk in wanted else "test"
            continue
        for stratum, gg in g.groupby("group"):
            spks = sorted(gg["speaker_id"].unique())
            r = np.random.default_rng(zlib.crc32(f"{seed}|{src}|{stratum}".encode()))
            r.shuffle(spks)
            n_cal = 0 if len(spks) < 2 else min(len(spks) - 1, max(1, round(frac * len(spks))))
            for i, spk in enumerate(spks):
                split[spk] = "calib" if i < n_cal else "test"
    df["split"] = df["speaker_id"].map(split)
    cal, test = set(df.loc[df.split == "calib", "speaker_id"]), set(df.loc[df.split == "test", "speaker_id"])
    assert not cal & test, f"speaker overlap between calib and test: {sorted(cal & test)[:5]}"
    return df


def add_degraded(df: pd.DataFrame, cfg, noise_pool: list[Path]) -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor

    dcfg = cfg["degrade"]
    base = df[(df.label_type == "positive") & df.source.isin(dcfg["sources"]) & ~df.degraded]
    base = base[(base.source != "own") | (base.condition == "normal")]  # never stack on whisper/soft/... takes
    if base.empty:
        return df
    jobs = [(cfg.data / r["path"], cfg.data / "degraded" / r["source"] / Path(r["path"]).parent.as_posix().replace("/", "__"))
            for _, r in base.iterrows()]
    with ProcessPoolExecutor() as pool:
        made = list(pool.map(degrade_file, [j[0] for j in jobs], [j[1] for j in jobs], [dcfg] * len(jobs),
                             [noise_pool] * len(jobs), chunksize=16))
    new = [{**r.to_dict(), "path": rel(dst, cfg.data), "condition": cond, "degraded": True, "derived_from": r["path"]}
           for (_, r), copies in zip(base.iterrows(), made) for cond, dst in copies]
    print(f"  degraded copies: {len(new)} from {len(base)} positives")
    return pd.concat([df, pd.DataFrame(new)], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    rng = np.random.default_rng(cfg["seed"])
    rows = own(cfg) + tts(cfg) + generated(cfg, rng) + librispeech(cfg, rng)
    ns, noise_pool = nonspeech_public(cfg, rng)
    rows += ns + torgo(cfg, rng)
    df = pd.DataFrame(rows)
    df = assign_splits(df, cfg, cfg["seed"])
    df = add_degraded(df, cfg, noise_pool)
    df["duration_s"] = [round(sf.info(str(cfg.data / p)).duration, 3) for p in df["path"]]
    df = df[COLUMNS]
    out = cfg.data / "manifest.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")
    summary = df.groupby(["source", "origin", "label_type", "split"]).size().rename("n").reset_index()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()  # the process pool re-imports this module; keep work under the main guard
