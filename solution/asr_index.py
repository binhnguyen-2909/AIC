#!/usr/bin/env python3
"""Vietnamese ASR index for the Vortex-style retrieval branch.

The official data contains MP4 audio but no transcript files.  This script
transcribes video audio with PhoWhisper and stores timestamped chunks.  The
``search`` command performs a lightweight BM25 search over chunks and maps the
best timestamp back to the nearest extracted keyframe frame index.

It is deliberately independent of the production FAISS/BM25 artifacts: an
incomplete ASR run can be resumed without corrupting the visual index.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXTRACTED = ROOT / "extracted"
MEDIA = EXTRACTED / "media-info-aic25-b1" / "media-info"
MAP_DIR = EXTRACTED / "map-keyframes-aic25-b1" / "map-keyframes"


def video_path(video_id: str) -> Path | None:
    batch = video_id.split("_")[0]
    candidates = sorted(EXTRACTED.glob(f"Videos_{batch}*/video/{video_id}.*"))
    return candidates[0] if candidates else None


def load_audio(path: Path) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
        "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1",
    ]
    raw = subprocess.check_output(cmd)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe(video_id: str, asr, batch_size: int = 8,
               workers: int = 0, timestamp_mode: str = "word") -> dict:
    path = video_path(video_id)
    if path is None:
        raise FileNotFoundError(video_id)
    audio = load_audio(path)
    # Pass sampling_rate as part of the audio input object; otherwise recent
    # Transformers versions forward it as an unsupported generate kwarg.
    result = asr(
        {"raw": audio, "sampling_rate": 16000},
        num_workers=workers,
        batch_size=batch_size,
        return_timestamps=("word" if timestamp_mode == "word" else True),
        generate_kwargs={"language": "vi", "task": "transcribe"},
    )
    words = []
    for chunk in result.get("chunks", []):
        ts = chunk.get("timestamp") or (None, None)
        text = re.sub(r"\s+", " ", str(chunk.get("text", ""))).strip()
        if not text:
            continue
        words.append({
            "start": None if ts[0] is None else float(ts[0]),
            "end": None if ts[1] is None else float(ts[1]),
            "text": text,
        })
    # Word timestamps are useful for mapping back to frames, but individual
    # words are poor BM25 documents.  Build short overlapping-search units and
    # collapse pathological consecutive repetitions common in music/noisy
    # audio with small ASR checkpoints.
    chunks = []
    cur = []
    for word in words:
        if cur and word["text"].lower() == cur[-1]["text"].lower():
            continue
        cur.append(word)
        start = cur[0]["start"]
        end = word["end"]
        if len(cur) >= 40 or (start is not None and end is not None and end - start >= 15):
            chunks.append({
                "start": start, "end": end,
                "text": " ".join(x["text"] for x in cur),
            })
            cur = []
    if cur:
        chunks.append({
            "start": cur[0]["start"], "end": cur[-1]["end"],
            "text": " ".join(x["text"] for x in cur),
        })
    return {"video_id": video_id, "chunks": chunks,
            "text": " ".join(c["text"] for c in chunks)}


def all_video_ids() -> list[str]:
    return sorted(p.stem for p in MEDIA.glob("*.json"))


def build(args):
    import torch
    from transformers import pipeline

    device = 0 if args.device.startswith("cuda") else -1
    dtype = torch.float16 if device >= 0 else torch.float32
    asr = pipeline(
        "automatic-speech-recognition", model=args.model,
        device=device, torch_dtype=dtype, chunk_length_s=30,
        stride_length_s=(5, 2),
    )
    out = Path(args.out)
    done = {}
    if out.exists() and not args.overwrite:
        with out.open() as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    done[row["video_id"]] = row
    ids = [args.video] if args.video else all_video_ids()
    if args.limit:
        ids = ids[:args.limit]
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "a"
    with out.open(mode) as f:
        for i, vid in enumerate(ids, 1):
            if vid in done:
                continue
            try:
                row = transcribe(vid, asr, batch_size=args.batch_size,
                                 workers=args.workers,
                                 timestamp_mode=args.timestamp_mode)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{i}/{len(ids)}] {vid}: {len(row['chunks'])} chunks", flush=True)
            except Exception as exc:
                print(f"[{i}/{len(ids)}] {vid}: ERROR {exc}", flush=True)


def tokenize(text: str) -> list[str]:
    return [x for x in re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)
            if len(x) > 1]


def frame_at_time(video_id: str, seconds: float | None) -> int:
    csv = MAP_DIR / f"{video_id}.csv"
    if seconds is None or not csv.exists():
        return 0
    rows = []
    with csv.open() as f:
        next(f, None)
        for line in f:
            p = line.strip().split(",")
            if len(p) >= 4:
                rows.append((float(p[1]), int(p[3])))
    if not rows:
        return 0
    pos = bisect_left([x[0] for x in rows], seconds)
    pos = min(pos, len(rows) - 1)
    if pos and abs(rows[pos - 1][0] - seconds) < abs(rows[pos][0] - seconds):
        pos -= 1
    return rows[pos][1]


def search(args):
    from rank_bm25 import BM25Okapi

    rows = [json.loads(x) for x in Path(args.index).read_text().splitlines() if x.strip()]
    chunks, owners = [], []
    for row in rows:
        for chunk in row.get("chunks", []):
            toks = tokenize(chunk.get("text", ""))
            if toks:
                chunks.append(toks)
                owners.append((row["video_id"], chunk))
    if not chunks:
        return
    bm25 = BM25Okapi(chunks)
    scores = bm25.get_scores(tokenize(args.query))
    best = defaultdict(lambda: (-1.0, None))
    for score, (vid, chunk) in zip(scores, owners):
        if score > best[vid][0]:
            best[vid] = (float(score), chunk)
    ranked = sorted(best.items(), key=lambda x: -x[1][0])[:args.top_k]
    for rank, (vid, (score, chunk)) in enumerate(ranked, 1):
        frame = frame_at_time(vid, chunk.get("start"))
        print(json.dumps({"rank": rank, "video_id": vid, "frame_id": frame,
                          "score": score, "chunk": chunk}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--model", default="vinai/PhoWhisper-tiny")
    b.add_argument("--device", default="cuda")
    b.add_argument("--batch-size", type=int, default=8,
                   help="number of 30s ASR chunks decoded together")
    b.add_argument("--workers", type=int, default=0,
                   help="pipeline preprocessing workers (8 needs a free server)")
    b.add_argument("--timestamp-mode", choices=("word", "chunk"), default="word",
                   help="word timestamps are precise; chunk timestamps are faster")
    b.add_argument("--limit", type=int)
    b.add_argument("--video")
    b.add_argument("--overwrite", action="store_true")
    b.set_defaults(func=build)
    s = sub.add_parser("search")
    s.add_argument("--index", required=True)
    s.add_argument("--query", required=True)
    s.add_argument("--top-k", type=int, default=100)
    s.set_defaults(func=search)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
