#!/usr/bin/env python3
"""Optional OCR index for AIC keyframes.

OCR is kept separate from the production visual index.  It is useful for
lower-thirds, signs, names and scoreboards that CLIP/object detectors do not
represent well.  The output can be consumed by the optional OCR branch in
``ensemble.py`` after a complete run.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXTRACTED = ROOT / "extracted"
MAP_DIR = EXTRACTED / "map-keyframes-aic25-b1" / "map-keyframes"
MEDIA = EXTRACTED / "media-info-aic25-b1" / "media-info"


def natural(path: Path):
    return (int(path.stem) if path.stem.isdigit() else path.name, path.name)


def frame_manifest(video_id: str) -> list[tuple[int, Path]]:
    batch = video_id.split("_")[0]
    dirs = sorted(EXTRACTED.glob(f"Keyframes_{batch}*/keyframes/{video_id}"))
    images = []
    for d in dirs:
        images.extend(d.glob("*.jpg"))
    images.sort(key=natural)
    csv = MAP_DIR / f"{video_id}.csv"
    frame_ids = []
    if csv.exists():
        with csv.open() as f:
            next(f, None)
            for line in f:
                p = line.strip().split(",")
                if len(p) >= 4:
                    frame_ids.append(int(p[3]))
    n = min(len(images), len(frame_ids))
    return [(frame_ids[i], images[i]) for i in range(n)]


def all_video_ids():
    return sorted(p.stem for p in MEDIA.glob("*.json"))


def build(args):
    import easyocr
    import torch

    reader = easyocr.Reader(["vi", "en"], gpu=args.device.startswith("cuda"),
                            verbose=False)
    ids = [args.video] if args.video else all_video_ids()
    if args.limit:
        ids = ids[:args.limit]
    out = Path(args.out)
    done = set()
    if out.exists() and not args.overwrite:
        with out.open() as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["video_id"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w" if args.overwrite else "a") as f:
        for i, vid in enumerate(ids, 1):
            if vid in done:
                continue
            frames = frame_manifest(vid)
            if args.sample and len(frames) > args.sample:
                picks = np.linspace(0, len(frames) - 1, args.sample, dtype=np.int64)
                frames = [frames[int(j)] for j in picks]
            row = {"video_id": vid, "frames": []}
            for start in range(0, len(frames), args.batch_size):
                chunk = frames[start:start + args.batch_size]
                paths = [str(image) for _frame_id, image in chunk]
                try:
                    hit_batches = reader.readtext_batched(
                        paths, batch_size=args.recognizer_batch_size,
                        workers=args.workers, detail=1, paragraph=False,
                    )
                except Exception as exc:
                    if isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower():
                        # Do not fall back to hundreds of one-image calls
                        # after a CUDA OOM: the allocator is already under
                        # pressure and that path can produce a misleading
                        # partial row. Resume later with a smaller batch.
                        raise
                    # Some corpora contain mixed resolutions. Keep the full
                    # run robust by falling back to the original per-image
                    # path for only the affected batch.
                    print(f"OCR batch fallback {vid} rows={start}:{start + len(chunk)}: {exc}", flush=True)
                    hit_batches = []
                    for _frame_id, image in chunk:
                        try:
                            hit_batches.append(reader.readtext(
                                str(image), detail=1, paragraph=False))
                        except Exception as inner:
                            print(f"OCR error {vid}/{_frame_id}: {inner}", flush=True)
                            hit_batches.append([])
                for (frame_id, _image), hits in zip(chunk, hit_batches):
                    texts = []
                    for _box, text, conf in hits:
                        text = re.sub(r"\s+", " ", str(text)).strip()
                        if text and float(conf) >= args.min_conf:
                            texts.append(text)
                    if texts:
                        row["frames"].append({"frame_idx": frame_id,
                                              "text": " ".join(texts)})
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(ids)}] {vid}: {len(row['frames'])} text frames", flush=True)


def tokenize(text):
    return [x for x in re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)
            if len(x) > 1]


def search(args):
    from rank_bm25 import BM25Okapi

    rows = [json.loads(x) for x in Path(args.index).read_text().splitlines() if x.strip()]
    docs, owners = [], []
    for row in rows:
        for frame in row.get("frames", []):
            toks = tokenize(frame.get("text", ""))
            if toks:
                docs.append(toks)
                owners.append((row["video_id"], frame))
    if not docs:
        return
    scores = BM25Okapi(docs).get_scores(tokenize(args.query))
    best = {}
    for score, owner in zip(scores, owners):
        if score > best.get(owner[0], (-1.0, None))[0]:
            best[owner[0]] = (float(score), owner[1])
    for rank, (vid, (score, frame)) in enumerate(
            sorted(best.items(), key=lambda x: -x[1][0])[:args.top_k], 1):
        print(json.dumps({"rank": rank, "video_id": vid,
                          "frame_id": frame["frame_idx"], "score": score,
                          "text": frame["text"]}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--device", default="cuda")
    b.add_argument("--limit", type=int)
    b.add_argument("--sample", type=int, help="max frames per video")
    b.add_argument("--batch-size", type=int, default=32,
                   help="images sent to EasyOCR detector at once")
    b.add_argument("--recognizer-batch-size", type=int, default=8,
                   help="recognition batch size inside EasyOCR")
    b.add_argument("--workers", type=int, default=0,
                   help="EasyOCR recognition workers")
    b.add_argument("--video")
    b.add_argument("--min-conf", type=float, default=0.25)
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
