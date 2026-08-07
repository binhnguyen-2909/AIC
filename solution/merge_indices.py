#!/usr/bin/env python3
"""Atomically merge disjoint resumable ASR/OCR JSONL shards.

Each input must contain one row per video. Duplicate video IDs are rejected
unless the serialized rows are identical. When media-info is available, the
merged output is also checked against the complete expected video list.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _rows(path: Path):
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if not row.get("video_id"):
                raise ValueError(f"missing video_id in {path}")
            yield row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--inputs", nargs="+", required=True, type=Path)
    ap.add_argument("--media-info", type=Path,
                    help="directory of <video_id>.json for completeness check")
    args = ap.parse_args()

    by_id = {}
    for path in args.inputs:
        if not path.exists():
            raise SystemExit(f"missing shard: {path}")
        for row in _rows(path):
            vid = str(row["video_id"])
            canonical = json.dumps(row, ensure_ascii=False, sort_keys=True)
            if vid in by_id and by_id[vid][0] != canonical:
                raise SystemExit(f"conflicting duplicate video_id={vid}")
            by_id[vid] = (canonical, row)

    expected = None
    if args.media_info:
        expected = {p.stem for p in args.media_info.glob("*.json")}
        missing = sorted(expected - set(by_id))
        extra = sorted(set(by_id) - expected)
        if missing or extra:
            raise SystemExit(
                f"incomplete merge: rows={len(by_id)} expected={len(expected)} "
                f"missing={len(missing)} extra={len(extra)}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    with tmp.open("w") as f:
        for vid in sorted(by_id):
            f.write(json.dumps(by_id[vid][1], ensure_ascii=False) + "\n")
    os.replace(tmp, args.out)
    print(f"merged={len(by_id)} output={args.out}")


if __name__ == "__main__":
    main()
