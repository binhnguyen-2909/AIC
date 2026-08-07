"""Rebuild retrieval artifacts from one extracted-data snapshot.

This uses the deduplicated ``(video_id, frame_idx)`` index and rebuilds the
per-video/object caches against the resulting metadata, avoiding silent frame
shifts caused by repeated frame IDs in source CSVs.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-objects", action="store_true")
    args = ap.parse_args()

    subprocess.run([sys.executable, str(HERE / "clean_index.py")], check=True)

    cache = HERE / "index" / "per_video.pkl"
    if cache.exists():
        cache.unlink()
    sys.path.insert(0, str(HERE))
    from trake_solver import build_per_video_cache
    per_video = build_per_video_cache()
    print(f"per_video: {len(per_video)} videos")

    if not args.skip_objects:
        from build_object_index import main as build_objects
        build_objects()
    print("Consistent retrieval artifacts rebuilt successfully.")


if __name__ == "__main__":
    main()
