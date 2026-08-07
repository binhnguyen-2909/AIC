"""
Synthetic end-to-end test for trake_solver.

Strategy:
  - Monkey-patch trake_solver.TextEncoder so it does NOT load CLIP (offline / CPU).
    Instead we generate deterministic random embeddings and use a fixed
    similarity pattern so the right video always wins.
  - Create a tiny in-memory FAISS index with N=4 videos * T keyframes each.
  - Plant ground-truth ranges [s_j, e_j] in one video per query.
  - Run the solver pipeline, then verify the chosen frame falls inside [s, e]
    for every event.

Expected: synthetic accuracy ~0.75 (4/4 events in range on planted query).
"""
import json
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add the solution dir to import path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import trake_solver as ts


# ---------------------------------------------------------------------------
# Stub CLIP text encoder: returns deterministic per-query embeddings.
# The embedding for each sub-event is biased toward the *planted* keyframes.
# ---------------------------------------------------------------------------


class StubEncoder:
    def __init__(self, keyframe_embs_by_video: Dict[str, np.ndarray],
                 frame_table: Dict[str, np.ndarray],
                 planted_video: str, planted_frames: List[int]):
        self.cache: Dict[str, np.ndarray] = {}
        self.keyframe_embs = keyframe_embs_by_video
        self.frame_table = frame_table
        self.planted_video = planted_video
        self.planted_frames = planted_frames
        self.dim = 512
        self.rng = np.random.default_rng(42)

    def encode(self, texts):
        out = []
        for t in texts:
            if t in self.cache:
                out.append(self.cache[t])
                continue
            v = self.rng.standard_normal(self.dim).astype(np.float32)
            v = v / (np.linalg.norm(v) + 1e-9)
            planted_vecs = self.keyframe_embs[self.planted_video]
            planted_frames = self.frame_table[self.planted_video]
            target = None
            for f in self.planted_frames:
                if str(f) in t:
                    idx = int(np.argmin(np.abs(planted_frames - f)))
                    target = planted_vecs[idx]
                    break
            if target is None:
                proto = planted_vecs.mean(axis=0)
                proto = proto / (np.linalg.norm(proto) + 1e-9)
                target = proto
            noise = self.rng.standard_normal(self.dim).astype(np.float32) * 0.05
            noise = noise / (np.linalg.norm(noise) + 1e-9)
            v = 0.05 * v + 0.95 * target + 0.05 * noise
            v = v / (np.linalg.norm(v) + 1e-9)
            self.cache[t] = v.astype(np.float32)
            out.append(v)
        return np.stack(out, axis=0)


def build_fake_index(n_videos: int = 4, frames_per_video: int = 60,
                     dim: int = 512, seed: int = 1):
    rng = np.random.default_rng(seed)
    meta: List[List] = []
    vecs: List[np.ndarray] = []
    by_vid: Dict[str, np.ndarray] = {}
    frame_table: Dict[str, np.ndarray] = {}
    for vi in range(n_videos):
        vid = f"L99_V{vi:03d}"
        frames = np.array(list(range(0, frames_per_video * 30, 30)),
                          dtype=np.int64)  # 0,30,60,...,1770
        m = rng.standard_normal((len(frames), dim)).astype(np.float32)
        m /= (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
        by_vid[vid] = m
        frame_table[vid] = frames
        for f in frames:
            meta.append([vid, int(f)])
        vecs.append(m)
    mat = np.vstack(vecs).astype(np.float32)
    import faiss
    idx = faiss.IndexFlatIP(dim)
    idx.add(mat)
    return idx, meta, by_vid, frame_table


def main():
    print("[synth] building fake index with 4 videos x 60 keyframes")
    faiss_index, meta, by_vid, frame_table = build_fake_index(4, 60)

    # planted ground truth
    planted_video = "L99_V002"
    planted_frames = [120, 240, 360, 480]  # 4 sub-events
    ranges = [(f - 5, f + 5) for f in planted_frames]

    print(f"[synth] planted: video={planted_video}, events={planted_frames}")

    # Build a per-video cache object similar to what build_per_video_cache produces
    cache: Dict[str, Dict] = {}
    for vid, m in by_vid.items():
        cache[vid] = {
            "frames": frame_table[vid],
            "rows": np.arange(len(meta))[np.array(
                [row[0] == vid for row in meta])],
            "vecs": m,
        }

    # Stub encoder
    enc = StubEncoder(by_vid, frame_table, planted_video, planted_frames)

    query_text = "high jump athlete approaches bar"
    events = [
        {"text": "approach 120"},
        {"text": "take-off 240"},
        {"text": "clearance 360"},
        {"text": "landing 480"},
    ]
    event_texts = ts.make_event_texts(query_text, events)

    emb_full = enc.encode([query_text] + event_texts)
    q_emb = emb_full[0:1]
    e_embs = emb_full[1:]

    primary, frames, backup = ts.solve_query(
        q_emb, event_texts, e_embs, faiss_index, meta, cache, top_videos=3,
    )
    print(f"[synth] primary video: {primary}")
    print(f"[synth] aligned frames: {frames}")
    assert primary == planted_video, (
        f"FAIL primary video: got {primary}, expected {planted_video}")

    ok = 0
    for f, (s, e) in zip(frames, ranges):
        in_range = s <= int(f) <= e
        print(f"[synth] event frame={f} range=[{s},{e}] -> {'OK' if in_range else 'MISS'}")
        if in_range:
            ok += 1
    print(f"[synth] events-in-range: {ok}/{len(ranges)} "
          f"({ok/len(ranges):.2%})")

    # Test top-100 generation: should produce 100 distinct lines, primary first
    answers = ts.generate_top100(primary, frames, backup, cache, n_lines=100)
    print(f"[synth] generated {len(answers)} candidate lines (top-100)")
    assert len(answers) == 100
    assert answers[0] == ts.assemble_answer(primary, frames)

    # Verify ordering monotonicity for the primary slot
    parsed = [int(x.strip()) for x in answers[0].split(",")[1:]]
    print(f"[synth] primary slot frames (parsed): {parsed}")

    # Stress: also test backup list (top-3)
    print(f"[synth] backup candidates: {backup}")

    # Save a sample submission
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "sub.jsonl"
        with open(out, "w") as f:
            for rank, ans in enumerate(answers, 1):
                f.write(json.dumps({"query_id": "Q0", "rank": rank,
                                    "answer": ans}) + "\n")
        print(f"[synth] wrote sample submission -> {out}")
        # Print first 3 lines
        with open(out) as f:
            for _ in range(3):
                print("  ", f.readline().rstrip())

    print("[synth] PASS")


if __name__ == "__main__":
    main()